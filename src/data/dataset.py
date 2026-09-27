"""Training chunks (x0, mel, s, b) read from the caches (design doc §4.4-4.5).
The file layout is in src/data/cache.py.

window="chunk"  every item is one of the fixed 384-cell chunks (row 0 at 384c)
window="bar"    every item starts at a random bar line inside chunk c
                (row 0 at 384c + 48u, u = 0..7), redrawn every time the item is
                read. Generation by continuation (§4.8) starts its windows every
                6 bars, which a model trained only on the fixed chunks has
                never seen; this shows it such windows during training. A
                window that runs past the song is filled with PAD and silence.

mel comes from mel.MelStore (the real log-Mel, put on the window's grid when the
item is read) or, with fake_mel=True, from mel.FakeMelStore (plumbing only).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.beat_grid import from_timing_points
from src.data.cache import FRAMES_PER_CELL, N_MELS, read_manifest
from src.data.mel import open_mel
from src.data.tokenizer import BAR, BEATS_PER_CHUNK, PAD, D, L


class ChunkDataset(Dataset):
    """Every chunk of every kept chart (manifest drop == "") in the given splits
    that has an SR label, tokens and mel. s comes from the manifest, so a new
    label only needs a new manifest, not a new token cache.

    item: {"x0": long [384, 4], "mel": float [1536, 80], "s": float [], "b": float [],
           "start": long []}  (start = token row of the window's row 0)
    Each chunk appears once per epoch. Mel files are memory-mapped, not loaded.
    """

    def __init__(self, manifest: Path, cache: Path, splits=("train",), *,
                 keys: list[str] | None = None, max_charts: int | None = None,
                 window: str = "chunk", fake_mel: bool = False):
        if window not in ("chunk", "bar"):
            raise ValueError(f"unknown window {window!r}")
        self.window = window
        rows = [r for r in read_manifest(manifest) if r["split"] in splits and r["sr"] != ""
                and r.get("drop", "") == ""]
        if keys is not None:
            wanted = set(keys)
            rows = [r for r in rows if r["key"] in wanted]
        self.tok_dir = Path(cache) / "tokens"
        self.mel = open_mel(cache, fake=fake_mel)
        rows = [r for r in rows if (self.tok_dir / f"{r['key']}.npz").exists()
                and self.mel.has(r["key"])]
        if max_charts is not None:
            rows = rows[:max_charts]

        self.charts, self.index = [], []
        for r in rows:
            z = np.load(self.tok_dir / f"{r['key']}.npz")
            tokens = z["tokens"]
            if fake_mel:
                frames = np.load(self.mel.dir / f"{r['key']}.npy", mmap_mode="r").shape
                if frames != (len(tokens) * L * FRAMES_PER_CELL, N_MELS):
                    raise ValueError(f"fake mel for {r['key']} has shape {frames}, expected "
                                     f"({len(tokens) * L * FRAMES_PER_CELL}, {N_MELS})")
            ci = len(self.charts)
            self.charts.append({
                "key": r["key"], "rows": tokens.reshape(-1, tokens.shape[-1]),
                "n_cells": int(z["n_cells"]), "b": z["beat_len_ms"], "s": float(r["sr"]),
                "cell_offset": int(z["cell_offset"]),
                "timing_points": [tuple(tp) for tp in z["timing_points"]],
            })
            self.index += [(ci, c) for c in range(len(tokens))]
        self._grids: dict[int, object] = {}

    def __len__(self) -> int:
        return len(self.index)

    def _start(self, chart: dict, c: int) -> int:
        start = c * L
        if self.window == "bar":
            last_bar = min(L // BAR - 1, (chart["n_cells"] - 1 - start) // BAR)
            start += BAR * int(torch.randint(0, last_bar + 1, ()))
        return start

    def _grid(self, ci: int):
        if ci not in self._grids:
            self._grids[ci] = from_timing_points(self.charts[ci]["timing_points"])
        return self._grids[ci]

    def _beat_len(self, ci: int, start: int) -> float:
        grid = self._grid(ci)
        cell = start - self.charts[ci]["cell_offset"]
        return float(grid.time_from_cell(cell + L, D) - grid.time_from_cell(cell, D)) / BEATS_PER_CHUNK

    def __getitem__(self, i: int) -> dict:
        ci, c = self.index[i]
        chart = self.charts[ci]
        start = self._start(chart, c)

        x0 = np.full((L, chart["rows"].shape[1]), PAD, dtype=np.int64)
        part = chart["rows"][start:start + L]
        x0[:len(part)] = part
        frames = self.mel.frames(chart["key"], self._grid(ci), chart["cell_offset"], start, L)
        b = chart["b"][c] if start == c * L else self._beat_len(ci, start)
        return {
            "x0": torch.from_numpy(x0),
            "mel": torch.from_numpy(np.ascontiguousarray(frames, dtype=np.float32)),
            "s": torch.tensor(chart["s"], dtype=torch.float32),
            "b": torch.tensor(float(b), dtype=torch.float32),
            "start": torch.tensor(start, dtype=torch.long),
        }

    def __getstate__(self) -> dict:                  # DataLoader workers rebuild the grids
        state = self.__dict__.copy()
        state["_grids"] = {}
        return state
