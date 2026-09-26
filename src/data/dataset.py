"""Training chunks (x0, mel, s, b) read from the caches (design doc §4.4-4.5).
The file layout is in src/data/cache.py.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.cache import FRAMES_PER_CHUNK, N_MELS, read_manifest


class ChunkDataset(Dataset):
    """Every chunk of every kept chart (manifest drop == "") in the given splits
    that has an SR label, tokens and mel. s comes from the manifest, so a new
    label only needs a new manifest, not a new token cache.

    item: {"x0": long [384, 4], "mel": float [1536, 80], "s": float [], "b": float []}
    Each chunk appears once per epoch. Mel files are memory-mapped, not loaded.
    """

    def __init__(self, manifest: Path, cache: Path, splits=("train",), *,
                 keys: list[str] | None = None, max_charts: int | None = None):
        rows = [r for r in read_manifest(manifest) if r["split"] in splits and r["sr"] != ""
                and r.get("drop", "") == ""]
        if keys is not None:
            wanted = set(keys)
            rows = [r for r in rows if r["key"] in wanted]
        self.tok_dir, self.mel_dir = Path(cache) / "tokens", Path(cache) / "mel"
        rows = [r for r in rows if (self.tok_dir / f"{r['key']}.npz").exists()
                and (self.mel_dir / f"{r['key']}.npy").exists()]
        if max_charts is not None:
            rows = rows[:max_charts]

        self.charts, self.index = [], []
        for r in rows:
            z = np.load(self.tok_dir / f"{r['key']}.npz")
            tokens = z["tokens"]
            frames = np.load(self.mel_dir / f"{r['key']}.npy", mmap_mode="r").shape
            if frames != (len(tokens) * FRAMES_PER_CHUNK, N_MELS):
                raise ValueError(f"mel for {r['key']} has shape {frames}, expected "
                                 f"({len(tokens) * FRAMES_PER_CHUNK}, {N_MELS})")
            ci = len(self.charts)
            self.charts.append({"key": r["key"], "tokens": tokens,
                                "b": z["beat_len_ms"], "s": float(r["sr"])})
            self.index += [(ci, c) for c in range(len(tokens))]
        self._mel: dict[int, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> dict:
        ci, c = self.index[i]
        chart = self.charts[ci]
        if ci not in self._mel:
            self._mel[ci] = np.load(self.mel_dir / f"{chart['key']}.npy", mmap_mode="r")
        frames = self._mel[ci][c * FRAMES_PER_CHUNK:(c + 1) * FRAMES_PER_CHUNK]
        return {
            "x0": torch.from_numpy(chart["tokens"][c].astype(np.int64)),
            "mel": torch.from_numpy(np.asarray(frames, dtype=np.float32)),
            "s": torch.tensor(chart["s"], dtype=torch.float32),
            "b": torch.tensor(float(chart["b"][c]), dtype=torch.float32),
        }

    def __getstate__(self) -> dict:                  # DataLoader workers reopen the memmaps
        state = self.__dict__.copy()
        state["_mel"] = {}
        return state
