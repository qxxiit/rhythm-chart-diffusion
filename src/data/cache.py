"""Cache file layout shared by the token stage, the mel stage and the Dataset.
No torch here, so preprocessing runs without it.

One pair of files per chart, named by the manifest key (scripts/build_manifest.py):

    data/cache/tokens/{key}.npz     written by scripts/preprocess_data.py
        tokens        int8    [n_chunks, 384, 4]
        start_cell    int64   [n_chunks]      BeatGrid cell of each chunk's row 0
        beat_len_ms   float32 [n_chunks]      b
        sr            float32 []              the label when the cache was built; the
                                              Dataset reads s from the manifest instead
        cell_offset   int64   []
        n_cells       int64   []              rows before PAD
        timing_points float64 [n_tp, 2]       (time_ms, ms_per_beat)

    data/cache/mel/{key}.npy        written by the mel pipeline (design doc §3)
        float16 [n_chunks * 1536, 80]: log-Mel of the whole chart, frame f at
        from_timing_points(timing_points).frame_times(start_cell[0], n_chunks * 384)[f].
        Frames 4r .. 4r+3 belong to token row r. Row 0 can be before 0 ms and
        the last chunk runs past the song, so frames outside the audio hold the
        silence value. About 3 MB per chart, ~55 GB for the whole set; charts
        with a non-empty manifest `drop` are never read and can be skipped.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from src.data.tokenizer import EncodeStats, K, L

FRAMES_PER_CELL = 4
N_MELS = 80
FRAMES_PER_CHUNK = L * FRAMES_PER_CELL        # 1536


def read_manifest(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save_tokens(path: Path, tokens: np.ndarray, metas, stats: EncodeStats) -> None:
    np.savez_compressed(
        path,
        tokens=tokens.astype(np.int8),
        start_cell=np.array([m.start_cell for m in metas], dtype=np.int64),
        beat_len_ms=np.array([m.beat_len_ms for m in metas], dtype=np.float32),
        sr=np.float32(metas[0].sr),
        cell_offset=np.int64(metas[0].cell_offset),
        n_cells=np.int64(stats.n_cells),
        timing_points=np.array(metas[0].timing_points, dtype=np.float64).reshape(-1, 2),
    )


def oracle_mel(tokens: np.ndarray, seed: int = 0) -> np.ndarray:
    """Stand-in mel that ENCODES THE ANSWER: each lane lights its own band of
    mel bins by token type. For checking that the pipeline runs end to end,
    never for results.  [n_chunks * 1536, 80] float16."""
    level = np.array([0.0, 6.0, 6.0, 2.0, 4.0, 0.0, 0.0], dtype=np.float32)   # by token id
    rows = tokens.reshape(-1, K).astype(np.int64)
    rng = np.random.default_rng(seed)
    mel = rng.normal(-8.0, 0.5, size=(len(rows) * FRAMES_PER_CELL, N_MELS)).astype(np.float32)
    for k in range(K):
        mel[:, 8 + 16 * k:20 + 16 * k] += np.repeat(level[rows[:, k]], FRAMES_PER_CELL)[:, None]
    return mel.astype(np.float16)
