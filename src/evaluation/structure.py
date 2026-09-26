"""Structure metric rho (design doc §4.11-2): does the chart repeat where the music repeats?

    rho = corr( triu(S_audio), triu(S_chart) )   over bar pairs i < j

    ssm_audio(mel, n_bars)       cosine between bars of song-standardized log-Mel
    ssm_chart(tokens, n_bars)    cosine between bars' note patterns
    structure_scores(...)        rho over all pairs and over the pair sets of §4.11:
        in     both bars in the same chunk (the fixed 8-bar blocks from row 0)
        cross  bars in different chunks
        far    j - i >= k bars (k: where the adjacency effect fades in human charts,
               see lag_profile and scripts/human_baselines.py)
    lag_profile(S)               mean similarity of bar pairs at each distance

Bars are 48 cells (4/4). Row 0 of the token grid is a bar line (tokenizer), so
bar m is rows 48m..48m+47 and mel frames 192m..192m+191, and chunk c is bars
8c..8c+7, the same blocks for human, AR and diffusion charts. Only whole bars
before PAD count. Per song, all pairs of a set are pooled into one rho (a chunk
alone has 28 pairs).

Choices made here, to confirm before results (§4.11 leaves them open):
  audio vector   the bar's log-Mel after per-bin standardization over the song,
                 averaged over the 4 frames of each cell: [48 cells x 80 bins]
  chart vector   "notes": onset cells (tap, hold start) and hold cells (start,
                 body, end) of the bar, [48 x 4 x 2] binary.
                 "types": notes per pattern rule in the bar (patterns.py).
  empty bars     two empty bars have similarity 1, an empty and a non-empty 0
                 (the rule proposed in §4.11)
"""

from __future__ import annotations

import numpy as np

from src.data.tokenizer import HOLD_BODY, HOLD_END, HOLD_START, TAP

BAR = 48                    # cells per bar (4/4)
FRAMES_PER_CELL = 4
CHUNK_BARS = 8              # 384 cells


def n_whole_bars(n_cells: int) -> int:
    return int(n_cells) // BAR


def _cosine(v: np.ndarray) -> np.ndarray:
    """[n, d] -> [n, n] cosine similarity; zero rows follow the empty-bar rule."""
    norm = np.linalg.norm(v, axis=1)
    empty = norm == 0
    u = v / np.where(empty, 1.0, norm)[:, None]
    s = u @ u.T
    s[np.ix_(empty, empty)] = 1.0          # empty vs empty: same
    return s                               # empty vs non-empty: 0 already (zero row)


def ssm_audio(mel: np.ndarray, n_bars: int) -> np.ndarray:
    """mel [n_frames, F] on the token grid (frame 4r + j belongs to row r)."""
    frames = np.asarray(mel[:n_bars * BAR * FRAMES_PER_CELL], dtype=np.float64)
    if len(frames) < n_bars * BAR * FRAMES_PER_CELL:
        raise ValueError("mel is shorter than the bars it should cover")
    z = (frames - frames.mean(axis=0)) / (frames.std(axis=0) + 1e-5)
    cells = z.reshape(n_bars * BAR, FRAMES_PER_CELL, -1).mean(axis=1)
    return _cosine(cells.reshape(n_bars, -1))


def chart_vectors(tokens: np.ndarray, n_bars: int) -> np.ndarray:
    """Note-level pattern vector per bar: [n_bars, 48 * 4 * 2]."""
    x = np.asarray(tokens).reshape(-1, np.asarray(tokens).shape[-1])[:n_bars * BAR]
    onset = np.isin(x, (TAP, HOLD_START))
    hold = np.isin(x, (HOLD_START, HOLD_BODY, HOLD_END))
    return np.stack([onset, hold], axis=-1).reshape(n_bars, -1).astype(np.float64)


def ssm_chart(tokens: np.ndarray, n_bars: int, kind: str = "notes") -> np.ndarray:
    if kind == "notes":
        return _cosine(chart_vectors(tokens, n_bars))
    if kind == "types":
        from src.evaluation.patterns import bar_type_vectors
        return _cosine(bar_type_vectors(tokens, n_bars))
    raise ValueError(f"unknown chart vector kind {kind!r}")


def pair_sets(n_bars: int, far_k: int) -> dict[str, np.ndarray]:
    """Boolean [n_bars, n_bars] masks over i < j."""
    i, j = np.triu_indices(n_bars, k=1)
    upper = np.zeros((n_bars, n_bars), dtype=bool)
    upper[i, j] = True
    chunk = np.arange(n_bars) // CHUNK_BARS
    same_chunk = chunk[:, None] == chunk[None, :]
    dist = np.arange(n_bars)[None, :] - np.arange(n_bars)[:, None]
    return {"all": upper, "in": upper & same_chunk, "cross": upper & ~same_chunk,
            "far": upper & (dist >= far_k)}


def rho(s_audio: np.ndarray, s_chart: np.ndarray, pairs: np.ndarray) -> float:
    """Pearson correlation of the two similarity matrices over the chosen pairs.
    NaN with fewer than 3 pairs or when either side is constant."""
    a, c = s_audio[pairs], s_chart[pairs]
    if a.size < 3 or a.std() == 0 or c.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, c)[0, 1])


def structure_scores(tokens: np.ndarray, mel: np.ndarray, n_cells: int, *,
                     far_k: int = 8, kind: str = "notes") -> dict:
    """rho_all / rho_in / rho_cross / rho_far for one song, with pair counts."""
    n_bars = n_whole_bars(n_cells)
    out = {"n_bars": n_bars}
    if n_bars < 3:
        return {**out, **{f"rho_{k}": float("nan") for k in ("all", "in", "cross", "far")}}
    s_a, s_c = ssm_audio(mel, n_bars), ssm_chart(tokens, n_bars, kind)
    for name, mask in pair_sets(n_bars, far_k).items():
        out[f"rho_{name}"] = rho(s_a, s_c, mask)
        out[f"pairs_{name}"] = int(mask.sum())
    return out


def lag_profile(s: np.ndarray, max_lag: int) -> np.ndarray:
    """Mean of S[i, i + d] for d = 1..max_lag (NaN where the song is too short)."""
    n = len(s)
    return np.array([np.diagonal(s, offset=d).mean() if d < n else np.nan
                     for d in range(1, max_lag + 1)])
