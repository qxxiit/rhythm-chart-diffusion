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
    dynamics(tokens, mel, n)     how the chart's density follows the music's loudness:
        loud_slope   slope of each bar's onsets / the song's mean bar on the bar's
                     loudness z-score (mean of the song-standardized log-Mel, z over the
                     song's bars); 40 val songs: human 0.22, fwd + ref2 + copies 0.20
        light_bars   share of the bars with notes that have at most 30% of the song's
                     90th-percentile bar: the rests inside a song
    quiet_rhythm(gen, human, mel, n)   against the human chart, by the bar's loudness z
                 (as in dynamics), EXPERIMENTS 2026-10-06 (quiet sections):
        off_rhythm_quiet, off_rhythm_rest
                     of the generated onset rows in quiet bars (z < -1) / in the other
                     bars, the share with no human onset row within one cell (1/12 beat);
                     fwd + ref2 on 40 val songs: 30% in quiet bars, 7% in the loudest
        empty_bar_fill
                     of the bars between the human chart's first and last onset where the
                     human chart has none, the share where the generated chart has one
                     (1.2% of the bars; the sampled charts fill 83-92% of them)
    bar_kinds(tokens, n)         choices a chart makes for a whole bar, which cell-by-cell
                 sampling makes less often than humans (240 val songs, 2026-10-07: human /
                 fwd + ref2 + gate + loudness 0.1 + lane guidance 2):
        rest_bars    of the bars between the chart's first and last onset, the share with
                     none (0.85% of the human bars)
        ln_bars      of the bars with LN_BAR_MIN+ note starts, the share where at least
                     LN_BAR of them start long notes (7.5% / 3.8%)
        steady_bars  of the bars with STEADY_MIN+ onset rows, the share where they are
                     evenly spaced: an unbroken stream (36.5% / 20.1%)

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
    # numpy's Accelerate BLAS on Apple Silicon raises bogus divide/overflow/invalid
    # flags in matmul; silence them, but check the result for real.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        s = u @ u.T
    if not np.isfinite(s).all():
        raise FloatingPointError("non-finite similarity")
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


def dynamics(tokens: np.ndarray, mel: np.ndarray, n_cells: int) -> dict:
    """loud_slope and light_bars (module docstring); nan for songs under 8 bars."""
    n_bars = n_whole_bars(n_cells)
    nan = {"loud_slope": float("nan"), "light_bars": float("nan")}
    if n_bars < 8:
        return nan
    x = np.asarray(tokens).reshape(-1, np.asarray(tokens).shape[-1])[:n_bars * BAR]
    per_bar = np.isin(x, (TAP, HOLD_START)).reshape(n_bars, -1).sum(axis=1).astype(np.float64)
    frames = np.asarray(mel[:n_bars * BAR * FRAMES_PER_CELL], dtype=np.float64)
    loud = frames.reshape(n_bars, BAR * FRAMES_PER_CELL, -1).mean(axis=(1, 2))
    if per_bar.mean() == 0 or loud.std() == 0:
        return nan
    z = (loud - loud.mean()) / loud.std()
    slope = float(np.polyfit(z, per_bar / per_bar.mean(), 1)[0])
    full = per_bar[per_bar > 0]
    light = float(np.mean(full <= 0.3 * np.percentile(full, 90))) if len(full) else float("nan")
    return {"loud_slope": slope, "light_bars": light}


LN_BAR = 0.8                    # bar_kinds: a long-note bar starts at least this share
LN_BAR_MIN = 4                  # of at least this many notes as long notes
STEADY_MIN = 8                  # bar_kinds: a steady bar has at least this many onset rows


def bar_kinds(tokens: np.ndarray, n_cells: int) -> dict:
    """rest_bars, ln_bars, steady_bars (module docstring); nan for songs under 8 bars or
    without such bars to count."""
    out = dict.fromkeys(("rest_bars", "ln_bars", "steady_bars"), float("nan"))
    n_bars = n_whole_bars(n_cells)
    if n_bars < 8:
        return out
    x = np.asarray(tokens).reshape(-1, np.asarray(tokens).shape[-1])[:n_bars * BAR]
    starts = np.isin(x, (TAP, HOLD_START)).reshape(n_bars, BAR, -1)
    rows = starts.any(axis=2)
    n_start = starts.sum(axis=(1, 2))
    n_long = (x == HOLD_START).reshape(n_bars, BAR, -1).sum(axis=(1, 2))
    full = np.flatnonzero(rows.any(axis=1))
    if len(full) >= 2:
        out["rest_bars"] = float(np.mean(~rows[full[0]:full[-1] + 1].any(axis=1)))
    busy = n_start >= LN_BAR_MIN
    if busy.any():
        out["ln_bars"] = float(np.mean(n_long[busy] >= LN_BAR * n_start[busy]))
    steady = [len(set(np.diff(np.flatnonzero(r)).tolist())) == 1
              for r in rows if r.sum() >= STEADY_MIN]
    if steady:
        out["steady_bars"] = float(np.mean(steady))
    return out


QUIET_Z = -1.0                  # quiet_rhythm: bars this far below the song's mean loudness


def quiet_rhythm(gen: np.ndarray, human: np.ndarray, mel: np.ndarray, n_cells: int,
                 quiet_z: float = QUIET_Z) -> dict:
    """off_rhythm_quiet, off_rhythm_rest, empty_bar_fill (module docstring); nan for songs
    under 8 bars, or where a part has no generated onsets / the human chart no empty bar."""
    keys = ("off_rhythm_quiet", "off_rhythm_rest", "empty_bar_fill")
    out = dict.fromkeys(keys, float("nan"))
    n_bars = n_whole_bars(n_cells)
    if n_bars < 8:
        return out
    rows = n_bars * BAR

    def onset_rows(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x).reshape(-1, np.asarray(x).shape[-1])[:rows]
        return np.isin(x, (TAP, HOLD_START)).any(axis=1)

    g, h = onset_rows(gen), onset_rows(human)
    frames = np.asarray(mel[:rows * FRAMES_PER_CELL], dtype=np.float64)
    if len(frames) < rows * FRAMES_PER_CELL:
        return out
    loud = frames.reshape(n_bars, BAR * FRAMES_PER_CELL, -1).mean(axis=(1, 2))
    if loud.std() == 0:
        return out
    quiet = np.repeat((loud - loud.mean()) / loud.std() < quiet_z, BAR)
    near = h.copy()
    near[1:] |= h[:-1]
    near[:-1] |= h[1:]
    off = g & ~near
    for key, part in (("off_rhythm_quiet", quiet), ("off_rhythm_rest", ~quiet)):
        if (g & part).any():
            out[key] = float((off & part).sum() / (g & part).sum())
    hb, gb = h.reshape(n_bars, BAR).any(axis=1), g.reshape(n_bars, BAR).any(axis=1)
    full = np.flatnonzero(hb)
    if len(full) >= 2:
        inside = np.zeros(n_bars, dtype=bool)
        inside[full[0]:full[-1] + 1] = True
        empty = inside & ~hb
        if empty.any():
            out["empty_bar_fill"] = float((empty & gb).sum() / empty.sum())
    return out


def lag_profile(s: np.ndarray, max_lag: int) -> np.ndarray:
    """Mean of S[i, i + d] for d = 1..max_lag (NaN where the song is too short)."""
    n = len(s)
    return np.array([np.diagonal(s, offset=d).mean() if d < n else np.nan
                     for d in range(1, max_lag + 1)])
