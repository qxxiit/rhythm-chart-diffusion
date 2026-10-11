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
    phrase_repeats(tokens, n)    rhythm_rep1/2/4/8/16: bars with the rhythm (onset rows) of
                 the bar 1, 2, 4, 8, 16 before; lane_rep1: of those 1 before, also its lanes
                 (EXPERIMENTS 2026-10-10, means over the 240 val songs: human 0.27 / 0.31 /
                 0.29 / 0.23 / 0.18 and 0.06, the candidate 0.14 / 0.15 / 0.12 / 0.09 / 0.07
                 and 0.03); tap_rep1 / ln_rep1: rhythm_rep1 over bars of taps / of long
                 notes; copy_near/mid/far: bars that copy a bar 1-2, 3-8, 9+ back, lanes too
                 (2026-10-11)
    section_runs(s_audio)        the stretches of 4+ bars whose audio comes back 8+ bars later
    section_repeats(tokens, s_audio, n)
                 section_rhythm / section_copy: in those stretches, the bars with the rhythm
                 / also the lanes of the bar where the music was before (human 0.55 / 0.24,
                 the default of 10-11 0.19 / 0.07; 2026-10-11)
    ln_placement(tokens, n)      ln_sections, ln_scattered: the share of long notes in bars
                 that are mostly long notes / in bars of taps (human 0.48 / 0.19, the
                 candidate 0.34 / 0.29)

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


REPEAT_LAGS = (1, 2, 4, 8, 16)  # phrase_repeats: bar distances
REPEAT_MIN = 4                  # phrase_repeats: bars with at least this many onset rows
COPY_NEAR, COPY_FAR = 2, 9      # phrase_repeats: a whole-bar copy 1-2 bars back is near,
                                # 9 or more far, between mid
_MIRROR = np.array([int(f"{m:04b}"[::-1], 2) for m in range(16)])


def phrase_repeats(tokens: np.ndarray, n_cells: int) -> dict:
    """rhythm_rep<k> for k in REPEAT_LAGS: of the whole bars with REPEAT_MIN+ onset rows whose
    bar k earlier has them too, the share whose onset rows are the same (the rhythm; chord
    sizes and lanes aside). lane_rep1: of the bars with the rhythm of the bar just before,
    the share whose lanes are the same too, as they are or mirrored (lane k <-> 3 - k).
    Human charts come back to a rhythm 1, 2, 4, 8 and 16 bars later about twice as often as
    the sampled ones, and 2 and 4 bars later most (EXPERIMENTS 2026-10-10).
    tap_rep1, ln_rep1: rhythm_rep1 over the bars of taps only (no long-note cell) and over
    the bars that start 2+ long notes: the carry acts on the first, humans repeat both
    (EXPERIMENTS 2026-10-11).
    copy_near, copy_mid, copy_far: of the bars with REPEAT_MIN+ onset rows after the first
    bar, the share whose nearest earlier copy (a bar with REPEAT_MIN+ onset rows and the
    same lanes, as they are or mirrored) is 1 to COPY_NEAR bars back, between, COPY_FAR or
    more: humans copy whole bars mostly from far back, a section that comes back
    (EXPERIMENTS 2026-10-11). nan where there is nothing to count."""
    x = np.asarray(tokens).reshape(-1, np.asarray(tokens).shape[-1])
    n_bars = n_whole_bars(n_cells)
    out = {f"rhythm_rep{k}": float("nan") for k in REPEAT_LAGS}
    out.update({"lane_rep1": float("nan"), "tap_rep1": float("nan"), "ln_rep1": float("nan"),
                "copy_near": float("nan"), "copy_mid": float("nan"), "copy_far": float("nan")})
    if n_bars < 2:
        return out
    starts = np.isin(x[:n_bars * BAR], (TAP, HOLD_START)).reshape(n_bars, BAR, -1)
    rows = starts.any(axis=2)
    masks = (starts * (1 << np.arange(starts.shape[2]))).sum(axis=2)       # [bars, BAR]
    busy = rows.sum(axis=1) >= REPEAT_MIN
    for k in REPEAT_LAGS:
        b = np.flatnonzero(busy[k:] & busy[:-k]) + k if n_bars > k else np.array([], int)
        if len(b):
            out[f"rhythm_rep{k}"] = float(np.mean([np.array_equal(rows[i], rows[i - k])
                                                    for i in b]))
    cells = x[:n_bars * BAR].reshape(n_bars, BAR, -1)
    pairs = np.flatnonzero(busy[1:] & busy[:-1]) + 1
    kinds = {"tap_rep1": ~np.isin(cells, (HOLD_START, HOLD_BODY, HOLD_END)).any(axis=(1, 2)),
             "ln_rep1": (cells == HOLD_START).sum(axis=(1, 2)) >= 2}
    for name, kind in kinds.items():
        b = pairs[kind[pairs]]
        if len(b):
            out[name] = float(np.mean([np.array_equal(rows[i], rows[i - 1]) for i in b]))
    same = [i for i in pairs if np.array_equal(rows[i], rows[i - 1])]
    if same:
        out["lane_rep1"] = float(np.mean([np.array_equal(masks[i], masks[i - 1])
                                          or np.array_equal(masks[i], _MIRROR[masks[i - 1]])
                                          for i in same]))
    later = np.flatnonzero(busy[1:]) + 1
    if len(later):
        mirrored = _MIRROR[masks]
        dist = np.zeros(len(later), dtype=int)                  # 0: no earlier copy
        for n, i in enumerate(later):
            hit = np.flatnonzero(busy[:i] & ((masks[:i] == masks[i]).all(axis=1)
                                             | (mirrored[:i] == masks[i]).all(axis=1)))
            if len(hit):
                dist[n] = i - hit[-1]
        out["copy_near"] = float(np.mean((dist >= 1) & (dist <= COPY_NEAR)))
        out["copy_mid"] = float(np.mean((dist > COPY_NEAR) & (dist < COPY_FAR)))
        out["copy_far"] = float(np.mean(dist >= COPY_FAR))
    return out


SECTION_SIM = 0.5               # section_runs: bars whose audio cosine with the bar `lag` back
SECTION_BARS = 4                # is at least this, this many or more in a row, at a lag of
SECTION_LAG = 8                 # this many bars or more: a stretch of music that comes back


def section_runs(s_audio: np.ndarray, *, sim: float = SECTION_SIM, min_bars: int = SECTION_BARS,
                 min_lag: int = SECTION_LAG) -> list[tuple[int, int, int]]:
    """(first bar, bars, lag) of the stretches where the music comes back, in time order.

    Along each diagonal of the bars' audio similarity (ssm_audio) at a lag of min_lag or
    more, the maximal runs of bars t whose similarity with the bar t - lag is at least sim,
    min_bars or longer. The longest are kept first (then the smaller lag, then the earlier);
    a bar is in one stretch at most, so a run that cuts across those kept before keeps its
    longest free part, if that is still min_bars long. A stretch may be longer than its lag
    (a passage that repeats every lag bars). On 240 val songs (2026-10-11) 40% of the bars
    are in one, 9.7 bars a stretch, 5 stretches a song; of the human charts' copies from 9+
    bars back, the source is the bar most like it in sound 8+ bars back for 57% (78% among
    the 3 most alike)."""
    n = len(s_audio)
    found = []
    for lag in range(max(min_lag, 1), n):
        hit = np.diagonal(s_audio, offset=-lag) >= sim          # s[t, t - lag], t = lag ..
        edges = np.flatnonzero(np.diff(np.concatenate([[0], hit.astype(np.int8), [0]])))
        for a, b in zip(edges[::2], edges[1::2], strict=True):
            if b - a >= min_bars:
                found.append((int(b - a), lag, int(a) + lag))
    taken = np.zeros(n, dtype=bool)
    out = []
    for length, lag, t0 in sorted(found, key=lambda f: (-f[0], f[1], f[2])):
        best, start, run = 0, 0, 0
        for k, free in enumerate(~taken[t0:t0 + length]):
            run = run + 1 if free else 0
            if run > best:
                best, start = run, k - run + 1
        if best >= min_bars:
            taken[t0 + start:t0 + start + best] = True
            out.append((t0 + start, best, lag))
    return sorted(out)


def section_repeats(tokens: np.ndarray, s_audio: np.ndarray, n_cells: int) -> dict:
    """section_rhythm, section_copy: of the bars of the stretches where the music comes back
    (section_runs) that have REPEAT_MIN+ onset rows, as the bar lag back has, the share with
    the rhythm (onset rows) of that bar, and with its lanes too, as they are or mirrored.
    Per song, human charts 0.55 / 0.24, the default of 10-11 0.19 / 0.07, without the carry
    and copies 0.12 / 0.002 (208 of 240 val songs, 2026-10-11): when a section comes back,
    humans bring its rhythm back, the sampled charts seldom. nan where there is no such
    bar."""
    x = np.asarray(tokens).reshape(-1, np.asarray(tokens).shape[-1])
    n_bars = min(n_whole_bars(n_cells), len(s_audio))
    out = {"section_rhythm": float("nan"), "section_copy": float("nan")}
    if n_bars < 2:
        return out
    starts = np.isin(x[:n_bars * BAR], (TAP, HOLD_START)).reshape(n_bars, BAR, -1)
    rows = starts.any(axis=2)
    masks = (starts * (1 << np.arange(starts.shape[2]))).sum(axis=2)
    busy = rows.sum(axis=1) >= REPEAT_MIN
    same_rows, same_lanes = [], []
    for t0, length, lag in section_runs(s_audio[:n_bars, :n_bars]):
        for t in range(t0, t0 + length):
            if busy[t] and busy[t - lag]:
                same_rows.append(np.array_equal(rows[t], rows[t - lag]))
                same_lanes.append(np.array_equal(masks[t], masks[t - lag])
                                  or np.array_equal(masks[t], _MIRROR[masks[t - lag]]))
    if same_rows:
        out["section_rhythm"] = float(np.mean(same_rows))
        out["section_copy"] = float(np.mean(same_lanes))
    return out


LN_SECTION = 0.5                # ln_placement: a long-note bar starts at least this share
LN_SCATTER = 0.25               # of its notes as long notes; a bar of taps under this share


def ln_placement(tokens: np.ndarray, n_cells: int) -> dict:
    """Where the long notes go. ln_sections: of the long notes started in the whole bars,
    the share in bars with 2+ note starts of which at least LN_SECTION are long notes;
    ln_scattered: the share in bars where under LN_SCATTER are (a long note among taps).
    Humans 0.48 / 0.19, the sampled charts 0.34 / 0.29 (EXPERIMENTS 2026-10-10). nan
    without long notes."""
    x = np.asarray(tokens).reshape(-1, np.asarray(tokens).shape[-1])
    n_bars = n_whole_bars(n_cells)
    out = {"ln_sections": float("nan"), "ln_scattered": float("nan")}
    if n_bars < 1:
        return out
    n_start = np.isin(x[:n_bars * BAR], (TAP, HOLD_START)).reshape(n_bars, -1).sum(axis=1)
    n_long = (x[:n_bars * BAR] == HOLD_START).reshape(n_bars, -1).sum(axis=1)
    total = n_long.sum()
    if total == 0:
        return out
    share = n_long / np.maximum(n_start, 1)
    out["ln_sections"] = float(n_long[(share >= LN_SECTION) & (n_start >= 2)].sum() / total)
    out["ln_scattered"] = float(n_long[share < LN_SCATTER].sum() / total)
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
