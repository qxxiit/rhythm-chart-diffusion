"""Long-note numbers for one chart, on tokens (generated and human charts alike).

    hold_share     long notes / onsets (taps + long notes): each long note counts once,
                   at its start, as the manifest's hold_ratio does for notes
    hold_beats     mean long-note length in beats (release cell - start cell, / 12)
    short_holds    share of long notes shorter than SHORT cells (1/4 beat)
    quick_regrab   share of long notes whose lane has an onset within QUICK cells after
                   the release: let go and press again at once (an onset on the release
                   cell itself cannot happen, the tokenizer moves the release back)
    release_on_onset  share of long notes released on a row where another lane has an
                   onset: let go as the next note comes (human charts 0.77, 2026-10-04)
The last four are nan for a chart without long notes.

ln_agreement(gen, human) compares two charts of one song on the same grid: of the
onsets both have (same row and lane), which ones each makes a long note; ln_f1 is the
F1 of the generated chart's long notes against the human chart's there.
"""

from __future__ import annotations

import numpy as np

from src.data.tokenizer import HOLD_END, HOLD_START, TAP

SHORT = 3                         # cells: shorter than 1/4 beat
QUICK = 2                         # cells after a release (1/6 beat)
CELLS_PER_BEAT = 12


def hold_spans(tokens: np.ndarray) -> list[tuple[int, int, int]]:
    """(lane, start cell, release cell) of every long note; the grammar pairs them in order."""
    x = np.asarray(tokens).reshape(-1, np.asarray(tokens).shape[-1])
    out = []
    for k in range(x.shape[1]):
        starts = np.flatnonzero(x[:, k] == HOLD_START)
        ends = np.flatnonzero(x[:, k] == HOLD_END)
        out += [(k, int(s), int(e)) for s, e in zip(starts, ends, strict=True)]
    return out


def release_gaps(tokens: np.ndarray, spans=None) -> np.ndarray:
    """Cells from each release to the next onset in its lane (-1 if there is none)."""
    x = np.asarray(tokens).reshape(-1, np.asarray(tokens).shape[-1])
    onset = np.isin(x, (TAP, HOLD_START))
    gaps = []
    for k, _, e in spans if spans is not None else hold_spans(x):
        nxt = np.flatnonzero(onset[e + 1:, k])
        gaps.append(int(nxt[0]) + 1 if len(nxt) else -1)
    return np.array(gaps, dtype=np.int64)


def ln_agreement(gen: np.ndarray, human: np.ndarray) -> dict:
    """{"ln_f1": ...} over the onset cells both charts share; nan without long notes."""
    g = np.asarray(gen).reshape(-1, np.asarray(gen).shape[-1])
    h = np.asarray(human).reshape(-1, np.asarray(human).shape[-1])
    n = min(len(g), len(h))
    g, h = g[:n], h[:n]
    shared = np.isin(g, (TAP, HOLD_START)) & np.isin(h, (TAP, HOLD_START))
    g_ln, h_ln = shared & (g == HOLD_START), shared & (h == HOLD_START)
    both = int((g_ln & h_ln).sum())
    if not g_ln.any() or not h_ln.any():
        return {"ln_f1": float("nan")}
    p, r = both / g_ln.sum(), both / h_ln.sum()
    return {"ln_f1": float(2 * p * r / (p + r)) if p + r > 0 else 0.0}


def hold_stats(tokens: np.ndarray) -> dict:
    x = np.asarray(tokens).reshape(-1, np.asarray(tokens).shape[-1])
    onsets = int(np.isin(x, (TAP, HOLD_START)).sum())
    spans = hold_spans(x)
    out = {"hold_share": len(spans) / onsets if onsets else float("nan")}
    if not spans:
        return out | {"hold_beats": float("nan"), "short_holds": float("nan"),
                      "quick_regrab": float("nan"), "release_on_onset": float("nan")}
    lens = np.array([e - s for _, s, e in spans])
    gaps = release_gaps(x, spans)
    onset_rows = np.isin(x, (TAP, HOLD_START)).any(axis=1)
    return out | {"hold_beats": float(lens.mean() / CELLS_PER_BEAT),
                  "short_holds": float(np.mean(lens < SHORT)),
                  "quick_regrab": float(np.mean((gaps > 0) & (gaps <= QUICK))),
                  "release_on_onset": float(np.mean([onset_rows[e] for _, _, e in spans]))}
