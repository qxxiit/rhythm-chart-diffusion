"""Long-note numbers for one chart, on tokens (generated and human charts alike).

    hold_share     long notes / onsets (taps + long notes): each long note counts once,
                   at its start, as the manifest's hold_ratio does for notes
    hold_beats     mean long-note length in beats (release cell - start cell, / 12)
    short_holds    share of long notes shorter than SHORT cells (1/4 beat)
    quick_regrab   share of long notes whose lane has an onset within QUICK cells after
                   the release: let go and press again at once (an onset on the release
                   cell itself cannot happen, the tokenizer moves the release back)
The last three are nan for a chart without long notes.
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


def hold_stats(tokens: np.ndarray) -> dict:
    x = np.asarray(tokens).reshape(-1, np.asarray(tokens).shape[-1])
    onsets = int(np.isin(x, (TAP, HOLD_START)).sum())
    spans = hold_spans(x)
    out = {"hold_share": len(spans) / onsets if onsets else float("nan")}
    if not spans:
        return out | {"hold_beats": float("nan"), "short_holds": float("nan"),
                      "quick_regrab": float("nan")}
    lens = np.array([e - s for _, s, e in spans])
    gaps = release_gaps(x, spans)
    return out | {"hold_beats": float(lens.mean() / CELLS_PER_BEAT),
                  "short_holds": float(np.mean(lens < SHORT)),
                  "quick_regrab": float(np.mean((gaps > 0) & (gaps <= QUICK)))}
