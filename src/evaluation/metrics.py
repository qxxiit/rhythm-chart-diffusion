"""Note-level metrics shared by every model (design doc §4.11 / §7).

    onset_f1(pred, ref, tol_ms, lanes=True)   precision / recall / F1 of onsets
    violation_rate(tokens)                    grammar violations per scored position

Onsets are tap times and hold starts; releases are not scored (tokenizer
docstring: the release-aware mAP@tIoU is deferred). Matching is greedy, as in
the design doc: every (pred, ref) pair within tol_ms (same lane when lanes=True)
is a candidate, candidates are taken in order of |dt|, and each note is matched
at most once. The answer is not unique, so F1 has a low ceiling: report the
round trip decode(encode(ref)) next to it as the tokenizer's own score.
"""

from __future__ import annotations

import numpy as np

from src.data.chart_parser import Chart
from src.data.tokenizer import PAD, grammar_violations


def onsets(chart: Chart) -> tuple[np.ndarray, np.ndarray]:
    """(times_ms, lanes) of every onset, sorted by time then lane."""
    notes = sorted(chart.notes, key=lambda n: (n.time_ms, n.lane))
    return (np.array([n.time_ms for n in notes], dtype=np.float64),
            np.array([n.lane for n in notes], dtype=np.int64))


def greedy_matches(pred_t: np.ndarray, pred_lane: np.ndarray, ref_t: np.ndarray,
                   ref_lane: np.ndarray, tol_ms: float, lanes: bool = True) -> int:
    """Number of one-to-one matches, closest pairs first."""
    order = np.argsort(ref_t, kind="stable")
    rt, rl = ref_t[order], ref_lane[order]
    lo = np.searchsorted(rt, pred_t - tol_ms, side="left")
    hi = np.searchsorted(rt, pred_t + tol_ms, side="right")
    candidates = []
    for i in range(len(pred_t)):
        for j in range(lo[i], hi[i]):
            if not lanes or rl[j] == pred_lane[i]:
                candidates.append((abs(rt[j] - pred_t[i]), i, j))
    candidates.sort()
    used_pred, used_ref = set(), set()
    for _, i, j in candidates:
        if i not in used_pred and j not in used_ref:
            used_pred.add(i)
            used_ref.add(j)
    return len(used_pred)


def onset_f1(pred: Chart, ref: Chart, tol_ms: float = 50.0, lanes: bool = True) -> dict:
    """Precision, recall and F1 of pred's onsets against ref's. Two empty charts score 1."""
    pt, pl = onsets(pred)
    rt, rl = onsets(ref)
    if len(pt) == 0 and len(rt) == 0:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    tp = greedy_matches(pt, pl, rt, rl, tol_ms, lanes)
    precision = tp / len(pt) if len(pt) else 0.0
    recall = tp / len(rt) if len(rt) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if tp else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def violation_rate(tokens: np.ndarray) -> float:
    """Grammar violations (whole song, closed edges) per non-PAD position.
    0 by construction for the constrained sampler; the number that matters for
    models that sample freely (the AR baselines)."""
    x = np.asarray(tokens).reshape(-1, np.asarray(tokens).shape[-1])
    scored = int((x != PAD).sum())
    return len(grammar_violations(x)) / scored if scored else 0.0
