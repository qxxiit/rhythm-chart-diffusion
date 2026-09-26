"""Pattern clarity (design doc §4.11-2b).

DRAFT RULE SET. §4.11 asks for the rules to be fixed before any model output is
scored: review them (with the team's rhythm-game player), change what is wrong,
then freeze them in DECISIONS.md. Changing them after seeing model results
makes the comparison meaningless.

Events are rows with at least one onset (tap or hold start); an event is the
set of lanes that start there. A run is a stretch of consecutive events, all
the same number of cells apart (at most MAX_GAP = one beat), where every step
follows one rule:

    jack       the same lane set again                                    min 3 events
    trill      single notes alternating between two lanes: a b a b        min 4 events
    stairs     single notes moving to the neighbouring lane in one
               direction, turning only at an edge lane: 0123, 3210, 0123210  min 4 events
    jumptrill  two-note chords alternating between disjoint pairs:
               [01] [23] [01] [23]                                         min 4 events

Per song (summarize):
    coverage          share of onsets inside a run of any rule (coverage_<rule> per rule)
    run_length        mean events per run (run_length_<rule> per rule)
    breaks_per_100    pairs of runs of the same rule on the same lanes (same pair for
                      trill and jumptrill, same lane set for jack, any stairs) with
                      exactly one event between them, per 100 events in runs:
                      a pattern cut by a one-note slip
The target is the level of human charts of the same grade, not the maximum
(§4.11): scripts/human_baselines.py measures it.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np

from src.data.tokenizer import HOLD_START, TAP

RULES = ("jack", "trill", "stairs", "jumptrill")
MIN_EVENTS = {"jack": 3, "trill": 4, "stairs": 4, "jumptrill": 4}
MAX_GAP = 12                       # cells: patterns at one beat or faster
BAR = 48
_POP = [bin(m).count("1") for m in range(16)]


def events(tokens: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(rows, lane bitmasks) of every row that has an onset."""
    x = np.asarray(tokens).reshape(-1, np.asarray(tokens).shape[-1])
    onset = np.isin(x, (TAP, HOLD_START))
    rows = np.flatnonzero(onset.any(axis=1))
    masks = (onset[rows] * (1 << np.arange(x.shape[1]))).sum(axis=1)
    return rows, masks.astype(np.int64)


def _lane(mask: int) -> int:
    return int(mask).bit_length() - 1


def _step_ok(rule: str, m: np.ndarray, s: int, i: int, n_lanes: int) -> bool:
    """Does the step from event i to i+1 continue a `rule` run that began at s?"""
    a, b = int(m[i]), int(m[i + 1])
    if rule == "jack":
        return a == b
    if rule in ("trill", "jumptrill"):
        size = 1 if rule == "trill" else 2
        if _POP[a] != size or _POP[b] != size or a & b:
            return False
        return i == s or b == int(m[i - 1])
    if rule == "stairs":
        if _POP[a] != 1 or _POP[b] != 1:
            return False
        d = _lane(b) - _lane(a)
        if abs(d) != 1:
            return False
        if i == s:
            return True
        prev = _lane(a) - _lane(int(m[i - 1]))
        return d == prev or (d == -prev and _lane(a) in (0, n_lanes - 1))
    raise ValueError(rule)


def _identity(rule: str, m: np.ndarray, s: int):
    if rule == "jack":
        return int(m[s])
    if rule == "trill":
        return int(m[s]) | int(m[s + 1])
    if rule == "jumptrill":
        return tuple(sorted((int(m[s]), int(m[s + 1]))))
    return "stairs"


def find_runs(rows: np.ndarray, masks: np.ndarray, rule: str,
              n_lanes: int = 4) -> list[tuple[int, int, object]]:
    """Maximal runs of one rule: (first event, last event, identity)."""
    n, out, s = len(masks), [], 0
    while s < n - 1:
        gap = rows[s + 1] - rows[s]
        if gap > MAX_GAP or not _step_ok(rule, masks, s, s, n_lanes):
            s += 1
            continue
        i = s + 1
        while i + 1 < n and rows[i + 1] - rows[i] == gap and _step_ok(rule, masks, s, i, n_lanes):
            i += 1
        if i - s + 1 >= MIN_EVENTS[rule]:
            out.append((s, i, _identity(rule, masks, s)))
        s = i                              # the last event may begin the next run
    return out


def label_events(masks: np.ndarray, runs: dict[str, list]) -> np.ndarray:
    """One label per event: the first rule in RULES order whose run covers it, else
    'single' or 'chord'."""
    label = np.where(np.array([_POP[int(m)] for m in masks]) > 1, "chord", "single")
    label = label.astype(object)
    for rule in reversed(RULES):           # earlier rules overwrite later ones
        for s, e, _ in runs[rule]:
            label[s:e + 1] = rule
    return label


def summarize(tokens: np.ndarray) -> dict:
    """Pattern clarity numbers for one chart (see module docstring)."""
    rows, masks = events(tokens)
    n_lanes = np.asarray(tokens).shape[-1]
    weight = np.array([_POP[int(m)] for m in masks], dtype=np.float64)
    total = weight.sum()
    runs = {r: find_runs(rows, masks, r, n_lanes) for r in RULES}
    out = {"onsets": int(total)}
    covered_any = np.zeros(len(masks), dtype=bool)
    all_lengths, breaks = [], 0
    for rule, rs in runs.items():
        covered = np.zeros(len(masks), dtype=bool)
        for s, e, _ in rs:
            covered[s:e + 1] = True
        covered_any |= covered
        lengths = [e - s + 1 for s, e, _ in rs]
        all_lengths += lengths
        out[f"coverage_{rule}"] = float(weight[covered].sum() / total) if total else 0.0
        out[f"run_length_{rule}"] = float(np.mean(lengths)) if lengths else 0.0
        breaks += sum(1 for (_, e1, id1), (s2, _, id2) in pairwise(rs)
                      if s2 - e1 == 2 and id1 == id2)
    out["coverage"] = float(weight[covered_any].sum() / total) if total else 0.0
    out["run_length"] = float(np.mean(all_lengths)) if all_lengths else 0.0
    out["runs"] = len(all_lengths)
    in_runs = int(covered_any.sum())
    out["breaks_per_100"] = 100.0 * breaks / in_runs if in_runs else 0.0
    return out


def bar_type_vectors(tokens: np.ndarray, n_bars: int) -> np.ndarray:
    """Onsets per label in each bar: [n_bars, len(RULES) + 2] (the "types" vector of rho)."""
    rows, masks = events(tokens)
    n_lanes = np.asarray(tokens).shape[-1]
    runs = {r: find_runs(rows, masks, r, n_lanes) for r in RULES}
    labels = label_events(masks, runs)
    names = (*RULES, "single", "chord")
    out = np.zeros((n_bars, len(names)))
    for row, mask, lab in zip(rows, masks, labels, strict=True):
        bar = row // BAR
        if bar < n_bars:
            out[bar, names.index(lab)] += _POP[int(mask)]
    return out
