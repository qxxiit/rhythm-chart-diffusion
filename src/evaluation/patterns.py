"""Pattern clarity (design doc §4.11-2b), defined by periodicity (decided 2026-09-26).

A pattern is a short lane motif that repeats at a steady snap. Jacks (period 1),
trills and jumptrills (period 2), rolls (period 4) and back-and-forth stairs
(period 6) are special cases, and so is any unnamed motif that repeats
(1 3 2 4 1 3 2 4). Clarity is how much of a chart such runs cover, how long
they last, and how often one note cuts a run that then carries on.

Events are rows with at least one onset (tap or hold start); an event is the
set of lanes that start there. Segments are maximal stretches of consecutive
events with one gap of at most MAX_GAP cells (one beat); a change of snap ends
a segment. Inside a segment, a run of period p is a maximal stretch where every
event repeats the lane set p events earlier:

    events e_s .. e_e with lanes(e_i) == lanes(e_{i-p}) for s + p <= i <= e,
    at least max(2p, p + 4) events (the motif shown twice, and at least four
    events repeating an earlier one), and not periodic with any proper divisor
    of p over its whole length (a trill is period 2, not also 4). p = 1 .. P_MAX.

The minimum length keeps chance low. Streams with lanes drawn at random reach
coverage 0.09 (0.18 if jacks are avoided, 0.05 for a random jumpstream); with
"the motif shown twice" alone it was 0.40 / 0.49 / 0.23, i.e. half of a random
stream counted as pattern. coverage_chance measures the same thing per chart.

Per chart (summarize):
    coverage          share of onsets inside a run
    coverage_p1       ... inside runs of period 1 (jacks), coverage_p2 (trills,
                      jumptrills), coverage_p3plus (longer motifs)
    coverage_chance   coverage of the same chart with every event's lanes redrawn
                      at random (same rhythm, same chord sizes), mean of a few
                      fixed seeds: what the rhythm alone gives by accident
    run_length        mean events per run
    breaks_per_100    breaks per 100 events in runs. A break is two runs with the
                      same period, the same motif (up to rotation) and the same
                      gap, with exactly one event between them: one note off the
                      motif, or one note off the snap, and then the pattern goes on.
    motion_pred       (with chance_seeds) how predictable the lane motion is, with no
                      pattern names: every event is written as its move from the
                      event before (the lane step -3..3 between single notes, else
                      the lane set), an online context model of orders 0-4 predicts
                      each from the ones before it, and motion_pred is the share of
                      bits it saves over the same chart with lanes redrawn at random
                      (same rhythm, same chord sizes). 0 = no more structure than
                      chance; stairs, trills and unnamed motifs that come back all
                      count, also when moved to other lanes.
    move_stair, move_trill, move_jack, move_leap, chord_share
                      (always) which kinds of motion make up the chart, so that a
                      sampler that turns everything into one motif shows. Among two
                      single-note moves in a row (within a beat each) that both change
                      lane: move_stair, the second repeats the first step (1-2-3,
                      4-3-2: stairs, rolls); move_trill, it goes straight back (1-3-1,
                      2-1-2: trills, bounces); the rest is irregular. Among single-note
                      moves within a beat: move_jack, the same lane again; move_leap,
                      1 to 4 or back. chord_share: events with 2+ notes.
    lone_chord        (always) of the events inside a single-note stream (both
                      neighbours single notes at the same gap, at most a beat), the
                      share that are chords: a chord that cuts a trill or a stair.
                      The lane passes keep chord sizes, so these come from sampling.
    bar_rhythm_repeat, bar_lane_repeat
                      (always) phrase-level repetition. Of the bars with at least 4
                      onsets, the share whose rhythm (onset cells and chord sizes)
                      equals an earlier bar's; and of those, the share whose lanes
                      also equal one such earlier bar's, as they are or mirrored
                      (lane k <-> 3 - k): a1 b1 c1 a2 b2 c2.
The target is the level of human charts of the same grade, not the maximum
(§4.11): scripts/human_baselines.py measures it.
Fix MAX_GAP, P_MAX and the minimum length before any model output is scored.
"""

from __future__ import annotations

from collections import defaultdict
from itertools import pairwise

import numpy as np

from src.data.tokenizer import HOLD_START, TAP

MAX_GAP = 12                      # cells: at most one beat between events of a run
P_MAX = 8                         # longest motif, in events
BAR = 48
_POP = np.array([bin(m).count("1") for m in range(16)])


def min_events(p: int) -> int:
    return max(2 * p, p + 4)


def events(tokens: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(rows, lane bitmasks) of every row that has an onset."""
    x = np.asarray(tokens).reshape(-1, np.asarray(tokens).shape[-1])
    onset = np.isin(x, (TAP, HOLD_START))
    rows = np.flatnonzero(onset.any(axis=1))
    masks = (onset[rows] * (1 << np.arange(x.shape[1]))).sum(axis=1)
    return rows, masks.astype(np.int64)


def segments(rows: np.ndarray) -> list[tuple[int, int, int]]:
    """(first event, last event, gap) of every stretch with one gap <= MAX_GAP.
    Neighbouring segments share their boundary event."""
    gaps = np.diff(rows)
    out, s = [], 0
    while s < len(gaps):
        g = int(gaps[s])
        if g > MAX_GAP:
            s += 1
            continue
        e = s + 1
        while e < len(gaps) and gaps[e] == g:
            e += 1
        out.append((s, e, g))
        s = e
    return out


def _true_stretches(ok: np.ndarray) -> list[tuple[int, int]]:
    edges = np.diff(np.concatenate([[0], ok.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1, strict=True))


def _canonical(motif: tuple) -> tuple:
    return min(motif[k:] + motif[:k] for k in range(len(motif)))


def find_runs(rows: np.ndarray, masks: np.ndarray) -> list[tuple[int, int, int, tuple, int]]:
    """Every run: (first event, last event, period, motif up to rotation, gap), by start."""
    runs = []
    for s, e, gap in segments(rows):
        seg = masks[s:e + 1]
        for p in range(1, P_MAX + 1):
            if len(seg) < min_events(p):
                break
            for a, b in _true_stretches(seg[p:] == seg[:-p]):
                lo, hi = a, b + p                   # the stretch covers events a .. b + p
                if hi - lo + 1 < min_events(p):
                    continue
                part = seg[lo:hi + 1]
                if any(p % d == 0 and np.array_equal(part[d:], part[:-d]) for d in range(1, p)):
                    continue                        # really a shorter period
                runs.append((int(s + lo), int(s + hi), p,
                             _canonical(tuple(int(m) for m in part[:p])), gap))
    return sorted(runs)


def summarize(tokens: np.ndarray, chance_seeds: int = 0) -> dict:
    """Pattern clarity numbers for one chart (see module docstring). With
    chance_seeds > 0, also coverage_chance and motion_pred over that many
    lane-redrawn copies."""
    rows, masks = events(tokens)
    out = _stats(rows, masks) | move_shares(rows, masks) | lone_chords(rows, masks) \
        | bar_repeats(rows, masks)
    if chance_seeds:
        redrawn = [_redraw_lanes(masks, np.random.default_rng(seed)) for seed in range(chance_seeds)]
        out["coverage_chance"] = float(np.mean([_stats(rows, m)["coverage"] for m in redrawn]))
        out["motion_pred"] = motion_predictability(masks, redrawn)
    return out


def move_shares(rows: np.ndarray, masks: np.ndarray) -> dict:
    """move_stair, move_trill, move_jack, move_leap, chord_share (module docstring);
    nan when a chart has nothing to count."""
    single = _POP[masks] == 1
    lane = np.where(single, np.array([int(m).bit_length() - 1 for m in masks.tolist()]), -1)
    near = np.diff(rows) <= MAX_GAP
    step = single[1:] & single[:-1] & near                   # single -> single, within a beat
    delta = np.where(step, np.diff(lane), 0)
    pair = step[1:] & step[:-1] & (delta[1:] != 0) & (delta[:-1] != 0)
    n_step, n_pair = int(step.sum()), int(pair.sum())
    nan = float("nan")
    return {"move_stair": float((delta[1:] == delta[:-1])[pair].mean()) if n_pair else nan,
            "move_trill": float((delta[1:] == -delta[:-1])[pair].mean()) if n_pair else nan,
            "move_jack": float((delta[step] == 0).mean()) if n_step else nan,
            "move_leap": float((np.abs(delta[step]) == 3).mean()) if n_step else nan,
            "chord_share": float((_POP[masks] >= 2).mean()) if len(masks) else nan}


def lone_chords(rows: np.ndarray, masks: np.ndarray) -> dict:
    """lone_chord (module docstring); nan without single-note streams."""
    if len(masks) < 3:
        return {"lone_chord": float("nan")}
    gap = np.diff(rows)
    single = _POP[masks] == 1
    inside = (gap[:-1] == gap[1:]) & (gap[1:] <= MAX_GAP) & single[:-2] & single[2:]
    if not inside.any():
        return {"lone_chord": float("nan")}
    return {"lone_chord": float((~single[1:-1])[inside].mean())}


_MIRROR = np.array([int(f"{m:04b}"[::-1], 2) for m in range(16)])
MIN_BAR_ONSETS = 4


def bar_repeats(rows: np.ndarray, masks: np.ndarray, bar: int = BAR) -> dict:
    """bar_rhythm_repeat, bar_lane_repeat (module docstring); nan when undefined."""
    nan = float("nan")
    bars: dict[int, list[tuple[int, int]]] = {}
    for r, m in zip(rows.tolist(), masks.tolist(), strict=True):
        bars.setdefault(r // bar, []).append((r % bar, m))
    seen: dict[tuple, list[tuple]] = {}           # rhythm -> lane patterns of earlier bars
    n = rhythm_hits = lane_hits = 0
    for b in sorted(bars):
        notes = bars[b]
        if len(notes) < MIN_BAR_ONSETS:
            continue
        n += 1
        rhythm = tuple((o, int(_POP[m])) for o, m in notes)
        lanes = tuple(m for _, m in notes)
        mirrored = tuple(int(_MIRROR[m]) for m in lanes)
        earlier = seen.setdefault(rhythm, [])
        if earlier:
            rhythm_hits += 1
            lane_hits += any(e in (lanes, mirrored) for e in earlier)
        earlier.append(lanes)
    return {"bar_rhythm_repeat": rhythm_hits / n if n else nan,
            "bar_lane_repeat": lane_hits / rhythm_hits if rhythm_hits else nan}


MOTION_ORDER = 4
_N_MOTION = 15 + 7                # lane sets 1..15, steps -3..3


def motion_symbols(masks: np.ndarray) -> list[int]:
    """Each event as its move: single note after single note -> 15 + 3 + lane step
    (15..21), anything else -> the lane set - 1 (0..14)."""
    out, prev = [], 0
    for m in masks.tolist():
        if _POP[m] == 1 and _POP[prev] == 1:
            out.append(15 + 3 + (m.bit_length() - prev.bit_length()))
        else:
            out.append(m - 1)
        prev = m
    return out


def motion_bits(symbols: list[int], order: int = MOTION_ORDER, alpha: float = 0.5) -> float:
    """Mean bits per event under an online mixture of order-0..order context models
    (each with add-alpha counts of what followed that context so far in this chart)."""
    counts: list[dict] = [{} for _ in range(order + 1)]
    total = 0.0
    for i, sym in enumerate(symbols):
        ps = []
        for k in range(min(order, i) + 1):
            seen = counts[k].get(tuple(symbols[i - k:i]))
            n = seen["n"] if seen else 0
            c = seen.get(sym, 0) if seen else 0
            ps.append((c + alpha) / (n + _N_MOTION * alpha))
        total -= np.log2(sum(ps) / len(ps))
        for k in range(min(order, i) + 1):
            seen = counts[k].setdefault(tuple(symbols[i - k:i]), {"n": 0})
            seen["n"] += 1
            seen[sym] = seen.get(sym, 0) + 1
    return total / max(len(symbols), 1)


def motion_predictability(masks: np.ndarray, redrawn: list[np.ndarray],
                          min_events: int = 32) -> float:
    """1 - bits(chart) / mean bits(lane-redrawn copies); nan for very short charts."""
    if len(masks) < min_events:
        return float("nan")
    base = float(np.mean([motion_bits(motion_symbols(m)) for m in redrawn]))
    return float(1.0 - motion_bits(motion_symbols(masks)) / base) if base > 0 else float("nan")


def _redraw_lanes(masks: np.ndarray, rng: np.random.Generator, n_lanes: int = 4) -> np.ndarray:
    """Same chord sizes, lanes chosen uniformly at random."""
    order = rng.permuted(np.tile(np.arange(n_lanes), (len(masks), 1)), axis=1)
    sizes = _POP[masks]
    out = np.zeros(len(masks), dtype=np.int64)
    for j in range(n_lanes):
        out |= np.where(j < sizes, 1 << order[:, j], 0)
    return out


def _stats(rows: np.ndarray, masks: np.ndarray) -> dict:
    weight = _POP[masks].astype(np.float64)
    total = weight.sum()
    runs = find_runs(rows, masks)
    covered = {k: np.zeros(len(masks), dtype=bool) for k in ("p1", "p2", "p3plus")}
    for s, e, p, _, _ in runs:
        covered["p1" if p == 1 else "p2" if p == 2 else "p3plus"][s:e + 1] = True
    any_run = covered["p1"] | covered["p2"] | covered["p3plus"]
    out = {"onsets": int(total), "runs": len(runs)}
    out["coverage"] = float(weight[any_run].sum() / total) if total else 0.0
    for k, cov in covered.items():
        out[f"coverage_{k}"] = float(weight[cov].sum() / total) if total else 0.0
    out["run_length"] = float(np.mean([e - s + 1 for s, e, *_ in runs])) if runs else 0.0
    by_key = defaultdict(list)
    for s, e, p, motif, gap in runs:
        by_key[(p, motif, gap)].append((s, e))
    breaks = sum(1 for spans in by_key.values()
                 for (_, e1), (s2, _) in pairwise(sorted(spans)) if s2 - e1 == 2)
    in_runs = int(any_run.sum())
    out["breaks_per_100"] = 100.0 * breaks / in_runs if in_runs else 0.0
    return out


TYPES = ("p1", "p2", "p3plus", "single", "chord")


def bar_type_vectors(tokens: np.ndarray, n_bars: int) -> np.ndarray:
    """Onsets per type in each bar, [n_bars, 5] (the "types" vector of rho): in a run of
    period 1 / 2 / 3+ (shortest period wins), else a lone single note or chord."""
    rows, masks = events(tokens)
    label = np.where(_POP[masks] > 1, TYPES.index("chord"), TYPES.index("single"))
    for s, e, p, _, _ in sorted(find_runs(rows, masks), key=lambda r: -r[2]):
        label[s:e + 1] = TYPES.index("p1" if p == 1 else "p2" if p == 2 else "p3plus")
    out = np.zeros((n_bars, len(TYPES)))
    bars = rows // BAR
    keep = bars < n_bars
    np.add.at(out, (bars[keep], label[keep]), _POP[masks[keep]])
    return out
