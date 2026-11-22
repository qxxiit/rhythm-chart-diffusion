"""The notes a chart puts where no other chart of its mapset has one, moved or taken out with
those charts as the oracle: how much of what a judge sees in the AI charts do those notes carry
(EXPERIMENTS 2026-11-22 (4); metrics.new_positions counts them)? Not a pass: the system never
sees the other charts of a set."""

from __future__ import annotations

import numpy as np

from src.data.tokenizer import EMPTY, HOLD_BODY, HOLD_END, HOLD_START, TAP

MODES = ("same", "drop", "snap")
WINDOW = 3          # rows a tap may move (1/4 beat at 12 cells a beat)
JACK = 2            # no other note start in the tap's lane this many rows either side of it


def on_positions(times, ref_times, tol_ms: float = 5.0) -> np.ndarray:
    """[n] bool: a reference onset within tol_ms of each time (new_positions' test turned
    round). All True without references: nothing is new then."""
    t = np.asarray(times, dtype=np.float64)
    ref = np.unique(np.asarray(ref_times, dtype=np.float64))
    if len(ref) == 0:
        return np.ones(len(t), dtype=bool)
    j = np.clip(np.searchsorted(ref, t), 1, max(len(ref) - 1, 1))
    lo = ref[np.clip(j - 1, 0, len(ref) - 1)]
    hi = ref[np.clip(j, 0, len(ref) - 1)]
    return np.minimum(np.abs(lo - t), np.abs(hi - t)) <= tol_ms


def _fits(x: np.ndarray, q: int, lanes: np.ndarray, r: int) -> bool:
    """The taps of row r may go to row q: its cells in those lanes EMPTY, no long note in them
    a row either side, and no other note start in them within JACK rows."""
    n = len(x)
    for k in lanes:
        if x[q, k] != EMPTY:
            return False
        for s in (q - 1, q + 1):
            if 0 <= s < n and x[s, k] in (HOLD_START, HOLD_BODY, HOLD_END):
                return False
        for s in range(max(q - JACK, 0), min(q + JACK + 1, n)):
            if s != r and x[s, k] in (TAP, HOLD_START):
                return False
    return True


def to_set_positions(x: np.ndarray, times: np.ndarray, ref_times: np.ndarray, mode: str, *,
                     window: int = WINDOW, tol_ms: float = 5.0) -> tuple[np.ndarray, dict]:
    """x [n_rows, K] tokens of a song's rows, times [n_rows] their ms, ref_times the onsets of
    the set's other charts; returns a new array and the counts. The onset rows with no
    reference onset within tol_ms (new rows) that start no long note: drop: their taps go;
    snap: they move, lanes kept, to the nearest row within `window` (the earlier on a tie) that
    is on a reference onset, starts no note and fits (_fits), and go where none does; same:
    nothing changes. The rows that start a long note stay (hold_rows). Taps taken out of or put
    into EMPTY cells keep the grammar."""
    if mode not in MODES:
        raise ValueError(f"mode {mode!r}: one of {MODES}")
    x = np.array(x, copy=True)
    n = len(times)
    counts = {"new_rows": 0, "hold_rows": 0, "moved": 0, "dropped": 0}
    on = on_positions(times, ref_times, tol_ms)
    starts = np.isin(x[:n], (TAP, HOLD_START)).any(axis=1)
    steps = sorted((d for d in range(-window, window + 1) if d), key=lambda d: (abs(d), d))
    for r in np.flatnonzero(starts & ~on):
        counts["new_rows"] += 1
        if (x[r] == HOLD_START).any():
            counts["hold_rows"] += 1
            continue
        if mode == "same":
            continue
        lanes = np.flatnonzero(x[r] == TAP)
        target = None
        if mode == "snap":
            for d in steps:
                q = r + d
                if 0 <= q < n and on[q] and not np.isin(x[q], (TAP, HOLD_START)).any() \
                        and _fits(x, q, lanes, r):
                    target = q
                    break
        x[r, lanes] = EMPTY
        if target is None:
            counts["dropped"] += 1
        else:
            x[target, lanes] = TAP
            counts["moved"] += 1
    return x, counts
