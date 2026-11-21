"""Controlled corruptions of a chart: does a metric see what it is meant to see?
(docs/EVALUATION.md; the method of ChartGenEval, arXiv 2607.12857.)

Each corruption takes the tokens [n, K] of a song's rows (no PAD) and a dose, and returns
new tokens that keep the grammar. Long notes are left alone (but by lnflood): only taps move
or appear.

    timing     a share of the taps moved one row (1/12 beat) earlier or later, onto an empty
               cell
    lanes      the lanes of a share of the onset rows without long notes permuted (the
               rhythm and the chord sizes stay)
    new        taps added on empty rows inside the song, as many as a share of its onset rows
    jumptrill  pairs of single taps in a row, at most a beat apart, made into a two-note chord
               of one hand and then one of the other ([12][34]: the flow among chords that
               the AI charts have 1.5x the human rate of)

What the players and modders call bad (claude/chart_quality_research_1011.md, 2026-11-22:
"jacks where they do not belong", one-hand trills, chords the music does not call for, short
long notes covering a busy part, the same bar pasted again):

    minijack   of single taps within half a beat of the single tap before, a share moved
               into that tap's lane (the rhythm stays)
    trill      of runs of 4+ single taps each within half a beat of the last, a share made a
               trill of one hand (the rhythm stays)
    weakchord  of single taps off the beat, a share made a two-note chord
    lnflood    of taps with the next note start within half a beat and their lane free for
               a quarter beat, a share made a long note of a quarter beat (the only
               corruption that adds long notes)
    copypaste  of bars after a bar, both with notes and no long notes, a share made the bar
               before again

corrupt(tokens, name, dose, rng) dispatches; dose 0 returns a copy.
"""

from __future__ import annotations

import numpy as np

from src.data.tokenizer import EMPTY, HOLD_BODY, HOLD_END, HOLD_START, TAP, D, K

CORRUPTIONS = ("timing", "lanes", "new", "jumptrill", "minijack", "trill", "weakchord", "lnflood",
               "copypaste")
DOSES = {"timing": (0.1, 0.2, 0.4), "lanes": (0.2, 0.4, 0.8), "new": (0.02, 0.04, 0.08),
         "jumptrill": (0.2, 0.4, 0.8), "minijack": (0.2, 0.4, 0.8), "trill": (0.2, 0.4, 0.8),
         "weakchord": (0.1, 0.2, 0.4), "lnflood": (0.2, 0.4, 0.8), "copypaste": (0.2, 0.4, 0.8)}
BAR = 4 * D
HANDS = ((0, 1), (2, 3))                # 4K: left hand lanes 1-2, right hand lanes 3-4


def _hold_rows(x: np.ndarray) -> np.ndarray:
    """[n] rows with any long-note cell."""
    return np.isin(x, (HOLD_START, HOLD_BODY, HOLD_END)).any(axis=1)


def _onset_rows(x: np.ndarray) -> np.ndarray:
    return np.isin(x, (TAP, HOLD_START)).any(axis=1)


def timing(x: np.ndarray, dose: float, rng: np.random.Generator) -> np.ndarray:
    """Taps moved one row, to an empty cell of the same lane (else left where they are).
    An empty cell is never next to a hold body, so the grammar holds."""
    x = x.copy()
    taps = np.argwhere(x == TAP)
    move = taps[rng.random(len(taps)) < dose]
    for r, k in move[rng.permutation(len(move))]:
        to = r + (1 if rng.random() < 0.5 else -1)
        if 0 <= to < len(x) and x[to, k] == EMPTY and x[r, k] == TAP:
            x[to, k], x[r, k] = TAP, EMPTY
    return x


def lanes(x: np.ndarray, dose: float, rng: np.random.Generator) -> np.ndarray:
    """The taps of a share of the onset rows without long-note cells, in other lanes."""
    x = x.copy()
    rows = np.flatnonzero(_onset_rows(x) & ~_hold_rows(x))
    for r in rows[rng.random(len(rows)) < dose]:
        taps = x[r] == TAP
        if taps.all():
            continue
        for _ in range(10):
            perm = rng.permutation(K)
            if not np.array_equal(taps[perm], taps):
                x[r] = np.where(taps[perm], TAP, EMPTY)
                break
    return x


def new(x: np.ndarray, dose: float, rng: np.random.Generator) -> np.ndarray:
    """Taps on empty rows between the first and last onset, dose x the onset rows of them."""
    x = x.copy()
    onsets = np.flatnonzero(_onset_rows(x))
    if len(onsets) < 2:
        return x
    empty = np.flatnonzero((x == EMPTY).all(axis=1))
    empty = empty[(empty > onsets[0]) & (empty < onsets[-1])]
    n_add = min(len(empty), round(dose * len(onsets)))
    for r in rng.choice(empty, size=n_add, replace=False):
        x[r, int(rng.integers(K))] = TAP
    return x


def jumptrill(x: np.ndarray, dose: float, rng: np.random.Generator) -> np.ndarray:
    """Pairs of single taps (rows without long-note cells, the next onset at most a beat on)
    made into one hand's chord and then the other's; pairs do not overlap."""
    x = x.copy()
    onsets = np.flatnonzero(_onset_rows(x))
    single = (x[onsets] == TAP).sum(axis=1) == 1
    free = ~_hold_rows(x)[onsets]
    i = 0
    while i + 1 < len(onsets):
        r1, r2 = onsets[i], onsets[i + 1]
        if single[i] and single[i + 1] and free[i] and free[i + 1] and r2 - r1 <= D:
            if rng.random() < dose:
                lane = int(np.flatnonzero(x[r1] == TAP)[0])
                first = HANDS[0] if lane in HANDS[0] else HANDS[1]
                second = HANDS[1] if first == HANDS[0] else HANDS[0]
                x[r1] = EMPTY
                x[r1, list(first)] = TAP
                x[r2] = EMPTY
                x[r2, list(second)] = TAP
            i += 2
        else:
            i += 1
    return x


def _singles(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The onset rows, and which of them are one tap with no long-note cell in the row."""
    onsets = np.flatnonzero(_onset_rows(x))
    single = ((x[onsets] == TAP).sum(axis=1) == 1) & ~_hold_rows(x)[onsets] \
        & ((x[onsets] != EMPTY).sum(axis=1) == 1)
    return onsets, single


def minijack(x: np.ndarray, dose: float, rng: np.random.Generator) -> np.ndarray:
    x = x.copy()
    onsets, single = _singles(x)
    for i in range(1, len(onsets)):
        r0, r1 = onsets[i - 1], onsets[i]
        if single[i - 1] and single[i] and r1 - r0 <= D // 2 and rng.random() < dose:
            lane = int(np.flatnonzero(x[r0] == TAP)[0])
            if x[r1, lane] != TAP:
                x[r1] = EMPTY
                x[r1, lane] = TAP
    return x


def trill(x: np.ndarray, dose: float, rng: np.random.Generator) -> np.ndarray:
    x = x.copy()
    onsets, single = _singles(x)
    i = 0
    while i < len(onsets):
        j = i
        while j + 1 < len(onsets) and single[j] and single[j + 1] \
                and onsets[j + 1] - onsets[j] <= D // 2:
            j += 1
        if single[i] and j - i + 1 >= 4 and rng.random() < dose:
            hand = HANDS[int(rng.integers(2))]
            start = int(rng.integers(2))
            for n, r in enumerate(onsets[i:j + 1]):
                x[r] = EMPTY
                x[r, hand[(start + n) % 2]] = TAP
        i = j + 1
    return x


def weakchord(x: np.ndarray, dose: float, rng: np.random.Generator) -> np.ndarray:
    x = x.copy()
    onsets, single = _singles(x)
    for r, one in zip(onsets, single):
        if one and r % D != 0 and rng.random() < dose:
            free = np.flatnonzero(x[r] == EMPTY)
            x[r, int(rng.choice(free))] = TAP
    return x


def lnflood(x: np.ndarray, dose: float, rng: np.random.Generator) -> np.ndarray:
    x = x.copy()
    q = D // 4                                     # a quarter beat
    onsets = np.flatnonzero(_onset_rows(x))
    nxt = {int(a): int(b) for a, b in zip(onsets[:-1], onsets[1:])}
    for r, k in np.argwhere(x == TAP):
        if r not in nxt or nxt[r] - r > D // 2 or r + q >= len(x):
            continue
        if (x[r + 1:r + q + 2, k] != EMPTY).any() or rng.random() >= dose:
            continue                               # the lane free to one row past the end
        x[r, k] = HOLD_START
        x[r + 1:r + q, k] = HOLD_BODY
        x[r + q, k] = HOLD_END
    return x


def copypaste(x: np.ndarray, dose: float, rng: np.random.Generator) -> np.ndarray:
    x = x.copy()
    n_bars = len(x) // BAR
    held = np.isin(x, (HOLD_START, HOLD_BODY, HOLD_END))
    def plain(b: int) -> bool:
        s = slice(b * BAR, (b + 1) * BAR)
        return bool((x[s] == TAP).any()) and not held[s].any()
    for b in range(1, n_bars):
        if plain(b - 1) and plain(b) and rng.random() < dose:
            x[b * BAR:(b + 1) * BAR] = x[(b - 1) * BAR:b * BAR]
    return x


_BY_NAME = {"timing": timing, "lanes": lanes, "new": new, "jumptrill": jumptrill,
            "minijack": minijack, "trill": trill, "weakchord": weakchord, "lnflood": lnflood,
            "copypaste": copypaste}


def corrupt(x: np.ndarray, name: str, dose: float, rng: np.random.Generator) -> np.ndarray:
    if name not in _BY_NAME:
        raise ValueError(f"unknown corruption {name!r}; one of {CORRUPTIONS}")
    if dose <= 0:
        return np.array(x, copy=True)
    return _BY_NAME[name](np.asarray(x), dose, rng)


def spearman_with_dose(values: np.ndarray) -> float:
    """Spearman correlation of [v_dose0, v_dose1, ...] with the dose order; nan when the values
    are all equal or any is nan (ties take their mean rank)."""
    v = np.asarray(values, dtype=np.float64)
    if not np.isfinite(v).all() or np.ptp(v) == 0:
        return float("nan")
    order = np.argsort(v, kind="stable")
    ranks = np.empty(len(v))
    ranks[order] = np.arange(len(v))
    for val in np.unique(v):                       # mean ranks for ties
        same = v == val
        ranks[same] = ranks[same].mean()
    dose = np.arange(len(v), dtype=np.float64)
    a, b = ranks - ranks.mean(), dose - dose.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))
