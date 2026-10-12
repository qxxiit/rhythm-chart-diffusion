"""What plays well, by the osu!mania ranking criteria and the players (research of 2026-10-11:
the UGRP project's claude/chart_quality_research_1011.md, EXPERIMENTS 2026-10-12).

Players in English, Japanese and Korean communities praise charts whose notes sit on sounds
they can hear, with the same pattern for the same sound and chords on the strong sounds, and
blame charts that tangle the hands: fast jacks, long anchors, holds under hammering. The
ranking criteria put numbers on the second part for the low difficulties (4K guidelines,
https://osu.ppy.sh/wiki/en/Ranking_criteria/osu!mania), written for 180 BPM: times here are
those beats at 180 BPM (BEAT_180), so a chart of another tempo is held to the same times.

    guideline_checks(tokens, times, grade)
        chord_over     of the onset rows, the share holding more notes than the grade allows
                       (note starts plus long notes held through the row): Easy and Normal
                       2, Hard 3, but in Normal a double on the other hand from the long
                       notes held is fine; nan for the other grades
        fast_jacks     notes per 100 that come back to their lane within 1/4 beat: kept out
                       of Normal and discouraged in Hard by the guidelines
        anchor_over    anchors longer than the grade allows (Normal 2 notes, Hard 4) per 100
                       onset rows: a lane taken on every other onset row, each within 1/2
                       beat of the one before; nan for the other grades
        ln_short       of the long notes, the share shorter than the grade's minimum (Easy 1
                       beat, Normal 1/2, Hard 1/4); nan for the other grades
        release_close  of the releases, the share within the grade's spacing of the release
                       before (Easy 1 beat, Normal 1/2), releases at the same time apart; nan
                       for the other grades
        run_over       notes past the 5th of a run of notes each within the grade's snap of
                       the one before (Easy 1/2 beat, Normal 1/4: "avoid using more than 5
                       consecutive 1/2 [1/4] notes") per 100 onset rows; snaps in the song's
                       beats, halved under 120 BPM and doubled over 240 (Scaling BPM: a
                       1/4 at 100 BPM is as fast as a 1/2 at 200); nan for the other grades
    chord_strength(tokens, strength)
                   chords on the strong sounds: the mean onset strength (z within the song,
                   sampler.onset_strength) of the rows that start 2+ notes, less that of the
                   rows that start one

All are guidelines, not rules: humans break them too (240 val songs, 2026-10-11, per song:
chord_over Easy 0.005, Normal 0.028, Hard 0.010; fast_jacks Hard 0.04, Expert 1.8; ln_short
Easy 0.29; run_over Easy 1.4, Normal 0.12, 2026-10-12). The numbers worth watching are the ones far from the human charts' (the default
of 10-11: fast_jacks Hard 0.14, Expert 3.3; chord_over Easy 0.009; chord_strength 0.76
against 0.83; EXPERIMENTS 2026-10-12).
"""

from __future__ import annotations

import math

import numpy as np

from src.data.tokenizer import HOLD_BODY, HOLD_START, TAP, D

BEAT_180 = 60000.0 / 180          # ms: the guidelines' beat
GRADES = (("Easy", 2.0), ("Normal", 2.7), ("Hard", 4.0), ("Insane", 5.3), ("Expert", 6.5),
          ("Expert+", math.inf))  # (name, SR below), as evaluate.GRADES
CHORD_CAP = {"Easy": 2, "Normal": 2, "Hard": 3}
ANCHOR_MAX = {"Normal": 2, "Hard": 4}            # notes
LN_MIN_BEATS = {"Easy": 1.0, "Normal": 0.5, "Hard": 0.25}
RELEASE_GAP_BEATS = {"Easy": 1.0, "Normal": 0.5}
JACK_BEATS = 0.25
ANCHOR_GAP_BEATS = 0.5
RUN_BEATS = {"Easy": 0.5, "Normal": 0.25}       # a run: notes within this snap of the last ...
RUN_MAX = 5                                     # ... at most this many of them in a row
CHECKS = ("chord_over", "fast_jacks", "anchor_over", "ln_short", "release_close", "run_over")


def grade_of(sr: float) -> str:
    return next(name for name, below in GRADES if sr < below)


def chord_cap(sr: float) -> int | None:
    """The largest chord the guidelines allow at this SR (None: no cap)."""
    return CHORD_CAP.get(grade_of(sr))


def over_cap(rows: np.ndarray, cap: int, other_hand_double: bool = False) -> np.ndarray:
    """[n] for rows [n, K] of tokens: True where the note starts plus the long notes held
    through the row (HOLD_BODY) pass cap; other_hand_double (the Normal guideline): a double
    on one hand while long notes are held on the other only is fine."""
    starts = np.isin(rows, (TAP, HOLD_START))
    held = rows == HOLD_BODY
    over = starts.sum(axis=1) + held.sum(axis=1) > cap
    if other_hand_double:
        half = rows.shape[1] // 2
        for a, b in ((slice(0, half), slice(half, None)), (slice(half, None), slice(0, half))):
            over &= ~(held[:, a].any(axis=1) & ~held[:, b].any(axis=1) & ~starts[:, a].any(axis=1)
                      & (starts[:, b].sum(axis=1) <= 2))
    return over


def _anchors(present: np.ndarray, t: np.ndarray, longer_than: int) -> int:
    """Anchors of more than longer_than notes: a lane taken on onset rows i, i + 2, i + 4 ...
    and not on the rows between, each within ANCHOR_GAP_BEATS of the one before."""
    count = 0
    gap = ANCHOR_GAP_BEATS * BEAT_180 + 1e-6
    for k in range(present.shape[1]):
        ok = np.zeros(len(present), dtype=bool)
        ok[2:] = present[2:, k] & present[:-2, k] & ~present[1:-1, k] & (t[2:] - t[:-2] <= gap)
        for parity in (0, 1):
            run = 0
            for i in range(2 + parity, len(present), 2):
                if ok[i]:
                    run += 1
                    continue
                count += run + 1 > longer_than and run > 0
                run = 0
            count += run + 1 > longer_than and run > 0
    return int(count)


def run_over(t_rows: np.ndarray, beat_ms: np.ndarray, snap: float, longest: int = RUN_MAX) -> int:
    """Notes past the longest-th of runs of notes each within snap beats of the one before.

    t_rows   [n] ms of the onset rows; beat_ms [n] the beat's length there (ms)
    The snap is in the song's beats, halved under 120 BPM and doubled over 240 (the ranking
    criteria's Scaling BPM)."""
    if len(t_rows) < 2:
        return 0
    bpm = 60000.0 / beat_ms[:-1]
    scale = np.where(bpm < 120, 0.5, np.where(bpm > 240, 2.0, 1.0))
    close = np.diff(t_rows) <= snap * scale * beat_ms[:-1] + 1e-6
    over, n = 0, 0
    for c in close:
        n = n + 1 if c else 0
        over += n + 1 > longest
    return over


def guideline_checks(tokens: np.ndarray, times: np.ndarray, grade: str) -> dict:
    """The module docstring's guideline_checks for one chart.

    tokens  [n_rows, K] (or [n_chunks, L, K]) tokens of the whole bars to check, no PAD
    times   [n_rows] ms of each row (BeatGrid.time_from_cell)
    grade   the chart's grade by its SR (grade_of)
    """
    x = np.asarray(tokens).reshape(-1, np.asarray(tokens).shape[-1])
    t = np.asarray(times, dtype=np.float64)[:len(x)]
    out = dict.fromkeys(CHECKS, float("nan"))
    onset = np.isin(x, (TAP, HOLD_START))
    rows = np.flatnonzero(onset.any(axis=1))
    if len(rows) < 2:
        return out
    cap = CHORD_CAP.get(grade)
    if cap is not None:
        out["chord_over"] = float(np.mean(over_cap(x[rows], cap, grade == "Normal")))
    jacks = 0
    for k in range(x.shape[1]):
        r = np.flatnonzero(onset[:, k])
        jacks += int((np.diff(t[r]) <= JACK_BEATS * BEAT_180 + 1e-6).sum())
    out["fast_jacks"] = 100.0 * jacks / int(onset.sum())
    if grade in ANCHOR_MAX:
        out["anchor_over"] = 100.0 * _anchors(onset[rows], t[rows], ANCHOR_MAX[grade]) / len(rows)
    if grade in RUN_BEATS and len(t) >= 2:
        cell = np.diff(t, append=2 * t[-1] - t[-2])          # ms of each row's cell
        out["run_over"] = 100.0 * run_over(t[rows], D * cell[rows], RUN_BEATS[grade]) / len(rows)
    starts = np.argwhere(x == HOLD_START)
    if len(starts) and grade in LN_MIN_BEATS:
        lengths, releases = [], []
        for r, k in starts:
            e = r + 1
            while e < len(x) and x[e, k] == HOLD_BODY:
                e += 1
            e = min(e, len(x) - 1)                  # the release row (HOLD_END)
            lengths.append(t[e] - t[r])
            releases.append(t[e])
        out["ln_short"] = float(np.mean(np.array(lengths) < LN_MIN_BEATS[grade] * BEAT_180 - 1e-6))
        if grade in RELEASE_GAP_BEATS and len(releases) >= 2:
            gaps = np.diff(np.unique(np.round(releases, 3)))
            if len(gaps):
                out["release_close"] = float(np.mean(gaps < RELEASE_GAP_BEATS[grade] * BEAT_180
                                                     - 1e-6))
    return out


def chord_strength(tokens: np.ndarray, strength: np.ndarray, min_rows: int = 3) -> float:
    """The module docstring's chord_strength; nan with fewer than min_rows rows of each."""
    x = np.asarray(tokens).reshape(-1, np.asarray(tokens).shape[-1])
    s = np.asarray(strength, dtype=np.float64)[:len(x)]
    if len(s) < len(x) or s.std() == 0:
        return float("nan")
    z = (s - s.mean()) / s.std()
    size = np.isin(x, (TAP, HOLD_START)).sum(axis=1)
    chords, singles = size >= 2, size == 1
    if chords.sum() < min_rows or singles.sum() < min_rows:
        return float("nan")
    return float(z[chords].mean() - z[singles].mean())
