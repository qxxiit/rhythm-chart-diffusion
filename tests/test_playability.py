"""The ranking criteria's 4K guidelines and the chords-on-strong-sounds check (playability.py)."""

import math

import numpy as np
import pytest

from src.data.tokenizer import EMPTY, HOLD_BODY, HOLD_END, HOLD_START, TAP
from src.evaluation.playability import (
    BEAT_180,
    chord_cap,
    chord_strength,
    grade_of,
    guideline_checks,
    over_cap,
)


def _times(n: int, bpm: float = 180.0) -> np.ndarray:
    return np.arange(n) * 60000.0 / bpm / 12                # 12 cells a beat


def _hold(x: np.ndarray, row: int, lane: int, n: int) -> None:
    x[row, lane], x[row + 1:row + n, lane], x[row + n, lane] = HOLD_START, HOLD_BODY, HOLD_END


def test_grades_and_caps() -> None:
    assert [grade_of(s) for s in (1.0, 2.0, 2.69, 3.9, 4.0, 6.0, 9.0)] == \
        ["Easy", "Normal", "Normal", "Hard", "Insane", "Expert", "Expert+"]
    assert (chord_cap(1.5), chord_cap(2.5), chord_cap(3.5), chord_cap(4.5)) == (2, 2, 3, None)
    assert abs(BEAT_180 - 333.33) < 0.01


def test_chords_over_the_cap_count_long_notes_held() -> None:
    x = np.full((10, 4), EMPTY)
    x[0, :3] = TAP                                          # a triple
    _hold(x, 2, 0, 4)                                       # held through rows 3-5
    x[4, [1, 2]] = TAP                                      # 2 taps under it: 3 in Easy
    x[5, [2, 3]] = TAP                                      # a double on the other hand
    over = over_cap(x, 2)
    assert list(np.flatnonzero(over)) == [0, 4, 5]
    normal = over_cap(x, 2, other_hand_double=True)         # Normal: the double on the right
    assert list(np.flatnonzero(normal)) == [0, 4]           # hand under a left-hand hold is fine
    assert not over_cap(x, 3).any()                         # Hard: 3 is fine
    r = guideline_checks(x, _times(10), "Easy")
    assert r["chord_over"] == pytest.approx(3 / 4)          # rows 0, 2, 4, 5 start notes
    assert math.isnan(guideline_checks(x, _times(10), "Insane")["chord_over"])


def test_fast_jacks_by_time_at_180_bpm() -> None:
    x = np.full((40, 4), EMPTY)
    x[[0, 3], 0] = TAP                                      # 1/4 beat apart: a fast jack
    x[[10, 16], 1] = TAP                                    # 1/2 beat: not
    x[[20, 21, 22], 2] = TAP                                # two more
    r = guideline_checks(x, _times(40), "Hard")
    assert r["fast_jacks"] == pytest.approx(100 * 3 / 7)
    slow = guideline_checks(x, _times(40, bpm=90), "Hard")  # at 90 BPM 1/4 beat is 167 ms
    assert slow["fast_jacks"] == pytest.approx(100 * 2 / 7)


def test_anchors_over_the_limit() -> None:
    x = np.full((48, 4), EMPTY)
    lanes = [0, 1, 0, 2, 0, 3, 1, 2]                        # lane 0 on 3 alternate rows
    for i, k in enumerate(lanes):
        x[3 * i, k] = TAP
    r = guideline_checks(x, _times(48), "Normal")
    assert r["anchor_over"] == pytest.approx(100 / 8)       # 3 notes: over Normal's 2
    assert guideline_checks(x, _times(48), "Hard")["anchor_over"] == 0.0
    assert math.isnan(guideline_checks(x, _times(48), "Easy")["anchor_over"])


def test_long_notes_too_short_and_releases_too_close() -> None:
    x = np.full((80, 4), EMPTY)
    _hold(x, 0, 0, 6)                                       # 1/2 beat: short in Easy
    _hold(x, 0, 3, 6)                                       # released with it: not "close"
    _hold(x, 12, 1, 12)                                     # 1 beat, released at row 24
    _hold(x, 20, 2, 12)                                     # released 1/2 beat after: close
    r = guideline_checks(x, _times(80), "Easy")
    assert r["ln_short"] == pytest.approx(2 / 4)
    assert r["release_close"] == pytest.approx(1 / 2)       # release times 6, 24, 32 (cells)
    n = guideline_checks(x, _times(80), "Normal")
    assert n["ln_short"] == 0.0 and n["release_close"] == 0.0
    assert math.isnan(guideline_checks(x, _times(80), "Insane")["ln_short"])


def test_runs_past_five_notes_in_the_songs_beats() -> None:
    x = np.full((200, 4), EMPTY)
    for i in range(8):
        x[6 * i, i % 4] = TAP                               # 8 notes 1/2 beat apart
    for i in range(4):
        x[100 + 12 * i, i % 4] = TAP                        # 4 notes a beat apart
    r = guideline_checks(x, _times(200), "Easy")
    assert r["run_over"] == pytest.approx(100 * 3 / 12)     # notes 6, 7, 8 of the run
    assert guideline_checks(x, _times(200, bpm=100), "Easy")["run_over"] == 0.0   # 1/2 at 100
    assert guideline_checks(x, _times(200, bpm=250), "Normal")["run_over"] == \
        pytest.approx(100 * 3 / 12)                         # 1/2 at 250 BPM is a 1/4
    assert guideline_checks(x, _times(200), "Normal")["run_over"] == 0.0
    assert math.isnan(guideline_checks(x, _times(200), "Hard")["run_over"])


def test_chord_strength() -> None:
    x = np.full((40, 4), EMPTY)
    for r in (0, 8, 16):
        x[r, [0, 3]] = TAP                                  # chords
    for r in (4, 12, 20, 28):
        x[r, 1] = TAP                                       # singles
    s = np.zeros(40)
    s[[0, 8, 16]] = 3.0
    s[[4, 12, 20, 28]] = 1.0
    assert chord_strength(x, s) > 0
    assert chord_strength(x, s[::-1].copy()) < chord_strength(x, s)
    assert math.isnan(chord_strength(x[:10], s[:10]))       # too few of each
