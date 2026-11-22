import numpy as np
import pytest

from src.data.tokenizer import EMPTY, HOLD_BODY, HOLD_END, HOLD_START, TAP, K, grammar_violations
from src.evaluation.metrics import new_positions
from src.evaluation.oracle import on_positions, to_set_positions


def chart() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """48 rows 10 ms apart; the set's other charts have onsets on every third row."""
    x = np.full((48, K), EMPTY, dtype=np.int64)
    times = np.arange(48) * 10.0
    ref = times[::3]
    x[0, 0] = TAP                    # on a set position
    x[7, 1] = TAP                    # new: row 6 is on the set, nothing near in lane 1
    x[13, 2] = x[13, 3] = TAP        # new chord: row 12 is on the set
    x[16, 0] = HOLD_START            # new, starts a long note: stays
    x[17, 0] = HOLD_BODY
    x[18, 0] = HOLD_END
    x[22, 3] = TAP                   # new: 21 holds a tap in lane 3's way (x[21, 3]), 24 is free
    x[21, 3] = TAP                   # on the set
    return x, times, ref


def test_on_positions_is_new_positions_turned_round():
    x, times, ref = chart()
    onset = np.isin(x, (TAP, HOLD_START)).any(axis=1)
    on = on_positions(times[onset], ref)
    assert np.mean(~on) == pytest.approx(new_positions(times[onset], ref))
    assert on_positions(times, np.zeros(0)).all()              # no references: nothing new


def test_same_counts_and_changes_nothing():
    x, times, ref = chart()
    y, counts = to_set_positions(x, times, ref, "same")
    assert np.array_equal(x, y)
    assert counts == {"new_rows": 4, "hold_rows": 1, "moved": 0, "dropped": 0}


def test_drop_takes_the_new_taps_out_and_keeps_the_long_note():
    x, times, ref = chart()
    y, counts = to_set_positions(x, times, ref, "drop")
    assert counts["dropped"] == 3 and counts["moved"] == 0
    assert (y[[7, 13, 22]] == EMPTY).all() and y[16, 0] == HOLD_START and y[0, 0] == TAP
    assert len(grammar_violations(y)) == 0


def test_snap_moves_to_the_nearest_set_row_that_fits():
    x, times, ref = chart()
    y, counts = to_set_positions(x, times, ref, "snap")
    assert counts["moved"] == 3 and counts["dropped"] == 0
    assert y[6, 1] == TAP and y[7, 1] == EMPTY                 # 7 -> 6 (one row back)
    assert (y[12, 2:] == TAP).all() and (y[13] == EMPTY).all()  # the chord as it was
    assert y[24, 3] == TAP and y[22, 3] == EMPTY               # 21 has a note: on to 24
    assert y[16, 0] == HOLD_START
    onset = np.isin(y, (TAP, HOLD_START)).any(axis=1)
    assert new_positions(times[onset], ref) == pytest.approx(1 / onset.sum())   # the hold row
    assert len(grammar_violations(y)) == 0
    assert np.isin(y, (TAP, HOLD_START)).sum() == np.isin(x, (TAP, HOLD_START)).sum()


def test_snap_makes_no_jack_and_drops_where_nothing_fits():
    x = np.full((12, K), EMPTY, dtype=np.int64)
    times = np.arange(12) * 10.0
    ref = times[[0, 2, 3, 6, 7]]
    x[4, 0] = TAP                    # new; 3 and 6 are on the set and free
    x[2, 0] = x[7, 0] = TAP          # on the set, lane 0 taken next to both (within 2 rows)
    x[2, 1] = x[7, 1] = TAP
    y, counts = to_set_positions(x, times, ref, "snap")
    assert counts["dropped"] == 1 and y[4, 0] == EMPTY and (y[[3, 6], 0] == EMPTY).all()


def test_an_unknown_mode():
    x, times, ref = chart()
    with pytest.raises(ValueError):
        to_set_positions(x, times, ref, "oracle")
