"""Long-note numbers (src/evaluation/holds.py) and the sampler's hold clean-up."""

import numpy as np

from src.data.tokenizer import EMPTY, HOLD_BODY, HOLD_END, HOLD_START, TAP, grammar_violations
from src.evaluation.holds import hold_spans, hold_stats, release_gaps
from src.models.sampler import clean_holds


def lane(n: int = 60, holds=(), taps=()) -> np.ndarray:
    """[n, 4] tokens with holds (start, end) and taps in lane 0, nothing elsewhere."""
    x = np.full((n, 4), EMPTY, dtype=np.int64)
    for s, e in holds:
        x[s, 0], x[s + 1:e, 0], x[e, 0] = HOLD_START, HOLD_BODY, HOLD_END
    for t in taps:
        x[t, 0] = TAP
    return x


def test_hold_numbers() -> None:
    x = lane(holds=[(0, 1), (4, 12), (20, 22), (25, 35)], taps=[13, 50])
    assert hold_spans(x) == [(0, 0, 1), (0, 4, 12), (0, 20, 22), (0, 25, 35)]
    assert release_gaps(x).tolist() == [3, 1, 3, 15]
    h = hold_stats(x)
    assert h["hold_share"] == 4 / 6
    assert h["hold_beats"] == (1 + 8 + 2 + 10) / 4 / 12
    assert h["short_holds"] == 2 / 4 and h["quick_regrab"] == 1 / 4
    none = hold_stats(lane(taps=[3, 9]))
    assert none["hold_share"] == 0 and np.isnan(none["hold_beats"])


def test_clean_holds_moves_releases_back_and_turns_short_holds_into_taps() -> None:
    x = lane(holds=[(0, 1), (4, 12), (20, 23), (30, 32), (40, 42)], taps=[13, 26, 43])
    onsets_before = np.isin(x, (TAP, HOLD_START))
    out = clean_holds(x.copy(), min_hold=3, release_gap=2)
    assert len(grammar_violations(out)) == 0
    assert np.array_equal(np.isin(out, (TAP, HOLD_START)), onsets_before)    # onsets stay
    assert out[0, 0] == TAP and out[1, 0] == EMPTY                           # 1 cell: a tap
    assert hold_spans(out) == [(0, 4, 10), (0, 20, 23)]                      # 12 -> 10 before 13
    assert out[30, 0] == TAP and np.all(out[31:33, 0] == EMPTY)              # 2 cells: a tap
    assert out[40, 0] == TAP and np.all(out[41:43, 0] == EMPTY)              # no room: a tap
    gaps = release_gaps(out)
    assert np.all(gaps > 2)
    assert np.array_equal(clean_holds(x.copy(), min_hold=0, release_gap=0), x)
