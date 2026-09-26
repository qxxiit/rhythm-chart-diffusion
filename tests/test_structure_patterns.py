"""rho (§4.11-2) and the pattern-clarity tagger (§4.11-2b)."""

import numpy as np
import pytest

from src.data.tokenizer import EMPTY, HOLD_BODY, HOLD_END, HOLD_START, TAP
from src.evaluation.patterns import events, find_runs, summarize
from src.evaluation.structure import (
    BAR,
    _cosine,
    lag_profile,
    pair_sets,
    rho,
    ssm_chart,
    structure_scores,
)

# --- rho -------------------------------------------------------------------


def test_empty_bars_rule() -> None:
    v = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 2.0]])
    s = _cosine(v)
    assert s[0, 1] == 1.0                    # empty vs empty
    assert s[0, 2] == 0.0 and s[1, 3] == 0.0  # empty vs non-empty
    assert s[2, 3] == 0.0 and s[2, 2] == pytest.approx(1.0)


def test_pair_sets() -> None:
    p = pair_sets(16, far_k=8)
    assert p["all"].sum() == 120                             # 16 * 15 / 2, no diagonal
    assert p["in"].sum() == 2 * 28                           # two 8-bar chunks
    assert p["cross"].sum() == 64
    assert p["far"].sum() == sum(16 - d for d in range(8, 16))
    assert not np.diagonal(p["all"]).any()


def _song(structure: list[int], chart_structure: list[int], seed: int = 0):
    """Bars built from templates: audio bar m uses template structure[m], chart bar m
    uses note template chart_structure[m]."""
    rng = np.random.default_rng(seed)
    audio_t = rng.normal(size=(3, BAR * 4, 80))
    chart_t = [rng.random((BAR, 4)) < 0.12 for _ in range(3)]
    mel = np.concatenate([audio_t[a] + 0.3 * rng.normal(size=(BAR * 4, 80)) for a in structure])
    tokens = np.concatenate([np.where(chart_t[c], TAP, EMPTY) for c in chart_structure])
    return tokens, mel


def test_rho_is_high_when_the_chart_follows_the_music() -> None:
    form = [0, 0, 1, 1, 0, 0, 2, 2, 0, 1, 2, 0, 1, 1, 2, 0]
    tokens, mel = _song(form, form)
    s = structure_scores(tokens, mel, len(tokens), far_k=4)
    assert s["n_bars"] == 16 and s["rho_all"] > 0.9 and s["rho_far"] > 0.9
    shuffled = list(np.random.default_rng(1).permutation(form))
    tokens2, _ = _song(form, shuffled)
    assert abs(structure_scores(tokens2, mel, len(tokens2))["rho_all"]) < 0.5


def test_rho_is_nan_for_constant_similarity() -> None:
    s = np.ones((5, 5))
    assert np.isnan(rho(s, s, pair_sets(5, 2)["all"]))


def test_holds_count_in_the_chart_vector() -> None:
    x = np.full((2 * BAR, 4), EMPTY)
    x[0, 0], x[1:BAR - 1, 0], x[BAR - 1, 0] = HOLD_START, HOLD_BODY, HOLD_END
    x[BAR, 0] = TAP
    s = ssm_chart(x, 2)
    assert 0 < s[0, 1] < 1                   # the shared onset cell, but only one bar holds
    assert lag_profile(s, 3)[0] == pytest.approx(s[0, 1]) and np.isnan(lag_profile(s, 3)[2])


# --- patterns (periodic motifs) ------------------------------------------------

def grid(seq, gap: int = 3) -> np.ndarray:
    x = np.full((len(seq) * gap + 12, 4), EMPTY)
    for i, lanes in enumerate(seq):
        for k in lanes:
            x[i * gap, k] = TAP
    return x


@pytest.mark.parametrize("seq, period, length", [
    ([(1,)] * 5, 1, 5),                                          # jack
    ([(k,) for k in [0, 2] * 5], 2, 10),                         # trill
    ([(0, 1), (2, 3)] * 4, 2, 8),                                # jumptrill
    ([(k,) for k in [0, 1, 2, 3] * 3], 4, 12),                   # roll
    ([(k,) for k in [0, 1, 2, 3, 2, 1] * 2 + [0]], 6, 13),       # stairs back and forth
    ([(k,) for k in [0, 2, 1, 3] * 3], 4, 12),                   # an unnamed motif
])
def test_periodic_runs(seq, period, length) -> None:
    rows, masks = events(grid(seq))
    runs = find_runs(rows, masks)
    assert [(s, e, p) for s, e, p, _, _ in runs] == [(0, length - 1, period)]
    r = summarize(grid(seq))
    assert r["coverage"] == 1.0 and r["run_length"] == length and r["breaks_per_100"] == 0.0


def test_a_shorter_period_wins() -> None:
    rows, masks = events(grid([(k,) for k in [0, 1] * 6]))       # also 4- and 6-periodic
    assert [p for _, _, p, _, _ in find_runs(rows, masks)] == [2]


def test_too_short_to_be_a_pattern() -> None:
    assert summarize(grid([(k,) for k in [0, 1, 2, 3]]))["coverage"] == 0.0   # shown once
    assert summarize(grid([(1,)] * 4))["coverage"] == 0.0                      # 4-note jack
    assert summarize(grid([(k,) for k in [0, 1] * 2 + [0]]))["coverage"] == 0.0  # 5-note trill


def test_a_one_note_slip_is_a_break() -> None:
    r = summarize(grid([(k,) for k in [0, 1, 0, 1, 0, 1, 0, 2, 0, 1, 0, 1, 0, 1, 0]]))
    assert r["runs"] == 2 and r["coverage"] == pytest.approx(14 / 15)
    assert r["breaks_per_100"] == pytest.approx(100 / 14)


def test_a_one_note_timing_slip_is_a_break_too() -> None:
    x = np.full((48, 4), EMPTY)
    rows = [3 * i for i in range(15)]
    rows[7] += 1                                                 # the 8th note is a cell late
    for i, row in enumerate(rows):
        x[row, i % 2] = TAP
    r = summarize(x)
    assert r["runs"] == 2 and r["breaks_per_100"] == pytest.approx(100 / 14)


def test_a_new_motif_is_not_a_break() -> None:
    r = summarize(grid([(k,) for k in [0, 1] * 3 + [2, 3] * 3]))
    assert r["runs"] == 2 and r["breaks_per_100"] == 0.0


def test_a_change_of_snap_ends_a_run() -> None:
    x = np.full((48, 4), EMPTY)
    for i, row in enumerate([0, 3, 6, 9, 12, 15, 21, 27, 33, 39, 45]):   # gap 3, then gap 6
        x[row, i % 2] = TAP
    rows, masks = events(x)
    assert [(s, e, gap) for s, e, _, _, gap in find_runs(rows, masks)] == [(0, 5, 3), (5, 10, 6)]


def test_random_lanes_are_rarely_patterns() -> None:
    rng = np.random.default_rng(0)
    x = np.full((3 * 4000, 4), EMPTY)
    x[np.arange(0, 3 * 4000, 3), rng.integers(0, 4, 4000)] = TAP   # a random 1/4 stream
    r = summarize(x, chance_seeds=2)
    assert r["coverage"] < 0.15 and abs(r["coverage"] - r["coverage_chance"]) < 0.03


def test_chance_keeps_rhythm_and_chord_sizes() -> None:
    from src.evaluation.patterns import _POP, _redraw_lanes
    masks = np.array([1, 3, 7, 15, 2, 12] * 50)
    redrawn = _redraw_lanes(masks, np.random.default_rng(0))
    assert np.array_equal(_POP[redrawn], _POP[masks]) and not np.array_equal(redrawn, masks)
    trill = grid([(k,) for k in [0, 2] * 20])
    r = summarize(trill, chance_seeds=3)
    assert r["coverage"] == 1.0 and r["coverage_chance"] < 0.5


def test_type_vectors_count_onsets_by_period() -> None:
    from src.evaluation.patterns import TYPES, bar_type_vectors
    x = np.full((96, 4), EMPTY)
    for i, k in enumerate([0, 2] * 4):                           # a trill in bar 0
        x[3 * i, k] = TAP
    x[60, 1] = TAP                                               # a lone note in bar 1
    v = bar_type_vectors(x, 2)
    assert v[0, TYPES.index("p2")] == 8 and v[1, TYPES.index("single")] == 1
