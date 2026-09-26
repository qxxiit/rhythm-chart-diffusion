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


# --- patterns ----------------------------------------------------------------

def grid(seq, gap: int = 3) -> np.ndarray:
    x = np.full((len(seq) * gap + 12, 4), EMPTY)
    for i, lanes in enumerate(seq):
        for k in lanes:
            x[i * gap, k] = TAP
    return x


@pytest.mark.parametrize("seq, rule, length", [
    ([(k,) for k in [0, 1, 2, 3, 2, 1, 0, 1, 2, 3]], "stairs", 10),
    ([(k,) for k in [0, 2] * 4], "trill", 8),
    ([(1,)] * 5, "jack", 5),
    ([(0, 1), (2, 3)] * 3, "jumptrill", 6),
])
def test_each_rule(seq, rule, length) -> None:
    r = summarize(grid(seq))
    assert r["coverage"] == 1.0 and r[f"coverage_{rule}"] == 1.0
    assert r["runs"] == 1 and r["run_length"] == length and r["breaks_per_100"] == 0.0


def test_turning_away_from_an_edge_is_not_stairs() -> None:
    rows, masks = events(grid([(k,) for k in [0, 1, 2, 1, 2, 1]]))
    assert find_runs(rows, masks, "stairs") == []             # turns at lanes 2 and 1
    assert find_runs(rows, masks, "trill") == [(1, 5, 0b0110)]  # 1 2 1 2 1 is a trill
    rows, masks = events(grid([(k,) for k in [2, 1, 0, 1, 2]]))
    assert find_runs(rows, masks, "stairs") == [(0, 4, "stairs")]  # turning at the edge is fine


def test_a_one_note_slip_is_a_break() -> None:
    r = summarize(grid([(k,) for k in [0, 1, 0, 1, 0, 2, 0, 1, 0, 1]]))
    assert r["runs"] == 2 and r["coverage"] == pytest.approx(0.9)
    assert r["breaks_per_100"] == pytest.approx(100 / 9)


def test_a_change_of_snap_ends_a_run() -> None:
    x = np.full((48, 4), EMPTY)
    for row, k in zip([0, 3, 6, 9, 15, 21, 27, 33], [0, 1, 0, 1, 0, 1, 0, 1], strict=True):
        x[row, k] = TAP                                      # gap 3, then gap 6
    rows, masks = events(x)
    assert find_runs(rows, masks, "trill") == [(0, 3, 0b11), (3, 7, 0b11)]


def test_random_charts_have_little_pattern() -> None:
    rng = np.random.default_rng(0)
    x = np.where(rng.random((2000, 4)) < 0.08, TAP, EMPTY)
    assert summarize(x)["coverage"] < 0.35
