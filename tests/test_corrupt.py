"""Corruptions for the metric checks (src/evaluation/corrupt.py) and the judge-against-
diagnostics fit (scripts/judge_vs_diag.py); docs/EVALUATION.md."""

import csv
import json

import numpy as np
import pytest

from src.data.tokenizer import EMPTY, HOLD_BODY, HOLD_END, HOLD_START, TAP, K, grammar_violations
from src.evaluation.corrupt import CORRUPTIONS, DOSES, corrupt, spearman_with_dose


def chart(seed: int = 0, rows: int = 2000, p_hold: float = 0.1) -> np.ndarray:
    """Taps and long notes on a 1/4-beat grid (3 rows), some chords, grammar intact."""
    rng = np.random.default_rng(seed)
    x = np.full((rows, K), EMPTY, dtype=np.int64)
    r = 6
    while r < rows - 30:
        if rng.random() < p_hold:                                 # a long note
            k, n = int(rng.integers(K)), int(rng.integers(3, 12))
            x[r, k], x[r + 1:r + n, k], x[r + n, k] = HOLD_START, HOLD_BODY, HOLD_END
            r += n + 3
            continue
        lanes = rng.choice(K, size=1 if rng.random() < 0.7 else 2, replace=False)
        x[r, lanes] = TAP
        r += 3 * int(rng.integers(1, 3))
    return x


def holds(x: np.ndarray) -> np.ndarray:
    return np.isin(x, (HOLD_START, HOLD_BODY, HOLD_END))


@pytest.mark.parametrize("name", CORRUPTIONS)
def test_corruptions_keep_the_grammar_and_the_long_notes(name: str) -> None:
    x = chart(p_hold=0.02 if name == "copypaste" else 0.1)
    assert len(grammar_violations(x)) == 0
    changed = []
    for dose in DOSES[name]:
        y = corrupt(x, name, dose, np.random.default_rng(1))
        assert len(grammar_violations(y)) == 0
        if name == "lnflood":                                 # the long notes it adds, only
            assert (holds(y) >= holds(x)).all() and np.array_equal(y[holds(x)], x[holds(x)])
        else:
            assert np.array_equal(holds(y), holds(x)) and np.array_equal(y[holds(y)], x[holds(x)])
        changed.append(int((y != x).sum()))
    assert 0 < changed[0] < changed[-1]                       # more with the dose
    assert np.array_equal(corrupt(x, name, 0.0, np.random.default_rng(1)), x)


def test_what_each_corruption_keeps() -> None:
    x = chart()
    rng = np.random.default_rng(2)
    taps = int((x == TAP).sum())
    onset_rows = np.isin(x, (TAP, HOLD_START)).any(axis=1)
    y = corrupt(x, "timing", 0.4, rng)                        # moved, none added or lost
    assert int((y == TAP).sum()) == taps
    y = corrupt(x, "lanes", 0.8, rng)                         # same rows, same chord sizes
    assert np.array_equal((y == TAP).sum(axis=1), (x == TAP).sum(axis=1))
    y = corrupt(x, "new", 0.08, rng)                          # new rows, one tap each
    added = np.isin(y, (TAP, HOLD_START)).any(axis=1) & ~onset_rows
    assert added.sum() == round(0.08 * onset_rows.sum())
    assert np.array_equal(y[onset_rows], x[onset_rows])
    y = corrupt(x, "jumptrill", 0.8, rng)                     # singles become 2-note chords
    pairs = ((y == TAP).sum(axis=1) == 2) & ((x == TAP).sum(axis=1) == 1)
    assert pairs.sum() > 20
    rows = np.flatnonzero(pairs)
    hands = [{0, 1}, {2, 3}]
    assert all(set(np.flatnonzero(y[r] == TAP)) in hands for r in rows)
    rhythm = np.isin(x, (TAP, HOLD_START)).any(axis=1)
    for name in ("minijack", "trill"):                        # lanes only: rhythm, chords stay
        y = corrupt(x, name, 0.8, rng)
        assert np.array_equal(np.isin(y, (TAP, HOLD_START)).any(axis=1), rhythm)
        assert np.array_equal((y == TAP).sum(axis=1), (x == TAP).sum(axis=1))
    y = corrupt(x, "minijack", 0.8, rng)                      # the same lane again, close by
    on = np.flatnonzero(np.isin(y, (TAP, HOLD_START)).any(axis=1))
    same = [(y[a] == TAP) @ (y[b] == TAP) for a, b in zip(on[:-1], on[1:]) if b - a <= 6]
    assert sum(same) > sum((x[a] == TAP) @ (x[b] == TAP) for a, b in zip(on[:-1], on[1:])
                           if b - a <= 6)
    y = corrupt(x, "trill", 0.8, rng)                         # one hand's two lanes in turn
    assert (y != x).any()
    y = corrupt(x, "weakchord", 0.4, rng)                     # chords off the beat, only
    grown = np.flatnonzero((y == TAP).sum(axis=1) > (x == TAP).sum(axis=1))
    assert len(grown) > 0 and all(r % 12 != 0 for r in grown)
    y = corrupt(x, "lnflood", 0.8, rng)                       # quarter-beat long notes
    starts = np.argwhere((y == HOLD_START) & (x == TAP))
    assert len(starts) > 0 and all(y[r + 3, k] == HOLD_END for r, k in starts)
    z = chart(p_hold=0.0)
    y = corrupt(z, "copypaste", 0.8, rng)                     # bars made the bar before
    same_bars = [np.array_equal(y[b * 48:(b + 1) * 48], y[(b - 1) * 48:b * 48])
                 for b in range(1, len(z) // 48)]
    assert sum(same_bars) > len(same_bars) // 2
    with pytest.raises(ValueError):
        corrupt(x, "mirror", 0.5, rng)


def test_spearman_with_dose() -> None:
    assert spearman_with_dose([0, 1, 2, 3]) == pytest.approx(1.0)
    assert spearman_with_dose([3, 2, 1, 0]) == pytest.approx(-1.0)
    assert np.isnan(spearman_with_dose([1, 1, 1, 1]))
    assert np.isnan(spearman_with_dose([0, np.nan, 1, 2]))
    assert spearman_with_dose([0, 1, 1, 2]) == pytest.approx(0.9487, abs=1e-3)


def test_judge_vs_diag_finds_the_gap_the_judge_follows(tmp_path) -> None:
    from scripts import judge_vs_diag
    rng = np.random.default_rng(0)
    n = 40
    gap = rng.normal(size=n)                                   # the run's jumptrill excess
    noise = rng.normal(size=n)                                 # a column the judge ignores
    judge = tmp_path / "judge"
    judge.mkdir()
    with open(judge / "windows.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["key", "grade", "fold", "start", "chart", "logit", "onset_rows"])
        for i in range(n):
            for s in range(3):
                w.writerow([f"k{i}", "Hard", i % 5, 384 * s, "human", 0.1 * rng.normal(), 9])
                w.writerow([f"k{i}", "Hard", i % 5, 384 * s, "ai",
                            1.0 + gap[i] + 0.1 * rng.normal(), 9])
    run = tmp_path / "run"
    run.mkdir()
    with open(run / "per_song.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["key", "grade", "jumptrill", "human_jumptrill", "other", "human_other",
                    "density_ratio"])
        for i in range(n):
            w.writerow([f"k{i}", "Hard", 0.06 + 0.01 * gap[i], 0.05, noise[i], 0.0, 1.0])
    assert judge_vs_diag.main(["--judge", str(judge), "--run", str(run)]) == 0
    out = json.loads((judge / "vs_diag.json").read_text())
    assert out["songs"] == n and out["gaps"]["jumptrill"]["rho_gap"] > 0.9
    assert abs(out["gaps"]["other"]["rho_gap"]) < 0.4
    assert next(iter(out["gaps"])) == "jumptrill" and out["fit"]["r2_5fold"] > 0.8
    assert "density_ratio" not in out["gaps"]                  # no spread, left out
