"""Chart stats as inputs (src/data/chartstats.py): values, buckets, parsing, sampling."""

import csv

import numpy as np
import pytest

from src.data import chartstats
from src.data.tokenizer import EMPTY, HOLD_BODY, HOLD_END, HOLD_START, TAP, K


def chart(lanes, step: int = 6, holds=()) -> np.ndarray:
    """Single notes in the given lanes, one every `step` cells; holds: indices made long."""
    x = np.full((len(lanes) * step + 8, K), EMPTY)
    for i, k in enumerate(lanes):
        r = i * step
        if i in holds:
            x[r, k], x[r + 1:r + 3, k], x[r + 3, k] = HOLD_START, HOLD_BODY, HOLD_END
        else:
            x[r, k] = TAP
    return x


def test_chart_stats_values() -> None:
    jacks = chartstats.chart_stats(chart([0, 0, 0, 0, 1, 1, 1, 1]))
    assert jacks["move_jack"] == pytest.approx(6 / 7) and jacks["hold_share"] == 0
    trill = chartstats.chart_stats(chart([0, 2, 0, 2, 0, 2, 0, 2], holds=(0, 2)))
    assert trill["move_trill"] == 1.0 and trill["move_jack"] == 0
    assert trill["hold_share"] == pytest.approx(2 / 8)
    stairs = chartstats.chart_stats(chart([0, 1, 2, 3, 0, 1, 2, 3]))
    assert stairs["move_trill"] == 0
    lone = chartstats.chart_stats(chart([1], step=6))
    assert np.isnan(lone["move_jack"]) and np.isnan(lone["move_trill"])


def test_spec_buckets_and_null() -> None:
    rng = np.random.default_rng(0)
    table = {f"k{i}": {"split": "train", "sr": 3.0, "hold_share": float(v),
                       "move_jack": float(rng.uniform(0, 0.2)), "move_trill": float("nan")}
             for i, v in enumerate(rng.uniform(0, 0.5, 400))}
    table["k0"]["hold_share"] = 0.0
    spec = chartstats.build_spec(table, list(table), bins=4)
    assert spec["names"] == list(chartstats.NAMES) and spec["bins"] == 4
    assert len(spec["edges"][0]) == 3 and spec["edges"][2] == []     # no values: no edges
    codes = np.array([chartstats.encode(spec, table[k])[0] for k in table])
    assert np.bincount(codes, minlength=4).min() >= 90               # equal shares
    assert chartstats.encode(spec, {"hold_share": 0.0})[0] == 0
    assert chartstats.encode(spec, {"hold_share": 1.0})[0] == 3
    null = chartstats.null_index(spec)
    assert chartstats.encode(spec, {"hold_share": 0.2})[1:] == (null, null)   # missing
    assert chartstats.encode(spec, [float("nan"), None, 0.1]) == (null, null, 0)
    assert chartstats.encode(spec, [0.0, 0.0, 0.0])[2] == 0          # no edges: one bucket
    ties = chartstats.build_spec({f"j{i}": {"split": "train", "sr": 3.0, "hold_share": 0.1,
                                            "move_jack": 0.0 if i < 30 else i / 1000,
                                            "move_trill": 0.1} for i in range(100)},
                                 [f"j{i}" for i in range(100)], bins=8)
    assert ties["edges"][1][0] == 0.0                                 # 30% of the charts: none
    assert chartstats.encode(ties, {"move_jack": 0.0})[1] == 0        # ... have bucket 0
    assert chartstats.encode(ties, {"move_jack": 0.031})[1] == 1


def test_parse() -> None:
    assert chartstats.parse("ln=0.4,jack=0.05") == {"hold_share": 0.4, "move_jack": 0.05}
    assert chartstats.parse("move_trill=0.2, ln=0") == {"move_trill": 0.2, "hold_share": 0.0}
    assert chartstats.parse("") == {}
    for bad in ("ln", "speed=1", "jack=fast"):
        with pytest.raises(ValueError):
            chartstats.parse(bad)


def test_sample_by_sr(tmp_path) -> None:
    path = tmp_path / "chart_stats.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=chartstats.FIELDS)
        w.writeheader()
        for i, sr in enumerate([1.0, 1.1, 3.0, 3.2, 3.1, 6.0]):
            w.writerow({"key": f"c{i}", "split": "val" if i == 4 else "train", "sr": sr,
                        "hold_share": i / 10, "move_jack": "", "move_trill": 0.1})
    table = chartstats.read_stats(path)
    assert np.isnan(table["c0"]["move_jack"]) and table["c3"]["sr"] == 3.2
    rng = np.random.default_rng(0)
    picks = {chartstats.sample(table, 3.05, rng)[0] for _ in range(50)}
    assert picks == {"c2", "c3"}                                     # within 0.3, train only
    key, values = chartstats.sample(table, 4.5, rng)                 # none within: the nearest
    assert key in table and set(values) == set(chartstats.NAMES)
