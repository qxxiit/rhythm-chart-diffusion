"""The diagnostics as a judge (scripts/diag_judge.py) and its columns in the results table."""

import csv
import json

import numpy as np
import pytest

pytest.importorskip("sklearn")


def write_run(path, columns: dict) -> None:
    path.mkdir(parents=True)
    n = len(next(iter(columns.values())))
    with open(path / "per_song.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["key", "grade", *columns])
        for i in range(n):
            w.writerow([f"k{i}", "Hard" if i % 2 else "Easy", *(v[i] for v in columns.values())])


def test_the_diagnostics_judge_finds_the_shifted_metric_and_not_the_alike_run(tmp_path) -> None:
    from scripts import diag_judge, results_table
    rng = np.random.default_rng(0)
    n = 80
    human = {m: rng.normal(size=n) for m in ("a", "b", "c")}
    human["gone"] = np.where(rng.random(n) < 0.3, rng.random(n), 0.0)  # a rare thing
    human_cols = {f"human_{m}": v for m, v in human.items()}
    far = {"a": rng.normal(2.0, 1.0, n), "b": rng.normal(size=n), "c": rng.normal(size=n),
           "gone": np.zeros(n),                       # a pass took it out altogether
           "extra": rng.normal(size=n), "human_extra": rng.normal(size=n)}
    alike = {"a": rng.normal(size=n), "b": rng.normal(size=n), "c": rng.normal(size=n),
             "gone": np.where(rng.random(n) < 0.3, rng.random(n), 0.0)}
    write_run(tmp_path / "far", {**far, **human_cols, "sr_error": np.full(n, 0.2)})
    write_run(tmp_path / "alike", {**alike, **human_cols, "sr_error": np.full(n, 0.2)})
    out = tmp_path / "diag.md"
    assert diag_judge.main([str(tmp_path / "far"), str(tmp_path / "alike"), "--repeats", "2",
                            "--out", str(out)]) == 0

    f = json.loads((tmp_path / "far" / "diag_judge.json").read_text())
    a = json.loads((tmp_path / "alike" / "diag_judge.json").read_text())
    assert f["metrics"] == 4 and f["dropped"] == ["extra"]   # only in one run: left out
    assert f["trees"]["auc"] > 0.8 and f["logistic"]["paired"] > 0.8
    assert next(iter(f["importance"])) in ("a", "gone")
    assert f["per_metric"]["a"]["effect"] > 1.0 and f["per_metric"]["a"]["run_lower"] < 0.2
    assert "gone" in f["constant"] and f["constant"]["gone"]["value"] == 0.0
    assert 0.1 < f["constant"]["gone"]["human_other"] < 0.5
    for model in ("logistic", "trees"):
        assert 0.3 < f[model]["null_auc"] < 0.7               # the labels swapped: no signal
        assert 0.3 < a[model]["auc"] < 0.7                    # drawn like the human charts
    assert set(f["trees"]["by_grade"]) == {"Easy", "Hard"}
    assert "| far |" in out.read_text() and "per_song" not in out.read_text()
    assert "Constant in the run" in (tmp_path / "far" / "diag_judge.md").read_text()

    table = tmp_path / "table.md"
    assert results_table.main([str(tmp_path / "far"), str(tmp_path / "alike"), "--metrics", "a",
                               "--out", str(table)]) == 0
    rows = json.loads(table.with_suffix(".json").read_text())
    assert rows[1]["diag:auc"] == pytest.approx(f["trees"]["auc"])
    assert rows[2]["diag:paired"] == pytest.approx(a["trees"]["paired"])
    assert "diag:auc" not in rows[0]                          # the human charts' row
