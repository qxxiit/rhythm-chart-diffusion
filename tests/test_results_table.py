"""The one results table (scripts/results_table.py, docs/EVALUATION.md)."""

import csv
import json

import numpy as np


def write_run(path, values, human, grades):
    path.mkdir(parents=True)
    with open(path / "per_song.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["key", "grade", "hold_share", "human_hold_share", "sr_error", "f1@50"])
        for i, (v, h, g) in enumerate(zip(values, human, grades, strict=True)):
            w.writerow([f"k{i}", g, v, h, 0.2, 0.38])


def test_band_shares_and_judge_columns(tmp_path) -> None:
    from scripts import results_table
    rng = np.random.default_rng(0)
    n = 60
    human = rng.uniform(0.0, 0.4, size=n)
    grades = ["Easy" if i < 30 else "Hard" for i in range(n)]
    write_run(tmp_path / "close", human + rng.normal(0, 0.01, n), human, grades)
    write_run(tmp_path / "far", human + 1.0, human, grades)         # outside every band
    judge = tmp_path / "judge"
    judge.mkdir()
    (judge / "summary.json").write_text(json.dumps({
        "labels": {"ai": str(tmp_path / "close"), "far": str(tmp_path / "far")},
        "ai": {"auc": 0.9, "paired": 0.88},
        "far": {"auc": 0.99, "paired": 0.97, "minus_ai": {"mean": 1.2, "lo": 1.0, "hi": 1.4}}}))
    out = tmp_path / "table.md"
    assert results_table.main([str(tmp_path / "close"), str(tmp_path / "far"), "--judge",
                               str(judge), "--metrics", "hold_share", "absent",
                               "--out", str(out)]) == 0
    rows = json.loads(out.with_suffix(".json").read_text())
    assert rows[0]["run"] == "human charts" and 0.7 < rows[0]["hold_share"] < 0.9
    assert rows[1]["hold_share"] > 0.6 and rows[2]["hold_share"] == 0.0
    assert rows[1]["judge:auc"] == 0.9 and rows[2]["judge:minus"].startswith("+1.200")
    assert "absent" not in rows[1] and "| far |" in out.read_text()
