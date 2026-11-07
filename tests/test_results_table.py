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


def test_songs_of_restricts_rows_and_recomputes_the_judge_from_its_windows(tmp_path) -> None:
    from scripts import results_table
    n = 40
    human = np.linspace(0.0, 0.4, n)
    grades = ["Easy"] * n
    write_run(tmp_path / "F", human, human, grades)
    (tmp_path / "S").mkdir()                                   # a run of the first 10 songs
    with open(tmp_path / "S" / "per_song.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["key", "grade", "hold_share", "human_hold_share", "sr_error", "f1@50"])
        for i in range(10):
            w.writerow([f"k{i}", "Easy", human[i], human[i], 0.1, 0.38])
    judge = tmp_path / "judge"
    judge.mkdir()
    (judge / "summary.json").write_text(json.dumps({
        "labels": {"ai": str(tmp_path / "F"), "S": str(tmp_path / "S")},
        "ai": {"auc": 0.5, "paired": 0.5}, "S": {"auc": 0.5, "paired": 0.5}}))
    with open(judge / "windows.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["key", "grade", "fold", "start", "chart", "logit", "onset_rows"])
        for i in range(n):
            for st in (0, 384):
                w.writerow([f"k{i}", "Easy", 0, st, "human", -1.0, 10])
                w.writerow([f"k{i}", "Easy", 0, st, "ai", 1.0 if i < 10 else -2.0, 10])
                if i < 10:
                    w.writerow([f"k{i}", "Easy", 0, st, "S", 0.5, 10])
    out = tmp_path / "t.md"
    assert results_table.main([str(tmp_path / "F"), str(tmp_path / "S"), "--judge", str(judge),
                               "--metrics", "hold_share", "--songs-of", str(tmp_path / "S"),
                               "--out", str(out)]) == 0
    rows = json.loads(out.with_suffix(".json").read_text())
    assert rows[1]["songs"] == 10 and rows[1]["sr_error"] == 0.2
    assert rows[1]["judge:auc"] == 1.0 and rows[1]["judge:paired"] == 1.0   # F on S's songs
    assert rows[2]["judge:auc"] == 1.0 and rows[2]["judge:minus"].startswith("-0.500")
    assert "judge:minus" not in rows[1]
    # without it the summary's numbers stand
    assert results_table.main([str(tmp_path / "F"), "--judge", str(judge), "--metrics",
                               "hold_share", "--out", str(out)]) == 0
    assert json.loads(out.with_suffix(".json").read_text())[1]["judge:auc"] == 0.5
