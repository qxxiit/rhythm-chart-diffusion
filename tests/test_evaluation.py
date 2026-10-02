"""Onset F1 (greedy matching), grammar violation rate and local SR (design doc §4.11)."""

import numpy as np
import pytest

from src.data.chart_parser import Chart, Note
from src.data.chart_writer import write_osu
from src.data.tokenizer import EMPTY, HOLD_BODY, PAD, TAP
from src.evaluation.metrics import onset_f1, violation_rate


def chart(notes) -> Chart:
    return Chart(4, "audio.mp3", [(0, 500.0)], [Note(t, k, e) for t, k, e in notes])


REF = chart([(0, 0, None), (250, 1, None), (500, 2, 900), (750, 3, None), (1000, 0, None)])


def test_identical_charts_score_one() -> None:
    assert onset_f1(REF, REF, 20) == {"precision": 1.0, "recall": 1.0, "f1": 1.0}


def test_tolerance() -> None:
    late = chart([(t + 30, k, e) for t, k, e in [(0, 0, None), (250, 1, None), (500, 2, 900),
                                                 (750, 3, None), (1000, 0, None)]])
    assert onset_f1(late, REF, 20)["f1"] == 0.0
    assert onset_f1(late, REF, 50)["f1"] == 1.0


def test_lanes_on_and_off() -> None:
    mirrored = chart([(0, 3, None), (250, 2, None), (500, 1, 900), (750, 0, None), (1000, 3, None)])
    assert onset_f1(mirrored, REF, 20)["f1"] == 0.0
    assert onset_f1(mirrored, REF, 20, lanes=False)["f1"] == 1.0


def test_greedy_matching_is_one_to_one_and_closest_first() -> None:
    ref = chart([(0, 0, None), (40, 0, None)])
    pred = chart([(25, 0, None), (45, 0, None)])        # 45 takes 40 (5 ms), 25 takes 0 (25 ms)
    assert onset_f1(pred, ref, 50)["f1"] == 1.0
    crowd = chart([(38, 0, None), (40, 0, None), (42, 0, None)])
    r = onset_f1(crowd, chart([(40, 0, None)]), 50)      # one reference note, three guesses
    assert r["precision"] == pytest.approx(1 / 3) and r["recall"] == 1.0


def test_empty_charts() -> None:
    empty = chart([])
    assert onset_f1(empty, empty)["f1"] == 1.0
    assert onset_f1(empty, REF)["f1"] == 0.0 and onset_f1(REF, empty)["f1"] == 0.0


def test_violation_rate() -> None:
    x = np.full((8, 4), EMPTY)
    x[2, 1] = TAP
    x[6:] = PAD
    assert violation_rate(x) == 0.0
    x[4, 3] = HOLD_BODY                                  # a hold body out of nowhere
    assert violation_rate(x) == pytest.approx(2 / 24)    # itself and the orphaned cell after it


def test_local_star_rating(tmp_path) -> None:
    sr = pytest.importorskip("src.evaluation.sr")
    sparse = chart([(t, t // 500 % 4, None) for t in range(0, 30_000, 500)])
    dense = chart([(t, t // 125 % 4, None) for t in range(0, 30_000, 125)])
    assert sr.star_rating(dense) > sr.star_rating(sparse) > 0
    write_osu(tmp_path / "d.osu", dense)
    assert sr.star_rating_file(tmp_path / "d.osu") == sr.star_rating(dense)


def test_compare_runs_pairs_songs_by_key(tmp_path, capsys) -> None:
    import csv

    from scripts import compare_runs
    fields = ["key", "path", "grade", "f1@50", "sr_target", "sr_gen", "coverage",
              "coverage_chance", "human_coverage", "human_coverage_chance", "motion_pred",
              "human_motion_pred", "density_ratio"]

    def write(name: str, shift: float, keys: list[str]):
        d = tmp_path / name
        d.mkdir()
        with open(d / "per_song.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for k in keys:
                i = int(k[1:])
                w.writerow({"key": k, "path": f"{k}.osu", "grade": "Hard" if i % 2 else "Easy",
                            "f1@50": 0.3 + 0.001 * i + shift, "sr_target": 3.0,
                            "sr_gen": 3.1 + shift, "coverage": 0.01 + shift / 10,
                            "coverage_chance": 0.005, "human_coverage": 0.04,
                            "human_coverage_chance": 0.006,
                            "motion_pred": "" if i == 0 else 0.05 + shift,
                            "human_motion_pred": 0.13, "density_ratio": 1.05})
        return d

    keys = [f"k{i}" for i in range(30)]
    ref = write("ref", 0.0, keys)
    run = write("run", 0.02, keys[::-1][:25])            # other order, 25 songs in common
    assert compare_runs.main([str(ref), str(run), "--by-grade"]) == 0
    out = capsys.readouterr().out
    assert "25 songs in common" in out
    lines = {ln.split()[0]: ln for ln in out.splitlines() if ln.startswith("  ") and ln.split()}
    assert "+0.0200 [+0.0200, +0.0200]" in lines["f1@50"] and "100%" in lines["f1@50"]
    assert "+0.0200" in lines["sr_bias"]
    assert "0.1300" in lines["motion_pred"]                 # the human level
    assert "x1.20" in lines["coverage/chance"]              # (0.012 / 0.005) / (0.010 / 0.005)
    assert "Easy" in lines and "Hard" in lines
    assert compare_runs.main([str(ref)]) == 2
