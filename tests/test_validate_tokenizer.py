from scripts.validate_tokenizer import grid_measures, main
from src.data.chart_parser import Chart, Note


def test_selftest_passes(tmp_path) -> None:
    summary, charts = tmp_path / "summary.csv", tmp_path / "charts.csv"

    assert main(["--selftest", "--summary", str(summary), "--charts", str(charts)]) == 0
    assert summary.exists() and charts.exists()


def test_independent_rounding_gap_is_counted() -> None:
    # sections of 10.37 and 5.37 beats: 124.44 -> 124 and 64.44 -> 64 cells,
    # but round(12 * 15.74) = 189 = 124 + 65, one cell more than the recursive grid.
    chart = Chart(4, "song.mp3", [(0, 500.0), (5185, 500.0), (7870, 500.0)], [Note(0, 0)])

    row = grid_measures(chart)

    assert (row["n_indep_gap"], row["n_indep_overlap"]) == (1, 0)
