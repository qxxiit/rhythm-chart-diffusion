import numpy as np
import pytest

from src.data.beat_grid import BeatGrid, TimingPoint, from_chart, from_timing_points, grid_index
from src.data.chart_parser import Chart


@pytest.fixture
def offbeat_grid() -> BeatGrid:
    # The red line at 5185 ms is 10.37 beats into section 0: 12 * 10.37 = 124.44 -> cell 124.
    return BeatGrid([TimingPoint(0.0, 500.0), TimingPoint(5185.0, 400.0)])


def test_reoriginated_cells_round_trip_across_offbeat_red_line(offbeat_grid: BeatGrid) -> None:
    grid = offbeat_grid
    times = np.array([0.0, 500.0, 5185.0, 5185.0 + 400.0 / 12])

    cells = grid.cell_index(times, 12)

    np.testing.assert_array_equal(cells, [0, 12, 124, 125])
    np.testing.assert_allclose(grid.time_from_cell(cells, 12), times)
    # the legacy cumulative grid misses the same note
    legacy = grid.time_from_global_beat(grid_index(grid.global_beat(times[-1]), 12) / 12)
    assert legacy != pytest.approx(times[-1])


def test_boundary_cell_belongs_to_following_section(offbeat_grid: BeatGrid) -> None:
    grid = offbeat_grid

    np.testing.assert_array_equal(grid.cell_starts(12), [0, 124])
    assert grid.cell_index(5185.0, 12) == 124
    assert grid.time_from_cell(124, 12) == pytest.approx(5185.0)


def test_cells_before_first_red_line_are_preserved(offbeat_grid: BeatGrid) -> None:
    grid = offbeat_grid

    assert grid.cell_index(-250.0, 12) == -6
    assert grid.time_from_cell(-6, 12) == pytest.approx(-250.0)


def test_from_chart_and_from_timing_points_build_the_same_grid() -> None:
    tps = [(0, 500.0), (5185, 400.0)]
    chart = Chart(key_count=4, audio_filename="song.mp3", timing_points=tps)

    for grid in (from_chart(chart), from_timing_points(tps)):
        np.testing.assert_array_equal(grid.times, [0.0, 5185.0])
        np.testing.assert_array_equal(grid.bls, [500.0, 400.0])


def test_time_from_cell_rejects_fractional_cells(offbeat_grid: BeatGrid) -> None:
    # Fractional cells used to be truncated silently: all four came back as 5185.0.
    with pytest.raises(ValueError):
        offbeat_grid.time_from_cell(np.array([124.0, 124.25, 124.5, 124.75]), 12)
    with pytest.raises(ValueError):
        offbeat_grid.time_from_cell(124.5, 12)
    assert offbeat_grid.time_from_cell(np.array([124.0]), 12)[0] == pytest.approx(5185.0)


def test_frame_times_land_on_cells_across_a_red_line(offbeat_grid: BeatGrid) -> None:
    frames = offbeat_grid.frame_times(100, 60)                 # cells 100..159 cross cell 124

    assert frames.shape == (240,)
    np.testing.assert_allclose(frames[::4], offbeat_grid.time_from_cell(np.arange(100, 160), 12))
    assert np.all(np.diff(frames) > 0)


def test_frame_times_are_1_48_beat_inside_a_section(offbeat_grid: BeatGrid) -> None:
    np.testing.assert_allclose(offbeat_grid.frame_times(0, 12), np.arange(48) * 500.0 / 48)


def test_a_separate_1_48_grid_drifts_from_the_cells(offbeat_grid: BeatGrid) -> None:
    # Why frame_times exists: re-originating a 1/48 grid on its own puts the
    # section start at 498 instead of 4 * 124 = 496, so frame 500 is 16.67 ms
    # away from token cell 125.
    assert offbeat_grid.cell_starts(48)[1] == 498
    assert offbeat_grid.time_from_cell(125, 12) - offbeat_grid.time_from_cell(500, 48) \
        == pytest.approx(400.0 * 2 / 48)
    assert offbeat_grid.frame_times(125, 1)[0] == pytest.approx(offbeat_grid.time_from_cell(125, 12))


def test_frame_times_random_grids() -> None:
    rng = np.random.default_rng(0)
    for _ in range(300):
        n = int(rng.integers(1, 15))
        times = np.sort(rng.choice(np.arange(-500, 240_000), size=n, replace=False))
        grid = BeatGrid([TimingPoint(float(t), 60000.0 / rng.uniform(80, 300)) for t in times])
        first = grid.cell_index(-1000.0, 12)
        n_cells = grid.cell_index(250_000.0, 12) - first

        frames = grid.frame_times(first, n_cells)

        np.testing.assert_allclose(
            frames[::4], grid.time_from_cell(first + np.arange(n_cells), 12))
        assert np.all(np.diff(frames) > 0)
