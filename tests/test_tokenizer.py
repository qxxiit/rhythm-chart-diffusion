import numpy as np
import pytest

from src.data.beat_grid import BeatGrid, TimingPoint, from_chart, grid_index
from src.data.chart_parser import Chart, Note
from scripts.tokenizer import (
    D,
    EMPTY,
    HOLD_BODY,
    HOLD_END,
    HOLD_START,
    L,
    PAD,
    TAP,
    encode,
)


@pytest.fixture
def variable_tempo_grid() -> BeatGrid:
    return BeatGrid([
        TimingPoint(0.0, 500.0),
        TimingPoint(5185.0, 400.0),
    ])


def test_reoriginated_cells_round_trip_across_offbeat_red_line(
    variable_tempo_grid: BeatGrid,
) -> None:
    grid = variable_tempo_grid
    times = np.array([0.0, 500.0, 5185.0, 5185.0 + 400.0 / 12])

    cells = grid.cell_index(times, 12)

    np.testing.assert_array_equal(cells, [0, 12, 124, 125])
    np.testing.assert_allclose(grid.time_from_cell(cells, 12), times)
    assert grid.time_from_global_beat(grid_index(grid.global_beat(times[-1]), 12) / 12) != pytest.approx(times[-1])


def test_boundary_cell_belongs_to_following_section(variable_tempo_grid: BeatGrid) -> None:
    grid = variable_tempo_grid

    np.testing.assert_array_equal(grid.cell_starts(12), [0, 124])
    assert grid.cell_index(5185.0, 12) == 124
    assert grid.time_from_cell(124, 12) == pytest.approx(5185.0)


def test_cells_before_first_red_line_are_preserved(variable_tempo_grid: BeatGrid) -> None:
    grid = variable_tempo_grid

    assert grid.cell_index(-250.0, 12) == -6
    assert grid.time_from_cell(-6, 12) == pytest.approx(-250.0)


def test_from_chart_builds_beat_grid() -> None:
    chart = Chart(
        key_count=4,
        audio_filename="song.mp3",
        timing_points=[(0, 500.0), (5185, 400.0)],
    )

    grid = from_chart(chart)

    np.testing.assert_array_equal(grid.times, [0.0, 5185.0])
    np.testing.assert_array_equal(grid.bls, [500.0, 400.0])


def test_encode_uses_reoriginated_cells_and_preserves_negative_cells() -> None:
    chart = Chart(
        key_count=4,
        audio_filename="song.mp3",
        timing_points=[(0, 500.0), (5185, 400.0)],
        notes=[
            Note(-250, 0),
            Note(500, 1, 510),
            Note(5185, 2, 5185 + round(2 * 400 / D)),
        ],
    )

    tokens, metas, stats = encode(chart, sr=4.5)
    full = tokens.reshape(-1, 4)

    assert tokens.shape == (1, L, 4)
    assert metas[0].cell_offset == 6
    assert metas[0].start_cell == -6
    assert full[0, 0] == TAP
    assert full[18, 1] == TAP
    assert full[130, 2] == HOLD_START
    assert full[131, 2] == HOLD_BODY
    assert full[132, 2] == HOLD_END
    assert np.all(full[133:] == PAD)
    assert stats.n_notes == 3
    assert stats.n_collisions == 0
    assert stats.n_demoted_holds == 1
    np.testing.assert_array_equal(stats.lane_hist, [1, 1, 1, 0])


def test_encode_keeps_first_token_when_notes_collide() -> None:
    chart = Chart(
        key_count=4,
        audio_filename="song.mp3",
        timing_points=[(0, 500.0)],
        notes=[Note(0, 0), Note(0, 0)],
    )

    tokens, _, stats = encode(chart, sr=1.0)

    assert tokens[0, 0, 0] == TAP
    assert stats.n_collisions == 1