"""Beat-grid utilities: ms <-> beat conversion under variable BPM.

Reused by:
  - scripts/analyze_dataset.py  (Aug 12-13, alignment statistics)
  - src/data/tokenizer.py       (Aug 15, encode/decode)
  - src/data/preprocess.py      (Aug 18, resampling log-Mel onto the beat grid)

Coordinates kept deliberately separate:

  local_beat(t)     beats elapsed since the *governing* uninherited timing point.
                    The metronome restarts at every red line, so snap/alignment
                    must be measured against this.

  global_beat(t)    cumulative beats from the first timing point. Monotone,
                    retained for diagnostics and legacy statistics only.

  cell_index(t, d) / time_from_cell(cell, d)
                    THE token time axis. Each timing section restarts on its own
                    metronome; integer cell offsets keep sections contiguous
                    without shifting those local gridlines.

  frame_times(start_cell, n_cells)
                    THE audio time axis: r frames per token cell, placed between
                    consecutive cell times. Never build frame times from a
                    separate 1/48 grid (cell_starts(48) != 4 * cell_starts(12)
                    after an off-beat red line) or by passing fractional cells
                    to time_from_cell (rejected).

global_beat and cell_index differ whenever a red line lands on a non-integer
global beat, which is common. Measuring snap against global_beat is a silent
way to destroy the alignment statistics, so don't.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.data.chart_parser import Chart


@dataclass(frozen=True)
class TimingPoint:
    """An *uninherited* (red) timing point. Inherited (green, SV) lines carry no
    timing information and must be filtered out before constructing BeatGrid."""

    time: float  # ms
    beat_length: float  # ms per beat, > 0
    meter: int = 4  # beats per measure


class BeatGrid:
    def __init__(self, timing_points: list[TimingPoint]):
        tps = sorted((tp for tp in timing_points if tp.beat_length > 0),
                     key=lambda tp: tp.time)
        if not tps:
            raise ValueError("no uninherited timing points")

        self.tps = tps
        self.times = np.array([tp.time for tp in tps], dtype=np.float64)
        self.bls = np.array([tp.beat_length for tp in tps], dtype=np.float64)

        # cumulative beats at the start of each section
        cum = np.zeros(len(tps), dtype=np.float64)
        if len(tps) > 1:
            spans = np.diff(self.times)
            cum[1:] = np.cumsum(spans / self.bls[:-1])
        self.cum = cum

        self._t_list = self.times.tolist()
        self._cum_list = cum.tolist()
        self._cell_starts: dict[int, np.ndarray] = {}

    # ---------- section lookup ----------

    def section_index(self, t: float) -> int:
        """Index of the timing point governing time t. Times before the first
        red line extrapolate backwards from section 0 (osu! does the same)."""
        return max(0, bisect_right(self._t_list, t) - 1)

    def section_index_array(self, t: np.ndarray) -> np.ndarray:
        return np.clip(np.searchsorted(self.times, t, side="right") - 1, 0, None)

    def beat_length_at(self, t) -> np.ndarray | float:
        if np.isscalar(t):
            return self.bls[self.section_index(t)]
        return self.bls[self.section_index_array(np.asarray(t, dtype=np.float64))]

    # ---------- ms -> beat ----------

    def local_beat(self, t) -> np.ndarray | float:
        """Beats since the governing red line. Use this for snap measurement."""
        if np.isscalar(t):
            i = self.section_index(t)
            return (t - self.times[i]) / self.bls[i]
        t = np.asarray(t, dtype=np.float64)
        i = self.section_index_array(t)
        return (t - self.times[i]) / self.bls[i]

    def global_beat(self, t) -> np.ndarray | float:
        """Monotone cumulative beat position. Use this for grid indexing."""
        if np.isscalar(t):
            i = self.section_index(t)
            return self.cum[i] + (t - self.times[i]) / self.bls[i]
        t = np.asarray(t, dtype=np.float64)
        i = self.section_index_array(t)
        return self.cum[i] + (t - self.times[i]) / self.bls[i]

    # ---------- beat -> ms (needed by decode and by mel resampling) ----------

    def time_from_global_beat(self, b) -> np.ndarray | float:
        scalar = np.isscalar(b)
        b = np.atleast_1d(np.asarray(b, dtype=np.float64))
        i = np.clip(np.searchsorted(self.cum, b, side="right") - 1, 0, None)
        t = self.times[i] + (b - self.cum[i]) * self.bls[i]
        return float(t[0]) if scalar else t

    # ---------- re-originated token/audio grid ----------

    def cell_starts(self, divisor: int) -> np.ndarray:
        """Integer cell offsets at timing-section starts for ``divisor`` cells/beat.

        Offsets are recursively accumulated from rounded section lengths rather
        than independently rounded cumulative beats. This gives adjacent timing
        sections one shared boundary cell and no missing cell between them.
        """
        if divisor <= 0:
            raise ValueError("divisor must be positive")
        if divisor not in self._cell_starts:
            starts = np.zeros(self.n_sections, dtype=np.int64)
            if self.n_sections > 1:
                spans = np.diff(self.times) / self.bls[:-1]
                starts[1:] = np.cumsum(np.round(spans * divisor).astype(np.int64))
            self._cell_starts[divisor] = starts
        return self._cell_starts[divisor]

    def cell_index(self, t, divisor: int) -> np.ndarray | int:
        """Map time to the re-originated integer cell coordinate.

        This is the coordinate shared by chart tokenization and audio
        resampling. It can be negative for times before the first red line.
        """
        scalar = np.isscalar(t)
        t = np.atleast_1d(np.asarray(t, dtype=np.float64))
        i = self.section_index_array(t)
        cells = self.cell_starts(divisor)[i] + np.round(
            (t - self.times[i]) / self.bls[i] * divisor
        ).astype(np.int64)
        return int(cells[0]) if scalar else cells

    def time_from_cell(self, cell, divisor: int) -> np.ndarray | float:
        """Map a re-originated integer cell coordinate back to milliseconds.

        A boundary cell belongs to the following timing section, matching the
        section lookup used by :meth:`cell_index` at the red-line time.

        Cells must be whole numbers. Fractional cells used to be truncated
        silently (124.25 -> 124), which turns sub-cell audio frames into a
        staircase. Use :meth:`frame_times` for positions between cells.
        """
        scalar = np.isscalar(cell)
        cell = np.atleast_1d(np.asarray(cell))
        if cell.dtype.kind == "f":
            if not np.all(np.isfinite(cell)) or not np.all(cell == np.round(cell)):
                raise ValueError(
                    "time_from_cell takes whole cells; use frame_times() for sub-cell positions"
                )
        elif cell.dtype.kind not in "iu":
            raise TypeError(f"integer cells expected, got dtype {cell.dtype}")
        cell = cell.astype(np.int64)
        starts = self.cell_starts(divisor)
        i = np.clip(np.searchsorted(starts, cell, side="right") - 1, 0, None)
        t = self.times[i] + (cell - starts[i]) / divisor * self.bls[i]
        return float(t[0]) if scalar else t

    def frame_times(self, start_cell: int, n_cells: int, divisor: int = 12,
                    frames_per_cell: int = 4) -> np.ndarray:
        """Times (ms) of the audio frames for cells [start_cell, start_cell + n_cells).

        With r = frames_per_cell, frame r*c + j sits j/r of the way from cell c
        to cell c + 1:

            t = time_from_cell(c) + (j / r) * (time_from_cell(c + 1) - time_from_cell(c))

        so frame r*c lands exactly on cell c (the alignment the Conv1D with
        kernel = stride = r relies on), and inside one timing section the
        spacing is beat_length / (divisor * r), i.e. 1/48 beat by default.

        For a chunk: frame_times(meta.start_cell, L) gives its L * r frames.
        """
        if n_cells < 0:
            raise ValueError("n_cells must be >= 0")
        if frames_per_cell < 1:
            raise ValueError("frames_per_cell must be >= 1")
        cells = int(start_cell) + np.arange(n_cells + 1, dtype=np.int64)
        tc = self.time_from_cell(cells, divisor)          # cell edges, n_cells + 1 of them
        j = np.arange(frames_per_cell, dtype=np.float64) / frames_per_cell
        return (tc[:-1, None] + j[None, :] * np.diff(tc)[:, None]).reshape(-1)

    # ---------- diagnostics ----------

    def offbeat_red_lines(self, tol: float = 1e-6) -> int:
        """How many red lines land on a non-integer global beat.

        Every one of these is a point where the global grid and the section's
        own metronome disagree. Feeds the Aug 15 decision on whether grid cells
        are re-origined at each red line."""
        frac = np.abs(self.cum - np.round(self.cum))
        return int(np.sum(frac > tol))

    @property
    def n_sections(self) -> int:
        return len(self.tps)

    @property
    def bpm_main(self) -> float:
        """BPM of the section covering the most time (not the first line)."""
        if len(self.tps) == 1:
            return 60000.0 / self.bls[0]
        edges = np.append(self.times, self.times[-1] + 1.0)
        durations = np.diff(edges)
        return 60000.0 / self.bls[int(np.argmax(durations))]


def from_timing_points(timing_points) -> BeatGrid:
    """Build a beat grid from (time_ms, ms_per_beat) tuples, the format used by
    Chart.timing_points and ChunkMeta.timing_points."""
    return BeatGrid([
        TimingPoint(float(time_ms), float(beat_length))
        for time_ms, beat_length in timing_points
    ])


def from_chart(chart: Chart) -> BeatGrid:
    """Build a beat grid from the parser's uninherited timing-point tuples."""
    return from_timing_points(chart.timing_points)


# ---------- snapping ----------

def snap_error_beats(local_beats: np.ndarray, divisor: int) -> np.ndarray:
    """Distance in beats from each note to the nearest 1/divisor gridline.

    Wraparound is handled by rounding in gridline units, so a note at 0.999
    beats correctly snaps to 1.0 rather than to divisor-1."""
    r = np.asarray(local_beats, dtype=np.float64) * divisor
    return np.abs(r - np.round(r)) / divisor


def snap_error_ms(local_beats: np.ndarray, beat_lengths: np.ndarray,
                  divisor: int) -> np.ndarray:
    return snap_error_beats(local_beats, divisor) * np.asarray(beat_lengths)


def grid_index(global_beats: np.ndarray, divisor: int) -> np.ndarray:
    """Legacy cumulative grid cell index for diagnostics and statistics.

    Tokenization and audio resampling must use :meth:`BeatGrid.cell_index`.
    """
    return np.round(np.asarray(global_beats, dtype=np.float64) * divisor).astype(np.int64)
