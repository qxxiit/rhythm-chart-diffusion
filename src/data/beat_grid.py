"""Beat-grid utilities: ms <-> beat conversion under variable BPM.

Reused by:
  - scripts/analyze_dataset.py  (Aug 12-13, alignment statistics)
  - src/data/tokenizer.py       (Aug 15, encode/decode)
  - src/data/preprocess.py      (Aug 18, resampling log-Mel onto the beat grid)

Two beat coordinates are deliberately kept separate:

  local_beat(t)   beats elapsed since the *governing* uninherited timing point.
                  The metronome restarts at every red line, so snap/alignment
                  must be measured against this.

  global_beat(t)  cumulative beats from the first timing point. Monotone,
                  used for grid cell indexing.

They differ whenever a red line lands on a non-integer global beat, which is
common. Measuring snap against global_beat is a silent way to destroy the
alignment statistics, so don't.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass

import numpy as np


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
    """Grid cell index for each note at the given resolution."""
    return np.round(np.asarray(global_beats, dtype=np.float64) * divisor).astype(np.int64)