"""Log-Mel on the token grid (design doc §4.5).

Two steps, so that the costly one runs once per audio file and the grid, which
depends on each chart's timing, is applied when a window is read:

  log_mel(samples)          fixed hop at 22,050 Hz: n_fft 512 (23.2 ms), hop 128
                            (5.8 ms), Hann, power, 80 Slaney mel bands 30 Hz - 11,025 Hz,
                            log(x + 1e-6). Frame k is centred on sample 128 k, so frame
                            0 is 0 ms of the chart (audio.decode_osu puts it there).
  on_grid(logmel, times)    linear interpolation at the grid times (ms). Before 0 ms
                            and after the audio: SILENCE = log(1e-6), what digital
                            silence gives.

The grid times are BeatGrid.frame_times: 4 frames per token cell, 1/48 beat inside a
timing section (frame 4r + j belongs to row r). A store serves them:

  MelStore       the real features: data/cache/logmel/{audio_id}.npy, one per audio
                 file, standardized per song and mel band when read (norm="song")
  FakeMelStore   cache.oracle_mel, which encodes the answer: data/cache/fake_mel/{key}.npy,
                 plumbing tests only

Both answer frames(key, grid, cell_offset, start_row, n_rows) with float32
[n_rows * 4, 80]. File layout: src/data/cache.py.

MelStore(stems=True) appends the four stems' features (scripts/stems.py:
data/cache/stems/{audio_id}.npy, htdemucs's drums, bass, other and vocals as 16 bands each
on the mixture's own frames), standardized per song with their own statistics: float32
[n_rows * 4, 80 + 64], the mixture first. A model reads the first config.n_mels channels
(Denoiser.hidden), so an 80-channel model takes the same frames.
"""

from __future__ import annotations

import csv
import json
from functools import cache
from pathlib import Path

import numpy as np

from src.data.audio import SAMPLE_RATE
from src.data.beat_grid import BeatGrid, from_timing_points

N_FFT = 512
HOP = 128
N_MELS = 80
FMIN, FMAX = 30.0, 11025.0
EPS = 1e-6
SILENCE = float(np.log(EPS))
FRAMES_PER_CELL = 4
STD_FLOOR = 0.1                 # a band that never moves (a low-pass mp3) is not amplified
STEM_BANDS = 64                 # scripts/stems.py: 4 stems x 16 bands, after the 80 of the mix


def _hz_to_mel(f):
    """Slaney mel scale (librosa's default): linear below 1 kHz, log above."""
    f = np.asarray(f, dtype=np.float64)
    mel = f / (200.0 / 3)
    log_part = 15.0 + np.log(np.maximum(f, 1e-10) / 1000.0) / (np.log(6.4) / 27.0)
    return np.where(f >= 1000.0, log_part, mel)


def _mel_to_hz(m):
    m = np.asarray(m, dtype=np.float64)
    return np.where(m >= 15.0, 1000.0 * np.exp((np.log(6.4) / 27.0) * (m - 15.0)),
                    m * (200.0 / 3))


@cache
def mel_filterbank(sr: int = SAMPLE_RATE, n_fft: int = N_FFT, n_mels: int = N_MELS,
                   fmin: float = FMIN, fmax: float = FMAX) -> np.ndarray:
    """[n_mels, n_fft // 2 + 1] triangles with Slaney area normalization, the same
    matrix as librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)."""
    freqs = np.linspace(0.0, sr / 2, n_fft // 2 + 1)
    edges = _mel_to_hz(np.linspace(_hz_to_mel(fmin), _hz_to_mel(fmax), n_mels + 2))
    ramps = edges[:, None] - freqs[None, :]
    widths = np.diff(edges)
    lower = -ramps[:-2] / widths[:-1, None]
    upper = ramps[2:] / widths[1:, None]
    weights = np.maximum(0.0, np.minimum(lower, upper))
    weights *= (2.0 / (edges[2:] - edges[:-2]))[:, None]
    weights.setflags(write=False)
    return weights


def _hann(n: int) -> np.ndarray:
    return (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n)).astype(np.float32)


def log_mel(samples: np.ndarray, *, hop: int = HOP, block: int = 4096) -> np.ndarray:
    """float32 [1 + len // hop, 80] from mono float32 at 22,050 Hz. Zero padding of
    n_fft / 2 on both ends (librosa center=True, pad_mode="constant")."""
    y = np.asarray(samples, dtype=np.float32)
    pad = np.zeros(N_FFT // 2, dtype=np.float32)
    frames = np.lib.stride_tricks.sliding_window_view(np.concatenate([pad, y, pad]), N_FFT)[::hop]
    n = 1 + len(y) // hop
    fb = mel_filterbank().T.astype(np.float32)
    window = _hann(N_FFT)
    out = np.empty((n, N_MELS), dtype=np.float32)
    for i in range(0, n, block):
        spec = np.fft.rfft(frames[i:min(i + block, n)] * window, axis=1)
        power = (spec.real ** 2 + spec.imag ** 2).astype(np.float32)
        # numpy's Accelerate BLAS on Apple Silicon raises bogus divide/overflow/invalid
        # flags in matmul (as in structure.py); silence them, but check the result.
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            mel = power @ fb
        out[i:i + len(power)] = np.log(mel + EPS)
    if not np.isfinite(out).all():
        raise FloatingPointError("non-finite log-Mel")
    return out


def frame_position(times_ms: np.ndarray, hop: int = HOP) -> np.ndarray:
    """Fractional log_mel frame index of each time."""
    return np.asarray(times_ms, dtype=np.float64) * (SAMPLE_RATE / 1000.0 / hop)


def on_grid(logmel: np.ndarray, times_ms: np.ndarray, hop: int = HOP) -> np.ndarray:
    """float32 [len(times), bands]: logmel linearly interpolated at each time.
    logmel may be a memmap: only the rows the times need are read."""
    pos = frame_position(times_ms, hop)
    out = np.full((len(pos), logmel.shape[1]), SILENCE, dtype=np.float32)
    inside = (pos >= 0) & (pos <= len(logmel) - 1)
    if not inside.any():
        return out
    p = pos[inside]
    i0 = np.floor(p).astype(np.int64)
    lo, hi = int(i0.min()), min(int(i0.max()) + 2, len(logmel))
    block = np.asarray(logmel[lo:hi], dtype=np.float32)
    k = i0 - lo
    k1 = np.minimum(k + 1, len(block) - 1)
    w = (p - i0).astype(np.float32)[:, None]
    out[inside] = (1 - w) * block[k] + w * block[k1]
    return out


def song_stats(logmel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-band mean and std over the whole song, for norm="song"."""
    x = np.asarray(logmel, dtype=np.float64)
    return x.mean(axis=0).astype(np.float32), np.maximum(x.std(axis=0), STD_FLOOR).astype(np.float32)


# ---------- stores ----------

class MelStore:
    """Real log-Mel, one file per audio file, read on each chart's grid.

    norm="song"   (x - mean) / std with the song's per-band statistics (the model input)
    norm="none"   raw log(x + 1e-6), for plots and checks
    """

    def __init__(self, cache: str | Path, norm: str = "song", stems: bool = False):
        if norm not in ("song", "none"):
            raise ValueError(f"unknown norm {norm!r}")
        self.dir = Path(cache) / "logmel"
        self.stems_dir = Path(cache) / "stems" if stems else None
        self.norm = norm
        self.audio_of: dict[str, str] = {}
        index = self.dir / "index.csv"
        if index.exists():
            with open(index, newline="", encoding="utf-8") as f:
                self.audio_of = {r["key"]: r["audio_id"] for r in csv.DictReader(f)}
        self._open: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self._stems: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    @property
    def channels(self) -> int:
        return N_MELS + (STEM_BANDS if self.stems_dir is not None else 0)

    def has(self, key: str) -> bool:
        if key not in self.audio_of:
            return False
        return self.stems_dir is None or (self.stems_dir / f"{self.audio_of[key]}.json").exists()

    @staticmethod
    def _load(folder: Path, aid: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        meta = json.loads((folder / f"{aid}.json").read_text(encoding="utf-8"))
        return (np.load(folder / f"{aid}.npy", mmap_mode="r"),
                np.asarray(meta["mean"], dtype=np.float32),
                np.asarray(meta["std"], dtype=np.float32))

    def audio(self, key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(memmap [n_frames, 80] float16, mean [80], std [80]) of the chart's audio."""
        aid = self.audio_of[key]
        if aid not in self._open:
            self._open[aid] = self._load(self.dir, aid)
        return self._open[aid]

    def stems(self, key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(memmap [n_frames, 64] float16, mean [64], std [64]): the stems' features."""
        aid = self.audio_of[key]
        if aid not in self._stems:
            self._stems[aid] = self._load(self.stems_dir, aid)
        return self._stems[aid]

    def frames(self, key: str, grid: BeatGrid, cell_offset: int, start_row: int,
               n_rows: int) -> np.ndarray:
        """float32 [n_rows * 4, 80] for token rows [start_row, start_row + n_rows)
        ([.., 80 + 64] with stems)."""
        logmel, mean, std = self.audio(key)
        times = grid.frame_times(start_row - cell_offset, n_rows, frames_per_cell=FRAMES_PER_CELL)
        x = on_grid(logmel, times)
        if self.norm == "song":
            x = (x - mean) / std
        if self.stems_dir is None:
            return x
        feats, s_mean, s_std = self.stems(key)
        y = on_grid(feats, times)
        if self.norm == "song":
            y = (y - s_mean) / s_std
        return np.concatenate([x, y], axis=1)

    def chart(self, key: str, timing_points, cell_offset: int, n_rows: int) -> np.ndarray:
        """The whole chart: rows 0 .. n_rows - 1 (n_rows may run past the song)."""
        return self.frames(key, from_timing_points(timing_points), cell_offset, 0, n_rows)

    def __getstate__(self) -> dict:              # DataLoader workers reopen the memmaps
        state = self.__dict__.copy()
        state["_open"] = {}
        state["_stems"] = {}
        return state


class FakeMelStore:
    """cache.oracle_mel per chart (it encodes the answer): plumbing tests only."""

    def __init__(self, cache: str | Path):
        self.dir = Path(cache) / "fake_mel"
        self._open: dict[str, np.ndarray] = {}

    def has(self, key: str) -> bool:
        return (self.dir / f"{key}.npy").exists()

    def frames(self, key: str, grid: BeatGrid | None, cell_offset: int, start_row: int,
               n_rows: int) -> np.ndarray:
        if key not in self._open:
            self._open[key] = np.load(self.dir / f"{key}.npy", mmap_mode="r")
        mel = self._open[key]
        part = np.asarray(mel[start_row * FRAMES_PER_CELL:(start_row + n_rows) * FRAMES_PER_CELL],
                          dtype=np.float32)
        short = n_rows * FRAMES_PER_CELL - len(part)
        if short > 0:                             # past the cache: the song's quietest value
            fill = float(np.min(mel[-384 * FRAMES_PER_CELL:]))
            part = np.concatenate([part, np.full((short, mel.shape[1]), fill, dtype=np.float32)])
        return part

    def chart(self, key: str, timing_points, cell_offset: int, n_rows: int) -> np.ndarray:
        return self.frames(key, None, cell_offset, 0, n_rows)

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_open"] = {}
        return state


def open_mel(cache: str | Path, fake: bool = False, norm: str = "song", stems: bool = False):
    """The store a model reads: stems=True for a model with N_MELS + STEM_BANDS channels
    (reads_stems); the fake store has no stems."""
    if fake and stems:
        raise ValueError("the fake mel has no stems")
    return FakeMelStore(cache) if fake else MelStore(cache, norm=norm, stems=stems)


def reads_stems(model) -> bool:
    """Whether a Denoiser (or anything with .config.n_mels) was trained on the stems too."""
    return int(getattr(getattr(model, "config", None), "n_mels", N_MELS)) > N_MELS
