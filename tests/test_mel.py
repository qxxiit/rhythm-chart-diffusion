"""mel.py: the fixed-hop log-Mel, its time axis, and reading it on the token grid."""

import csv
import json

import numpy as np
import pytest

from src.data.beat_grid import from_timing_points
from src.data.mel import (
    HOP,
    N_MELS,
    SILENCE,
    STD_FLOOR,
    FakeMelStore,
    MelStore,
    log_mel,
    mel_filterbank,
    on_grid,
    song_stats,
)

MS_PER_FRAME = 1000 * HOP / 22050            # 5.805 ms


def test_filterbank_is_librosa_slaney():
    fb = mel_filterbank()
    assert fb.shape == (N_MELS, 257) and (fb.sum(axis=1) > 0).all()     # no empty band
    assert np.all(np.diff(np.argmax(fb, axis=1)) >= 0)                    # bands go up
    librosa = pytest.importorskip("librosa")
    ref = librosa.filters.mel(sr=22050, n_fft=512, n_mels=80, fmin=30, fmax=11025)
    np.testing.assert_allclose(fb, ref, atol=1e-7)


def test_log_mel_matches_librosa_and_is_centred():
    rng = np.random.default_rng(0)
    y = rng.normal(0, 0.1, 22050 * 3).astype(np.float32)
    m = log_mel(y)
    assert m.shape == (1 + len(y) // HOP, N_MELS) and m.dtype == np.float32
    librosa = pytest.importorskip("librosa")
    ref = librosa.feature.melspectrogram(y=y, sr=22050, n_fft=512, hop_length=HOP, n_mels=80,
                                         fmin=30, fmax=11025, center=True, pad_mode="constant")
    np.testing.assert_allclose(m, np.log(ref + 1e-6).T, atol=1e-3)


@pytest.mark.parametrize("n", [12800, 13000, 31111])
def test_frame_k_is_centred_on_sample_128k(n):
    y = np.zeros(22050 * 2, dtype=np.float32)
    y[n] = 1.0
    energy = log_mel(y).sum(axis=1)
    assert abs(int(np.argmax(energy)) - n / HOP) <= 0.5


def test_silence_and_on_grid():
    assert log_mel(np.zeros(4096, np.float32)).max() == pytest.approx(SILENCE)
    ramp = np.repeat(np.arange(100, dtype=np.float32)[:, None], N_MELS, axis=1)
    got = on_grid(ramp, np.array([-1.0, 0.0, MS_PER_FRAME, 2.5 * MS_PER_FRAME,
                                  99 * MS_PER_FRAME, 99.5 * MS_PER_FRAME]))[:, 0]
    np.testing.assert_allclose(got, [SILENCE, 0, 1, 2.5, 99, SILENCE], atol=1e-4)
    mmap_like = ramp.astype(np.float16)
    np.testing.assert_allclose(on_grid(mmap_like, np.array([10 * MS_PER_FRAME]))[0], 10)


def test_song_stats_floor():
    x = np.zeros((50, N_MELS), np.float32)
    x[:, 0] = np.arange(50)
    mean, std = song_stats(x)
    assert mean[0] == pytest.approx(24.5) and std[0] == pytest.approx(np.arange(50).std())
    assert std[1] == pytest.approx(STD_FLOOR)        # float32 vs 0.1: numpy 1.x compares in float64


def make_store(tmp_path, logmel, key="k" * 16, aid="a" * 16):
    d = tmp_path / "logmel"
    d.mkdir()
    np.save(d / f"{aid}.npy", logmel.astype(np.float16))
    mean, std = song_stats(logmel)
    (d / f"{aid}.json").write_text(json.dumps({"mean": mean.tolist(), "std": std.tolist()}))
    with open(d / "index.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["key", "audio_id", "audio"])
        w.writeheader()
        w.writerow({"key": key, "audio_id": aid, "audio": "x/audio.mp3"})
    return key


def test_store_reads_frames_at_the_grid_times(tmp_path):
    n = 3000                                              # 17.4 s of frames
    logmel = np.repeat((np.arange(n, dtype=np.float32) % 97)[:, None] / 10, N_MELS, axis=1)
    logmel[:, 1] *= -1
    key = make_store(tmp_path, logmel)
    tps = [(1000.0, 400.0), (5000.0, 300.0)]              # a red line at 1 s, a change at 5 s
    grid = from_timing_points(tps)
    raw, song = MelStore(tmp_path, norm="none"), MelStore(tmp_path)
    assert raw.has(key) and not raw.has("x" * 16)
    cell_offset = 48                                      # row 0 is one bar before the red line
    got = raw.frames(key, grid, cell_offset, 96, 384)
    want = on_grid(logmel, grid.frame_times(96 - cell_offset, 384))
    assert got.shape == (1536, N_MELS)
    np.testing.assert_allclose(got, want, atol=0.02)      # float16 storage
    mean, std = song_stats(logmel.astype(np.float16).astype(np.float32))
    np.testing.assert_allclose(song.frames(key, grid, cell_offset, 96, 384), (got - mean) / std,
                               atol=0.01)
    whole = raw.chart(key, tps, cell_offset, 2000)       # runs past the audio: silence
    assert whole.shape == (8000, N_MELS)
    t = grid.frame_times(-cell_offset, 2000)
    assert np.all(whole[t < 0] == SILENCE) and np.all(whole[t > (n - 1) * MS_PER_FRAME] == SILENCE)
    np.testing.assert_allclose(whole[4 * 96:4 * 480], got, atol=1e-6)


def test_fake_store_pads_with_the_quietest_value(tmp_path):
    d = tmp_path / "fake_mel"
    d.mkdir()
    fake = np.random.default_rng(0).normal(size=(1536, N_MELS)).astype(np.float16)
    np.save(d / "abc.npy", fake)
    store = FakeMelStore(tmp_path)
    got = store.frames("abc", None, 0, 200, 384)
    np.testing.assert_allclose(got[:4 * 184], fake[800:].astype(np.float32))
    assert np.all(got[4 * 184:] == fake.astype(np.float32).min())
