"""scripts/stems.py: the stems' features on the mel cache's frames, the order, resuming."""

import csv
import json
import wave

import numpy as np

from scripts import stems
from src.data.audio import SAMPLE_RATE, decode_osu, resample
from src.data.mel import log_mel
from tests.synth_audio import clicks


def _wav(path, x, sr):
    y = (np.clip(np.stack([x, x], axis=1), -1, 1) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(y.tobytes())


def _cache(tmp_path, n_audio=3):
    """A raw folder of short wavs, their logmel sidecars and index, a manifest."""
    raw, cache = tmp_path / "raw", tmp_path / "cache"
    (cache / "logmel").mkdir(parents=True)
    splits = ["train", "val", "test"]
    index, manifest = [], []
    for i in range(n_audio):
        rel = f"{i}/audio.wav"
        (raw / str(i)).mkdir(parents=True)
        x = clicks([200 + 150 * j for j in range(10)], 44100, 2.0, seed=i)
        _wav(raw / rel, x, 44100)
        d = decode_osu(raw / rel)
        frames = len(log_mel(resample(d.samples, d.sample_rate, SAMPLE_RATE)))
        aid = f"aid{i}"
        (cache / "logmel" / f"{aid}.json").write_text(json.dumps({"audio": rel, "frames": frames}))
        for c in range(2):                                # two charts per audio file
            key = f"k{i}{c}"
            index.append({"key": key, "audio_id": aid, "audio": rel})
            manifest.append({"key": key, "split": splits[i % 3]})
    for name, rows in (("logmel/index.csv", index), ("manifest.csv", manifest)):
        path = cache / name if name.startswith("logmel") else tmp_path / name
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    return raw, cache, tmp_path / "manifest.csv"


def _fake_separator(mono, rate):
    """Four 'stems' at 44.1 kHz: the mix scaled, so each has the mix's frames."""
    wav = resample(mono, rate, 44100)
    return np.stack([wav * g for g in (0.5, 0.2, 0.2, 0.1)]), 44100


def test_pool_bands_sums_the_power():
    rng = np.random.default_rng(0)
    lm = rng.normal(-3, 2, size=(7, 80)).astype(np.float32)
    got = stems.pool_bands(lm)
    want = np.log(np.exp(lm.astype(np.float64)).reshape(7, 16, 5).sum(axis=2))
    assert got.shape == (7, 16)
    assert np.allclose(got, want, atol=1e-4)


def test_todo_val_and_test_first_each_audio_once(tmp_path):
    _, cache, manifest = _cache(tmp_path)
    got = stems.todo(cache, manifest, ("val", "test", "train"))
    assert [t[0] for t in got] == ["aid1", "aid2", "aid0"]     # val, test, then train
    assert [t[2] for t in got] == ["val", "test", "train"]
    assert stems.todo(cache, manifest, ("train",)) == [("aid0", "0/audio.wav", "train")]


def test_process_writes_the_mixtures_frames_and_resumes(tmp_path, monkeypatch):
    raw, cache, manifest = _cache(tmp_path)
    sources = ("drums", "bass", "other", "vocals")
    monkeypatch.setattr(stems, "demucs_separator", lambda device: (_fake_separator, sources))
    argv = ["--cache", str(cache), "--raw", str(raw), "--manifest", str(manifest),
            "--min-free", "0"]
    assert stems.main([*argv, "--limit", "2"]) == 0
    out = cache / "stems"
    done = sorted(p.stem for p in out.glob("*.json"))
    assert done == ["aid1", "aid2"]                            # val and test first
    for aid in done:
        meta = json.loads((out / f"{aid}.json").read_text())
        x = np.load(out / f"{aid}.npy")
        mix = json.loads((cache / "logmel" / f"{aid}.json").read_text())
        assert x.dtype == np.float16 and x.shape == (mix["frames"], 4 * 16)
        assert meta["sources"] == list(sources) and len(meta["mean"]) == len(meta["std"]) == 64
        assert np.isfinite(x.astype(np.float32)).all()
    before = (out / "aid1.npy").stat().st_mtime_ns
    assert stems.main(argv) == 0                               # the rest, the done ones kept
    assert sorted(p.stem for p in out.glob("*.json")) == ["aid0", "aid1", "aid2"]
    assert (out / "aid1.npy").stat().st_mtime_ns == before
    assert not list(out.glob("*.tmp.npy"))


def test_stops_when_the_disk_is_full(tmp_path, monkeypatch):
    raw, cache, manifest = _cache(tmp_path)
    monkeypatch.setattr(stems, "demucs_separator",
                        lambda device: (_fake_separator, ("d", "b", "o", "v")))
    argv = ["--cache", str(cache), "--raw", str(raw), "--manifest", str(manifest),
            "--min-free", "1e9"]
    assert stems.main(argv) == 3
    assert not list((cache / "stems").glob("*.json"))
