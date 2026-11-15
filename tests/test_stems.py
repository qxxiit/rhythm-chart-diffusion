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


def _store_cache(tmp_path, n_frames=4000, stems=True):
    """A logmel cache of one audio file for two charts, with or without its stems."""
    cache = tmp_path / "cache"
    (cache / "logmel").mkdir(parents=True)
    rng = np.random.default_rng(1)
    mix = rng.normal(-4, 2, size=(n_frames, 80)).astype(np.float16)
    np.save(cache / "logmel" / "a1.npy", mix)
    (cache / "logmel" / "a1.json").write_text(json.dumps(
        {"mean": mix.astype(np.float32).mean(0).tolist(),
         "std": mix.astype(np.float32).std(0).tolist()}))
    with open(cache / "logmel" / "index.csv", "w", newline="") as f:
        f.write("key,audio_id,audio\nk1,a1,x/a.mp3\nk2,a2,y/b.mp3\n")
    st = rng.normal(-6, 3, size=(n_frames, 64)).astype(np.float16)
    if stems:
        (cache / "stems").mkdir()
        np.save(cache / "stems" / "a1.npy", st)
        (cache / "stems" / "a1.json").write_text(json.dumps(
            {"mean": st.astype(np.float32).mean(0).tolist(),
             "std": st.astype(np.float32).std(0).tolist()}))
    return cache, st


def test_mel_store_puts_the_stems_after_the_mix(tmp_path):
    from src.data.beat_grid import from_timing_points
    from src.data.mel import N_MELS, STEM_BANDS, MelStore, on_grid, open_mel
    cache, st = _store_cache(tmp_path)
    grid = from_timing_points([(0.0, 500.0)])
    mix_only, with_stems = MelStore(cache), open_mel(cache, stems=True)
    assert mix_only.channels == N_MELS and with_stems.channels == N_MELS + STEM_BANDS
    a = mix_only.frames("k1", grid, 0, 24, 96)
    b = with_stems.frames("k1", grid, 0, 24, 96)
    assert a.shape == (96 * 4, 80) and b.shape == (96 * 4, 144)
    assert np.array_equal(b[:, :80], a)
    times = grid.frame_times(24, 96, frames_per_cell=4)
    s32 = st.astype(np.float32)
    want = (on_grid(st, times) - s32.mean(0)) / s32.std(0)
    assert np.allclose(b[:, 80:], want, atol=1e-4)
    assert mix_only.has("k1") and with_stems.has("k1")
    assert not with_stems.has("k2") and not with_stems.has("nope")    # a2 has no stems
    cache2, _ = _store_cache(tmp_path / "two", stems=False)
    assert MelStore(cache2).has("k1") and not open_mel(cache2, stems=True).has("k1")


def test_a_widened_model_starts_as_the_narrow_one():
    import pytest
    import torch

    from src.data.tokenizer import MASK, K, L
    from src.models.diffusion import Denoiser, DenoiserConfig, widen_audio
    torch.manual_seed(0)
    small = dict(d_model=64, lane_dim=16, n_layers=2, n_heads=2, d_ff=128)
    narrow = Denoiser(DenoiserConfig(**small)).eval()
    wide = Denoiser(DenoiserConfig(**small, n_mels=144)).eval()
    wide.load_state_dict(widen_audio(narrow.state_dict(), 144))
    x = torch.full((2, L, K), MASK, dtype=torch.long)
    x[:, :100] = torch.randint(0, 3, (2, 100, K))
    mel80 = torch.randn(2, L * 4, 80)
    mel144 = torch.cat([mel80, 5 * torch.randn(2, L * 4, 64)], dim=-1)
    s, b = torch.tensor([3.0, 4.0]), torch.tensor([400.0, 500.0])
    with torch.no_grad():
        want = narrow(x, mel80, s, b)
        assert torch.allclose(wide(x, mel144, s, b), want, atol=1e-5)    # stems weigh 0 at first
        assert torch.allclose(narrow(x, mel144, s, b), want, atol=1e-6)  # the stems are skipped
        with pytest.raises(ValueError, match="144 audio channels"):
            wide(x, mel80, s, b)
    assert widen_audio(narrow.state_dict(), 80) is not None
    with pytest.raises(ValueError):
        widen_audio(wide.state_dict(), 80)
