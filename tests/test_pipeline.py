"""manifest -> token cache -> ChunkDataset -> train.py, on synthetic .osu files.

Every set also gets an audio.mp3 with a click at each note of its first chart, so
the real log-Mel path (decode as osu! does, fixed hop, token grid) is checked for
alignment end to end. Without PyAV's mp3 encoder those tests are skipped."""

import csv
import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from scripts import build_manifest, preprocess_data, sample, tempo_density, train  # noqa: E402
from src.data.chart_parser import Chart, Note, parse_osu  # noqa: E402
from src.data.chart_writer import write_osu  # noqa: E402
from src.data.dataset import ChunkDataset  # noqa: E402
from src.data.mel import MelStore, on_grid  # noqa: E402
from src.data.tokenizer import HOLD_START, PAD, TAP, L  # noqa: E402
from tests.synth_audio import can_write_mp3, clicks, write_mp3  # noqa: E402

requires_mp3 = pytest.mark.skipif(not can_write_mp3(), reason="PyAV without libmp3lame")


def synthetic_song(rng: np.random.Generator, seconds: float = 40.0) -> tuple[list, list]:
    bl = 60000 / rng.uniform(120, 200)
    notes = []
    for k in range(4):
        t = bl * rng.integers(0, 4) / 4
        while t < seconds * 1000:
            is_ln = rng.random() < 0.25
            dur = bl * rng.choice([0.5, 1.0])
            notes.append(Note(round(float(t)), k, round(float(t + dur)) if is_ln else None))
            t += (dur if is_ln else 0) + bl * rng.choice([0.25, 0.5, 1.0])
    notes.sort(key=lambda n: (n.time_ms, n.lane))
    return [(0, bl)], notes


@pytest.fixture(scope="module")
def data(tmp_path_factory) -> Path:
    base = tmp_path_factory.mktemp("data")
    raw = base / "raw"
    rng = np.random.default_rng(0)
    sets = []
    for s in range(8):
        artist, title = f"Artist {s}", f"Song {s % 7}"            # set 7 re-uploads song 0
        beatmaps = []
        for v in range(2):
            tps, notes = synthetic_song(rng)
            bid = 1000 + 10 * s + v
            chart = Chart(4, "audio.mp3", tps, notes)
            folder = raw / f"{100 + s} {artist} - {title}"
            write_osu(folder / f"v{v}.osu", chart, title=title,
                      artist=artist if s != 7 else "Artist 0", version=f"v{v}",
                      beatmap_id=bid if (s, v) != (3, 1) else 0, set_id=100 + s)
            beatmaps.append({"id": bid, "difficulty_rating": 2.0 + s * 0.3 + v})
            if v == 0 and can_write_mp3():            # the set's audio follows chart v0
                onsets = sorted({n.time_ms for n in notes})
                write_mp3(folder / "audio.mp3", clicks(onsets, 44100, 42.0, seed=s), 44100)
        sets.append({"id": 100 + s, "beatmaps": beatmaps})
    meta = base / "metadata" / "beatmapsets.jsonl"
    meta.parent.mkdir(parents=True)
    meta.write_text("\n".join(json.dumps(x) for x in sets) + "\n")

    assert build_manifest.main(["--root", str(raw), "--metadata", str(meta),
                                "--out", str(base / "manifest.csv"), "--workers", "1"]) == 0
    assert preprocess_data.main(["--manifest", str(base / "manifest.csv"), "--root", str(raw),
                                 "--cache", str(base / "cache"), "--fake-mel", "--limit", "100",
                                 "--workers", "1"]) == 0
    if can_write_mp3():
        assert preprocess_data.main(["--manifest", str(base / "manifest.csv"), "--root", str(raw),
                                     "--cache", str(base / "cache"), "--mel", "--workers", "1"]) == 0
    return base


def test_manifest(data: Path) -> None:
    with open(data / "manifest.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 16 and len({r["key"] for r in rows}) == 16
    by_song = {}
    for r in rows:
        by_song.setdefault(r["audio_key"], set()).add(r["split"])
    assert all(len(splits) == 1 for splits in by_song.values())      # a song never straddles
    reupload = [r for r in rows if r["set_id"] in ("100", "107")]
    assert len({r["audio_key"] for r in reupload}) == 1                # grouped across sets
    no_id = [r for r in rows if r["beatmap_id"] == "0"]
    assert len(no_id) == 1 and no_id[0]["sr_api"] == ""               # no API SR ...
    assert no_id[0]["sr"] == no_id[0]["sr_local"] != ""                # ... but a local label
    assert all(r["sr_source"] == "local" and r["drop"] == "" for r in rows)


def test_token_cache_and_dataset(data: Path) -> None:
    ds = ChunkDataset(data / "manifest.csv", data / "cache", ("train", "val", "test"),
                      fake_mel=True)
    assert len(ds.charts) == 16                                        # local SR labels all
    item = ds[0]
    assert item["x0"].shape == (L, 4) and item["x0"].dtype == torch.long
    assert item["mel"].shape == (L * 4, 80) and item["mel"].dtype == torch.float32
    z = np.load(data / "cache" / "tokens" / f"{ds.charts[0]['key']}.npz")
    assert float(item["b"]) == pytest.approx(float(z["beat_len_ms"][0]))
    assert z["cell_offset"] % 48 == 0
    last = ds[len(ds) - 1]["x0"]
    assert (last == PAD).any()                                         # the final chunk pads


def test_bar_windows_stay_aligned(data: Path) -> None:
    chunk = ChunkDataset(data / "manifest.csv", data / "cache", ("train", "val", "test"),
                         fake_mel=True)
    bar = ChunkDataset(data / "manifest.csv", data / "cache", ("train", "val", "test"),
                       window="bar", fake_mel=True)
    level = np.array([0.0, 6.0, 6.0, 2.0, 4.0, 0.0, 0.0])        # cache.oracle_mel
    torch.manual_seed(0)
    shifted = 0
    for i in range(len(bar)):
        ci, c = bar.index[i]
        chart = bar.charts[ci]
        for _ in range(3):
            item = bar[i]
            start = int(item["start"])
            assert start % 48 == 0 and c * L <= start < min((c + 1) * L, chart["n_cells"])
            shifted += start != c * L
            x0 = item["x0"].numpy()
            want = chart["rows"][start:start + L]
            assert np.array_equal(x0[:len(want)], want) and np.all(x0[len(want):] == PAD)
            cells = item["mel"].numpy().reshape(L, 4, 80).mean(axis=1)   # [384, 80]
            for k in range(4):
                band = cells[:, 8 + 16 * k:20 + 16 * k].mean(axis=1)
                real = x0[:, k] != PAD
                assert np.abs(band[real] - (-8 + level[x0[real, k]])).max() < 0.5
            z = np.load(data / "cache" / "tokens" / f"{chart['key']}.npz")
            assert float(item["b"]) == pytest.approx(float(z["beat_len_ms"][0]), rel=1e-6)
        if start == c * L:
            assert torch.equal(item["x0"], chunk[i]["x0"])
    assert shifted > 0


def test_train_overfit_runs_and_learns(data: Path, tmp_path: Path) -> None:
    args = ["--manifest", str(data / "manifest.csv"), "--cache", str(data / "cache"),
            "--out", str(tmp_path), "--run", "t", "--overfit", "2", "--steps", "150",
            "--batch-size", "4", "--lr", "2e-3", "--warmup", "10", "--d-model", "64",
            "--layers", "2", "--heads", "2", "--d-ff", "128", "--log-every", "10",
            "--val-every", "150", "--sample-steps", "8", "--window", "bar", "--fake-mel",
            "--device", "cpu"]
    assert train.main(args) == 0
    with open(tmp_path / "t" / "log.csv", newline="") as f:
        losses = [float(r["loss"]) for r in csv.DictReader(f)]
    assert losses[-1] < 0.5 * losses[0]
    check = json.loads((tmp_path / "t" / "overfit_check.json").read_text())
    assert check["grammar_violations"] == 0
    assert (tmp_path / "t" / "last.pt").exists() and (tmp_path / "t" / "best.pt").exists()

    resumed = train.main([*args[:-2], "--device", "cpu",
                          "--resume", str(tmp_path / "t" / "last.pt"), "--steps", "160"])
    assert resumed == 0

    key = json.loads((tmp_path / "t" / "config.json").read_text())["overfit_keys"][0]
    assert sample.main(["--ckpt", str(tmp_path / "t" / "best.pt"), "--key", key,
                        "--manifest", str(data / "manifest.csv"), "--root", str(data / "raw"),
                        "--cache", str(data / "cache"), "--steps", "8", "--fake-mel",
                        "--device", "cpu"]) == 0
    written = list((tmp_path / "t" / "samples").glob("*.osu"))
    assert len(written) == 1
    chart = parse_osu(written[0])                                     # osu! format round trip
    assert chart.key_count == 4 and chart.audio_filename == "audio.mp3" and chart.notes

    pytest.importorskip("rosu_pp_py")
    from scripts import evaluate
    assert evaluate.main(["--ckpt", str(tmp_path / "t" / "best.pt"), "--split", "train",
                          "--n", "2", "--steps", "4", "--manifest", str(data / "manifest.csv"),
                          "--root", str(data / "raw"), "--cache", str(data / "cache"),
                          "--fake-mel", "--device", "cpu"]) == 0
    summary = json.loads(next((tmp_path / "t").glob("eval_*/summary.json")).read_text())
    assert summary["songs"] == 2 and summary["mean_violation_rate"] == 0.0
    assert summary["mean_ceiling_f1@20"] > 0.99                        # tokenizer round trip
    assert summary["mean_human_rho_all"] > 0          # the fake mel is made from the human chart
    assert 0.0 <= summary["mean_coverage"] <= 1.0


def test_human_baselines(data: Path, tmp_path: Path) -> None:
    from scripts import human_baselines
    assert human_baselines.main(["--manifest", str(data / "manifest.csv"),
                                 "--cache", str(data / "cache"), "--stats", str(tmp_path),
                                 "--split", "train", "--workers", "1"]) == 0
    with open(tmp_path / "chart_ssm_lag.csv", newline="") as f:
        lag = list(csv.DictReader(f))
    assert len(lag) == 32 and float(lag[0]["mean_chart_similarity"]) > 0
    assert (tmp_path / "pattern_baseline.csv").exists()


def test_sr_check_and_tempo_density(data: Path, tmp_path: Path) -> None:
    pytest.importorskip("rosu_pp_py")
    from scripts import check_sr
    assert check_sr.main(["--manifest", str(data / "manifest.csv"), "--root", str(data / "raw"),
                          "--out", str(tmp_path / "sr.csv"), "--stats", str(tmp_path / "st.csv"),
                          "--workers", "1"]) == 0
    with open(tmp_path / "st.csv", newline="") as f:
        stats = {r["metric"]: r["value"] for r in csv.DictReader(f)}
    assert stats["errors"] == "0" and stats["without_api_sr_but_local"] == "1"
    assert float(stats["parsed_vs_local.max_abs"]) == 0.0              # the parser keeps everything
    assert tempo_density.main(["--manifest", str(data / "manifest.csv"),
                               "--cache", str(data / "cache"), "--out", str(tmp_path / "td.csv"),
                               "--longest", "3"]) == 0
    with open(tmp_path / "td.csv", newline="") as f:
        assert sum(int(r["charts"]) for r in csv.DictReader(f)) == 16


def onset_flux_profile(mel: np.ndarray, x0: np.ndarray, reach: int = 6) -> np.ndarray:
    """Mean spectral flux at frames 4r + d (d = -reach..reach) around every onset row r."""
    flux = np.concatenate([[0.0], np.maximum(np.diff(mel, axis=0), 0).sum(axis=1)])
    rows = np.flatnonzero(np.isin(x0, (TAP, HOLD_START)).any(axis=1))
    rows = rows[(4 * rows - reach >= 1) & (4 * rows + reach < len(mel))]
    return np.mean([flux[4 * r - reach:4 * r + reach + 1] for r in rows], axis=0)


@requires_mp3
def test_real_mel_is_aligned_with_the_notes(data: Path) -> None:
    with open(data / "cache" / "logmel" / "index.csv", newline="") as f:
        index = list(csv.DictReader(f))
    assert len(index) == 16 and len({r["audio_id"] for r in index}) == 8
    info = json.loads((data / "cache" / "logmel" / f"{index[0]['audio_id']}.json").read_text())
    assert info["mp3_tag"] == "lame" and info["mp3_trim"] == 1105 and len(info["mean"]) == 80

    with open(data / "manifest.csv", newline="") as f:
        first = [r["key"] for r in csv.DictReader(f) if r["path"].endswith("v0.osu")]
    ds = ChunkDataset(data / "manifest.csv", data / "cache", ("train", "val", "test"),
                      keys=first)
    assert len(ds.charts) == 8
    profiles = [onset_flux_profile(item["mel"].numpy(), item["x0"].numpy())
                for item in (ds[i] for i in range(len(ds)))]
    peak = int(np.argmax(np.mean(profiles, axis=0))) - 6
    assert peak in (-1, 0)                    # the click's rise lands on the onset frame

    store, chart, grid = MelStore(data / "cache", norm="none"), ds.charts[0], ds._grid(0)
    late = []                                 # control: the same audio read 25 ms late
    for c in range(len(chart["rows"]) // L):
        times = grid.frame_times(c * L - chart["cell_offset"], L) - 25.0
        mel = on_grid(store.audio(chart["key"])[0], times)
        late.append(onset_flux_profile(mel, chart["rows"][c * L:(c + 1) * L]))
    assert int(np.argmax(np.mean(late, axis=0))) - 6 >= peak + 2


@requires_mp3
def test_train_and_sample_on_real_mel(data: Path, tmp_path: Path) -> None:
    args = ["--manifest", str(data / "manifest.csv"), "--cache", str(data / "cache"),
            "--out", str(tmp_path), "--run", "r", "--overfit", "2", "--steps", "20",
            "--batch-size", "4", "--warmup", "5", "--d-model", "64", "--layers", "2",
            "--heads", "2", "--d-ff", "128", "--log-every", "10", "--val-every", "20",
            "--sample-steps", "4", "--device", "cpu"]
    assert train.main(args) == 0
    key = json.loads((tmp_path / "r" / "config.json").read_text())["overfit_keys"][0]
    assert sample.main(["--ckpt", str(tmp_path / "r" / "best.pt"), "--key", key,
                        "--manifest", str(data / "manifest.csv"), "--root", str(data / "raw"),
                        "--cache", str(data / "cache"), "--steps", "4", "--device", "cpu"]) == 0


def test_find_audio_uses_the_name_on_disk(tmp_path: Path) -> None:
    folder = tmp_path / "1 A - B"
    write_osu(folder / "x.osu", Chart(4, "Audio.MP3", [(0, 500.0)], [Note(0, 0)]),
              title="B", artist="A", version="x")
    row = {"key": "k", "path": "1 A - B/x.osu"}
    assert preprocess_data.find_audio((row, tmp_path)) == ("k", None)
    (folder / "audio.mp3").write_bytes(b"")
    preprocess_data._files.cache_clear()
    assert preprocess_data.find_audio((row, tmp_path)) == ("k", "1 A - B/audio.mp3")
    if not (folder / "Audio.MP3").exists():                    # case-sensitive file system
        (folder / "Audio.MP3").write_bytes(b"")
        preprocess_data._files.cache_clear()
        assert preprocess_data.find_audio((row, tmp_path)) == ("k", "1 A - B/Audio.MP3")
