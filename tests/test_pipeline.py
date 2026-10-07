"""manifest -> token cache -> ChunkDataset -> train.py, on synthetic .osu files.

Every set also gets an audio.mp3 with a click at each note of its first chart, so
the real log-Mel path (decode as osu! does, fixed hop, token grid) is checked for
alignment end to end. Without PyAV's mp3 encoder those tests are skipped."""

import csv
import json
import zipfile
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
            beatmaps.append({"id": bid, "difficulty_rating": 2.0 + s * 0.3 + v,
                             "user_id": 500 + (s + v) % 3})                # the mapper
            if v == 0 and can_write_mp3():            # the set's audio follows chart v0
                onsets = sorted({n.time_ms for n in notes})
                write_mp3(folder / "audio.mp3", clicks(onsets, 44100, 42.0, seed=s), 44100)
        sets.append({"id": 100 + s, "beatmaps": beatmaps, "genre_id": [10, 3, 5][s % 3],
                     "user_id": 500 + s % 3, "creator": f"host{s % 3}"})
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


def test_style_conditioning_end_to_end(data: Path, tmp_path: Path) -> None:
    from scripts import build_style
    from src.data.style import read_style
    style_csv = tmp_path / "style.csv"
    assert build_style.main(["--manifest", str(data / "manifest.csv"),
                             "--metadata", str(data / "metadata" / "beatmapsets.jsonl"),
                             "--out", str(style_csv), "--min-charts", "2"]) == 0
    labels = read_style(style_csv)
    assert len(labels) == 16
    with_id = [v for v in labels.values() if v["mapper_id"] is not None]
    assert len(with_id) == 15                                 # one .osu has no beatmap id
    assert {v["genre_id"] for v in labels.values()} == {10, 3, 5}
    assert any(v["mapper_name"].startswith("host") for v in with_id)

    args = ["--manifest", str(data / "manifest.csv"), "--cache", str(data / "cache"),
            "--out", str(tmp_path), "--run", "st", "--overfit", "4", "--steps", "30",
            "--batch-size", "4", "--warmup", "5", "--d-model", "64", "--layers", "2",
            "--heads", "2", "--d-ff", "128", "--log-every", "10", "--val-every", "30",
            "--sample-steps", "4", "--fake-mel", "--device", "cpu",
            "--style", str(style_csv), "--min-mapper-charts", "1", "--style-drop", "0.3"]
    assert train.main(args) == 0
    ckpt = torch.load(tmp_path / "st" / "best.pt", weights_only=True)
    vocab = ckpt["style"]
    assert ckpt["config"]["n_genres"] == len(vocab["genres"]) == 13
    assert ckpt["config"]["n_mappers"] == len(vocab["mappers"]) + 1 and vocab["mappers"]
    with open(tmp_path / "st" / "val.csv", newline="") as f:
        assert float(next(csv.DictReader(f))["val_ce_null"]) > 0
    assert train.main([*args[:-6], "--resume", str(tmp_path / "st" / "last.pt"),
                       "--steps", "40"]) == 2               # a style run resumes with --style
    assert train.main([*args, "--resume", str(tmp_path / "st" / "last.pt"),
                       "--steps", "40"]) == 0

    key = json.loads((tmp_path / "st" / "config.json").read_text())["overfit_keys"][0]
    common = ["--ckpt", str(tmp_path / "st" / "best.pt"), "--manifest", str(data / "manifest.csv"),
              "--root", str(data / "raw"), "--cache", str(data / "cache"), "--fake-mel",
              "--device", "cpu", "--steps", "4"]
    assert sample.main([*common, "--key", key, "--style", "oracle", "--style-csv", str(style_csv),
                        "--style-guidance", "1"]) == 0
    assert list((tmp_path / "st" / "samples").glob("*-style-sg1_*.osu"))
    pytest.importorskip("rosu_pp_py")
    from scripts import evaluate
    assert evaluate.main([*common, "--split", "train", "--n", "2", "--style", "oracle",
                          "--style-csv", str(style_csv)]) == 0
    run = next((tmp_path / "st").glob("eval_*_style"))
    with open(run / "per_song.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert all(r["genre"] in ("electronic", "anime", "pop") for r in rows)
    assert all(r["mapper_known"] in ("0", "1", "") for r in rows)


def test_chart_stats_end_to_end(data: Path, tmp_path: Path) -> None:
    from scripts import build_chart_stats
    from src.data import chartstats
    stats_csv = tmp_path / "chart_stats.csv"
    assert build_chart_stats.main(["--manifest", str(data / "manifest.csv"),
                                   "--cache", str(data / "cache"), "--out", str(stats_csv)]) == 0
    table = chartstats.read_stats(stats_csv)
    assert len(table) == 16
    assert all(0 < v["hold_share"] < 1 for v in table.values())     # 25% long notes, synthetic

    args = ["--manifest", str(data / "manifest.csv"), "--cache", str(data / "cache"),
            "--out", str(tmp_path), "--run", "cs", "--overfit", "4", "--steps", "30",
            "--batch-size", "4", "--warmup", "5", "--d-model", "64", "--layers", "2",
            "--heads", "2", "--d-ff", "128", "--log-every", "10", "--val-every", "30",
            "--sample-steps", "4", "--fake-mel", "--device", "cpu", "--row-mask", "0.5",
            "--chart-stats", str(stats_csv), "--stats-drop", "0.3"]
    assert train.main(args) == 0
    ckpt = torch.load(tmp_path / "cs" / "best.pt", weights_only=True)
    spec = ckpt["chart_stats"]
    assert spec["names"] == list(chartstats.NAMES) and len(spec["edges"]) == 3
    assert ckpt["config"]["n_stats"] == 3 and ckpt["config"]["stat_bins"] == spec["bins"]
    with open(tmp_path / "cs" / "val.csv", newline="") as f:
        assert float(next(csv.DictReader(f))["val_ce_null"]) > 0
    assert train.main([*args[:-4], "--resume", str(tmp_path / "cs" / "last.pt"),
                       "--steps", "40"]) == 2               # a stats run resumes with them
    assert train.main([*args, "--resume", str(tmp_path / "cs" / "last.pt"),
                       "--steps", "40"]) == 0

    key = json.loads((tmp_path / "cs" / "config.json").read_text())["overfit_keys"][0]
    common = ["--ckpt", str(tmp_path / "cs" / "best.pt"), "--manifest", str(data / "manifest.csv"),
              "--root", str(data / "raw"), "--cache", str(data / "cache"), "--fake-mel",
              "--device", "cpu", "--steps", "4"]
    assert sample.main([*common, "--key", key, "--stats", "oracle"]) == 0
    assert sample.main([*common, "--key", key, "--stats", "ln=0.4,jack=0.1"]) == 0
    assert list((tmp_path / "cs" / "samples").glob("*-stln0.4jack0.1_*.osu"))
    assert sample.main([*common, "--key", key, "--stats", "bogus=1"]) == 2
    assert sample.main([*common, "--key", key, "--stats", "ln=0.4", "--style-guidance", "1"]) == 0
    assert list((tmp_path / "cs" / "samples").glob("*-sg1-stln0.4_*.osu"))
    pytest.importorskip("rosu_pp_py")
    from scripts import evaluate
    for mode in ("oracle", "sample"):
        assert evaluate.main([*common, "--split", "train", "--n", "2", "--stats", mode,
                              "--stats-csv", str(stats_csv)]) == 0
        run = next((tmp_path / "cs").glob(f"eval_*_st-{mode}"))
        with open(run / "per_song.csv", newline="") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 2 and all(r["stat_hold_share"] != "" for r in rows)
        if mode == "oracle":                              # the human chart's own long notes
            assert all(abs(float(r["stat_hold_share"]) - float(r["human_hold_share"])) < 1e-6
                       for r in rows)
        else:
            assert all(table[r["stat_key"]]["split"] == "train" for r in rows)
    oracle_run = next((tmp_path / "cs").glob("eval_*_st-oracle"))
    with open(oracle_run / "per_song.csv", newline="") as f:      # bar kinds, both charts
        row = next(csv.DictReader(f))
    assert all(k in row for k in ("rest_bars", "human_rest_bars", "ln_bars", "steady_bars"))
    assert evaluate.main([*common, "--split", "train", "--n", "2", "--stats", "oracle",
                          "--from-charts", str(oracle_run), "--rest", "1000"]) == 0
    rested = oracle_run.parent / (oracle_run.name + "_rb1000_st-oracle")
    with open(rested / "per_song.csv", newline="") as f:          # every whole bar emptied
        rows = list(csv.DictReader(f))
    with open(oracle_run / "per_song.csv", newline="") as f:
        before = {r["key"]: float(r["density_ratio"]) for r in csv.DictReader(f)}
    assert all(float(r["density_ratio"]) < before[r["key"]] for r in rows)
    assert all(int(r["rest_bars_cleared"]) > 0 for r in rows)
    assert evaluate.main([*common, "--split", "train", "--n", "1", "--stats", "oracle",
                          "--from-charts", str(oracle_run), "--sr-offset", "0.2"]) == 2
    assert evaluate.main([*common, "--split", "train", "--n", "1", "--stats", "oracle",
                          "--sr-offset", "0.2", "--rest", "0.5"]) == 0
    assert next((tmp_path / "cs").glob("eval_*_rb0.5_so0.2_st-oracle"))
    from scripts import probe_bars
    probe_args = ["--ckpt", str(tmp_path / "cs" / "best.pt"), "--split", "train", "--n", "2",
                  "--stats", "oracle", "--fake-mel", "--manifest", str(data / "manifest.csv"),
                  "--cache", str(data / "cache"), "--device", "cpu"]
    assert probe_bars.main(probe_args) == 0
    probe = json.loads((tmp_path / "cs" / "probe_bars.json").read_text())
    assert probe["songs"] == 2 and probe["bars"] > 0 and "1" in probe["thresholds"]
    assert probe_bars.main([*probe_args, "--charts", str(oracle_run / "charts.npz")]) == 0
    assert (tmp_path / "cs" / f"probe_bars_{oracle_run.name}.csv").exists()
    if can_write_mp3():                       # the meeting pack of 10-06: stats + lane guidance
        from scripts import playtest_pack
        pp = tmp_path / "cs_pairs"
        assert playtest_pack.main([
            "--ckpt", str(tmp_path / "cs" / "best.pt"), "--split", "train", "--songs", "2",
            "--sr-range", "0", "10", "--pairs", "--stats-csv", str(stats_csv),
            "--settings", "random:4:continue:fwd+ref1+cp0+lbq0.1+st+lg1",
            "random:4:continue:fwd+ref1+cp0+lbq0.1+stsample+lg1",
            "--manifest", str(data / "manifest.csv"), "--root", str(data / "raw"),
            "--cache", str(data / "cache"), "--out", str(pp), "--device", "cpu"]) == 0
        with open(pp / "answers.csv", encoding="utf-8-sig") as f:
            sources = {r["source"] for r in csv.DictReader(f)}
        assert sources == {"human", "ai random T4 continue fwd ref1@0.5 lbq0.1 cp0 st lg1",
                           "ai random T4 continue fwd ref1@0.5 lbq0.1 cp0 stsample lg1"}
    from scripts import compare_runs
    runs = sorted((tmp_path / "cs").glob("eval_*_st-*"))
    assert compare_runs.main([str(r) for r in runs]) == 0
    assert evaluate.main([*common, "--split", "train", "--n", "1", "--stats", "jack=x"]) == 2
    assert evaluate.main([*common, "--split", "train", "--n", "1", "--style-guidance", "1"]) == 2
    assert evaluate.main([*common, "--split", "train", "--n", "1", "--stats", "oracle",
                          "--style-guidance", "1"]) == 0
    summary = json.loads(next((tmp_path / "cs").glob("eval_*_st-oracle_sg1/summary.json"))
                         .read_text())
    assert summary["style_guidance"] == 1 and summary["mean_passes"] > 0


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
                        "--cache", str(data / "cache"), "--steps", "4", "--order", "noisy",
                        "--temperature", "2", "--device", "cpu"]) == 0
    assert list((tmp_path / "r" / "samples").glob("*_noisy2-fwd-ref2t0.5-lbq0.1-og1-cp0_T4_*.osu"))
    import zipfile
    osz = next((tmp_path / "r" / "samples").glob("*_noisy2-fwd-ref2t0.5-lbq0.1-og1-cp0_T4_*.osz"))
    names = zipfile.ZipFile(osz).namelist()
    assert "audio.mp3" in names and "v0.osu" in names and any("noisy2" in n for n in names)

    # a "new" mp3: only audio + timing
    from scripts import generate
    folder = next((data / "raw").glob("100 *"))
    v0 = parse_osu(folder / "v0.osu")
    t0, bl = v0.timing_points[0]
    out = tmp_path / "gen"
    common = ["--ckpt", str(tmp_path / "r" / "best.pt"), "--audio", str(folder / "audio.mp3"),
              "--steps", "4", "--device", "cpu"]
    assert generate.main([*common, "--bpm", str(60000 / bl), "--offset", str(t0),
                          "--sr", "2", "4", "--out", str(out)]) == 0
    charts = sorted(out.glob("*.osu"))
    assert len(charts) == 2 and (out / "audio.osz").exists()
    for p in charts:
        c = parse_osu(p)
        assert c.audio_filename == "audio.mp3" and c.timing_points == [(t0, bl)]
    assert all("-lbq0.1-og1-cp0]" in p.name for p in charts)    # quiet bars, onset gate, copies
    assert generate.main([*common, "--timing", str(folder / "v0.osu"), "--mode", "independent",
                          "--order", "confidence", "--no-copy", "--out", str(tmp_path / "gen2")]) == 0
    made = list((tmp_path / "gen2").glob("*independent-confidence*.osu"))
    assert len(made) == 1 and "-cp" not in made[0].name
    # the model input from the audio file equals the one training read from the cache
    with open(data / "manifest.csv", newline="") as f:
        key = next(r["key"] for r in csv.DictReader(f)
                   if r["path"] == f"{folder.name}/v0.osu")
    from src.data.audio import load_osu
    from src.data.beat_grid import from_timing_points
    samples, _ = load_osu(folder / "audio.mp3")
    co, _ = generate.song_range([(t0, bl)], 1000 * len(samples) / 22050)
    fresh = generate.model_input(samples, [(t0, bl)], co, 2 * L)
    cached = MelStore(data / "cache").frames(key, from_timing_points([(t0, bl)]), co, 0, 2 * L)
    np.testing.assert_allclose(fresh, cached, atol=1e-3)

    # blind pack: labels hide who made what
    from scripts import playtest_pack
    pt = tmp_path / "pt"
    assert playtest_pack.main(["--ckpt", str(tmp_path / "r" / "best.pt"), "--split", "train",
                               "--songs", "2", "--sr-range", "0", "10",
                               "--settings", "random:4", "confidence:4:independent",
                               "--manifest", str(data / "manifest.csv"),
                               "--root", str(data / "raw"), "--cache", str(data / "cache"),
                               "--out", str(pt), "--device", "cpu"]) == 0
    with open(pt / "answers.csv", encoding="utf-8-sig") as f:
        answers = list(csv.DictReader(f))
    assert len(answers) == 6
    assert {r["source"] for r in answers} == {"human", "ai random T4 continue",
                                             "ai confidence T4 independent"}
    packs = sorted((pt / "pack").glob("*.osz"))
    assert len(packs) == 2 and (pt / "playtest_pack.zip").exists()
    for osz in packs:
        with zipfile.ZipFile(osz) as zf:
            names = zf.namelist()
            charts = [n for n in names if n.endswith(".osu")]
            assert "audio.mp3" in names and len(charts) == 3
            texts = [zf.read(n).decode() for n in charts]
        versions = sorted(line for t in texts for line in t.splitlines()
                          if line.startswith("Version:"))
        assert versions == ["Version:A", "Version:B", "Version:C"]
        assert all("Creator:playtest" in t for t in texts)
        assert len({t.split("[TimingPoints]")[1].split("[HitObjects]")[0] for t in texts}) == 1
    song = answers[0]["song"]
    human = next(r["label"] for r in answers if r["song"] == song and r["source"] == "human")
    filled = tmp_path / "ratings_me.csv"
    with open(filled, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=playtest_pack.RATING_FIELDS)
        w.writeheader()
        for r in answers:
            if r["song"] == song:
                is_h = r["label"] == human
                w.writerow({"song": song, "label": r["label"], "rank": 1 if is_h else 2,
                            "human?": "y" if is_h else "n"})
    assert playtest_pack.main(["--score", str(pt / "answers.csv"), str(filled)]) == 0

    # pairs: the human chart and one AI chart per song, the settings taking turns
    pp = tmp_path / "pairs"
    assert playtest_pack.main(["--ckpt", str(tmp_path / "r" / "best.pt"), "--split", "train",
                               "--songs", "2", "--sr-range", "0", "10", "--pairs",
                               "--settings", "random:4", "random:4:continue:fwd+cp2",
                               "--manifest", str(data / "manifest.csv"),
                               "--root", str(data / "raw"), "--cache", str(data / "cache"),
                               "--out", str(pp), "--device", "cpu"]) == 0
    with open(pp / "answers.csv", encoding="utf-8-sig") as f:
        answers = list(csv.DictReader(f))
    by_song = {}
    for r in answers:
        by_song.setdefault(r["song"], {})[r["label"]] = r["source"]
    assert all(sorted(v) == ["A", "B"] and "human" in v.values() for v in by_song.values())
    assert sorted(src for v in by_song.values() for src in v.values() if src != "human") == \
        ["ai random T4 continue", "ai random T4 continue fwd@0.5 cp2"]
    with open(pp / "pack" / "ratings.csv", encoding="utf-8-sig") as f:
        sheet = list(csv.DictReader(f))
    assert [r["song"] for r in sheet] == sorted(by_song) and list(sheet[0]) == playtest_pack.PAIR_FIELDS
    filled = tmp_path / "ratings_pairs.csv"
    with open(filled, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=playtest_pack.PAIR_FIELDS)
        w.writeheader()
        for song, labels in sorted(by_song.items()):
            ai = next(lb for lb, src in labels.items() if src != "human")
            w.writerow({"song": song, "human": ai, "better": "="})       # fooled every time
    assert playtest_pack.main(["--score", str(pp / "answers.csv"), str(filled)]) == 0
    px = tmp_path / "pairs_exclude"                    # a later pack leaves those songs out
    assert playtest_pack.main(["--ckpt", str(tmp_path / "r" / "best.pt"), "--split", "train",
                               "--songs", "2", "--sr-range", "0", "10", "--pairs",
                               "--settings", "random:4", "--exclude", str(pp / "answers.csv"),
                               "--manifest", str(data / "manifest.csv"),
                               "--root", str(data / "raw"), "--cache", str(data / "cache"),
                               "--out", str(px), "--device", "cpu"]) == 0
    folders = []
    for run in (pp, px):
        with open(run / "answers.csv", encoding="utf-8-sig") as f:
            folders.append({r["path"].split("/")[0] for r in csv.DictReader(f)})
    assert folders[1] and not folders[0] & folders[1]

    from scripts import audio_ablation
    assert audio_ablation.main(["--ckpt", str(tmp_path / "r" / "best.pt"), "--split", "train",
                                "--manifest", str(data / "manifest.csv"),
                                "--cache", str(data / "cache"), "--batches", "2",
                                "--batch-size", "4", "--device", "cpu"]) == 0
    res = json.loads((tmp_path / "r" / "audio_ablation_train.json").read_text())
    assert len(res) == 4 * 5 * 2 and all(v >= 0 for v in res.values())

    from scripts import pattern_probe
    assert pattern_probe.main(["--ckpt", str(tmp_path / "r" / "best.pt"), "--split", "train",
                               "--manifest", str(data / "manifest.csv"),
                               "--cache", str(data / "cache"), "--batches", "2",
                               "--batch-size", "4", "--device", "cpu"]) == 0
    probe = json.loads((tmp_path / "r" / "pattern_probe_train.json").read_text())
    assert set(probe) == {"full", "thin50", "thin90", "chance", "previous", "best of 8",
                          "full (1 row)", "past+rhythm", "past", "chance (1 row)",
                          "move: jack", "move: step", "move: skip", "move: leap",
                          "jack rate (human)", "jack rate (model)", "jack prob (t=1)",
                          "jack prob (t=0.5)"}
    assert probe["full"]["rows"] == probe["chance"]["rows"] > 0
    assert probe["past"]["rows"] == probe["past+rhythm"]["rows"] == probe["full (1 row)"]["rows"]
    assert 0 < probe["past"]["rows"] <= probe["full"]["rows"]
    assert all(0 <= v <= 1 for d in probe.values() for k, v in d.items() if k != "rows")

    from scripts import hold_stats
    assert hold_stats.main(["--manifest", str(data / "manifest.csv"), "--cache", str(data / "cache"),
                            "--split", "train", "--out", str(tmp_path / "holds.csv")]) == 0
    with open(tmp_path / "holds.csv") as f:
        rows = list(csv.DictReader(f))
    assert rows[-1]["grade"] == "all" and int(rows[-1]["long_notes"]) > 0

    pytest.importorskip("rosu_pp_py")
    from scripts import evaluate
    assert evaluate.main(["--ckpt", str(tmp_path / "r" / "best.pt"), "--split", "train",
                          "--per-song", "--n", "0", "--steps", "4", "--order", "noisy",
                          "--manifest", str(data / "manifest.csv"), "--root", str(data / "raw"),
                          "--cache", str(data / "cache"), "--device", "cpu"]) == 0
    summary = json.loads((tmp_path / "r" / "eval_train_songs_continue_noisy1_T4"
                          / "summary.json").read_text())
    assert summary["songs"] == summary["distinct_songs"] == 6          # 14 train charts, 6 songs
    lo, hi = summary["ci95_f1@50"]
    assert lo <= summary["mean_f1@50"] <= hi
    assert "mean_motion_pred" in summary and "mean_human_motion_pred" in summary
    assert summary["min_hold"] is None and summary["release_gap"] is None   # by SR
    assert summary["mean_short_holds"] == 0 and 0 < summary["mean_human_hold_share"] < 1
    assert evaluate.main(["--ckpt", str(tmp_path / "r" / "best.pt"), "--split", "train",
                          "--per-song", "--n", "2", "--steps", "4", "--refine", "1",
                          "--lane-temp", "0", "--manifest", str(data / "manifest.csv"),
                          "--root", str(data / "raw"), "--cache", str(data / "cache"),
                          "--device", "cpu"]) == 0
    refined = json.loads((tmp_path / "r" / "eval_train_songs_continue_random_T4_ref1t0"
                          / "summary.json").read_text())
    assert refined["refine"] == 1 and refined["lane_temp"] == 0
    assert refined["lanes"] == "sampled" and not refined["spread"] and refined["empty_bias"] == 0
    assert refined["mean_passes"] > 0 and refined["mean_seconds"] > 0
    common = ["--ckpt", str(tmp_path / "r" / "best.pt"), "--split", "train", "--per-song",
              "--manifest", str(data / "manifest.csv"), "--root", str(data / "raw"),
              "--cache", str(data / "cache"), "--device", "cpu"]
    assert evaluate.main([*common, "--n", "2", "--steps", "4", "--order", "block", "--spread",
                          "--lanes", "forward", "--forward-temp", "1", "--refine", "1",
                          "--empty-bias", "0.5"]) == 0
    fwd = json.loads((tmp_path / "r" / "eval_train_songs_continue_block_T4_spread_fwdt1_ref1t0.5_eb0.5"
                      / "summary.json").read_text())
    assert fwd["lanes"] == "forward" and fwd["spread"] and fwd["empty_bias"] == 0.5
    assert fwd["lane_temp"] == 0.5 and fwd["forward_temp"] == 1 and fwd["mean_violation_rate"] == 0
    assert all(f"mean_{k}" in fwd for k in ("move_stair", "human_move_trill", "chord_share"))
    saved = np.load(tmp_path / "r" / "eval_train_songs_continue_block_T4_spread_fwdt1_ref1t0.5_eb0.5"
                    / "charts.npz")
    assert len(saved.files) == 2 and saved[saved.files[0]].shape[1:] == (L, 4)
    from scripts import rescore
    run_dir = tmp_path / "r" / "eval_train_songs_continue_block_T4_spread_fwdt1_ref1t0.5_eb0.5"
    assert rescore.main([str(run_dir), "--cache", str(data / "cache")]) == 0
    with open(run_dir / "per_song.csv", newline="") as f:
        rescored = list(csv.DictReader(f))
    assert len(rescored) == 2 and "bar_lane_repeat" in rescored[0] and "human_lone_chord" in rescored[0]
    assert (run_dir / "per_song.orig.csv").exists()
    assert "mean_bar_rhythm_repeat" in json.loads((run_dir / "summary.json").read_text())
    assert evaluate.main([*common, "--n", "2", "--copy-bias", "0",
                          "--from-charts", str(run_dir)]) == 0             # no new sampling
    copied = json.loads((run_dir.parent / (run_dir.name + "_cp0") / "summary.json").read_text())
    assert copied["songs"] == 2 and copied["mean_violation_rate"] == 0
    assert "mean_copied_bars" in copied and copied["copy_bias"] == 0
    assert evaluate.main([*common, "--n", "3", "--from-charts", str(run_dir)]) == 2   # not saved
    assert evaluate.main([*common, "--n", "2", "--refine-holds", "--copy-bias", "0",
                          "--from-charts", str(run_dir)]) == 0
    both = json.loads((run_dir.parent / (run_dir.name + "_hr_cp0") / "summary.json").read_text())
    assert both["refine_holds"] and "mean_changed_holds" in both and both["mean_violation_rate"] == 0
    assert "mean_release_on_onset" in both or "mean_human_release_on_onset" in both
    assert evaluate.main([*common, "--n", "2", "--hold-share", "oracle",
                          "--from-charts", str(run_dir)]) == 0
    oracle = json.loads((run_dir.parent / (run_dir.name + "_hr-oracle") / "summary.json").read_text())
    assert oracle["hold_share"] == "oracle" and abs(oracle["mean_hold_share"]
                                                   - oracle["mean_human_hold_share"]) < 0.05
    assert evaluate.main(["--ckpt", str(tmp_path / "r" / "best.pt"), "--split", "train",
                          "--per-song", "--n", "0", "--steps", "4", "--order", "noisy",
                          "--sample-seed", "1",
                          "--manifest", str(data / "manifest.csv"), "--root", str(data / "raw"),
                          "--cache", str(data / "cache"), "--device", "cpu"]) == 0
    base = tmp_path / "r" / "eval_train_songs_continue_noisy1_T4"
    again = tmp_path / "r" / "eval_train_songs_continue_noisy1_T4_s1"
    keys = [line.split(",")[0] for line in (base / "per_song.csv").read_text().splitlines()]
    assert keys == [line.split(",")[0] for line in (again / "per_song.csv").read_text().splitlines()]
    from scripts import compare_runs
    assert compare_runs.main([str(base), str(again), str(tmp_path / "r" / "eval_train_songs_"
                              "continue_block_T4_spread_fwdt1_ref1t0.5_eb0.5"), "--by-grade"]) == 0


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


def test_playtest_settings() -> None:
    from scripts.playtest_pack import parse_setting
    plain = parse_setting("random:128")
    assert plain["lanes"] == "sampled" and plain["refine"] == 0 and plain["name"] == "ai random T128 continue"
    old = parse_setting("random:128:continue:ref2@0.5")                 # the name earlier packs used
    assert old["name"] == "ai random T128 continue ref2@0.5" and old["lane_temperature"] == 0.5
    both = parse_setting("random:128:continue:fwd@1+ref2@0.5")
    assert (both["lanes"], both["forward_temperature"], both["lane_temperature"]) == ("forward", 1, 0.5)
    assert both["name"] == "ai random T128 continue fwd@1 ref2@0.5"
    shared = parse_setting("random:128:continue:fwd+ref2")
    assert shared["forward_temperature"] is None and shared["name"].endswith("fwd ref2@0.5")
    alone = parse_setting("random:0:continue:fwd@0.3+spread+eb0.2")
    assert alone["lane_temperature"] == 0.3 and alone["forward_temperature"] is None
    assert alone["spread"] and alone["empty_bias"] == 0.2 and alone["steps"] == 0
    assert alone["name"] == "ai random Tseq continue spread fwd@0.3 eb0.2"
    assert parse_setting("random:128:continue:fwd+ln")["hold_share"] == "human"
    assert parse_setting("random:128:continue:fwd+ln0.2")["name"].endswith("fwd@0.5 ln0.2")
    passes = parse_setting("random:128:continue:fwd+ref2+lb0.2+hr+cp0")
    assert passes["holds"] and passes["loud_bias"] == 0.2 and passes["copy_bias"] == 0
    assert passes["name"] == "ai random T128 continue fwd ref2@0.5 lb0.2 hr cp0"
    quiet = parse_setting("random:128:continue:fwd+ref2+lbq0.1+st")
    assert (quiet["loud_bias"], quiet["loud_side"], quiet["stats"]) == (0.1, "quiet", "human")
    assert quiet["name"] == "ai random T128 continue fwd ref2@0.5 lbq0.1 st"
    assert parse_setting("random:128:continue:fwd+stsample")["stats"] == "sample"
    gated = parse_setting("random:128:continue:fwd+ref2+st+lg2+og1")
    assert gated["onset_bias"] == 1 and gated["name"].endswith("st lg2 og1")
    assert parse_setting("random:128")["onset_bias"] == 0
    guided = parse_setting("random:128:continue:fwd+ref2+st+sg1.5")
    assert guided["style_guidance"] == 1.5 and guided["name"].endswith("sg1.5 st")
    from scripts.playtest_pack import style_args
    assert style_args(guided, None, {}) == {"style_guidance": 1.5}     # towards the stats
    assert style_args(passes, None, {}) == {}
    with pytest.raises(ValueError):
        parse_setting("random:128:continue:fwd+sg1")                  # guidance towards nothing
    assert passes["loud_side"] == "both"
    copies = parse_setting("random:128:continue:fwd+ref2+cp4")
    assert copies["copy_bias"] == 4 and copies["name"] == "ai random T128 continue fwd ref2@0.5 cp4"
    assert parse_setting("random:128")["copy_bias"] is None
    with pytest.raises(ValueError):
        parse_setting("random:128:continue:bogus")


def test_playtest_retag(tmp_path: Path) -> None:
    from scripts import playtest_pack
    out = tmp_path / "playtest"
    (out / "pack").mkdir(parents=True)
    chart = Chart(4, "audio.mp3", [(0.0, 500.0)], [Note(0, 0, None), Note(500, 1, 1000)])
    osu = tmp_path / "x.osu"
    write_osu(osu, chart, title="PT01 Song", artist="Artist", version="A", creator="playtest")
    with zipfile.ZipFile(out / "pack" / "PT01 Artist - Song.osz", "w") as zf:
        zf.writestr("audio.mp3", b"")
        zf.write(osu, "Artist - PT01 Song (playtest) [A].osu")
    for path, row in ((out / "answers.csv", {"song": "PT01", "label": "A", "source": "human"}),
                      (out / "pack" / "ratings.csv", {"song": "PT01", "human": "", "better": ""})):
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            w.writeheader()
            w.writerow(row)
    assert playtest_pack.main(["--retag", str(out), "--tag", "1004"]) == 0
    packs = sorted((out / "pack").glob("*.osz"))
    assert [p.name for p in packs] == ["1004-PT01 Artist - Song.osz"]
    with zipfile.ZipFile(packs[0]) as zf:
        names = zf.namelist()
        text = zf.read("Artist - 1004-PT01 Song (playtest) [A].osu").decode()
    assert "audio.mp3" in names and "Title:1004-PT01 Song" in text
    assert "TitleUnicode:1004-PT01 Song" in text
    assert parse_osu_text_notes(text) == 2
    for path in (out / "answers.csv", out / "pack" / "ratings.csv"):
        with open(path, encoding="utf-8-sig") as f:
            assert next(csv.DictReader(f))["song"] == "1004-PT01"
    assert (out / "playtest_pack.zip").exists()
    assert playtest_pack.main(["--retag", str(out), "--tag", "1005"]) == 0     # already tagged
    assert [p.name for p in (out / "pack").glob("*.osz")] == ["1004-PT01 Artist - Song.osz"]


def parse_osu_text_notes(text: str) -> int:
    return len(text.split("[HitObjects]")[1].strip().splitlines())
