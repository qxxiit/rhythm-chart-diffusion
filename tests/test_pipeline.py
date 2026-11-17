"""manifest -> token cache -> ChunkDataset -> train.py, on synthetic .osu files.

Every set also gets an audio.mp3 with a click at each note of its first chart, so
the real log-Mel path (decode as osu! does, fixed hop, token grid) is checked for
alignment end to end. Without PyAV's mp3 encoder those tests are skipped."""

import csv
import importlib.util
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
from src.data.tokenizer import HOLD_START, PAD, TAP, K, L  # noqa: E402
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


def test_train_with_the_mask_options(data: Path, tmp_path: Path) -> None:
    args = ["--manifest", str(data / "manifest.csv"), "--cache", str(data / "cache"),
            "--out", str(tmp_path), "--run", "m", "--overfit", "2", "--steps", "60",
            "--batch-size", "4", "--lr", "2e-3", "--warmup", "10", "--d-model", "64",
            "--layers", "2", "--heads", "2", "--d-ff", "128", "--log-every", "10",
            "--val-every", "60", "--sample-steps", "8", "--window", "bar", "--fake-mel",
            "--row-mask", "0.5", "--loss-weight", "count", "--complementary",
            "--block-mask", "0.5", "--device", "cpu"]
    assert train.main(args) == 0
    with open(tmp_path / "m" / "log.csv", newline="") as f:
        losses = [float(r["loss"]) for r in csv.DictReader(f)]
    assert losses[-1] < losses[0]
    with open(tmp_path / "m" / "log_more.csv", newline="") as f:
        more = list(csv.DictReader(f))
    assert len(more) == len(losses) and float(more[0]["weight_max"]) > 0
    assert 0.0 <= float(more[-1]["clipped_share"]) <= 1.0
    with open(tmp_path / "m" / "val_more.csv", newline="") as f:
        val_more = next(csv.DictReader(f))
    assert {"val_ce@0.02", "val_ce@1.0", "on@0.5", "off@0.5", "on@1.0"} <= set(val_more)
    with open(tmp_path / "m" / "val.csv", newline="") as f:
        assert "val_ce@1.0" not in next(csv.DictReader(f))          # val.csv keeps its columns


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
    assert evaluate.main([*common, "--split", "train", "--n", "2", "--stats", "oracle",
                          "--from-charts", str(oracle_run), "--rest", "1000",
                          "--rest-max-run", "8", "--answer", "0"]) == 0
    guarded = oracle_run.parent / (oracle_run.name + "_ab0u1d_rb1000m8_st-oracle")
    with open(guarded / "per_song.csv", newline="") as f:
        g_rows = list(csv.DictReader(f))
    assert all("answered_beats" in r and "rest_bars_kept" in r for r in g_rows)
    assert evaluate.main([*common, "--split", "train", "--n", "1", "--stats", "oracle",
                          "--answer", "0"]) == 2                  # passes on saved charts only
    assert evaluate.main([*common, "--split", "train", "--n", "1", "--stats", "oracle",
                          "--from-charts", str(oracle_run), "--rest-max-run", "8"]) == 2
    with open(rested / "per_song.csv", newline="") as f:          # every whole bar emptied
        rows = list(csv.DictReader(f))
    with open(oracle_run / "per_song.csv", newline="") as f:
        before = {r["key"]: float(r["density_ratio"]) for r in csv.DictReader(f)}
    assert all(float(r["density_ratio"]) < before[r["key"]] for r in rows)
    assert all(int(r["rest_bars_cleared"]) > 0 for r in rows)
    assert all(k in rows[0] for k in ("new_onsets", "human_new_onsets", "set_charts"))
    assert evaluate.main([*common, "--split", "train", "--n", "2", "--stats", "oracle",
                          "--from-charts", str(oracle_run), "--recheck", "1.01"]) == 0
    with open(oracle_run.parent / (oracle_run.name + "_rk1.01_st-oracle") / "per_song.csv",
              newline="") as f:                           # every row with taps asked and emptied
        rows = list(csv.DictReader(f))
    assert all(int(r["recheck_out"]) > 0 and float(r["density_ratio"]) < before[r["key"]]
               for r in rows)
    assert evaluate.main([*common, "--split", "train", "--n", "1", "--stats", "oracle",
                          "--from-charts", str(oracle_run), "--recheck-add", "0.5"]) == 2
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
    cal = probe["calibration"]
    assert cal["expected_over_human"] > 0 and sum(b["bars"] for b in cal["bins"]) == \
        pytest.approx(1.0)
    assert probe_bars.main([*probe_args, "--charts", str(oracle_run / "charts.npz")]) == 0
    assert (tmp_path / "cs" / f"probe_bars_{oracle_run.name}.csv").exists()
    assert probe_bars.main([*probe_args, "--context", "none"]) == 0
    none = json.loads((tmp_path / "cs" / "probe_bars_none.json").read_text())
    assert none["context"] == "none" and "auc_rest_expected_starts_within" in none
    assert evaluate.main([*common, "--split", "train", "--n", "2", "--stats", "oracle",
                          "--from-charts", str(oracle_run), "--fill-holes", "0.0",
                          "--rest", "1000"]) == 0
    assert (oracle_run.parent / (oracle_run.name + "_fh0_rb1000_st-oracle")).exists()
    assert evaluate.main([*common, "--split", "train", "--n", "2", "--stats", "oracle",
                          "--from-charts", str(oracle_run), "--carry-rhythm", "100",
                          "--carry-keep", "--carry-rows", "--carry-holds", "-1",
                          "--lane-guidance", "1", "--ln-tidy", "1", "--copy-bias", "0",
                          "--copy-min-lag", "4"]) == 0
    carried = oracle_run.parent / (oracle_run.name + "_rc100krh-1_cp0l4_lt1_st-oracle_lg1")
    with open(carried / "per_song.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2 and all(int(r["carried_bars"]) >= 0 and int(r["tidied_holds"]) >= 0
                                  for r in rows)
    assert all(k in rows[0] for k in ("rhythm_rep1", "human_rhythm_rep16", "lane_rep1",
                                      "ln_sections", "human_ln_scattered", "copy_far",
                                      "human_copy_near", "copied_bars", "carried_ln_bars",
                                      "tap_rep1", "human_ln_rep1"))
    summary = json.loads((carried / "summary.json").read_text())
    assert summary["carry_rhythm"] == 100 and summary["ln_tidy"] == 1
    assert evaluate.main([*common, "--split", "train", "--n", "2", "--stats", "oracle",
                          "--from-charts", str(oracle_run), "--copy-sections", "4",
                          "--section-keep", "--section-back", "--untangle", "--untangle-move",
                          "--untangle-grades", "--chord-cap", "2", "--alike", "2",
                          "--alike-keep", "--strong-chords", "1", "--hold-share", "oracle",
                          "--hold-up", "--light", "0.5", "--light-alike",
                          "--untangle-wide"]) == 0
    late = oracle_run.parent / (oracle_run.name
                                + "_hu-oracle_sc1_cs4kb_al2k_li0.5a_ujmgw_cc2_st-oracle")
    with open(late / "per_song.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert all(int(r["section_back"]) <= int(r["section_bars"]) for r in rows)
    assert all(int(r["untangle_moved"]) <= int(r["untangled"]) and int(r["capped_rows"]) >= 0
               and int(r["aligned_bars"]) >= 0 and int(r["strong_moved"]) >= 0
               and int(r["more_holds"]) >= 0 and int(r["thinned_bars"]) >= 0 for r in rows)
    assert all("run_over" in r and "human_run_over" in r for r in rows)
    late_summary = json.loads((late / "summary.json").read_text())
    assert late_summary["chord_cap"] == 2 and late_summary["section_back"]
    assert late_summary["untangle_move"] and late_summary["untangle_grades"]
    assert late_summary["alike"] == 2 and late_summary["alike_keep"] == 0
    assert late_summary["strong_chords"] == 1 and late_summary["hold_up"]
    assert late_summary["light"] == 0.5 and late_summary["light_alike"]
    assert late_summary["untangle_wide"]
    assert evaluate.main([*common, "--split", "train", "--n", "2", "--stats", "oracle",
                          "--from-charts", str(oracle_run), "--hold-up"]) == 2   # no share
    assert summary["copy_min_lag"] == 4 and summary["carry_keep"] and summary["carry_holds"] == -1
    assert summary["carry_rows"]
    assert sample.main([*common, "--key", key, "--stats", "oracle", "--carry-rhythm", "0",
                        "--ln-tidy", "1"]) == 0
    assert list((tmp_path / "cs" / "samples").glob(
        "*-rc0kh1-sc1-cs4kb-cp0l8-al2k4-lt1-*-li0.5a-ujmgi-cc2-*.osu"))
    assert sample.main([*common, "--key", key, "--stats", "oracle", "--no-carry-keep",
                        "--carry-rows", "--no-carry-holds", "--ln-tidy", "1"]) == 0
    assert list((tmp_path / "cs" / "samples").glob("*-rc0r-sc1-cs4kb-cp0l8-al2k4-lt1-*.osu"))
    assert evaluate.main([*common, "--split", "train", "--n", "2", "--stats", "oracle",
                          "--onset-bias", "1", "--onset-taper", "2.7,5,0.35",
                          "--sr-min", "4.5"]) == 0
    tapered = next((tmp_path / "cs").glob("eval_*_og1_ot2.7-5-0.35_st-oracle_min4.5"))
    with open(tapered / "per_song.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1 and float(rows[0]["sr_target"]) >= 4.5   # 4.92 and 4.11 above
    assert "weak_onsets" in rows[0] and "human_weak_onsets" in rows[0]
    assert evaluate.main([*common, "--split", "train", "--n", "1", "--onset-bias", "1",
                          "--onset-taper", "5,2,0.3"]) == 2
    named = tmp_path / "cs" / "probe_bars_t0.5_x"                 # a dot in the name
    assert probe_bars.main([*probe_args, "--out", str(named)]) == 0
    assert (tmp_path / "cs" / "probe_bars_t0.5_x.json").exists()
    if can_write_mp3():                       # the meeting pack of 10-06: stats + lane guidance
        from scripts import playtest_pack
        pp = tmp_path / "cs_pairs"
        assert playtest_pack.main([
            "--ckpt", str(tmp_path / "cs" / "best.pt"), "--split", "train", "--songs", "2",
            "--sr-range", "0", "10", "--pairs", "--stats-csv", str(stats_csv),
            "--settings", "random:4:continue:fwd+ref1+cp0+lbq0.1+st+lg1",
            "random:4:continue:fwd+ref1+cp0+lbq0.1+stsample+lg1+og1+ot+rc0+lt1+fh0.5+rb1",
            "--manifest", str(data / "manifest.csv"), "--root", str(data / "raw"),
            "--cache", str(data / "cache"), "--out", str(pp), "--device", "cpu"]) == 0
        with open(pp / "answers.csv", encoding="utf-8-sig") as f:
            sources = {r["source"] for r in csv.DictReader(f)}
        assert sources == {"human", "ai random T4 continue fwd ref1@0.5 lbq0.1 cp0 st lg1",
                           "ai random T4 continue fwd ref1@0.5 lbq0.1 rc0 cp0 lt1 stsample lg1 "
                           "og1 ot fh0.5 rb1"}
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
    made = ("*_noisy2-fwd-ref2t0.5-lbq0.1-og1-ot-rc0kh1-sc1-cs4kb-cp0l8-al2k4-lt0.5-fh0.5-rb1"
            "-li0.5a-ujmgi-cc2_T4_*")
    assert list((tmp_path / "r" / "samples").glob(made + ".osu"))
    import zipfile
    osz = next((tmp_path / "r" / "samples").glob(made + ".osz"))
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
    assert all("-lbq0.1-og1-ot-rc0kh1-sc1-cs4kb-cp0l8-al2k4-lt0.5-fh0.5-rb1-li0.5a-ujmgi-cc2]"
               in p.name for p in charts)
    # quiet bars, onset gate tapered by SR, rhythms carried over, copies from 8 back, short
    # long notes among taps made taps, stream holes, rests
    assert generate.main([*common, "--bpm", "120", "--sr", "2", "--onset-taper", "5,2,1",
                          "--out", str(out)]) == 2
    assert generate.main([*common, "--timing", str(folder / "v0.osu"), "--mode", "independent",
                          "--order", "confidence", "--no-copy", "--no-carry",
                          "--out", str(tmp_path / "gen2")]) == 0
    made = list((tmp_path / "gen2").glob("*independent-confidence*.osu"))
    assert len(made) == 1 and "-cp" not in made[0].name and "-rc" not in made[0].name
    assert generate.main([*common, "--bpm", str(60000 / bl), "--offset", str(t0), "--sr", "2",
                          "--copy-min-lag", "4", "--carry-holds", "-2", "--ln-tidy", "0",
                          "--out", str(tmp_path / "gen3")]) == 0
    made = list((tmp_path / "gen3").glob("*.osu"))
    assert made and all("-rc0kh-2-sc1-cs4kb-cp0l4-al2k4-fh" in p.name for p in made)
    assert generate.main([*common, "--bpm", str(60000 / bl), "--offset", str(t0), "--sr", "2",
                          "--no-sections", "--chord-cap", "0", "--section-back", "--untangle",
                          "--untangle-move", "--untangle-grades",
                          "--out", str(tmp_path / "gen4")]) == 0
    made = list((tmp_path / "gen4").glob("*.osu"))
    assert made and all("-rc0kh1-sc1-cp0l8-al2k4-lt0.5-fh0.5-rb1-li0.5a-ujmgi]" in p.name
                        for p in made)
    assert generate.main([*common, "--bpm", str(60000 / bl), "--offset", str(t0), "--sr", "2",
                          "--no-section-back", "--no-untangle", "--alike", "2", "--alike-keep",
                          "--strong-chords", "1", "--out", str(tmp_path / "gen5")]) == 0
    made = list((tmp_path / "gen5").glob("*.osu"))
    assert made and all("-rc0kh1-sc1-cs4k-cp0l8-al2k-lt0.5-fh0.5-rb1-li0.5a-cc2]" in p.name
                        for p in made)
    assert generate.main([*common, "--bpm", str(60000 / bl), "--offset", str(t0), "--sr", "2",
                          "--no-alike", "--strong-chords", "0", "--hold-share", "0.3",
                          "--light-alike", "--untangle-wide",
                          "--out", str(tmp_path / "gen6")]) == 0
    made = list((tmp_path / "gen6").glob("*.osu"))
    assert made and all("-hr0.3-rc0kh1-cs4kb-cp0l8-lt0.5-fh0.5-rb1-li0.5a-ujmgw-cc2]" in p.name
                        for p in made)
    assert generate.main([*common, "--bpm", str(60000 / bl), "--offset", str(t0), "--sr", "2",
                          "--no-alike-keep", "--light", "0", "--out", str(tmp_path / "gen7")]) == 0
    made = list((tmp_path / "gen7").glob("*.osu"))
    assert made and all("-cp0l8-al2-lt0.5-" in p.name and "-li" not in p.name for p in made)
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
    # one chart per song, the human one or a saved run's, and the other in the second version
    fake = tmp_path / "run_ai"                         # a saved run: the human charts mirrored
    fake.mkdir()
    with open(data / "manifest.csv", newline="") as f:
        train_keys = [r["key"] for r in csv.DictReader(f) if r["split"] == "train"]
    saved = {}
    for key in train_keys:
        tok_file = data / "cache" / "tokens" / f"{key}.npz"
        if tok_file.exists():
            saved[key] = np.load(tok_file)["tokens"][..., ::-1].copy()
    np.savez_compressed(fake / "charts.npz", **saved)
    ps = tmp_path / "single"
    single = ["--split", "train", "--songs", "2", "--sr-range", "0", "10",
              "--manifest", str(data / "manifest.csv"), "--root", str(data / "raw"),
              "--cache", str(data / "cache"), "--out", str(ps)]
    assert playtest_pack.main(["--single", *single]) == 2                  # no --from-run
    assert playtest_pack.main(["--single", "--from-run", str(fake), "--versions", "2",
                               *single]) == 0
    with open(ps / "answers.csv", encoding="utf-8-sig") as f:
        answers = list(csv.DictReader(f))
    assert len(answers) == 4
    per_song = {}
    for r in answers:
        per_song.setdefault(r["song"], {})[r["version"]] = r["source"]
    assert all(sorted(v.values()) == ["ai run_ai", "human"] for v in per_song.values())
    assert sorted(r["source"] for r in answers if r["version"] == "1") == ["ai run_ai", "human"]
    assert playtest_pack.single_humans(5, 0).sum() in (2, 3)   # one of each two, a coin left
    rated = [("p", "human", True, 5, 4.0, False, "S01"), ("p", "ai x", False, 5, 4.0, False, "S02")]
    assert playtest_pack.cluster_accuracy(rated, n=50) == (1.0, 1.0)
    for v in ("1", "2"):
        packs = sorted((ps / f"pack_v{v}").glob("*.osz"))
        assert len(packs) == 2 and (ps / f"playtest_pack_v{v}.zip").exists()
        for osz in packs:
            with zipfile.ZipFile(osz) as zf:
                charts = [n for n in zf.namelist() if n.endswith(".osu")]
                text = zf.read(charts[0]).decode()
            assert len(charts) == 1 and "Version:chart" in text and "Creator:playtest" in text
        with open(ps / f"pack_v{v}" / "ratings.csv", encoding="utf-8-sig") as f:
            sheet = list(csv.DictReader(f))
        assert list(sheet[0]) == playtest_pack.SINGLE_FIELDS and {r["version"] for r in sheet} == {v}
    filled = tmp_path / "ratings_single.csv"           # one player, right every time, sure
    with open(filled, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=playtest_pack.SINGLE_FIELDS)
        w.writeheader()
        for r in answers:
            if r["version"] == "1":
                w.writerow({"song": r["song"], "version": "1", "sure": "5", "fun": "4",
                            "human": "y" if r["source"] == "human" else "n", "knew": "n"})
    assert playtest_pack.main(["--score", str(ps / "answers.csv"), str(filled)]) == 0
    assert playtest_pack.answer_auc([3, 1], [-3, 1]) == 0.875
    assert playtest_pack.answer_auc([2], [2]) == 0.5

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
    # shards: N processes, each on its share of the songs, give the unsharded run's charts
    from scripts import merge_shards
    sharded = ["--ckpt", str(tmp_path / "r" / "best.pt"), "--split", "train", "--per-song",
               "--n", "3", "--steps", "4", "--sample-seed", "3", "--manifest",
               str(data / "manifest.csv"), "--root", str(data / "raw"),
               "--cache", str(data / "cache"), "--device", "cpu"]
    assert evaluate.main(sharded) == 0
    whole = tmp_path / "r" / "eval_train_songs_continue_random_T4_s3"
    whole.rename(whole.with_name(whole.name + "_ref"))
    ref = whole.with_name(whole.name + "_ref")
    assert evaluate.main([*sharded, "--shard", "2/2"]) == 2               # K < N
    assert merge_shards.main([str(whole)]) == 2                           # no shards yet
    assert evaluate.main([*sharded, "--shard", "1/2"]) == 0
    assert merge_shards.main([str(whole)]) == 2                           # shard 0 missing
    assert evaluate.main([*sharded, "--shard", "0/2"]) == 0
    assert merge_shards.main([str(whole)]) == 0
    one, two = np.load(ref / "charts.npz"), np.load(whole / "charts.npz")
    assert sorted(one.files) == sorted(two.files) and len(one.files) == 3
    assert all((one[k] == two[k]).all() for k in one.files)
    with open(ref / "per_song.csv", newline="") as f:
        rows_ref = list(csv.DictReader(f))
    with open(whole / "per_song.csv", newline="") as f:
        rows_merged = list(csv.DictReader(f))
    assert list(rows_ref[0]) == list(rows_merged[0])                      # no _index left
    timing = {"seconds"}
    assert [{k: v for k, v in r.items() if k not in timing} for r in rows_ref] == \
        [{k: v for k, v in r.items() if k not in timing} for r in rows_merged]
    merged = json.loads((whole / "summary.json").read_text())
    single = json.loads((ref / "summary.json").read_text())
    assert merged["songs"] == 3 and merged["shards"] == 2 and "shard" not in merged
    assert merged["mean_f1@50"] == single["mean_f1@50"]
    assert merged["ci95_f1@50"] == single["ci95_f1@50"]
    assert merge_shards.main([str(whole)]) == 2                           # already there
    assert evaluate.main([*sharded, "--from-charts", str(whole), "--rest", "1",
                          "--shard", "0/2"]) == 0                          # passes, sharded too
    assert (tmp_path / "r" / "eval_train_songs_continue_random_T4_s3_rb1_shard0of2").exists()
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
    # a run grown to more songs: the two it has kept as they are, the third drawn
    grow = ["--steps", "4", "--order", "block", "--spread", "--lanes", "forward",
            "--forward-temp", "1", "--refine", "1", "--empty-bias", "0.5"]
    kept = {k: saved[k].copy() for k in saved.files}
    with open(run_dir / "per_song.csv", newline="") as f:
        before = {r["key"]: r for r in csv.DictReader(f)}
    assert evaluate.main([*common, "--n", "3", "--steps", "4", "--reuse", str(run_dir)]) == 2
    assert evaluate.main([*common, "--n", "3", *grow, "--reuse", str(run_dir)]) == 0
    with np.load(run_dir / "charts.npz") as grown:
        assert len(grown.files) == 3 and all(np.array_equal(grown[k], kept[k]) for k in kept)
    with open(run_dir / "per_song.csv", newline="") as f:
        after = {r["key"]: r for r in csv.DictReader(f)}
    assert all(after[k]["seconds"] == before[k]["seconds"] and after[k]["f1@50"] == before[k]["f1@50"]
               for k in before)
    from scripts.evaluate import shard_of
    with open(data / "manifest.csv", newline="") as f:
        some = [r for r in csv.DictReader(f) if (data / "cache" / "tokens" / f"{r['key']}.npz").exists()][:4]
    split = shard_of(some, data / "cache", 2, skip=frozenset(r["key"] for r in some[:3]))
    assert split == [1, 1, 1, 0]          # the one drawn alone on a shard, the kept weigh nothing
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



@requires_mp3
def test_stems_model_end_to_end(data: Path, tmp_path: Path, monkeypatch) -> None:
    """stems.py (a fake separator) -> train.py --init C --stems on the real mel -> evaluate.py
    and sample.py with the 144-channel model; the --resume / --init checks."""
    from scripts import evaluate, stems
    from src.data.audio import resample
    from src.models.diffusion import load_denoiser

    def fake(mono, rate):
        wav = resample(mono, rate, 44100)
        return np.stack([wav * g for g in (0.5, 0.2, 0.2, 0.1)]), 44100

    monkeypatch.setattr(stems, "demucs_separator",
                        lambda device: (fake, ("drums", "bass", "other", "vocals")))
    assert stems.main(["--cache", str(data / "cache"), "--raw", str(data / "raw"),
                       "--manifest", str(data / "manifest.csv"), "--min-free", "0"]) == 0
    base = ["--manifest", str(data / "manifest.csv"), "--cache", str(data / "cache"),
            "--out", str(tmp_path), "--overfit", "2", "--batch-size", "4", "--warmup", "2",
            "--d-model", "64", "--layers", "2", "--heads", "2", "--d-ff", "128",
            "--log-every", "2", "--val-every", "4", "--sample-steps", "4", "--device", "cpu"]
    assert train.main([*base, "--run", "mix", "--steps", "4"]) == 0
    assert train.main([*base, "--run", "st", "--steps", "4", "--stems",
                       "--init", str(tmp_path / "mix" / "best.pt")]) == 0
    model = load_denoiser(tmp_path / "st" / "best.pt")
    assert model.config.n_mels == 144 and model.config.d_model == 64
    cfg = json.loads((tmp_path / "st" / "config.json").read_text())
    assert cfg["args"]["stems"] and cfg["args"]["init"].endswith("best.pt")
    assert train.main([*base, "--run", "st", "--steps", "6",
                       "--resume", str(tmp_path / "st" / "last.pt")]) == 2        # needs --stems
    assert train.main([*base, "--run", "st", "--steps", "6", "--stems",
                       "--resume", str(tmp_path / "st" / "last.pt")]) == 0
    assert train.main([*base, "--run", "x", "--steps", "4",
                       "--init", str(tmp_path / "st" / "best.pt")]) == 2          # needs --stems
    assert train.main([*base, "--run", "x", "--steps", "4", "--stems",
                       "--init", str(tmp_path / "mix" / "best.pt"),
                       "--resume", str(tmp_path / "st" / "last.pt")]) == 2
    common = ["--ckpt", str(tmp_path / "st" / "best.pt"), "--manifest", str(data / "manifest.csv"),
              "--root", str(data / "raw"), "--cache", str(data / "cache"), "--device", "cpu",
              "--steps", "4"]
    assert evaluate.main([*common, "--split", "train", "--n", "1"]) == 0
    key = cfg["overfit_keys"][0]
    assert sample.main([*common, "--key", key]) == 0
    if importlib.util.find_spec("rosu_pp_py") is None:
        return
    from scripts import judge                                    # a judge on the stems trunk
    assert evaluate.main([*common, "--split", "train", "--per-song", "--n", "2"]) == 0
    run = next((tmp_path / "st").glob("eval_train_songs_*"))
    assert judge.main(["--ckpt", str(tmp_path / "st" / "best.pt"), "--ai", str(run),
                       "--manifest", str(data / "manifest.csv"), "--cache", str(data / "cache"),
                       "--device", "cpu", "--out", str(tmp_path / "judge"), "--folds", "2",
                       "--epochs", "1", "--batch-pairs", "2", "--name", "st"]) == 0
    summary = json.loads((tmp_path / "judge" / "st" / "summary.json").read_text())
    assert summary["ai"]["songs"] == 2


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
    rested = parse_setting("random:128:continue:fwd+ref2+st+lg2+og1+rb1")
    assert rested["rest"] == 1 and rested["name"].endswith("og1 rb1")
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
    full = parse_setting("random:128:continue:fwd+ref2+st+lg2+og1+ot+rc0+cp0+lt1+fh0.5+rb1")
    assert (full["onset_taper"], full["carry"], full["ln_tidy"], full["holes"]) == \
        ((2.7, 5.0, 0.35), 0, 1, 0.5)
    assert full["name"] == "ai random T128 continue fwd ref2@0.5 rc0 cp0 lt1 st lg2 og1 ot fh0.5 rb1"
    assert parse_setting("random:128")["carry"] is None
    lagged = parse_setting("random:128:continue:fwd+cp0l4")
    assert (lagged["copy_bias"], lagged["copy_min_lag"]) == (0, 4) and lagged["name"].endswith("cp0l4")
    both = parse_setting("random:128:continue:fwd+rc0kh-2")
    assert (both["carry"], both["carry_keep"], both["carry_holds"]) == (0, True, -2)
    assert both["name"].endswith("rc0kh-2")
    assert parse_setting("random:128:continue:fwd+rc2k")["carry_holds"] is None
    rows = parse_setting("random:128:continue:fwd+rc1.5krh-1")
    assert (rows["carry"], rows["carry_keep"], rows["carry_rows"], rows["carry_holds"]) == \
        (1.5, True, True, -1)
    assert rows["name"].endswith("rc1.5krh-1") and not both["carry_rows"]
    assert parse_setting("random:128:continue:fwd+cp2")["copy_min_lag"] == 1
    back = parse_setting("random:128:continue:fwd+rc0kh-1+cs1k+cp0l8")
    assert (back["sections"], back["section_keep"]) == (1, True)
    assert back["name"].endswith("rc0kh-1 cs1k cp0l8")
    assert parse_setting("random:128:continue:fwd+cs-0.5")["section_keep"] is False
    lanes = parse_setting("random:128:continue:fwd+cs4kn")
    assert (lanes["sections"], lanes["section_keep"], lanes["section_lanes"]) == (4, True, True)
    assert lanes["name"].endswith("cs4kn") and not back["section_lanes"]
    capped = parse_setting("random:128:continue:fwd+ref2+st+lg2+og1+ot+rc0kh1+cp0l8+fh0.5+rb1+uj+cc")
    assert capped["chord_cap"] == 4 and capped["untangle"] and not capped["untangle_move"]
    assert capped["name"].endswith("fh0.5 rb1 uj cc4")
    assert parse_setting("random:128")["chord_cap"] is None
    later = parse_setting("random:128:continue:fwd+rc0kh1+cs4kb+cp0l8+ujmg+cc2")
    assert (later["sections"], later["section_keep"], later["section_back"]) == (4, True, True)
    assert later["untangle"] and later["untangle_move"] and later["untangle_grades"]
    assert later["chord_cap"] == 2 and later["name"].endswith("cs4kb cp0l8 ujmg cc2")
    alike = parse_setting("random:128:continue:fwd+rc0kh1+cs4kb+cp0l8+al2k+rb1+sc1+ujmg+cc2")
    assert (alike["alike"], alike["alike_keep"], alike["strong"]) == (2, 0, 1)
    assert alike["name"].endswith("rc0kh1 sc1 cs4kb cp0l8 al2k rb1 ujmg cc2")
    assert parse_setting("random:128")["alike"] is None
    keep4 = parse_setting("random:128:continue:fwd+hu+rc0kh1+cs4kb+cp0l8+al2k4+rb1+li0.5+ujmg")
    assert (keep4["alike"], keep4["alike_keep"], keep4["light"]) == (2, 4, 0.5)
    assert keep4["hold_up"] and keep4["hold_share"] == "human"
    assert keep4["name"].endswith("hu rc0kh1 cs4kb cp0l8 al2k4 rb1 li0.5 ujmg")
    wide = parse_setting("random:128:continue:fwd+rb1+li0.5a+ujmgw")
    assert wide["light_alike"] and wide["untangle_wide"] and wide["untangle_grades"]
    assert wide["name"].endswith("rb1 li0.5a ujmgw") and not keep4["light_alike"]
    insane = parse_setting("random:128:continue:fwd+rb1+li0.5a+ujmgi")
    assert insane["untangle_insane"] and not insane["untangle_wide"]
    assert insane["name"].endswith("rb1 li0.5a ujmgi") and not wide["untangle_insane"]
    assert parse_setting("random:128:continue:al2")["alike_keep"] is None
    assert parse_setting("random:128:continue:hu0.3")["hold_share"] == 0.3
    assert parse_setting("random:128")["sections"] is None
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


def test_fit_plane_recovers_a_planar_path(monkeypatch) -> None:
    from scripts import fit_viz
    monkeypatch.setattr(fit_viz, "BLOCK", 1024)           # several passes over 5000 columns
    rng = np.random.default_rng(0)
    e = np.linalg.qr(rng.normal(size=(5000, 2)))[0].T     # two orthonormal directions
    base = rng.normal(size=5000)
    t = np.linspace(0, 1, 12)
    a, b = 3 * (1 - t) ** 2, np.sin(3 * t)                # a curved path in that plane
    X = [(base + ai * e[0] + bi * e[1]).astype(np.float32) for ai, bi in zip(a, b, strict=True)]
    pl = fit_viz.plane(X)
    assert sum(pl["explained"]) > 0.999 and pl["explained"][0] >= pl["explained"][1]
    assert np.allclose(pl["d"] @ pl["d"].T, np.eye(2), atol=1e-6)
    c = pl["coords"]
    assert np.allclose(c[-1], 0)
    true = np.stack([a, b], axis=1)
    dist = lambda p: np.linalg.norm(p[:, None] - p[None], axis=-1)   # noqa: E731
    assert np.allclose(dist(c), dist(true), atol=1e-3)    # the plane keeps every distance
    from_init, length = fit_viz.distances(pl["gram"])
    assert np.allclose(from_init, np.linalg.norm(true - true[0], axis=1), atol=1e-3)
    assert np.isclose(length, np.linalg.norm(np.diff(true, axis=0), axis=1).sum(), rtol=1e-3)
    with pytest.raises(ValueError):
        fit_viz.plane(X[:3])


def test_snapshots_and_fit_viz(data: Path, tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    from scripts import fit_viz
    from src.models.diffusion import load_denoiser
    common = ["--manifest", str(data / "manifest.csv"), "--cache", str(data / "cache"),
              "--out", str(tmp_path), "--overfit", "4", "--batch-size", "4", "--warmup", "5",
              "--d-model", "64", "--layers", "2", "--heads", "2", "--d-ff", "128",
              "--log-every", "10", "--sample-steps", "4", "--fake-mel", "--device", "cpu",
              "--lr", "2e-3"]
    assert train.main([*common, "--run", "a", "--steps", "40", "--val-every", "40",
                       "--snapshot-every", "20"]) == 0
    traj = tmp_path / "a" / "traj"
    steps = sorted(int(f.stem.split("_")[1]) for f in traj.glob("params_*.npy"))
    assert steps == train.snapshot_steps(20, 40) == [0, 2, 4, 10, 20, 40]
    assert sorted(int(f.stem.split("_")[1]) for f in traj.glob("adam_*.npz")) == steps[1:]
    assert train.snapshot_steps(1000, 2500) == [0, 100, 200, 500, 1000, 2000, 2500]
    assert train.snapshot_steps(0, 100) == []
    model = load_denoiser(tmp_path / "a" / "last.pt")
    last = torch.nn.utils.parameters_to_vector(model.parameters()).detach().numpy()
    assert np.array_equal(np.load(traj / "params_000040.npy"), last)
    z = np.load(traj / "adam_000040.npz")
    assert len(z["m"]) == len(z["v"]) == len(np.load(traj / "sample_idx.npy")) and z["t"] == 40

    # snapshots draw no random numbers: the same run without them ends at the same point
    assert train.main([*common, "--run", "b", "--steps", "40", "--val-every", "40"]) == 0
    other = torch.load(tmp_path / "b" / "last.pt", weights_only=True)["model"]
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in other.items())

    short = [*common, "--steps", "20", "--val-every", "20", "--snapshot-every", "10",
             "--weight-decay", "0"]
    assert train.main([*short, "--run", "sgd", "--optim", "sgd", "--lr", "0.05"]) == 0
    assert train.main([*short, "--run", "norm", "--betas", "0", "0", "--lr", "1e-3"]) == 0
    assert not list((tmp_path / "sgd" / "traj").glob("adam_*.npz"))

    viz = ["--grid", "4", "--line", "3", "--chunks", "3", "--batch-size", "8",
           "--device", "cpu"]
    assert fit_viz.main([str(tmp_path / "a"), *viz, "--ref", str(tmp_path / "b" / "last.pt")]) == 0
    out = tmp_path / "a" / "fit_viz"
    res = json.loads((out / "fit_viz.json").read_text())
    r = res["runs"][0]
    assert r["steps"] == steps and r["coords"][-1] == [0.0, 0.0]
    assert 0 < r["explained"][1] <= r["explained"][0] and sum(r["explained"]) <= 1 + 1e-9
    assert all(np.isfinite(r["loss_true"])) and r["loss_true"][-1] < r["loss_true"][0]
    assert np.isclose(r["loss_plane"][-1], r["loss_true"][-1], rtol=1e-5)  # the end is on it
    assert len(r["grid"]["loss"]) == 4 and len(r["line"]["loss"]) == 3
    assert np.isclose(r["line"]["loss"][0], r["loss_true"][0], rtol=1e-5)
    assert r["ref"]["dist_to_final"] < 1e-6                  # run b ended where run a did
    assert r["adam"]["steps"] == steps[1:] and set(r["adam"]["groups"]) >= {"embeddings",
                                                                            "LayerNorms"}
    for name in ("fit_path.png", "fit_curves.png", "fit_adam.png"):
        assert (out / name).stat().st_size > 0

    cmp_out = tmp_path / "cmp"
    assert fit_viz.main([str(tmp_path / "a"), str(tmp_path / "sgd"), str(tmp_path / "norm"),
                         *viz, "--out", str(cmp_out)]) == 0
    runs = json.loads((cmp_out / "fit_viz.json").read_text())["runs"]
    assert [r["optim"] for r in runs] == ["adamw", "sgd", "adamw"] and runs[1]["adam"] is None
    assert np.allclose(runs[2]["adam"]["u_median"], 1.0, atol=1e-3)   # betas 0 0: the sign
    a_loss = [r["loss_true"][0] for r in runs]
    assert np.allclose(a_loss, a_loss[0])            # same start, same chunks and masks
    before = (cmp_out / "fit_path.png").stat().st_mtime_ns
    assert fit_viz.main([str(tmp_path / "a"), "--out", str(cmp_out), "--replot"]) == 0
    assert (cmp_out / "fit_path.png").stat().st_mtime_ns >= before
    assert fit_viz.main([str(tmp_path / "b"), *viz]) == 2          # no snapshots


def test_judge_end_to_end(data: Path, tmp_path: Path) -> None:
    pytest.importorskip("rosu_pp_py")
    from scripts import evaluate, judge
    assert train.main(["--manifest", str(data / "manifest.csv"), "--cache", str(data / "cache"),
                       "--out", str(tmp_path), "--run", "j", "--overfit", "4", "--steps", "20",
                       "--batch-size", "4", "--warmup", "5", "--d-model", "64", "--layers", "2",
                       "--heads", "2", "--d-ff", "128", "--log-every", "10", "--val-every", "20",
                       "--sample-steps", "4", "--fake-mel", "--device", "cpu"]) == 0
    ckpt = tmp_path / "j" / "best.pt"
    common = ["--ckpt", str(ckpt), "--manifest", str(data / "manifest.csv"),
              "--root", str(data / "raw"), "--cache", str(data / "cache"), "--fake-mel",
              "--device", "cpu", "--steps", "4", "--split", "train", "--per-song", "--n", "3"]
    assert evaluate.main(common) == 0
    run = next((tmp_path / "j").glob("eval_*"))
    assert evaluate.main([*common, "--from-charts", str(run), "--rest", "1000"]) == 0
    rested = run.parent / (run.name + "_rb1000")
    args = ["--ckpt", str(ckpt), "--ai", str(run), "--score", str(rested),
            "--manifest", str(data / "manifest.csv"), "--cache", str(data / "cache"),
            "--fake-mel", "--device", "cpu", "--out", str(tmp_path / "judge"), "--folds", "2",
            "--epochs", "1", "--batch-pairs", "2", "--log-every", "1"]
    assert judge.main([*args, "--name", "a"]) == 0
    out = tmp_path / "judge" / "a"
    summary = json.loads((out / "summary.json").read_text())
    assert summary["ai"]["songs"] == 3 and 0 <= summary["ai"]["auc"] <= 1
    assert "minus_ai" in summary["+rb1000"] and summary["labels"]["+rb1000"] == str(rested)
    with open(out / "windows.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert {r["chart"] for r in rows} == {"human", "ai", "+rb1000"}
    assert {r["fold"] for r in rows} == {"0", "1"}                  # every song held out once
    def onsets(chart: str) -> int:
        return sum(int(r["onset_rows"]) for r in rows if r["chart"] == chart)
    assert onsets("+rb1000") < onsets("ai")                         # the whole bars emptied
    scores = np.load(out / "rows.npz")
    for key in {r["key"] for r in rows}:
        n = len(scores[f"{key}|starts"])
        assert scores[f"{key}|ai"].shape == (n, L) and scores[f"{key}|human"].shape == (n, L)
    with open(out / "log.csv", newline="") as f:
        assert {r["fold"] for r in csv.DictReader(f)} == {"0", "1"}
    assert judge.main([*args, "--name", "b", "--no-audio", "--folds", "5",       # as many as songs
                       "--score", str(rested), str(tmp_path / "none")]) == 0    # one left out
    assert json.loads((tmp_path / "judge" / "b" / "summary.json").read_text())["args"]["no_audio"]
    # the epochs scored on the way, a view, and a stop once the question is answered
    two = [*args, "--epochs", "2"]
    assert judge.main([*two, "--name", "e0"]) == 0
    assert judge.main([*two, "--name", "e1", "--eval-epochs"]) == 0

    def windows(name: str) -> list[dict]:
        with open(tmp_path / "judge" / name / "windows.csv", newline="") as f:
            return list(csv.DictReader(f))
    assert [r["logit"] for r in windows("e0")] == [r["logit"] for r in windows("e1")]  # the same
    e1 = json.loads((tmp_path / "judge" / "e1" / "summary.json").read_text())
    assert [e["epoch"] for e in e1["by_epoch"]] == [1, 2]
    assert e1["by_epoch"][-1]["auc"] == e1["ai"]["auc"] and len(e1["ai"]["auc_ci"]) == 2
    assert e1["folds_run"] == 2 and e1["stopped"] is None
    assert judge.main([*args, "--name", "v", "--view", "rhythm", "--sr", "view",
                       "--stop-bounds", "2", "--min-folds", "1"]) == 0
    v = json.loads((tmp_path / "judge" / "v" / "summary.json").read_text())
    assert v["stopped"]["after_folds"] == 1 and v["folds_run"] == 1 and v["ai"]["songs"] < 3
    assert v["args"]["view"] == "rhythm" and v["args"]["sr"] == "view"
    # --sr view: the SR of the chart as shown, as evaluate.py has it for the full chart
    with open(run / "per_song.csv", newline="") as f:
        first = next(csv.DictReader(f))
    z = np.load(data / "cache" / "tokens" / f"{first['key']}.npz")
    tps = [(float(t), float(b)) for t, b in z["timing_points"]]
    drawn = np.load(run / "charts.npz")[first["key"]].reshape(-1, K)
    assert abs(judge.shown_sr(drawn, tps, int(z["cell_offset"])) - float(first["sr_gen"])) < 1e-3
    taps = judge.view_tokens(drawn, "taps")
    assert np.isfinite(judge.shown_sr(taps, tps, int(z["cell_offset"])))
    held = judge.view_tokens(drawn, "holds", (0, 1, 2))           # the holds view decodes too
    assert np.isfinite(judge.shown_sr(held, tps, int(z["cell_offset"])))
    seen = {(r["key"], r["start"], r["chart"]): r["onset_rows"] for r in windows("a")}
    kept = windows("v")
    assert {r["fold"] for r in kept} == {"0"} and kept
    assert all(seen[(r["key"], r["start"], r["chart"])] == r["onset_rows"] for r in kept)
    assert judge.main([*args, "--name", "x", "--stop-bounds", "a,b"]) == 2
    assert judge.main([*args, "--name", "m", "--max-folds", "1", "--view", "holds", "--sr",
                       "view"]) == 0
    m = json.loads((tmp_path / "judge" / "m" / "summary.json").read_text())
    assert m["folds_run"] == 1 and m["stopped"]["max_folds"] == 1 and m["args"]["view"] == "holds"
    # the fold judges kept, scoring as they did; a drawing that chooses by them
    from src.data.beat_grid import from_timing_points
    from src.data.mel import open_mel
    from src.models.judge import stable_hash
    from src.models.select import WindowJudge
    assert judge.main([*args, "--name", "keep", "--save-models"]) == 0
    keep = tmp_path / "judge" / "keep"
    meta = json.loads((keep / "models.json").read_text())
    assert meta["saved"] == [0, 1] and len(meta["fold_of"]) == 3 and meta["view"] == "full"
    assert (keep / "fold0.pt").exists() and (keep / "fold1.pt").exists()
    seen = next(r for r in windows("keep") if r["chart"] == "human")
    key, start = seen["key"], int(seen["start"])
    with open(run / "per_song.csv", newline="") as f:
        sr_human = float(next(r for r in csv.DictReader(f) if r["key"] == key)["sr_human_local"])
    z = np.load(data / "cache" / "tokens" / f"{key}.npz")
    grid = from_timing_points([(float(t), float(b)) for t, b in z["timing_points"]])
    cell = start - int(z["cell_offset"])
    beat = float(grid.time_from_cell(cell + L, 12) - grid.time_from_cell(cell, 12)) / (L / 12)
    mel = torch.as_tensor(open_mel(data / "cache", fake=True).frames(
        key, grid, int(z["cell_offset"]), start, L), dtype=torch.float32)
    human = judge.window(z["tokens"].reshape(-1, K), start)
    score = WindowJudge(keep, torch.device("cpu")).scorer(key, sr_human)
    assert abs(float(score([human], mel, beat)[0]) - float(seen["logit"])) < 0.05
    other = WindowJudge(keep, torch.device("cpu"))     # a song of another split: a kept fold
    unseen = other.scorer("no such song", 3.0)
    assert unseen is not None and np.isfinite(unseen([human], mel, beat)).all()
    assert other.saved[stable_hash("no such song") % 2] in other._models
    unkept = [k for k, f in meta["fold_of"].items() if f not in meta["saved"]]
    for k in unkept:                                  # its own song in a fold not kept: none
        assert other.scorer(k, 3.0) is None
    assert judge.main([*args, "--name", "pick", "--view", "rhythm", "--sr", "human",
                       "--save-models"]) == 0
    pick = tmp_path / "judge" / "pick"
    assert evaluate.main([*common, "--select-judge", str(pick), "--select-k", "2"]) == 0
    chosen = next((tmp_path / "j").glob("eval_*_sel2pick*"))
    with open(chosen / "per_song.csv", newline="") as f:
        picked = list(csv.DictReader(f))
    assert len(picked) == 3 and all(float(r["select_windows"]) >= 1 for r in picked)
    assert all(float(r["select_gain"]) >= 0 for r in picked)
    assert json.loads((chosen / "summary.json").read_text())["select_k"] == 2
    assert evaluate.main([*common, "--select-judge", str(pick), str(pick), "--select-k", "2"]) == 0
    two = next((tmp_path / "j").glob(f"eval_*_sel2{pick.name}+{pick.name}*"))
    assert len(json.loads((two / "summary.json").read_text())["select_judge"]) == 2
    assert evaluate.main([*common, "--from-charts", str(run), "--select-judge", str(pick)]) == 2
    # the lanes chosen by a full-view judge: while drawing, and again on the saved charts
    lanes = ["--lanes", "forward", "--refine", "1", "--lane-select-judge", str(keep),
             "--lane-select-k", "2"]
    assert evaluate.main([*common, *lanes]) == 0
    drawn = next((tmp_path / "j").glob("eval_*_lsel2keep*"))
    assert json.loads((drawn / "summary.json").read_text())["lane_select_k"] == 2
    assert evaluate.main([*common, "--from-charts", str(run), *lanes]) == 0
    again = run.parent / (run.name + "_lsel2keep")
    with open(again / "per_song.csv", newline="") as f:
        relaned = list(csv.DictReader(f))
    assert len(relaned) == 3 and all(float(r["lane_select_windows"]) >= 1 for r in relaned)
    assert all(float(r["lane_select_gain"]) >= 0 and int(r["lane_seams"]) >= 0
               for r in relaned)
    old, new = np.load(run / "charts.npz"), np.load(again / "charts.npz")
    for r in relaned:                               # the rhythm and the long notes stay
        a, b = old[r["key"]], new[r["key"]]
        assert np.array_equal(np.isin(a, (TAP, HOLD_START)).any(-1),
                              np.isin(b, (TAP, HOLD_START)).any(-1))
        assert np.array_equal(a == HOLD_START, b == HOLD_START)
        assert np.array_equal((a == TAP).sum(-1), (b == TAP).sum(-1))
    assert evaluate.main([*common, "--lane-select-judge", str(keep)]) == 2   # no lane passes
    # the lanes drawn again once at other settings, no judge (a lane setting without a draw)
    relane = ["--from-charts", str(run), "--relane", "--lanes", "forward", "--refine", "1",
              "--lane-temp", "1"]
    assert evaluate.main([*common, *relane]) == 0
    other = run.parent / (run.name + "_rlfr1t1")
    new = np.load(other / "charts.npz")
    changed = 0
    for r in relaned:
        a, b = old[r["key"]], new[r["key"]]
        assert np.array_equal(np.isin(a, (TAP, HOLD_START)).any(-1),
                              np.isin(b, (TAP, HOLD_START)).any(-1))
        assert np.array_equal(a == HOLD_START, b == HOLD_START)
        changed += int(not np.array_equal(a, b))
    assert changed >= 1                              # the lanes are drawn again
    assert evaluate.main([*common, *relane, "--lane-inner-temp", "0.5"]) == 0   # by unit
    unit = run.parent / (run.name + "_rlfr1t1i0.5u12")
    assert (unit / "charts.npz").exists()
    by_unit = np.load(unit / "charts.npz")
    for r in relaned:                                 # still the rhythm and the long notes
        assert np.array_equal(old[r["key"]] == HOLD_START, by_unit[r["key"]] == HOLD_START)
        assert np.array_equal((old[r["key"]] == TAP).sum(-1), (by_unit[r["key"]] == TAP).sum(-1))
    assert evaluate.main([*common, "--from-charts", str(run), "--lane-inner-temp", "0.5"]) == 2
    assert evaluate.main([*common, *relane, "--lane-inner-temp", "0.5", "--lane-unit", "0"]) == 2
    assert evaluate.main([*common, "--relane", "--refine", "1"]) == 2       # no saved charts
    # the releases of the long notes added at another temperature (tag rt<x>)
    assert evaluate.main([*common, "--from-charts", str(run), "--hold-share", "0.3", "--hold-up",
                          "--release-temp", "1"]) == 0
    assert (run.parent / (run.name + "_hu-0.3rt1") / "charts.npz").exists()
    assert evaluate.main([*common, "--from-charts", str(run), "--release-temp", "1"]) == 2
    # short long notes where the chart is busy, in songs whose hold share target is under 1
    assert evaluate.main([*common, "--from-charts", str(run), "--hold-share", "0.3",
                          "--dense-taps", "1"]) == 0
    dense = run.parent / (run.name + "_hs0.3_dt1")
    if not dense.exists():
        dense = next(run.parent.glob(run.name + "_*dt1"))
    with open(dense / "per_song.csv", newline="") as f:
        assert all(int(float(r["dense_taps"])) >= 0 for r in csv.DictReader(f))
    assert evaluate.main([*common, "--from-charts", str(run), "--dense-taps", "1"]) == 2
    assert evaluate.main([*common, "--from-charts", str(run), "--hold-share", "0.3",
                          "--dense-taps", "1", "--dense-repeats"]) == 0
    assert next(run.parent.glob(run.name + "_*dt1r"), None) is not None
    assert evaluate.main([*common, "--from-charts", str(run), "--hold-share", "0.3",
                          "--dense-repeats"]) == 2
    # the chords to the human chart's share, the long notes down to the target, the releases
    assert evaluate.main([*common, "--from-charts", str(run), "--hold-share", "0.3",
                          "--chord-share", "oracle", "--hold-down", "--snap-releases", "3"]) == 0
    late = next(run.parent.glob(run.name + "_*choracle_hd_sn3*"))
    with open(late / "per_song.csv", newline="") as f:
        for r in csv.DictReader(f):
            assert all(int(float(r[c])) >= 0 for c in
                       ("chords_thinned", "chords_added", "held_down", "releases_snapped"))
    from src.data.tokenizer import grammar_violations
    late_charts = np.load(late / "charts.npz")
    for r in relaned:
        assert len(grammar_violations(late_charts[r["key"]])) == 0
    assert evaluate.main([*common, "--from-charts", str(run), "--hold-down"]) == 2
    assert evaluate.main([*common, "--chord-share", "oracle"]) == 2
    assert evaluate.main([*common, "--from-charts", str(run), "--chord-share", "1.5"]) == 2
    assert evaluate.main([*common, "--from-charts", str(run), "--snap-releases", "0"]) == 2
    assert evaluate.main([*common, "--hold-share", "0.3", "--dense-taps", "1"]) == 2


def test_corruptions_and_the_judge_checks(data: Path, tmp_path: Path) -> None:
    pytest.importorskip("rosu_pp_py")
    from scripts import corrupt_test, evaluate, judge, judge_vs_diag
    assert train.main(["--manifest", str(data / "manifest.csv"), "--cache", str(data / "cache"),
                       "--out", str(tmp_path), "--run", "c", "--overfit", "4", "--steps", "20",
                       "--batch-size", "4", "--warmup", "5", "--d-model", "64", "--layers", "2",
                       "--heads", "2", "--d-ff", "128", "--log-every", "10", "--val-every", "20",
                       "--sample-steps", "4", "--fake-mel", "--device", "cpu"]) == 0
    ckpt = tmp_path / "c" / "best.pt"
    songs = ["--split", "train", "--per-song", "--manifest", str(data / "manifest.csv"),
             "--cache", str(data / "cache"), "--fake-mel"]
    assert evaluate.main(["--ckpt", str(ckpt), "--root", str(data / "raw"), "--device", "cpu",
                          "--steps", "4", *songs, "--n", "3"]) == 0
    run = next((tmp_path / "c").glob("eval_*"))
    out = tmp_path / "corrupt"
    assert corrupt_test.main([*songs, "--n", "3", "--out", str(out), "--write-runs",
                              "--corruptions", "lanes", "new"]) == 0
    summary = json.loads((out / "summary.json").read_text())
    assert summary["songs"] == 3 and set(summary["metrics"]["notes"]) == {"lanes", "new"}
    assert summary["metrics"]["notes"]["new"]["rho"] == pytest.approx(1.0)   # taps added
    assert "lanes" not in summary["verdict"]["notes"]["sensitive_to"]          # rows kept
    assert (out / "summary.md").exists()
    with open(out / "runs" / "new_0.08" / "per_song.csv", newline="") as f:
        assert {"key", "grade", "sr_gen", "sr_human_local"} <= set(next(csv.DictReader(f)))
    args = ["--ckpt", str(ckpt), "--ai", str(run), "--score", str(out / "runs" / "new_0.08"),
            "--manifest", str(data / "manifest.csv"), "--cache", str(data / "cache"),
            "--fake-mel", "--device", "cpu", "--out", str(tmp_path / "judge"), "--folds", "3",
            "--epochs", "1", "--batch-pairs", "2", "--log-every", "1"]
    assert judge.main([*args, "--name", "null", "--shuffle-labels"]) == 0
    null = json.loads((tmp_path / "judge" / "null" / "summary.json").read_text())
    assert null["args"]["shuffle_labels"] and "new_0.08" in null
    assert corrupt_test.main([*songs, "--n", "2", "--out", str(tmp_path / "two"),
                              "--write-runs", "--corruptions", "new"]) == 0
    partial = tmp_path / "partial"                            # a --score run of 2 of the 3 songs
    (tmp_path / "two" / "runs" / "new_0.08").rename(partial)
    assert judge.main([*args, "--score", str(partial), "--name", "real"]) == 0
    real = json.loads((tmp_path / "judge" / "real" / "summary.json").read_text())
    assert real["ai"]["songs"] == 3 and real["partial"]["songs"] == 2   # trained on all 3
    with open(run / "per_song.csv", newline="") as f:       # judge_vs_diag needs 10+ songs
        assert len(list(csv.DictReader(f))) == 3
    assert judge_vs_diag.main(["--judge", str(tmp_path / "judge" / "real"),
                               "--run", str(run)]) == 2
    assert judge.main([*args, "--name", "c", "--limit", "1"]) == 2   # one song: none to hold out
