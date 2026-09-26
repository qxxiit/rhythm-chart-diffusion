"""manifest -> token cache -> ChunkDataset -> train.py, on synthetic .osu files."""

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
from src.data.tokenizer import PAD, L  # noqa: E402


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
            write_osu(raw / f"{100 + s} {artist} - {title}" / f"v{v}.osu", chart, title=title,
                      artist=artist if s != 7 else "Artist 0", version=f"v{v}",
                      beatmap_id=bid if (s, v) != (3, 1) else 0, set_id=100 + s)
            beatmaps.append({"id": bid, "difficulty_rating": 2.0 + s * 0.3 + v})
        sets.append({"id": 100 + s, "beatmaps": beatmaps})
    meta = base / "metadata" / "beatmapsets.jsonl"
    meta.parent.mkdir(parents=True)
    meta.write_text("\n".join(json.dumps(x) for x in sets) + "\n")

    assert build_manifest.main(["--root", str(raw), "--metadata", str(meta),
                                "--out", str(base / "manifest.csv"), "--workers", "1"]) == 0
    assert preprocess_data.main(["--manifest", str(base / "manifest.csv"), "--root", str(raw),
                                 "--cache", str(base / "cache"), "--fake-mel", "--limit", "100",
                                 "--workers", "1"]) == 0
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
    assert sum(r["sr"] == "" for r in rows) == 1                       # the chart without an ID


def test_token_cache_and_dataset(data: Path) -> None:
    ds = ChunkDataset(data / "manifest.csv", data / "cache", ("train", "val", "test"))
    assert len(ds.charts) == 15                                        # one has no SR
    item = ds[0]
    assert item["x0"].shape == (L, 4) and item["x0"].dtype == torch.long
    assert item["mel"].shape == (L * 4, 80) and item["mel"].dtype == torch.float32
    z = np.load(data / "cache" / "tokens" / f"{ds.charts[0]['key']}.npz")
    assert float(item["b"]) == pytest.approx(float(z["beat_len_ms"][0]))
    assert z["cell_offset"] % 48 == 0
    last = ds[len(ds) - 1]["x0"]
    assert (last == PAD).any()                                         # the final chunk pads


def test_train_overfit_runs_and_learns(data: Path, tmp_path: Path) -> None:
    args = ["--manifest", str(data / "manifest.csv"), "--cache", str(data / "cache"),
            "--out", str(tmp_path), "--run", "t", "--overfit", "2", "--steps", "150",
            "--batch-size", "4", "--lr", "2e-3", "--warmup", "10", "--d-model", "64",
            "--layers", "2", "--heads", "2", "--d-ff", "128", "--log-every", "10",
            "--val-every", "150", "--sample-steps", "8", "--device", "cpu"]
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
                        "--cache", str(data / "cache"), "--steps", "8", "--device", "cpu"]) == 0
    written = list((tmp_path / "t" / "samples").glob("*.osu"))
    assert len(written) == 1
    chart = parse_osu(written[0])                                     # osu! format round trip
    assert chart.key_count == 4 and chart.audio_filename == "audio.mp3" and chart.notes

    pytest.importorskip("rosu_pp_py")
    from scripts import evaluate
    assert evaluate.main(["--ckpt", str(tmp_path / "t" / "best.pt"), "--split", "train",
                          "--n", "2", "--steps", "4", "--manifest", str(data / "manifest.csv"),
                          "--root", str(data / "raw"), "--cache", str(data / "cache"),
                          "--device", "cpu"]) == 0
    summary = json.loads(next((tmp_path / "t").glob("eval_*/summary.json")).read_text())
    assert summary["songs"] == 2 and summary["mean_violation_rate"] == 0.0
    assert summary["mean_ceiling_f1@20"] > 0.99                        # tokenizer round trip


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
        assert sum(int(r["charts"]) for r in csv.DictReader(f)) == 15
