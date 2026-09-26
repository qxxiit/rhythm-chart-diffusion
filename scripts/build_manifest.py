"""One row per 4K chart: cache key, split, SR label and the data filter (design doc §4.5).

    python scripts/build_manifest.py                  # data/raw -> data/manifest.csv
    python scripts/build_manifest.py --sr-source api  # label = osu! API SR instead of local

key         md5 of the path under data/raw; the file name used by the caches
path        path under data/raw
beatmap_id  [Metadata] BeatmapID, 0 if missing
set_id      [Metadata] BeatmapSetID, else the number the folder name starts with
audio_key   normalized artist|title (analyze_dataset._audio_key), the unit of the split
split       train / val / test from md5(audio_key): buckets 0-89 / 90-94 / 95-99 of 100
sr          the label s: local SR (rosu-pp, the calculator evaluation uses), else the API's
sr_local, sr_api, sr_source
n_notes, duration_s, nps, hold_ratio, lane_imbalance, api_notes
drop        why the chart is left out of training and evaluation ("" = kept), §4.5 filter:
            nps < 0.5, duration < 30 s, duration > 400 s, lane imbalance > 2,
            note count different from the API's, no SR at all
flags       kept but marked: ln_heavy (hold ratio > 0.9, undecided in §4.5)

duration, nps, hold ratio and lane imbalance follow analyze_dataset.py (span of
note starts), so the counts match the §4.5 table. The split is a pure function
of audio_key: every difficulty and re-upload of one song lands in the same
split, and it never moves when charts are added or dropped.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from scripts.analyze_dataset import _audio_key, _read_metadata
from scripts.validate_tokenizer import load

FIELDS = ["key", "path", "beatmap_id", "set_id", "audio_key", "split", "sr", "sr_local",
          "sr_api", "sr_source", "n_notes", "duration_s", "nps", "hold_ratio",
          "lane_imbalance", "api_notes", "drop", "flags"]


def split_of(audio_key: str) -> str:
    bucket = int(hashlib.md5(audio_key.encode("utf-8")).hexdigest(), 16) % 100
    return "train" if bucket < 90 else ("val" if bucket < 95 else "test")


def chart_key(relpath: str) -> str:
    return hashlib.md5(relpath.encode("utf-8")).hexdigest()[:16]


def load_api(metadata: Path) -> dict[int, dict]:
    """BeatmapID -> {"sr", "notes"} from the API dump."""
    if not metadata.exists():
        print(f"[warn] {metadata} not found: no API SR or note counts")
        return {}
    out = {}
    with open(metadata, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            for bm in json.loads(line).get("beatmaps", []):
                if bm.get("id") is None:
                    continue
                notes = None
                if bm.get("count_circles") is not None and bm.get("count_sliders") is not None:
                    notes = int(bm["count_circles"]) + int(bm["count_sliders"])  # mania holds = sliders
                out[int(bm["id"])] = {"sr": bm.get("difficulty_rating"), "notes": notes}
    return out


def local_sr(path: Path) -> float | None:
    try:
        from src.evaluation.sr import star_rating_file
    except ImportError:                          # rosu-pp-py not installed
        return None
    try:
        return star_rating_file(path)
    except Exception:
        return None


def row_for(args: tuple[str, str, bool]) -> dict | None:
    path, root, want_local = args
    _, chart = load(Path(path))
    if chart is None:
        return None
    rel = Path(path).relative_to(root).as_posix()
    meta = _read_metadata(Path(path))
    try:
        beatmap_id = int(meta.get("BeatmapID") or 0)
    except ValueError:
        beatmap_id = 0
    try:
        set_id = int(meta.get("BeatmapSetID") or 0)
    except ValueError:
        set_id = 0
    if set_id <= 0:
        head = Path(rel).parts[0].split()[0] if Path(rel).parts else ""
        set_id = int(head) if head.isdigit() else 0
    names = SimpleNamespace(artist=meta.get("ArtistUnicode") or meta.get("Artist", ""),
                            title=meta.get("TitleUnicode") or meta.get("Title", ""))
    audio_key = _audio_key(names)

    t = np.array([n.time_ms for n in chart.notes], dtype=np.float64)
    lanes = np.bincount([n.lane for n in chart.notes], minlength=chart.key_count)
    span = (t.max() - t.min()) / 1000.0
    sr_loc = local_sr(Path(path)) if want_local else None
    return {"key": chart_key(rel), "path": rel, "beatmap_id": beatmap_id, "set_id": set_id,
            "audio_key": audio_key, "split": split_of(audio_key), "n_notes": len(chart.notes),
            "sr_local": "" if sr_loc is None else f"{sr_loc:.5f}",
            "duration_s": f"{span:.3f}",
            "nps": f"{len(t) / span if span > 0 else 0.0:.4f}",
            "hold_ratio": f"{np.mean([n.end_ms is not None for n in chart.notes]):.4f}",
            "lane_imbalance": f"{lanes.max() / max(lanes.min(), 1):.4f}"}


def label_and_filter(r: dict, api: dict[int, dict], sr_source: str) -> None:
    info = api.get(r["beatmap_id"], {}) if r["beatmap_id"] else {}
    r["sr_api"] = "" if info.get("sr") is None else f"{float(info['sr']):.5f}"
    r["api_notes"] = "" if info.get("notes") is None else str(info["notes"])
    order = ("sr_local", "sr_api") if sr_source == "local" else ("sr_api", "sr_local")
    source = next((c for c in order if r[c] != ""), "")
    r["sr"] = r[source] if source else ""
    r["sr_source"] = source.removeprefix("sr_")

    drop = []
    if float(r["nps"]) < 0.5:
        drop.append("nps")
    if float(r["duration_s"]) < 30:
        drop.append("short")
    if float(r["duration_s"]) > 400:
        drop.append("long")
    if float(r["lane_imbalance"]) > 2:
        drop.append("lane_imbalance")
    if r["api_notes"] != "" and int(r["api_notes"]) != r["n_notes"]:
        drop.append("api_mismatch")
    if r["sr"] == "":
        drop.append("no_sr")
    r["drop"] = ";".join(drop)
    r["flags"] = "ln_heavy" if float(r["hold_ratio"]) > 0.9 else ""


def build(root: Path, metadata: Path, workers: int, sr_source: str = "local") -> list[dict]:
    paths = sorted(root.rglob("*.osu"))
    jobs = [(str(p), str(root), sr_source == "local") for p in paths]
    if workers > 1:
        with Pool(workers) as pool:
            rows = [r for r in pool.imap(row_for, jobs, chunksize=32) if r]
    else:
        rows = [r for r in map(row_for, jobs) if r]
    api = load_api(metadata)
    for r in rows:
        label_and_filter(r, api, sr_source)
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", type=Path, default=Path("data/raw"))
    ap.add_argument("--metadata", type=Path, default=Path("data/metadata/beatmapsets.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("data/manifest.csv"))
    ap.add_argument("--sr-source", choices=["local", "api"], default="local")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    a = ap.parse_args(argv)

    rows = build(a.root, a.metadata, a.workers, a.sr_source)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)

    kept = [r for r in rows if r["drop"] == ""]
    print(f"{len(rows):,} charts -> {a.out}  ({len(kept):,} kept)")
    for split in ("train", "val", "test"):
        rs = [r for r in kept if r["split"] == split]
        print(f"  {split:5}  charts {len(rs):6,}  songs {len({r['audio_key'] for r in rs}):5,}"
              f"  sets {len({r['set_id'] for r in rs}):5,}")
    reasons: dict[str, int] = {}
    for r in rows:
        for reason in filter(None, r["drop"].split(";")):
            reasons[reason] = reasons.get(reason, 0) + 1
    print("  dropped (a chart can have several reasons): "
          + (", ".join(f"{k} {v}" for k, v in sorted(reasons.items())) or "none")
          + f"; total {len(rows) - len(kept)}")
    print(f"  flagged ln_heavy (kept): {sum(r['flags'] == 'ln_heavy' for r in rows)}")
    by_source = {s: sum(r["sr_source"] == s for r in rows) for s in ("local", "api", "")}
    print(f"  SR label from local {by_source['local']:,}, API {by_source['api']:,}, "
          f"none {by_source['']:,}")
    keys = [r["key"] for r in rows]
    if len(set(keys)) != len(keys):
        print("[error] duplicate keys", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
