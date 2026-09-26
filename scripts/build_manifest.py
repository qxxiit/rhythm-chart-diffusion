"""One row per 4K chart: its cache key, split and SR label.

    python scripts/build_manifest.py          # data/raw -> data/manifest.csv

key        md5 of the path under data/raw; the file name used by the caches
path       path under data/raw
beatmap_id [Metadata] BeatmapID, 0 if missing
set_id     [Metadata] BeatmapSetID, else the number the folder name starts with
audio_key  normalized artist|title (analyze_dataset._audio_key), the unit of the split
split      train / val / test from md5(audio_key): buckets 0-89 / 90-94 / 95-99 of 100
sr         osu! API difficulty_rating from data/metadata/beatmapsets.jsonl; blank if missing
n_notes

The split is a pure function of audio_key: every difficulty and every re-upload
of one song lands in the same split, and it never changes when charts are added
(design doc §4.5, 분할).
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
from scripts.analyze_dataset import _audio_key, _read_metadata
from scripts.validate_tokenizer import load

FIELDS = ["key", "path", "beatmap_id", "set_id", "audio_key", "split", "sr", "n_notes"]


def split_of(audio_key: str) -> str:
    bucket = int(hashlib.md5(audio_key.encode("utf-8")).hexdigest(), 16) % 100
    return "train" if bucket < 90 else ("val" if bucket < 95 else "test")


def chart_key(relpath: str) -> str:
    return hashlib.md5(relpath.encode("utf-8")).hexdigest()[:16]


def load_sr(metadata: Path) -> dict[int, float]:
    if not metadata.exists():
        print(f"[warn] {metadata} not found: sr will be blank")
        return {}
    sr = {}
    with open(metadata, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                for bm in json.loads(line).get("beatmaps", []):
                    if bm.get("id") is not None and bm.get("difficulty_rating") is not None:
                        sr[int(bm["id"])] = float(bm["difficulty_rating"])
    return sr


def row_for(args: tuple[str, str]) -> dict | None:
    path, root = args
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
    return {"key": chart_key(rel), "path": rel, "beatmap_id": beatmap_id, "set_id": set_id,
            "audio_key": audio_key, "split": split_of(audio_key), "n_notes": len(chart.notes)}


def build(root: Path, metadata: Path, workers: int) -> list[dict]:
    paths = sorted(root.rglob("*.osu"))
    jobs = [(str(p), str(root)) for p in paths]
    if workers > 1:
        with Pool(workers) as pool:
            rows = [r for r in pool.imap(row_for, jobs, chunksize=32) if r]
    else:
        rows = [r for r in map(row_for, jobs) if r]
    sr = load_sr(metadata)
    for r in rows:
        value = sr.get(r["beatmap_id"]) if r["beatmap_id"] else None
        r["sr"] = "" if value is None else f"{value:.5f}"
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", type=Path, default=Path("data/raw"))
    ap.add_argument("--metadata", type=Path, default=Path("data/metadata/beatmapsets.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("data/manifest.csv"))
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    a = ap.parse_args(argv)

    rows = build(a.root, a.metadata, a.workers)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)

    print(f"{len(rows):,} charts -> {a.out}")
    for split in ("train", "val", "test"):
        rs = [r for r in rows if r["split"] == split]
        print(f"  {split:5}  charts {len(rs):6,}  songs {len({r['audio_key'] for r in rs}):5,}"
              f"  sets {len({r['set_id'] for r in rs}):5,}")
    print(f"  without SR: {sum(r['sr'] == '' for r in rows):,} (not used for training)")
    keys = [r["key"] for r in rows]
    if len(set(keys)) != len(keys):
        print("[error] duplicate keys", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
