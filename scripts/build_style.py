"""Write data/style.csv: genre and mapper of every manifest chart (src/data/style.py).

    python scripts/build_style.py
    python scripts/build_style.py --manifest data/manifest.csv \
        --metadata data/metadata/beatmapsets.jsonl --out data/style.csv

Reads the API metadata fetched for the dataset (one beatmap set per line): the
set's genre_id, and each beatmap's user_id (the mapper of that difficulty; for a
guest difficulty, the guest). Charts whose beatmap is not in the metadata (no
beatmap id in the .osu) get the set's genre if the set is there and no mapper.
Prints how many train charts the mappers with at least --min-charts train charts
cover: the vocab train.py --style builds.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.cache import read_manifest
from src.data.style import FIELDS, GENRES, MIN_MAPPER_CHARTS, build_vocab, read_style


def build(manifest: Path, metadata: Path) -> list[dict]:
    sets, beatmaps = {}, {}
    with open(metadata, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            sets[int(d["id"])] = d
            for bm in d.get("beatmaps", []):
                beatmaps[int(bm["id"])] = (int(d["id"]), bm.get("user_id"))
    rows = []
    for r in read_manifest(manifest):
        set_id = int(r["set_id"]) if r["set_id"] else None
        bid = int(r["beatmap_id"]) if r["beatmap_id"] else 0
        if bid in beatmaps:
            set_id = beatmaps[bid][0]
        s = sets.get(set_id, {})
        mapper = beatmaps[bid][1] if bid in beatmaps else None
        name = s.get("creator", "") if mapper is not None and mapper == s.get("user_id") else ""
        rows.append({"key": r["key"], "set_id": set_id or "", "beatmap_id": bid or "",
                     "genre_id": s.get("genre_id", ""), "mapper_id": "" if mapper is None else mapper,
                     "mapper_name": name})
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"))
    ap.add_argument("--metadata", type=Path, default=Path("data/metadata/beatmapsets.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("data/style.csv"))
    ap.add_argument("--min-charts", type=int, default=MIN_MAPPER_CHARTS)
    a = ap.parse_args(argv)
    rows = build(a.manifest, a.metadata)
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    kept = [r for r in read_manifest(a.manifest) if r.get("drop", "") == "" and r["sr"] != ""]
    train = [r["key"] for r in kept if r["split"] == "train"]
    style = read_style(a.out)
    vocab = build_vocab(style, train, a.min_charts)
    known = set(vocab["mappers"])
    covered = sum(style[k]["mapper_id"] in known for k in train if k in style)
    genres = Counter(GENRES.get(style.get(r["key"], {}).get("genre_id"), "none") for r in kept)
    print(f"{a.out}: {len(rows)} charts, {sum(r['mapper_id'] != '' for r in rows)} with a mapper")
    print(f"mappers with >= {a.min_charts} train charts: {len(known)}, covering {covered} of "
          f"{len(train)} train charts")
    print("genres (kept charts): " + ", ".join(f"{g} {n}" for g, n in genres.most_common()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
