"""Write data/chart_stats.csv: long-note share, jack and trill rate of every chart (src/data/chartstats.py).

    python scripts/build_chart_stats.py
    python scripts/build_chart_stats.py --manifest data/manifest.csv --cache data/cache \
        --out data/chart_stats.csv

Reads each kept chart's tokens from the cache (no audio needed). Prints the train
quantile edges train.py --chart-stats will use, and how much the stats vary.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from src.data.cache import read_manifest
from src.data.chartstats import BINS, FIELDS, NAMES, build_spec, chart_stats, read_stats
from src.data.tokenizer import K


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"))
    ap.add_argument("--cache", type=Path, default=Path("data/cache"))
    ap.add_argument("--out", type=Path, default=Path("data/chart_stats.csv"))
    ap.add_argument("--bins", type=int, default=BINS)
    a = ap.parse_args(argv)
    kept = [r for r in read_manifest(a.manifest) if r.get("drop", "") == "" and r["sr"] != ""]
    rows, missing = [], 0
    for r in kept:
        path = a.cache / "tokens" / f"{r['key']}.npz"
        if not path.exists():
            missing += 1
            continue
        z = np.load(path)
        st = chart_stats(z["tokens"].reshape(-1, K)[:int(z["n_cells"])])
        rows.append({"key": r["key"], "split": r["split"], "sr": r["sr"],
                     **{n: "" if not np.isfinite(st[n]) else f"{st[n]:.5f}" for n in NAMES}})
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    stats = read_stats(a.out)
    train = [k for k, v in stats.items() if v["split"] == "train"]
    spec = build_spec(stats, train, a.bins)
    print(f"{a.out}: {len(rows)} charts ({missing} without cached tokens), {len(train)} train")
    for n, e in zip(NAMES, spec["edges"], strict=True):
        v = np.array([stats[k][n] for k in train])
        v = v[np.isfinite(v)]
        print(f"  {n:11s} mean {v.mean():.3f} sd {v.std():.3f} p10/50/90 "
              f"{np.percentile(v, 10):.3f}/{np.median(v):.3f}/{np.percentile(v, 90):.3f}  "
              f"{len(e) + 1} buckets, edges " + " ".join(f"{x:.3f}" for x in e))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
