"""Local SR (rosu-pp) against the osu! API's SR, for every chart in the manifest.

    python scripts/check_sr.py              # data/sr_local.csv + docs/_stats/sr_check.csv
    python scripts/check_sr.py --limit 500

Per chart:
    sr_api        what training conditions on (manifest; blank for a few charts)
    sr_local      rosu-pp on the .osu file as downloaded
    sr_parsed     rosu-pp on the parsed chart written back out (should equal sr_local)
    sr_roundtrip  rosu-pp on decode(encode(chart)): what tokenization alone moves.
                  A generated chart's SR error cannot be read below this floor.

Answers the W0-1 questions: can the local number stand in where the API has
none, and how far does the tokenizer itself move SR.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd

from src.data.cache import read_manifest
from src.data.chart_parser import parse_osu
from src.data.tokenizer import ChartTooLong, decode, encode
from src.evaluation.sr import ROSU_VERSION, star_rating, star_rating_file

COLUMNS = ["key", "path", "sr_api", "sr_local", "sr_parsed", "sr_roundtrip", "error"]


def one(job: tuple) -> dict:
    row, root = job
    out = {"key": row["key"], "path": row["path"], "sr_api": row.get("sr_api", row["sr"]),
           "sr_local": "",
           "sr_parsed": "", "sr_roundtrip": "", "error": ""}
    path = root / row["path"]
    try:
        out["sr_local"] = f"{star_rating_file(path):.5f}"
        chart = parse_osu(path)
        out["sr_parsed"] = f"{star_rating(chart):.5f}"
        tokens, metas, _ = encode(chart)
        back = decode(tokens, metas)
        back.audio_filename = chart.audio_filename
        out["sr_roundtrip"] = f"{star_rating(back):.5f}"
    except ChartTooLong:
        out["error"] = "too_long"
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
    return out


def compare(a: pd.Series, b: pd.Series) -> dict:
    """Agreement of b with a, over charts that have both."""
    ok = a.notna() & b.notna()
    a, b = a[ok], b[ok]
    d = (b - a).abs()
    if len(a) < 2:
        return {"n": len(a)}
    return {
        "n": len(a),
        "pearson": round(float(np.corrcoef(a, b)[0, 1]), 5),
        "spearman": round(float(a.rank().corr(b.rank())), 5),
        "mean_signed": round(float((b - a).mean()), 5),
        "mae": round(float(d.mean()), 5),
        "p50_abs": round(float(d.median()), 5),
        "p95_abs": round(float(d.quantile(0.95)), 5),
        "max_abs": round(float(d.max()), 5),
        "share_abs_le_0.01": round(float((d <= 0.01).mean()), 5),
        "share_abs_le_0.1": round(float((d <= 0.1).mean()), 5),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"))
    ap.add_argument("--root", type=Path, default=Path("data/raw"))
    ap.add_argument("--out", type=Path, default=Path("data/sr_local.csv"))
    ap.add_argument("--stats", type=Path, default=Path("docs/_stats/sr_check.csv"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    a = ap.parse_args(argv)

    rows = read_manifest(a.manifest)
    if a.limit:
        rows = rows[:a.limit]
    jobs = [(r, a.root) for r in rows]
    if a.workers > 1:
        with Pool(a.workers) as pool:
            results = list(pool.imap(one, jobs, chunksize=32))
    else:
        results = [one(j) for j in jobs]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, lineterminator="\n")
        w.writeheader()
        w.writerows(results)

    df = pd.DataFrame(results)
    for c in ("sr_api", "sr_local", "sr_parsed", "sr_roundtrip"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    stats = {"rosu_pp_py": ROSU_VERSION, "charts": len(df),
             "errors": int((df["error"] != "").sum()),
             "without_api_sr": int(df["sr_api"].isna().sum()),
             "without_api_sr_but_local": int((df["sr_api"].isna() & df["sr_local"].notna()).sum())}
    blocks = {"local_vs_api": compare(df["sr_api"], df["sr_local"]),
              "parsed_vs_local": compare(df["sr_local"], df["sr_parsed"]),
              "roundtrip_vs_local": compare(df["sr_local"], df["sr_roundtrip"])}
    a.stats.parent.mkdir(parents=True, exist_ok=True)
    with open(a.stats, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["metric", "value"])
        for k, v in stats.items():
            w.writerow([k, v])
        for name, block in blocks.items():
            for k, v in block.items():
                w.writerow([f"{name}.{k}", v])

    print(f"rosu-pp-py {ROSU_VERSION}: {len(df):,} charts, {stats['errors']} errors, "
          f"{stats['without_api_sr']} without API SR "
          f"({stats['without_api_sr_but_local']} of them have a local SR)")
    for name, block in blocks.items():
        print(f"  {name:20} " + "  ".join(f"{k} {v}" for k, v in block.items()))
    d = (df["sr_local"] - df["sr_api"]).abs()
    worst = df.assign(diff=d).dropna(subset=["diff"]).nlargest(10, "diff")
    if len(worst):
        print("  largest |local - api|:")
        for _, r in worst.iterrows():
            print(f"    {r['diff']:6.3f}  api {r['sr_api']:.3f}  local {r['sr_local']:.3f}  {r['path']}")
    errors = df[df["error"] != ""]
    for _, r in errors.head(5).iterrows():
        print(f"  error: {r['path']}: {r['error']}")
    return 1 if len(df) and stats["errors"] == len(df) else 0


if __name__ == "__main__":
    raise SystemExit(main())
