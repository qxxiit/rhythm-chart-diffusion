"""Long notes in human charts, by SR grade: how many, how long, how soon the next press.

    python scripts/hold_stats.py                    # train split -> docs/_stats/hold_stats.csv
    python scripts/hold_stats.py --split val --out /tmp/holds_val.csv

Read from the token cache (1/12-beat cells, the grid the model writes on), so the
numbers compare directly with generated charts (evaluate.py reports the same
quantities per chart: src/evaluation/holds.py). Per grade:
    charts, no_ln            charts, and the share of them without any long note
    ln_share_p50/p75/p90     long notes / onsets, over charts
    len_p10/p25/p50          long-note length in cells (12 = one beat), over long notes
    short                    share of long notes shorter than 3 cells (1/4 beat)
    gap_1, gap_2, gap_3_5    share of long notes whose lane's next onset comes 1, 2 or
                             3-5 cells after the release (gap_1: the very next cell)
The sampler's clean_holds defaults (min_hold, release_gap) should sit where these
put human charts.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from src.data.cache import read_manifest
from src.data.tokenizer import HOLD_START, TAP, K
from src.evaluation.holds import SHORT, hold_spans, release_gaps

GRADES = [("Easy", 0.0, 2.0), ("Normal", 2.0, 2.7), ("Hard", 2.7, 4.0),       # evaluate.GRADES
          ("Insane", 4.0, 5.3), ("Expert", 5.3, 6.5), ("Expert+", 6.5, float("inf"))]
FIELDS = ["grade", "charts", "no_ln", "ln_share_p50", "ln_share_p75", "ln_share_p90",
          "long_notes", "len_p10", "len_p25", "len_p50", "short", "gap_1", "gap_2", "gap_3_5"]


def chart_numbers(tokens: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    flat = tokens.reshape(-1, K)
    onsets = int(np.isin(flat, (TAP, HOLD_START)).sum())
    spans = hold_spans(flat)
    lens = np.array([e - s for _, s, e in spans], dtype=np.int64)
    return (len(spans) / onsets if onsets else 0.0), lens, release_gaps(flat, spans)


def summarize(name: str, shares: list[float], lens: list[np.ndarray], gaps: list[np.ndarray]) -> dict:
    lens_all = np.concatenate(lens) if lens else np.zeros(0, np.int64)
    gaps_all = np.concatenate(gaps) if gaps else np.zeros(0, np.int64)
    sh = np.array(shares)
    n = max(len(lens_all), 1)

    def pct(v, q):
        return round(float(np.percentile(v, q)), 3) if len(v) else ""

    return {"grade": name, "charts": len(sh), "no_ln": round(float(np.mean(sh == 0)), 3),
            "ln_share_p50": pct(sh, 50), "ln_share_p75": pct(sh, 75), "ln_share_p90": pct(sh, 90),
            "long_notes": len(lens_all), "len_p10": pct(lens_all, 10), "len_p25": pct(lens_all, 25),
            "len_p50": pct(lens_all, 50), "short": round(float(np.sum(lens_all < SHORT) / n), 4),
            "gap_1": round(float(np.sum(gaps_all == 1) / n), 4),
            "gap_2": round(float(np.sum(gaps_all == 2) / n), 4),
            "gap_3_5": round(float(np.sum((gaps_all >= 3) & (gaps_all <= 5)) / n), 4)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"))
    ap.add_argument("--cache", type=Path, default=Path("data/cache"))
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", type=Path, default=Path("docs/_stats/hold_stats.csv"))
    a = ap.parse_args(argv)

    by_grade = {g: ([], [], []) for g, *_ in GRADES}
    for r in read_manifest(a.manifest):
        path = a.cache / "tokens" / f"{r['key']}.npz"
        if r["split"] != a.split or r.get("drop", "") != "" or not r["sr"] or not path.exists():
            continue
        sr = float(r["sr"])
        grade = next(g for g, lo, hi in GRADES if lo <= sr < hi)
        share, lens, gaps = chart_numbers(np.load(path)["tokens"])
        for bucket, value in zip(by_grade[grade], (share, lens, gaps), strict=True):
            bucket.append(value)
    rows = [summarize(g, *by_grade[g]) for g, *_ in GRADES if by_grade[g][0]]
    every = [[v for g, *_ in GRADES for v in by_grade[g][i]] for i in range(3)]
    rows.append(summarize("all", *every))
    if not rows[-1]["charts"]:
        print(f"no {a.split} charts with tokens", file=sys.stderr)
        return 2
    print(" ".join(f"{f:>12s}" for f in FIELDS))
    for row in rows:
        print(" ".join(f"{row[f]!s:>12s}" for f in FIELDS))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
