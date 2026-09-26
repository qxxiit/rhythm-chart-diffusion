"""Tempo against density inside each SR grade, from the token cache (W0-1).

    python scripts/tempo_density.py               # docs/_stats/tempo_density.csv
    python scripts/tempo_density.py --longest 10  # also the charts with the most cells

Why it matters: s (SR) and b (tempo) are separate conditions (§4.10). If, inside
one grade, faster songs get fewer notes per beat (difficulty holds notes per
second), SR alone does not say how many cells to fill and b carries real
information. The slope of log(notes per beat) on log(BPM) inside a grade reads
directly: -1 means notes per second are fixed, 0 means notes per beat are.

Per chart, over the span from the first to the last onset:
    nps  onsets / seconds,  npb  onsets / beats,  bpm  beats / minutes
so nps = npb * bpm / 60 exactly. Beats are counted in cells (12 per beat),
which stays exact across tempo changes.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd

from src.data.beat_grid import from_timing_points
from src.data.cache import read_manifest
from src.data.tokenizer import HOLD_START, TAP, D, K

GRADES = [("Easy", 0.0, 2.0), ("Normal", 2.0, 2.7), ("Hard", 2.7, 4.0),
          ("Insane", 4.0, 5.3), ("Expert", 5.3, 6.5), ("Expert+", 6.5, np.inf)]


def chart_row(row: dict, cache: Path) -> dict | None:
    path = cache / "tokens" / f"{row['key']}.npz"
    if not path.exists():
        return None
    z = np.load(path)
    onset = np.isin(z["tokens"].reshape(-1, K), (TAP, HOLD_START))
    rows = np.flatnonzero(onset.any(axis=1))
    counts = int(onset.sum())
    out = {"key": row["key"], "path": row["path"], "sr": float(row["sr"]) if row["sr"] else np.nan,
           "n_cells": int(z["n_cells"]), "onsets": counts}
    if len(rows) < 2:
        return out
    grid = from_timing_points([tuple(tp) for tp in z["timing_points"]])
    offset = int(z["cell_offset"])
    t0, t1 = grid.time_from_cell(np.array([rows[0], rows[-1]]) - offset, D)
    seconds, beats = (t1 - t0) / 1000.0, (rows[-1] - rows[0]) / D
    if seconds <= 0 or beats <= 0:
        return out
    out.update(seconds=seconds, nps=counts / seconds, npb=counts / beats,
               bpm=60.0 * beats / seconds)
    return out


def grade_table(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for name, lo, hi in GRADES:
        g = df[(df["sr"] >= lo) & (df["sr"] < hi)].dropna(subset=["bpm", "npb", "nps"])
        rec = {"grade": name, "charts": len(g)}
        if len(g) >= 3:
            x, y = np.log(g["bpm"]), np.log(g["npb"])
            rec.update(
                median_bpm=round(float(g["bpm"].median()), 1),
                median_nps=round(float(g["nps"].median()), 3),
                median_npb=round(float(g["npb"].median()), 3),
                spearman_bpm_nps=round(float(g["bpm"].rank().corr(g["nps"].rank())), 3),
                spearman_bpm_npb=round(float(g["bpm"].rank().corr(g["npb"].rank())), 3),
                slope_log_npb_on_log_bpm=round(float(np.polyfit(x, y, 1)[0]), 3),
            )
        out.append(rec)
    return pd.DataFrame(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"))
    ap.add_argument("--cache", type=Path, default=Path("data/cache"))
    ap.add_argument("--out", type=Path, default=Path("docs/_stats/tempo_density.csv"))
    ap.add_argument("--longest", type=int, default=0, help="list the N charts with the most cells")
    a = ap.parse_args(argv)

    recs = [r for r in (chart_row(row, a.cache) for row in read_manifest(a.manifest)) if r]
    if not recs:
        print("no token cache found: run scripts/preprocess_data.py first", file=sys.stderr)
        return 2
    df = pd.DataFrame(recs)
    table = grade_table(df)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(a.out, index=False, quoting=csv.QUOTE_MINIMAL)
    print(f"{len(df):,} charts with a token cache, {int(df['sr'].notna().sum()):,} with SR")
    print(table.to_string(index=False))
    if a.longest:
        cols = [c for c in ("n_cells", "bpm", "seconds", "onsets", "path") if c in df]
        print(f"\n{a.longest} charts with the most cells:")
        print(df.nlargest(a.longest, "n_cells")[cols].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
