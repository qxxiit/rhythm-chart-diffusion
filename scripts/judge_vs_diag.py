"""What does the judge see that the diagnostics do not? (docs/EVALUATION.md, judge checks)

    python scripts/judge_vs_diag.py --judge outputs/judge/F --run outputs/full-v4/eval_<F>
    python scripts/judge_vs_diag.py --judge outputs/judge/F --run <a scored run> --chart <label>

Per song, how much more AI the judge finds the run's chart than the human one: the mean
logit of its windows minus the human chart's (judge.py windows.csv; held-out folds only).
Against it, each diagnostic of evaluate.py's per_song.csv as the run's gap to the human chart
(column minus human_column; density_ratio minus 1; sr_error as it is): the Spearman
correlation over songs of the judge's difference with the gap and with its size. Then a ridge
fit of the judge's difference on the 8 gaps that correlate most (standardized), with R^2 in
sample and over 5 folds of songs: what the diagnostics explain of what the judge sees; the rest
is a tell no diagnostic measures yet (the next diagnostic to look for). Aggregate only: no
per-song output (the playtest songs not played yet stay blind). Writes <judge>/vs_diag.json
and vs_diag.md, or --out.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

SKIP = {"key", "path", "grade", "genre", "mapper_known", "stat_key", "seconds", "passes",
        "bpm", "set_charts"}


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--judge", type=Path, required=True, help="judge.py output directory")
    ap.add_argument("--run", type=Path, required=True,
                    help="the evaluate.py run of the chart being related (per_song.csv)")
    ap.add_argument("--chart", default="ai", help="its label in windows.csv (ai, or +<tag>)")
    ap.add_argument("--top", type=int, default=8, help="gaps in the ridge fit")
    ap.add_argument("--ridge", type=float, default=1.0)
    ap.add_argument("--exclude", type=Path, default=None,
                    help="a file of chart keys to leave out, one a line")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args(argv)


def ranks(v: np.ndarray) -> np.ndarray:
    """Ranks with ties at their mean."""
    order = np.argsort(v, kind="stable")
    r = np.empty(len(v))
    r[order] = np.arange(len(v))
    _, inv, counts = np.unique(v, return_inverse=True, return_counts=True)
    if (counts > 1).any():
        r = (np.bincount(inv, weights=r) / counts)[inv]
    return r


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 5 or np.ptp(a[ok]) == 0 or np.ptp(b[ok]) == 0:
        return float("nan")
    x, y = ranks(a[ok]), ranks(b[ok])
    x, y = x - x.mean(), y - y.mean()
    return float((x * y).sum() / math.sqrt((x * x).sum() * (y * y).sum()))


def ridge_r2(x: np.ndarray, y: np.ndarray, lam: float, folds: int,
             seed: int) -> tuple[float, float, np.ndarray]:
    """(R^2 in sample, R^2 over folds, coefficients) of y on standardized x."""
    mu, sd = x.mean(axis=0), x.std(axis=0)
    sd[sd == 0] = 1.0
    z = (x - mu) / sd

    def fit(zt, yt):
        a = zt.T @ zt + lam * np.eye(zt.shape[1])
        beta = np.linalg.solve(a, zt.T @ (yt - yt.mean()))
        return beta, yt.mean()

    beta, c = fit(z, y)
    pred = z @ beta + c
    r2_in = 1 - float(((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())
    idx = np.random.default_rng(seed).permutation(len(y))
    out = np.empty(len(y))
    for f in range(folds):
        test = idx[f::folds]
        train = np.setdiff1d(idx, test)
        b, cc = fit(z[train], y[train])
        out[test] = z[test] @ b + cc
    r2_cv = 1 - float(((y - out) ** 2).sum() / ((y - y.mean()) ** 2).sum())
    return r2_in, r2_cv, beta


def main(argv: list[str] | None = None) -> int:
    a = parse_args(argv)
    drop = set()
    if a.exclude is not None:
        drop = {line.strip() for line in a.exclude.read_text().splitlines() if line.strip()}
    logits: dict[tuple[str, str], list[float]] = {}
    with open(a.judge / "windows.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            logits.setdefault((r["key"], r["chart"]), []).append(float(r["logit"]))
    with open(a.run / "per_song.csv", newline="", encoding="utf-8") as f:
        per_song = {r["key"]: r for r in csv.DictReader(f) if r["key"] not in drop}
    keys = [k for k in per_song if (k, a.chart) in logits and (k, "human") in logits]
    if len(keys) < 10:
        print(f"{len(keys)} songs with judge windows for {a.chart!r} and a per_song row; "
              "need 10 or more", file=sys.stderr)
        return 2
    diff = np.array([np.mean(logits[(k, a.chart)]) - np.mean(logits[(k, "human")])
                     for k in keys])

    def num(k: str, col: str) -> float:
        try:
            v = float(per_song[k].get(col, "nan"))
        except ValueError:
            return float("nan")
        return v if math.isfinite(v) else float("nan")

    columns = [c for c in per_song[keys[0]] if c not in SKIP and not c.startswith("human_")]
    gaps: dict[str, np.ndarray] = {}
    for c in columns:
        if f"human_{c}" in per_song[keys[0]]:
            g = np.array([num(k, c) - num(k, f"human_{c}") for k in keys])
        elif c == "density_ratio":
            g = np.array([num(k, c) - 1 for k in keys])
        elif c in ("sr_error", "sr_error_vs_local", "new_onsets"):
            g = np.array([num(k, c) for k in keys])
        else:
            continue
        if np.isfinite(g).sum() >= 10 and np.nanstd(g) > 0:
            gaps[c] = g
    table = {c: {"rho_gap": spearman(diff, g), "rho_size": spearman(diff, np.abs(g)),
                 "mean_gap": float(np.nanmean(g))} for c, g in gaps.items()}
    ranked = sorted(table, key=lambda c: -max(abs(table[c]["rho_gap"] if math.isfinite(
        table[c]["rho_gap"]) else 0), abs(table[c]["rho_size"] if math.isfinite(
        table[c]["rho_size"]) else 0)))
    top = ranked[:a.top]
    x = np.column_stack([np.where(np.isfinite(gaps[c]), gaps[c], np.nanmean(gaps[c]))
                         for c in top]) if top else np.zeros((len(keys), 0))
    fit = {}
    if top:
        r2_in, r2_cv, beta = ridge_r2(x, diff, a.ridge, 5, a.seed)
        fit = {"columns": top, "r2_in_sample": r2_in, "r2_5fold": r2_cv,
               "unexplained": 1 - max(r2_cv, 0.0),
               "coef_per_sd": {c: float(b) for c, b in zip(top, beta, strict=True)}}
    summary = {"songs": len(keys), "chart": a.chart, "judge": str(a.judge), "run": str(a.run),
               "judge_minus_human": {"mean": float(diff.mean()),
                                     "share_above_0": float((diff > 0).mean())},
               "fit": fit, "gaps": {c: table[c] for c in ranked}}
    out = a.out or a.judge
    out.mkdir(parents=True, exist_ok=True)
    (out / "vs_diag.json").write_text(json.dumps(summary, indent=2))
    lines = [f"# Judge against the diagnostics ({a.chart}, {len(keys)} songs)", "",
             f"Judge logit, run minus human: mean {diff.mean():+.3f}, above 0 in "
             f"{(diff > 0).mean():.0%} of the songs.", ""]
    if fit:
        lines += [f"Ridge on {len(top)} gaps: R^2 {fit['r2_in_sample']:.2f} in sample, "
                  f"{fit['r2_5fold']:.2f} over 5 folds; unexplained {fit['unexplained']:.0%}.",
                  ""]
    lines += ["| gap | rho with the gap | rho with its size | mean gap |", "|---|---|---|---|"]
    for c in ranked[:20]:
        t = table[c]
        lines.append(f"| {c} | {t['rho_gap']:+.2f} | {t['rho_size']:+.2f} | "
                     f"{t['mean_gap']:+.4g} |")
    (out / "vs_diag.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:6]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
