"""The diagnostics as a judge: can a run's charts be told from the human ones by the per-song
diagnostics alone? (docs/EVALUATION.md; EXPERIMENTS 2026-11-07)

    python scripts/diag_judge.py RUN [RUN ...] [--out outputs/run_1105/diag_judge.md]

RUN: evaluate.py directories (per_song.csv with the columns X and human_X). Each song gives
two rows, the diagnostics of the run's chart and of the human chart: every X with a human_X
that all the runs given have (the others are named and left out, so the runs of one call are
judged on the same columns). Two classifiers tell the rows apart, 5 folds by song (a song's two
charts always in the same fold), 10 times over: logistic regression (median-imputed,
standardised, C 0.1) and boosted trees (sklearn's HistGradientBoostingClassifier, which takes
the gaps as they are). Numbers: auc over the charts and paired (the run's chart above the
human chart of the same song: the playtest's question asked of whole songs), by grade; and
the same with the two charts swapped in a random half of the songs, which must stay near 0.5
(a check of the script, not of the run).

Per metric: the run's mean and the human charts', effect (median of run minus human over the
human charts' sd, corrupt_test.py's unit), the share of songs where the run's chart is lower,
the run's spread over the human charts' within grade (sd ratio: under 1, too alike from song
to song), the share inside the human band of the grade (results_table.py's), the metric's
own auc; the metrics constant in the run while the human charts vary (a pass that takes a
thing out altogether); the trees' permutation importance (auc lost on the held-out songs when
the metric is shuffled there).

The judge (scripts/judge.py) takes a GPU's 40 minutes and sees 8 bars at a time; this takes
seconds once the charts are scored, and names what gives a run away, from the diagnostics
only. On F (240 val songs, 2026-11-07): logistic 0.87, trees 0.93 (the judge, songs: 0.97),
no single metric above 0.66, every band share 0.72 or more.

Writes <run>/diag_judge.json and <run>/diag_judge.md for each run, and with --out one table
of the runs (and .json beside). Aggregate only: no per-song rows (the playtest songs stay
blind).
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

FOLDS, REPEATS = 5, 10
LOGISTIC_C = 0.1
SHUFFLES = 3                     # permutation importance: shuffles per metric and fold
MIN_GRADE_SONGS = 10             # results_table.py's: a smaller grade takes the pooled band
TOP = 12
GRADES = ("Easy", "Normal", "Hard", "Insane", "Expert", "Expert+")


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", type=Path, nargs="+")
    ap.add_argument("--repeats", type=int, default=REPEATS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None,
                    help="markdown table of the runs (and .json beside)")
    return ap.parse_args(argv)


def paired_metrics(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if f"human_{c}" in df.columns]


def logistic():
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(C=LOGISTIC_C, max_iter=5000))


def trees():
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=8,
                                          min_samples_leaf=10, random_state=0)


MODELS = {"logistic": logistic, "trees": trees}


def auc(y: np.ndarray, p: np.ndarray) -> float:
    """Rank auc (ties count half); nan without both classes."""
    pos, neg = p[y == 1], p[y == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    order = np.argsort(np.r_[pos, neg], kind="mergesort")
    ranks = np.empty(len(order))
    ranks[order] = np.arange(1, len(order) + 1)
    allp = np.r_[pos, neg]
    for v in np.unique(allp):                        # mean ranks for ties
        same = allp == v
        ranks[same] = ranks[same].mean()
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2)
                 / (len(pos) * len(neg)))


def paired(p: np.ndarray, n: int) -> float:
    """Share of songs whose run chart (rows :n) scores above its human chart (rows n:)."""
    return float(np.mean(p[:n] > p[n:]) + 0.5 * np.mean(p[:n] == p[n:]))


def song_folds(n: int, rng: np.random.Generator) -> np.ndarray:
    f = np.empty(n, dtype=np.int64)
    f[rng.permutation(n)] = np.arange(n) % FOLDS
    return f


def held_out(make, X: np.ndarray, y: np.ndarray, folds: np.ndarray) -> np.ndarray:
    """Each row's score from a model that never saw its song."""
    both = np.r_[folds, folds]
    p = np.zeros(len(y))
    for k in range(FOLDS):
        test = both == k
        if not test.any():
            continue
        p[test] = make().fit(X[~test], y[~test]).predict_proba(X[test])[:, 1]
    return p


def importance(X: np.ndarray, y: np.ndarray, folds: np.ndarray,
               rng: np.random.Generator) -> np.ndarray:
    """The trees' held-out auc lost when one metric is shuffled among the held-out rows."""
    both = np.r_[folds, folds]
    lost = np.zeros(X.shape[1])
    for k in range(FOLDS):
        test = both == k
        model = trees().fit(X[~test], y[~test])
        Xt, yt = X[test], y[test]
        base = auc(yt, model.predict_proba(Xt)[:, 1])
        for j in range(X.shape[1]):
            for _ in range(SHUFFLES):
                Xs = Xt.copy()
                Xs[:, j] = Xs[rng.permutation(len(Xs)), j]
                lost[j] += (base - auc(yt, model.predict_proba(Xs)[:, 1])) / (SHUFFLES * FOLDS)
    return lost


def band_share(run_v: np.ndarray, human_v: np.ndarray, grades: np.ndarray) -> float:
    """Share of the run's values inside the 10th-90th percentile of the human charts of the
    grade (pooled for grades under MIN_GRADE_SONGS songs), as results_table.py."""
    ok = np.isfinite(human_v)
    pooled = np.percentile(human_v[ok], [10, 90]) if ok.any() else None
    inside = []
    for g in np.unique(grades):
        sel = grades == g
        h = human_v[sel & ok]
        lo_hi = np.percentile(h, [10, 90]) if len(h) >= MIN_GRADE_SONGS else pooled
        v = run_v[sel]
        v = v[np.isfinite(v)]
        if lo_hi is not None and len(v):
            inside += list((v >= lo_hi[0]) & (v <= lo_hi[1]))
    return float(np.mean(inside)) if inside else float("nan")


def sd_ratio(run_v: np.ndarray, human_v: np.ndarray, grades: np.ndarray) -> float:
    """The run's spread over the human charts' within grade (pooled variances)."""
    num = den = 0.0
    for g in np.unique(grades):
        sel = grades == g
        r, h = run_v[sel], human_v[sel]
        r, h = r[np.isfinite(r)], h[np.isfinite(h)]
        if len(r) > 4 and len(h) > 4:
            num += len(r) * r.var()
            den += len(h) * h.var()
    return float(math.sqrt(num / den)) if den > 0 else float("nan")


def per_metric(run_m: np.ndarray, human_m: np.ndarray, grades: np.ndarray,
               metrics: list[str]) -> dict:
    out = {}
    for j, m in enumerate(metrics):
        r, h = run_m[:, j], human_m[:, j]
        ok = np.isfinite(r) & np.isfinite(h)
        d = r[ok] - h[ok]
        sd = float(np.nanstd(h)) if np.isfinite(h).sum() > 1 else float("nan")
        both = np.r_[r[ok], h[ok]]
        out[m] = {
            "songs": int(ok.sum()),
            "run": float(np.nanmean(r)) if np.isfinite(r).any() else float("nan"),
            "human": float(np.nanmean(h)) if np.isfinite(h).any() else float("nan"),
            "effect": float(np.median(d) / sd) if len(d) and sd > 0 else float("nan"),
            "run_lower": float(np.mean(d < 0)) if len(d) else float("nan"),
            "sd_ratio": sd_ratio(r, h, grades),
            "band": band_share(r, h, grades),
            "auc": auc(np.r_[np.ones(ok.sum()), np.zeros(ok.sum())], both)
            if ok.sum() > 1 else float("nan"),
        }
    return out


def constant(run_m: np.ndarray, human_m: np.ndarray, metrics: list[str]) -> dict:
    """Metrics with one value in every run chart where the human charts vary: the value and
    the share of songs whose human chart has another."""
    out = {}
    for j, m in enumerate(metrics):
        r, h = run_m[:, j], human_m[:, j]
        r_ok, h_ok = r[np.isfinite(r)], h[np.isfinite(h)]
        if len(r_ok) > 1 and np.ptp(r_ok) == 0 and len(h_ok) > 1 and np.ptp(h_ok) > 0:
            out[m] = {"value": float(r_ok[0]), "human_other": float(np.mean(h_ok != r_ok[0]))}
    return out


def judge_run(df: pd.DataFrame, metrics: list[str], repeats: int, seed: int) -> dict:
    n = len(df)
    run_m = df[metrics].to_numpy(np.float64)
    human_m = df[[f"human_{m}" for m in metrics]].to_numpy(np.float64)
    grades = df["grade"].astype(str).to_numpy() if "grade" in df else np.array([""] * n)
    X = np.r_[run_m, human_m]
    y = np.r_[np.ones(n), np.zeros(n)]
    rng = np.random.default_rng(seed)
    folds = [song_folds(n, rng) for _ in range(repeats)]
    out = {"songs": n, "metrics": len(metrics)}
    for name, make in MODELS.items():
        ps = [held_out(make, X, y, f) for f in folds]
        aucs = [auc(y, p) for p in ps]
        by_grade = {}
        for g in sorted(set(grades), key=lambda g: (GRADES.index(g) if g in GRADES else 99, g)):
            sel = grades == g
            yy = np.r_[np.ones(sel.sum()), np.zeros(sel.sum())]
            parts = [np.r_[p[:n][sel], p[n:][sel]] for p in ps]
            by_grade[str(g)] = {"songs": int(sel.sum()),
                                "auc": float(np.mean([auc(yy, pp) for pp in parts])),
                                "paired": float(np.mean([paired(pp, int(sel.sum()))
                                                         for pp in parts]))}
        swap = rng.random(n) < 0.5                   # the check: labels without meaning
        Xs = np.r_[np.where(swap[:, None], human_m, run_m), np.where(swap[:, None], run_m, human_m)]
        pn = held_out(make, Xs, y, folds[0])
        out[name] = {"auc": float(np.mean(aucs)), "sd": float(np.std(aucs)),
                     "paired": float(np.mean([paired(p, n) for p in ps])),
                     "null_auc": auc(y, pn), "null_paired": paired(pn, n),
                     "by_grade": by_grade}
    out["importance"] = dict(sorted(zip(metrics, importance(X, y, folds[0], rng).tolist(),
                                        strict=True), key=lambda kv: -kv[1]))
    out["per_metric"] = per_metric(run_m, human_m, grades, metrics)
    out["constant"] = constant(run_m, human_m, metrics)
    return out


def f3(v) -> str:
    return "" if v is None or (isinstance(v, float) and not math.isfinite(v)) else f"{v:.3f}"


def run_markdown(name: str, r: dict, dropped: list[str]) -> str:
    lines = [f"# The diagnostics as a judge: {name}", "",
             f"{r['songs']} songs, {r['metrics']} metrics (5 folds by song x repeats). "
             "Aggregate only.", ""]
    if dropped:
        lines += [f"Left out (not in every run of the call): {', '.join(dropped)}", ""]
    lines += ["| classifier | auc (sd) | paired | check: labels swapped in half (auc / paired) |",
              "|---|---|---|---|"]
    for m in MODELS:
        s = r[m]
        lines.append(f"| {m} | {f3(s['auc'])} ({f3(s['sd'])}) | {f3(s['paired'])} | "
                     f"{f3(s['null_auc'])} / {f3(s['null_paired'])} |")
    lines += ["", "By grade (mean over the repeats): auc / paired / songs", "",
              "| grade | logistic | trees |", "|---|---|---|"]
    for g in r["trees"]["by_grade"]:
        lo, tr = r["logistic"]["by_grade"][g], r["trees"]["by_grade"][g]
        lines.append(f"| {g} | {f3(lo['auc'])} / {f3(lo['paired'])} | "
                     f"{f3(tr['auc'])} / {f3(tr['paired'])} / {tr['songs']} |")
    pm = r["per_metric"]
    lines += ["", f"What the trees lean on (held-out auc lost when shuffled), top {TOP}:", "",
              "| metric | auc lost | run | human | effect | run lower | sd ratio | band | "
              "own auc |", "|---|---|---|---|---|---|---|---|---|"]
    for m, v in list(r["importance"].items())[:TOP]:
        s = pm[m]
        lines.append(f"| {m} | {f3(v)} | {f3(s['run'])} | {f3(s['human'])} | "
                     f"{f3(s['effect'])} | {f3(s['run_lower'])} | {f3(s['sd_ratio'])} | "
                     f"{f3(s['band'])} | {f3(s['auc'])} |")
    big = sorted((m for m in pm if math.isfinite(pm[m]["effect"])),
                 key=lambda m: -abs(pm[m]["effect"]))[:TOP]
    lines += ["", f"Largest effects (median run - human over the human sd), top {TOP}:", "",
              "| metric | effect | run lower | run | human | band |", "|---|---|---|---|---|---|"]
    for m in big:
        s = pm[m]
        lines.append(f"| {m} | {f3(s['effect'])} | {f3(s['run_lower'])} | {f3(s['run'])} | "
                     f"{f3(s['human'])} | {f3(s['band'])} |")
    bands = [s["band"] for s in pm.values() if math.isfinite(s["band"])]
    ratios = [s["sd_ratio"] for s in pm.values() if math.isfinite(s["sd_ratio"])]
    lines += ["", f"Band shares: median {f3(float(np.median(bands)) if bands else float('nan'))}"
              f" (the human charts 0.80), lowest {f3(min(bands) if bands else float('nan'))}; "
              f"sd ratio: median {f3(float(np.median(ratios)) if ratios else float('nan'))}, "
              f"under 1 for {sum(x < 1 for x in ratios)} of {len(ratios)}."]
    if r["constant"]:
        lines += ["", "Constant in the run while the human charts vary: " + "; ".join(
            f"{m} = {f3(c['value'])} (human charts differ in {c['human_other']:.0%} of songs)"
            for m, c in r["constant"].items())]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    a = parse_args(argv)
    warnings.filterwarnings("ignore", category=RuntimeWarning)   # nanmean of empty columns
    frames = {run: pd.read_csv(run / "per_song.csv") for run in a.runs}
    common = [m for m in paired_metrics(frames[a.runs[0]])
              if all(m in paired_metrics(df) for df in frames.values())]
    if not common:
        print("no X / human_X columns common to the runs")
        return 2
    rows = []
    for run, df in frames.items():
        dropped = [m for m in paired_metrics(df) if m not in common]
        r = judge_run(df, common, a.repeats, a.seed)
        r["run"], r["dropped"] = str(run), dropped
        (run / "diag_judge.json").write_text(json.dumps(r, indent=1))
        (run / "diag_judge.md").write_text(run_markdown(run.name, r, dropped))
        top = list(r["importance"])[:3]
        rows.append({"run": run.name, "songs": r["songs"], "metrics": r["metrics"],
                     "logistic auc": r["logistic"]["auc"],
                     "logistic paired": r["logistic"]["paired"],
                     "trees auc": r["trees"]["auc"], "trees paired": r["trees"]["paired"],
                     "check auc": max(abs(r[m]["null_auc"] - 0.5) for m in MODELS) + 0.5,
                     "leans on": ", ".join(top)})
        print(f"{run.name}: logistic {r['logistic']['auc']:.3f} / {r['logistic']['paired']:.3f}"
              f", trees {r['trees']['auc']:.3f} / {r['trees']['paired']:.3f} "
              f"({r['songs']} songs, {r['metrics']} metrics; leans on {', '.join(top)})",
              flush=True)
    cols = list(rows[0])
    text = "\n".join(["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)] + [
        "| " + " | ".join(f3(row[c]) if isinstance(row[c], float) else str(row[c])
                          for c in cols) + " |" for row in rows]) + "\n"
    print(text)
    if a.out is not None:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(text)
        a.out.with_suffix(".json").write_text(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
