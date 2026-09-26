"""Human-chart baselines for the §4.11 metrics, from the token cache (train split).

    python scripts/human_baselines.py                # docs/_stats/pattern_baseline.csv
                                                     # docs/_stats/chart_ssm_lag.csv
    python scripts/human_baselines.py --limit 3000 --far-k 8

Pattern clarity by SR grade: the level the generated charts should match
(§4.11-2b: "같은 등급 사람 채보와 비슷한 수준"). Read it before freezing the
tagger rules; never tune the rules on model output.

Chart self-similarity by bar distance, averaged over charts: where the
adjacency effect fades is k for rho_far (§4.11: 사람 채보에서 인접 효과가
사라지는 거리로 미리 고정). The suggestion printed is only a starting point.

Charts with mel in the cache also get human rho (all / in / cross / far): the
upper reference line of §4.11. Uses only the train split, so nothing here
peeks at validation or test.
"""

from __future__ import annotations

import argparse
import os
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd

from src.data.cache import read_manifest
from src.evaluation.patterns import summarize
from src.evaluation.structure import lag_profile, n_whole_bars, ssm_chart, structure_scores

GRADES = [("Easy", 0.0, 2.0), ("Normal", 2.0, 2.7), ("Hard", 2.7, 4.0),
          ("Insane", 4.0, 5.3), ("Expert", 5.3, 6.5), ("Expert+", 6.5, np.inf)]
MAX_LAG = 32


def one(job: tuple) -> dict | None:
    row, cache, far_k = job
    path = cache / "tokens" / f"{row['key']}.npz"
    if not path.exists():
        return None
    z = np.load(path)
    n_cells = int(z["n_cells"])
    tokens = z["tokens"].reshape(-1, z["tokens"].shape[-1])[:n_cells]
    out = {"key": row["key"], "sr": float(row["sr"]), **summarize(tokens)}
    n_bars = n_whole_bars(n_cells)
    if n_bars >= 2:
        out["lag"] = lag_profile(ssm_chart(tokens, n_bars), MAX_LAG)
    mel_path = cache / "mel" / f"{row['key']}.npy"
    if mel_path.exists():
        out.update(structure_scores(tokens, np.load(mel_path), n_cells, far_k=far_k))
    return out


def grade_of(sr: float) -> str:
    return next(name for name, lo, hi in GRADES if lo <= sr < hi)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"))
    ap.add_argument("--cache", type=Path, default=Path("data/cache"))
    ap.add_argument("--split", default="train")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--far-k", type=int, default=8)
    ap.add_argument("--stats", type=Path, default=Path("docs/_stats"))
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    a = ap.parse_args(argv)

    rows = [r for r in read_manifest(a.manifest)
            if r["split"] == a.split and r["sr"] and r.get("drop", "") == ""]
    if a.limit:
        rows = rows[:a.limit]
    jobs = [(r, a.cache, a.far_k) for r in rows]
    if a.workers > 1:
        with Pool(a.workers) as pool:
            recs = [x for x in pool.imap(one, jobs, chunksize=16) if x]
    else:
        recs = [x for x in map(one, jobs) if x]
    if not recs:
        print("no token cache for this split: run scripts/preprocess_data.py", file=sys.stderr)
        return 2
    df = pd.DataFrame([{k: v for k, v in r.items() if k != "lag"} for r in recs])
    df["grade"] = df["sr"].map(grade_of)

    cols = ["coverage", "coverage_jack", "coverage_trill", "coverage_stairs",
            "coverage_jumptrill", "run_length", "breaks_per_100"]
    table = df.groupby("grade", sort=False)[cols].median().reindex([g for g, *_ in GRADES])
    table.insert(0, "charts", df.groupby("grade").size().reindex(table.index))
    a.stats.mkdir(parents=True, exist_ok=True)
    table.round(4).to_csv(a.stats / "pattern_baseline.csv")
    print(f"{len(df):,} {a.split} charts. Pattern clarity, median per grade:")
    print(table.round(3).to_string())

    lags = np.array([r["lag"] for r in recs if "lag" in r])
    mean = np.nanmean(lags, axis=0)
    count = np.sum(~np.isnan(lags), axis=0)
    tail = float(np.nanmean(mean[16:]))
    k_hint = next((d + 1 for d, v in enumerate(mean) if v <= tail + 0.02), MAX_LAG)
    lag_df = pd.DataFrame({"lag_bars": np.arange(1, MAX_LAG + 1), "mean_chart_similarity":
                           mean.round(4), "charts": count})
    lag_df.to_csv(a.stats / "chart_ssm_lag.csv", index=False)
    print("\nChart self-similarity by bar distance (mean over charts):")
    print("  " + "  ".join(f"{d}:{v:.3f}" for d, v in zip(range(1, MAX_LAG + 1), mean,
                                                           strict=True)))
    print(f"  tail mean (lags 17-32) {tail:.3f}; first lag within 0.02 of it: {k_hint}"
          " (a hint for k, not the decision)")

    if "rho_all" in df:
        have = df.dropna(subset=["rho_all"])
        print(f"\nHuman rho over {len(have)} charts with mel (far k = {a.far_k}): " + "  ".join(
            f"{c} {have[c].mean():.3f}" for c in ("rho_all", "rho_in", "rho_cross", "rho_far")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
