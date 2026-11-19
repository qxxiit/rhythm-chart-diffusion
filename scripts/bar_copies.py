"""Bars played again: of the bar pairs 1, 2 and 4 bars apart with the same onset rows (3+), the
share whose onsets are the earlier bar's mirrored (lane k -> 3 - k), the same, or 75%+ mirrored,
pooled over the songs; for each run and the human charts of its songs (EXPERIMENTS 2026-11-19
(1), 2026-11-20 (2)). Aggregate only.

    python scripts/bar_copies.py RUN [RUN ...] [--cache data/cache] [--out results.md]

RUN: evaluate.py directories (charts.npz); the human charts from the token cache, of the first
run's songs, and each run on those songs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.tokenizer import BAR, HOLD_START, K, TAP  # noqa: E402

LAGS = (1, 2, 4)


def counts(tokens: np.ndarray, acc: dict | None = None) -> dict:
    """Adds one chart's pairs to acc: {lag: [pairs, mirrored, same, near mirrored]}."""
    acc = {g: [0, 0, 0, 0] for g in LAGS} if acc is None else acc
    on = np.isin(np.asarray(tokens).reshape(-1, K), (TAP, HOLD_START))
    rows = on.any(axis=1)
    n_bars = len(on) // BAR
    for lag in LAGS:
        for b in range(lag, n_bars):
            ra, rb = rows[(b - lag) * BAR:(b - lag + 1) * BAR], rows[b * BAR:(b + 1) * BAR]
            if rb.sum() < 3 or not np.array_equal(ra, rb):
                continue
            src, dst = on[(b - lag) * BAR:(b - lag + 1) * BAR], on[b * BAR:(b + 1) * BAR]
            c = acc[lag]
            c[0] += 1
            c[1] += int(np.array_equal(src[:, ::-1], dst))
            c[2] += int(np.array_equal(src, dst))
            c[3] += int((src[:, ::-1] & dst).sum() / max(1, dst.sum()) >= 0.75)
    return acc


def shares(acc: dict) -> dict:
    """{"mirrored" | "same" | "near": [share at each lag]}, {"pairs": [...]}."""
    out = {"pairs": [acc[g][0] for g in LAGS]}
    for i, name in ((1, "mirrored"), (2, "same"), (3, "near")):
        out[name] = [acc[g][i] / acc[g][0] if acc[g][0] else float("nan") for g in LAGS]
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--cache", type=Path, default=Path("data/cache"))
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    for r in a.runs:
        if not (r / "charts.npz").exists():
            print(f"no {r / 'charts.npz'}", file=sys.stderr)
            return 2
    with np.load(a.runs[0] / "charts.npz") as z:
        keys = list(z.keys())
    n_cells, human = {}, None
    for k in keys:
        with np.load(a.cache / "tokens" / f"{k}.npz") as h:
            n_cells[k] = int(h["n_cells"])
            human = counts(h["tokens"].reshape(-1, K)[:n_cells[k]], human)
    table = [("human charts", shares(human))]
    for r in a.runs:
        acc = None
        with np.load(r / "charts.npz") as z:
            for k in keys:
                if k in z:
                    acc = counts(z[k].reshape(-1, K)[:n_cells[k]], acc)
        table.append((r.name, shares(acc)))
    pct = lambda v: " / ".join(f"{100 * x:.1f}" for x in v)       # noqa: E731
    lines = [f"Bar pairs 1 / 2 / 4 bars apart with the same onset rows (3+), {len(keys)} songs, "
             "onsets only, pooled (%).", "",
             "| run | mirrored | same | 75%+ mirrored | pairs |", "|---|---|---|---|---|"]
    lines += [f"| {name} | {pct(s['mirrored'])} | {pct(s['same'])} | {pct(s['near'])} | "
              + " / ".join(map(str, s["pairs"])) + " |" for name, s in table]
    text = "\n".join(lines) + "\n"
    print(text, end="")
    if a.out is not None:
        a.out.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
