"""Compare evaluate.py runs song by song, paired over the songs both scored.

    python scripts/compare_runs.py outputs/full-v1/eval_val_songs_continue_random_T128 \
        outputs/full-v1/eval_val_songs_continue_random_T128_ref2t0.5
    python scripts/compare_runs.py <reference dir> <dir> <dir> ... --by-grade

The first directory is the reference. For every other run, on the songs both have
(per_song.csv rows matched by key, so the same charts): each metric's mean in the
run and in the reference, the mean per-song difference with a 95% bootstrap
interval over songs, and the share of songs where it went up or down; the human
charts' level where there is one (density 1). sr_bias is sr_gen - sr_target.
Pattern coverage is also given over its chance level (ratio of means), with an
interval for the run's ratio over the reference's. Cost (passes, seconds) where
both runs recorded it. --by-grade adds the mean differences of the main numbers
per SR grade, --by-genre per genre of the song (evaluate.py's genre column), with
the human level of each in brackets.
Read the intervals over songs, not the means alone: one sampled chart per song
moves by chance too (--sample-seed replicates show how much).
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np

# (column, human level: a column of the same file, a constant, or None)
METRICS = [
    ("f1@50", None), ("f1@50_any_lane", None), ("density_ratio", 1.0), ("sr_error", None),
    ("sr_bias", None), ("rho_in", "human_rho_in"), ("coverage", "human_coverage"),
    ("coverage_chance", "human_coverage_chance"), ("coverage_p1", "human_coverage_p1"),
    ("coverage_p2", "human_coverage_p2"), ("coverage_p3plus", "human_coverage_p3plus"),
    ("run_length", "human_run_length"),
    ("breaks_per_100", "human_breaks_per_100"), ("motion_pred", "human_motion_pred"),
    ("move_stair", "human_move_stair"), ("move_trill", "human_move_trill"),
    ("move_jack", "human_move_jack"),
    ("move_leap", "human_move_leap"), ("chord_share", "human_chord_share"),
    ("lone_chord", "human_lone_chord"), ("bar_rhythm_repeat", "human_bar_rhythm_repeat"),
    ("bar_lane_repeat", "human_bar_lane_repeat"),
    ("hold_share", "human_hold_share"), ("passes", None), ("seconds", None),
]
GRADE_METRICS = ("f1@50", "sr_bias", "coverage", "motion_pred")
GRADES = ("Easy", "Normal", "Hard", "Insane", "Expert", "Expert+")


def read_run(path: Path) -> dict[str, dict]:
    """key -> row with numbers as floats (blank or missing -> nan); sr_bias added."""
    csv_path = path / "per_song.csv" if path.is_dir() else path
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out = {}
    for r in rows:
        row = {}
        for k, v in r.items():
            try:
                row[k] = float(v)
            except (TypeError, ValueError):
                row[k] = v
        if isinstance(row.get("sr_gen"), float) and isinstance(row.get("sr_target"), float):
            row["sr_bias"] = row["sr_gen"] - row["sr_target"]
        out[r["key"]] = row
    return out


def _num(row: dict, key: str) -> float:
    v = row.get(key, float("nan"))
    return v if isinstance(v, float) else float("nan")


def bootstrap(values: np.ndarray, stat, n: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95% percentile interval of stat(values[idx]) over song resamples."""
    if len(values) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(len(values), size=(n, len(values)))
    stats = np.array([stat(values[i]) for i in idx])
    stats = stats[np.isfinite(stats)]
    if len(stats) < n // 2:
        return float("nan"), float("nan")
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(lo), float(hi)


def paired(run: dict, ref: dict, keys: list[str], metric: str) -> dict | None:
    a = np.array([_num(run[k], metric) for k in keys])
    b = np.array([_num(ref[k], metric) for k in keys])
    ok = np.isfinite(a) & np.isfinite(b)
    if not ok.any():
        return None
    d = a[ok] - b[ok]
    lo, hi = bootstrap(d, np.mean)
    return {"run": float(a[ok].mean()), "ref": float(b[ok].mean()), "diff": float(d.mean()),
            "lo": lo, "hi": hi, "up": float((d > 1e-12).mean()), "down": float((d < -1e-12).mean()),
            "n": int(ok.sum())}


def coverage_ratio(run: dict, ref: dict, keys: list[str]) -> dict | None:
    cols = ("coverage", "coverage_chance")
    m = np.array([[_num(r[k], c) for r in (run, ref) for c in cols] for k in keys])
    m = m[np.isfinite(m).all(axis=1)]
    if len(m) < 2 or (m[:, [0, 1, 2, 3]].sum(axis=0) <= 0).any():
        return None

    def ratio(x: np.ndarray) -> float:
        den = x[:, 1].mean() * x[:, 2].mean()
        return float(x[:, 0].mean() * x[:, 3].mean() / den) if den > 0 else float("nan")

    lo, hi = bootstrap(m, ratio)
    return {"run": m[:, 0].mean() / m[:, 1].mean(), "ref": m[:, 2].mean() / m[:, 3].mean(),
            "times": ratio(m), "lo": lo, "hi": hi}


def human_level(run: dict, keys: list[str], human) -> float | None:
    if human is None:
        return None
    if isinstance(human, float):
        return human
    v = np.array([_num(run[k], human) for k in keys])
    return float(np.nanmean(v)) if np.isfinite(v).any() else None


def fmt(x: float, signed: bool = False) -> str:
    if x is None or not math.isfinite(x):
        return "-"
    big = abs(x) >= 100
    return f"{x:+.1f}" if signed and big else f"{x:.1f}" if big else \
        f"{x:+.4f}" if signed else f"{x:.4f}"


def compare(run: dict, ref: dict, by_grade: bool, by_genre: bool = False) -> list[str]:
    keys = [k for k in ref if k in run]
    lines = [f"  {'metric':16s} {'run':>8s} {'ref':>8s}   {'run - ref [95% CI]':30s} "
             f"{'up':>5s} {'down':>5s} {'human':>8s}"]
    for metric, human in METRICS:
        p = paired(run, ref, keys, metric)
        if p is None:
            continue
        h = human_level(run, keys, human)
        ci = f"{fmt(p['diff'], True)} [{fmt(p['lo'], True)}, {fmt(p['hi'], True)}]"
        lines.append(f"  {metric:16s} {fmt(p['run']):>8s} {fmt(p['ref']):>8s}   {ci:30s} "
                     f"{p['up']:5.0%} {p['down']:5.0%} {fmt(h) if h is not None else '':>8s}")
    c = coverage_ratio(run, ref, keys)
    if c is not None:
        h_cov = human_level(run, keys, "human_coverage")
        h_ch = human_level(run, keys, "human_coverage_chance")
        h = f"{h_cov / h_ch:.2f}" if h_cov is not None and h_ch else ""
        times = f"x{c['times']:.2f} [{c['lo']:.2f}, {c['hi']:.2f}]"
        lines.append(f"  {'coverage/chance':16s} {c['run']:8.2f} {c['ref']:8.2f}   {times:30s} "
                     f"{'':>5s} {'':>5s} {h:>8s}")
    if by_grade:
        lines += _groups(run, ref, keys, "grade", list(GRADES))
    if by_genre:
        counts: dict[str, int] = {}
        for k in keys:
            g = run[k].get("genre") or ref[k].get("genre")
            if isinstance(g, str) and g:
                counts[g] = counts.get(g, 0) + 1
        lines += _groups(run, ref, keys, "genre", sorted(counts, key=lambda g: -counts[g]))
    return lines


def _groups(run: dict, ref: dict, keys: list[str], column: str, values: list[str]) -> list[str]:
    """Mean run - ref of GRADE_METRICS per value of a column; [human level] for coverage
    and motion_pred."""
    out = [f"  by {column} (mean run - ref [human]): {'n':>4s} " +
           " ".join(f"{m:>22s}" for m in GRADE_METRICS)]
    for v in values:
        gk = [k for k in keys if (run[k].get(column) or ref[k].get(column)) == v]
        if not gk:
            continue
        cells = []
        for m in GRADE_METRICS:
            p = paired(run, ref, gk, m)
            h = human_level(run, gk, dict(METRICS).get(m))
            cell = (fmt(p["diff"], True) if p else "-") + (f" [{h:.3f}]" if h is not None else "")
            cells.append(f"{cell:>22s}")
        out.append(f"  {v:35s} {len(gk):4d} " + " ".join(cells))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", type=Path, nargs="+",
                    help="evaluate.py output directories (or per_song.csv files); the first "
                         "is the reference")
    ap.add_argument("--by-grade", action="store_true")
    ap.add_argument("--by-genre", action="store_true")
    a = ap.parse_args(argv)
    if len(a.runs) < 2:
        print("give a reference and at least one run", file=sys.stderr)
        return 2
    ref = read_run(a.runs[0])
    print(f"reference: {a.runs[0]} ({len(ref)} songs)")
    for path in a.runs[1:]:
        run = read_run(path)
        common = sum(k in run for k in ref)
        print(f"\n{path}: {common} songs in common")
        if common == 0:
            continue
        print("\n".join(compare(run, ref, a.by_grade, a.by_genre)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
