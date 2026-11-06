"""The one results table of docs/EVALUATION.md: a row per run, the same columns for every row.

    python scripts/results_table.py RUN [RUN ...] [--judge outputs/judge/<name> ...] \
        [--out outputs/results_1105.md]

RUN: evaluate.py directories (per_song.csv). Columns: the songs; with --judge, each judge
run's auc and paired for the runs it judged or scored (the run's own judge when it was --ai),
and a scored run's mean logit minus the judged run's per song with its 95% interval; the
diagnostics together, when scripts/diag_judge.py has been run on the run (its boosted trees'
auc and paired, diag:auc and diag:paired: the run's charts told from the human ones by all the
diagnostics at once); the diagnostics one by one as the share of the run's charts inside the
human band (the 10th-90th percentile of the human charts of the same grade, from the human_*
columns; grades with fewer than 10 songs pooled; 80% for the human charts themselves, the
first row); SR error and F1@50 as sanity checks. Aggregate only.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

DIAGNOSTICS = ("new_onsets", "move_stair", "move_trill", "move_jack", "chord_share",
               "lone_chord", "section_rhythm", "section_copy", "hold_share", "rhythm_rep1",
               "copy_near")
MIN_GRADE_SONGS = 10


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", type=Path, nargs="+")
    ap.add_argument("--judge", type=Path, nargs="*", default=[])
    ap.add_argument("--metrics", nargs="*", default=list(DIAGNOSTICS))
    ap.add_argument("--out", type=Path, default=None, help="markdown file (and .json beside)")
    return ap.parse_args(argv)


def read_rows(run: Path) -> list[dict]:
    with open(run / "per_song.csv", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def num(row: dict, col: str) -> float:
    try:
        v = float(row.get(col, "nan"))
    except ValueError:
        return float("nan")
    return v if math.isfinite(v) else float("nan")


def bands(rows: list[dict], metric: str) -> dict[str, tuple[float, float]]:
    """grade -> (10th, 90th percentile) of the human charts' values; "*" for all grades."""
    by_grade: dict[str, list[float]] = {}
    for r in rows:
        v = num(r, f"human_{metric}")
        if math.isfinite(v):
            by_grade.setdefault(r.get("grade", ""), []).append(v)
            by_grade.setdefault("*", []).append(v)
    return {g: (float(np.percentile(v, 10)), float(np.percentile(v, 90)))
            for g, v in by_grade.items() if len(v) >= MIN_GRADE_SONGS or g == "*"}


def band_share(rows: list[dict], metric: str, band: dict, column: str | None = None) -> float:
    inside = []
    for r in rows:
        v = num(r, column or metric)
        b = band.get(r.get("grade", ""), band.get("*"))
        if b is not None and math.isfinite(v):
            inside.append(b[0] <= v <= b[1])
    return float(np.mean(inside)) if inside else float("nan")


def judge_columns(judges: list[Path], run: Path) -> dict:
    out = {}
    for j in judges:
        summary = json.loads((j / "summary.json").read_text())
        for label, path in summary.get("labels", {}).items():
            if Path(path).resolve() != run.resolve() or label not in summary:
                continue
            s = summary[label]
            out[f"{j.name}:auc"] = s.get("auc")
            out[f"{j.name}:paired"] = s.get("paired")
            if "minus_ai" in s:
                m = s["minus_ai"]
                out[f"{j.name}:minus"] = f"{m['mean']:+.3f} [{m['lo']:+.3f}, {m['hi']:+.3f}]"
    return out


def diag_columns(run: Path) -> dict:
    """diag_judge.py's trees on the run, when it has been run there."""
    path = run / "diag_judge.json"
    if not path.exists():
        return {}
    d = json.loads(path.read_text())
    return {"diag:auc": d["trees"]["auc"], "diag:paired": d["trees"]["paired"]}


def fmt(v) -> str:
    if isinstance(v, str):
        return v
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return ""
    return f"{v:.3f}"


def main(argv: list[str] | None = None) -> int:
    a = parse_args(argv)
    table = []
    first = read_rows(a.runs[0])
    metrics = [m for m in a.metrics if f"human_{m}" in first[0] and m in first[0]]
    ref = {m: bands(first, m) for m in metrics}
    human = {"run": "human charts", "songs": len(first)}
    human.update({m: band_share(first, m, ref[m], f"human_{m}") for m in metrics})
    table.append(human)
    for run in a.runs:
        rows = read_rows(run)
        row = {"run": run.name, "songs": len(rows)}
        row.update(judge_columns(a.judge, run))
        row.update(diag_columns(run))
        row.update({m: band_share(rows, m, bands(rows, m)) for m in metrics})
        row["sr_error"] = float(np.nanmean([num(r, "sr_error") for r in rows]))
        row["f1@50"] = float(np.nanmean([num(r, "f1@50") for r in rows]))
        table.append(row)
    cols = ["run", "songs"]
    for row in table:
        cols += [c for c in row if c not in cols]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for row in table:
        lines.append("| " + " | ".join(fmt(row.get(c)) if c not in ("run", "songs")
                                        else str(row.get(c, "")) for c in cols) + " |")
    text = "\n".join(lines) + "\n"
    print(text)
    if a.out is not None:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(text)
        a.out.with_suffix(".json").write_text(json.dumps(table, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
