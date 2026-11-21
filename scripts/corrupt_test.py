"""Do the metrics see what they are meant to? The human charts, corrupted on purpose
(docs/EVALUATION.md, src/evaluation/corrupt.py; the method of ChartGenEval, arXiv 2607.12857).

    python scripts/corrupt_test.py --split val --per-song --n 0
    python scripts/corrupt_test.py --split val --per-song --n 0 --write-runs    # + judge runs

The songs of evaluate.py (same --split, --per-song, --seed and --n). For each song, the human
chart and every corruption at its three doses (each draw seeded by song, corruption and
dose), scored with evaluate.py's columns (structure_and_patterns, new_onsets), the local SR
(sr) and the note starts (notes). Per metric and corruption: the within-song Spearman
correlation of the metric with the dose (the human chart = dose 0), its median over the songs
(rho), the share of songs whose sign agrees with the median's (agree), and the median change
at the largest dose over the spread of the human charts across songs (effect). A metric is

    sensitive to a corruption   |rho| >= 0.8, agree >= 0.8 and |effect| >= 0.2
    invariant to it             |rho| < 0.2 or |effect| < 0.05

and blind when it is sensitive to none. Writes <out>/summary.json and <out>/summary.md, both
aggregate only: no per-song rows, so the playtest songs not played yet stay blind.
--write-runs also writes <out>/runs/<corruption>_<dose>/ (charts.npz, per_song.csv: key,
path, grade, sr_gen, sr_human_local), to score with judge.py --score: does the judge find the
corruption? Use --n 0 then, since judge.py keeps only the songs every run has (or the same
--n as the judged run's first songs). --other-human with it also writes <out>/runs/other_human/:
for each song the human chart of another difficulty of its set nearest in SR, on the same beat
grid (the same timing points and offset), cut to the song's rows; a song with none is left
out. A judge should take it for human: the control against a judge that only tells this
chart from any other.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from scripts.evaluate import (
    GRADES,
    onset_times,
    pick_per_song,
    row_times,
    set_onset_times,
    structure_and_patterns,
)
from src.data.cache import read_manifest
from src.data.mel import open_mel
from src.data.tokenizer import EMPTY, HOLD_START, TAP, K, L, decode, grammar_violations, make_metas
from src.evaluation.corrupt import CORRUPTIONS, DOSES, corrupt, spearman_with_dose
from src.evaluation.metrics import new_positions
from src.evaluation.sr import star_rating

SENSITIVE = {"rho": 0.8, "agree": 0.8, "effect": 0.2}
INVARIANT = {"rho": 0.2, "effect": 0.05}


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--split", default="val")
    ap.add_argument("--per-song", action="store_true")
    ap.add_argument("--n", type=int, default=0, help="first N songs (0 = all)")
    ap.add_argument("--seed", type=int, default=0, help="evaluate.py's --seed (which charts)")
    ap.add_argument("--corruptions", nargs="*", default=list(CORRUPTIONS),
                    choices=list(CORRUPTIONS))
    ap.add_argument("--far-k", type=int, default=8)
    ap.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"))
    ap.add_argument("--cache", type=Path, default=Path("data/cache"))
    ap.add_argument("--fake-mel", action="store_true", help="plumbing tests only")
    ap.add_argument("--exclude", type=Path, default=None,
                    help="a file of chart keys to leave out, one a line")
    ap.add_argument("--out", type=Path, default=None,
                    help="default outputs/corrupt/<split>")
    ap.add_argument("--write-runs", action="store_true",
                    help="also write a judge.py --score run per corruption and dose")
    ap.add_argument("--other-human", action="store_true",
                    help="with --write-runs: the run of another human chart of each song")
    a = ap.parse_args(argv)
    if a.other_human and not a.write_runs:
        ap.error("--other-human needs --write-runs")
    return a


def other_human(r: dict, mates: list[str], z, by_key: dict, cache: Path):
    """(tokens shaped as the song's own, SR) of the mate nearest in SR on the same beat
    grid, cut to the song's rows; None when no mate fits."""
    human, n_cells, sr = z["tokens"], int(z["n_cells"]), float(r["sr"])
    for k in sorted((m for m in mates if by_key.get(m, {}).get("sr")),
                    key=lambda m: abs(float(by_key[m]["sr"]) - sr)):
        o = np.load(cache / "tokens" / f"{k}.npz")
        if int(o["cell_offset"]) != int(z["cell_offset"]) \
                or not np.array_equal(o["timing_points"], z["timing_points"]):
            continue
        flat = o["tokens"].reshape(-1, K)[:int(o["n_cells"])]
        out = human.copy().reshape(-1, K)
        out[:n_cells] = EMPTY
        m = min(n_cells, len(flat))
        out[:m] = flat[:m]
        if len(grammar_violations(out[:n_cells])) == 0:
            return out.reshape(human.shape), float(by_key[k]["sr"])
    return None


def song_columns(full: np.ndarray, human: np.ndarray, mel, n_cells: int, far_k: int,
                 times: np.ndarray, grade: str, ref_t: np.ndarray, metas) -> dict:
    """evaluate.py's columns of one chart (full [n_chunks, L, K]) against the human chart."""
    cols = structure_and_patterns(full, human, mel, n_cells, far_k, times, grade)
    out = {k: v for k, v in cols.items() if not k.startswith("human_")}
    out["new_onsets"] = new_positions(onset_times(full, times), ref_t)
    flat = full.reshape(-1, K)[:n_cells]
    out["notes"] = float(np.isin(flat, (TAP, HOLD_START)).sum())
    out["sr"] = star_rating(decode(full, metas))
    return {k: float(v) for k, v in out.items() if isinstance(v, (int, float, np.number))}


def summarize_metric(per_song: list[list[float]]) -> dict:
    """per_song: for each song, [value at dose 0, dose 1, ...] of one metric and corruption."""
    v = np.array(per_song, dtype=np.float64)
    rhos = np.array([spearman_with_dose(row) for row in v])
    ok = np.isfinite(rhos)
    rho = float(np.median(rhos[ok])) if ok.any() else float("nan")
    sign = np.sign(rho) if np.isfinite(rho) else 0.0
    agree = float(np.mean(np.sign(rhos[ok]) == sign)) if ok.any() and sign != 0 else float("nan")
    base = v[:, 0][np.isfinite(v[:, 0])]
    spread = float(np.std(base)) if len(base) > 1 else float("nan")
    change = v[:, -1] - v[:, 0]
    change = change[np.isfinite(change)]
    effect = float(np.median(change) / spread) if len(change) and spread and spread > 0 \
        else float("nan")
    sensitive = bool(abs(rho) >= SENSITIVE["rho"] and agree >= SENSITIVE["agree"]
                     and abs(effect) >= SENSITIVE["effect"]) if np.isfinite(rho) \
        and np.isfinite(agree) and np.isfinite(effect) else False
    invariant = bool(not np.isfinite(rho) or abs(rho) < INVARIANT["rho"]
                     or (np.isfinite(effect) and abs(effect) < INVARIANT["effect"]))
    return {"rho": rho, "agree": agree, "effect": effect, "songs": int(ok.sum()),
            "sensitive": sensitive, "invariant": invariant}


def main(argv: list[str] | None = None) -> int:
    a = parse_args(argv)
    out = a.out or Path("outputs/corrupt") / a.split
    out.mkdir(parents=True, exist_ok=True)
    store = open_mel(a.cache, fake=a.fake_mel)
    rows = [r for r in read_manifest(a.manifest) if r["split"] == a.split and r["sr"]
            and r.get("drop", "") == ""
            and (a.cache / "tokens" / f"{r['key']}.npz").exists()
            and store.has(r["key"])]
    by_set: dict[str, list[str]] = {}
    by_key = {m["key"]: m for m in read_manifest(a.manifest)}
    others: tuple[dict, list] = ({}, [])
    for m in read_manifest(a.manifest):
        if (a.cache / "tokens" / f"{m['key']}.npz").exists():
            by_set.setdefault(m["set_id"], []).append(m["key"])
    if a.per_song:
        rows = pick_per_song(rows, a.seed)
    if a.n:
        rows = rows[:a.n]
    if a.exclude is not None:
        drop = {line.strip() for line in a.exclude.read_text().splitlines() if line.strip()}
        rows = [r for r in rows if r["key"] not in drop]
    if not rows:
        print(f"no {a.split} songs with SR, tokens and mel", file=sys.stderr)
        return 2

    versions = [(c, d) for c in a.corruptions for d in DOSES[c]]
    values: dict[tuple[str, str], list[list[float]]] = {}     # (metric, corruption) -> songs
    runs: dict[tuple[str, float], tuple[dict, list]] = {v: ({}, []) for v in versions}
    t0 = time.perf_counter()
    for i, r in enumerate(rows):
        z = np.load(a.cache / "tokens" / f"{r['key']}.npz")
        tps = [(float(t), float(bl)) for t, bl in z["timing_points"]]
        offset, n_cells, sr = int(z["cell_offset"]), int(z["n_cells"]), float(r["sr"])
        human = z["tokens"]
        mel = store.chart(r["key"], tps, offset, (len(human) + 1) * L)
        grade = next(g for g, lo, hi in GRADES if lo <= sr < hi)
        times = row_times(tps, offset, n_cells)
        mates = [k for k in by_set.get(r["set_id"], []) if k != r["key"]]
        ref_t = set_onset_times(mates, a.cache)
        metas = make_metas(tps, offset, len(human), sr)
        base = song_columns(human, human, mel, n_cells, a.far_k, times, grade, ref_t, metas)
        song: dict[str, dict[str, list[float]]] = {}
        for c in a.corruptions:
            for d in DOSES[c]:
                rng = np.random.default_rng([a.seed, i, CORRUPTIONS.index(c), int(d * 1000)])
                full = human.copy()
                flat = full.reshape(-1, K)
                flat[:n_cells] = corrupt(flat[:n_cells], c, d, rng)
                cols = song_columns(full, human, mel, n_cells, a.far_k, times, grade, ref_t,
                                    metas)
                for k, v in cols.items():
                    song.setdefault(c, {}).setdefault(k, [base.get(k, float("nan"))]).append(v)
                if a.write_runs:
                    charts, table = runs[(c, d)]
                    charts[r["key"]] = full
                    table.append({"key": r["key"], "path": r["path"], "grade": grade,
                                  "sr_gen": round(cols["sr"], 4),
                                  "sr_human_local": round(base["sr"], 4)})
        if a.other_human:
            got = other_human(r, mates, z, by_key, a.cache)
            if got is not None:
                others[0][r["key"]] = got[0]
                others[1].append({"key": r["key"], "path": r["path"], "grade": grade,
                                  "sr_gen": round(got[1], 4), "sr_human_local": round(base["sr"], 4)})
        for c, cols in song.items():
            for k, seq in cols.items():
                if len(seq) == 1 + len(DOSES[c]):
                    values.setdefault((k, c), []).append(seq)
        print(f"[{i + 1}/{len(rows)}] {time.perf_counter() - t0:.0f} s", flush=True)

    table = {}
    for (k, c), per_song in sorted(values.items()):
        table.setdefault(k, {})[c] = summarize_metric(per_song)
    verdict = {k: {"sensitive_to": [c for c, s in by_c.items() if s["sensitive"]],
                   "invariant_to": [c for c, s in by_c.items() if s["invariant"]]}
               for k, by_c in table.items()}
    summary = {"songs": len(rows), "split": a.split, "seed": a.seed, "doses": DOSES,
               "criteria": {"sensitive": SENSITIVE, "invariant": INVARIANT},
               "metrics": table, "verdict": verdict,
               "blind": sorted(k for k, v in verdict.items() if not v["sensitive_to"])}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    lines = [f"# Corruption test ({a.split}, {len(rows)} songs)", "",
             "rho / agree / effect per corruption; **bold** = sensitive, plain = neither, "
             "dim (·) = invariant.", "",
             "| metric | " + " | ".join(a.corruptions) + " |",
             "|---|" + "---|" * len(a.corruptions)]
    for k in sorted(table):
        cells = []
        for c in a.corruptions:
            s = table[k].get(c)
            if s is None:
                cells.append("")
                continue
            txt = f"{s['rho']:+.2f} / {s['agree']:.2f} / {s['effect']:+.2f}"
            cells.append(f"**{txt}**" if s["sensitive"] else f"· {txt}" if s["invariant"]
                         else txt)
        lines.append(f"| {k} | " + " | ".join(cells) + " |")
    lines += ["", "Blind (sensitive to none): " + (", ".join(summary["blind"]) or "none")]
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    if a.write_runs:
        named = {f"{c}_{d:g}": v for (c, d), v in runs.items()}
        if a.other_human and others[1]:
            named["other_human"] = others
            print(f"other_human: {len(others[1])} of {len(rows)} songs")
        for name, (charts, rows_out) in named.items():
            run = out / "runs" / name
            run.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(run / "charts.npz", **charts)
            with open(run / "per_song.csv", "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(rows_out[0]), lineterminator="\n")
                w.writeheader()
                w.writerows(rows_out)
    print(f"{len(table)} metrics, {len(summary['blind'])} blind; {out / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
