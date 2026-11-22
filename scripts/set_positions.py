"""A run's charts with their notes on new positions moved or taken out, the mapset's other
charts as the oracle (src/evaluation/oracle.py), written as a run for judge.py --ai.

    python scripts/set_positions.py --run "$(cat outputs/run_1105/mbh.path)" --mode snap \
        --out outputs/positions_1139/snap

The run's first --n songs (its per_song.csv order). same: the charts as they are (a control:
its judge should be the run's); drop: the taps of the new rows go; snap: they move to the
nearest row within 1/4 beat that another chart of the set uses, and go where none fits. Rows
that start a long note stay. Writes <out>/charts.npz, <out>/per_song.csv (key, path, grade,
sr_gen, sr_human_local, evaluate.py's columns of the chart as corrupt_test.py scores them, the
counts) and <out>/summary.json (means over the songs with other charts in their set). Not on
test songs: the oracle reads the other human charts.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from scripts.corrupt_test import song_columns
from scripts.evaluate import GRADES, row_times, set_onset_times
from src.data.cache import read_manifest
from src.data.mel import open_mel
from src.data.tokenizer import K, L, grammar_violations, make_metas
from src.evaluation.oracle import MODES, to_set_positions

MEANS = ("new_onsets", "notes", "sr_gen", "new_rows", "hold_rows", "moved", "dropped")


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", type=Path, required=True, help="a run's directory (charts.npz)")
    ap.add_argument("--mode", choices=MODES, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=0, help="the run's first N songs (0 = all)")
    ap.add_argument("--far-k", type=int, default=8)
    ap.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"))
    ap.add_argument("--cache", type=Path, default=Path("data/cache"))
    ap.add_argument("--fake-mel", action="store_true", help="plumbing tests only")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    a = parse_args(argv)
    with open(a.run / "per_song.csv", newline="") as f:
        run_rows = list(csv.DictReader(f))
    if a.n:
        run_rows = run_rows[:a.n]
    saved = np.load(a.run / "charts.npz")
    manifest = read_manifest(a.manifest)
    by_key = {m["key"]: m for m in manifest}
    if any(by_key[r["key"]]["split"] == "test" for r in run_rows):
        print("test songs: the oracle reads their other human charts; val only", file=sys.stderr)
        return 2
    by_set: dict[str, list[str]] = {}
    for m in manifest:
        if (a.cache / "tokens" / f"{m['key']}.npz").exists():
            by_set.setdefault(m["set_id"], []).append(m["key"])
    store = open_mel(a.cache, fake=a.fake_mel)
    a.out.mkdir(parents=True, exist_ok=True)
    charts, table = {}, []
    for r in run_rows:
        key = r["key"]
        z = np.load(a.cache / "tokens" / f"{key}.npz")
        tps = [(float(t), float(bl)) for t, bl in z["timing_points"]]
        offset, n_cells, sr = int(z["cell_offset"]), int(z["n_cells"]), float(by_key[key]["sr"])
        human, tokens = z["tokens"], saved[key]
        mel = store.chart(key, tps, offset, (len(human) + 1) * L)
        times = row_times(tps, offset, n_cells)
        mates = [k for k in by_set.get(by_key[key]["set_id"], []) if k != key]
        ref_t = set_onset_times(mates, a.cache)
        flat = tokens.reshape(-1, K).copy()
        flat[:n_cells], counts = to_set_positions(flat[:n_cells], times, ref_t, a.mode)
        full = flat.reshape(tokens.shape)
        if len(grammar_violations(full)):                    # cannot happen: taps into EMPTY
            raise RuntimeError(f"{key}: the grammar broke")
        grade = next(g for g, lo, hi in GRADES if lo <= sr < hi)
        cols = song_columns(full, human, mel, n_cells, a.far_k, times, grade, ref_t,
                            make_metas(tps, offset, len(full), sr))
        charts[key] = full
        table.append({"key": key, "path": r.get("path", by_key[key].get("path", "")),
                      "grade": grade, "sr_gen": round(cols.pop("sr"), 4),
                      "sr_human_local": r.get("sr_human_local", ""), "set_charts": len(mates),
                      "sr_gen_run": r.get("sr_gen", ""), **cols, **counts})
    np.savez_compressed(a.out / "charts.npz", **charts)
    names = list(dict.fromkeys(k for row in table for k in row))
    with open(a.out / "per_song.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=names)
        w.writeheader()
        w.writerows(table)
    with_set = [row for row in table if row["set_charts"]]
    summary = {"run": str(a.run), "mode": a.mode, "songs": len(table),
               "songs_with_set": len(with_set),
               **{c: float(np.nanmean([float(row[c]) for row in with_set])) if with_set
                  else float("nan") for c in MEANS}}
    if a.mode == "same":                  # the SR as evaluate.py had it, to the 4th decimal
        diffs = [abs(float(row["sr_gen"]) - float(row["sr_gen_run"])) for row in table
                 if row["sr_gen_run"] not in ("", "nan")]
        summary["sr_gen_max_diff"] = max(diffs) if diffs else float("nan")
    (a.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
