"""Put the shards of an evaluate.py run together (evaluate.py --shard K/N, N processes sharing
the GPU, each on the songs whose index is K modulo N, with the seeds of the full run).

    python scripts/merge_shards.py RUN_DIR      # reads RUN_DIR_shard0ofN ... RUN_DIR_shard{N-1}ofN

Writes RUN_DIR as the unsharded run would have: charts.npz (every song's chart), per_song.csv
(every row, in the run's order), summary.json (the shards' settings, the means and intervals
again over all songs, and shards: N), so --from-charts, judge.py, diag_judge.py and
results_table.py take it as any run. The shard directories stay (nothing is deleted). Refuses a
RUN_DIR that already has a per_song.csv unless --force, and shards that are missing, unfinished,
of two different N or that hold the same song twice.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from scripts.evaluate import CI_KEYS, NOT_NUMERIC, bootstrap_ci, nan_mean


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run", type=Path, help="the run's directory, without _shardKofN")
    ap.add_argument("--force", action="store_true", help="write over a run already there")
    return ap.parse_args(argv)


def find_shards(run: Path) -> dict[int, dict[int, Path]]:
    """N -> {K: directory} for the directories named run_shardKofN."""
    pattern = re.compile(re.escape(run.name) + r"_shard(\d+)of(\d+)")
    found: dict[int, dict[int, Path]] = {}
    for d in run.parent.glob(f"{run.name}_shard*of*"):
        m = pattern.fullmatch(d.name)
        if m and d.is_dir():
            found.setdefault(int(m[2]), {})[int(m[1])] = d
    return found


def number(v: str) -> float:
    try:
        return float(v)
    except ValueError:
        return float("nan")


def main(argv: list[str] | None = None) -> int:
    a = parse_args(argv)
    found = find_shards(a.run)
    if len(found) != 1:
        print(f"{a.run}: {'no shards' if not found else 'shards of more than one N'}"
              f" ({', '.join(f'of{n}' for n in sorted(found))})", file=sys.stderr)
        return 2
    n, parts = next(iter(found.items()))
    unfinished = [k for k in range(n) if k not in parts
                  or not (parts[k] / "per_song.csv").exists()
                  or not (parts[k] / "charts.npz").exists()]
    if unfinished:
        print(f"{a.run}: shards {unfinished} of {n} missing or unfinished", file=sys.stderr)
        return 2
    if (a.run / "per_song.csv").exists() and not a.force:
        print(f"{a.run}: a run is already there (--force writes over it)", file=sys.stderr)
        return 2

    rows: list[dict] = []
    charts: dict[str, np.ndarray] = {}
    for k in range(n):
        with open(parts[k] / "per_song.csv", newline="", encoding="utf-8") as f:
            rows += list(csv.DictReader(f))
        z = np.load(parts[k] / "charts.npz")
        for key in z.files:
            if key in charts:
                print(f"{a.run}: {key} in two shards", file=sys.stderr)
                return 2
            charts[key] = z[key]
    if any("_index" not in r for r in rows):
        print(f"{a.run}: a shard without _index (made before --shard?)", file=sys.stderr)
        return 2
    rows.sort(key=lambda r: int(r["_index"]))
    fields = [c for c in rows[0] if c != "_index"]

    a.run.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.run / "charts.npz", **charts)
    with open(a.run / "per_song.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)

    summary = json.loads((parts[0] / "summary.json").read_text())
    summary = {k: v for k, v in summary.items()
               if not k.startswith(("mean_", "ci95_")) and k != "shard"}
    summary["songs"], summary["shards"] = len(rows), n
    seed = summary.get("seed", 0)
    for c in (c for c in fields if c not in NOT_NUMERIC):
        summary[f"mean_{c}"] = nan_mean([number(r[c]) for r in rows])
    for c in (c for c in CI_KEYS if c in fields):
        summary[f"ci95_{c}"] = bootstrap_ci([number(r[c]) for r in rows], seed=seed)
    (a.run / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"{a.run}: {len(rows)} songs from {n} shards")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
