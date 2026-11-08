"""Judges of one run side by side: what each was shown and how well it told (EXPERIMENTS
2026-11-08 (2), the views).

    python scripts/judge_table.py outputs/judge/cosine outputs/judge/view_* --out table.md

One row a judge (judge.py output directories): its view, SR and audio, the folds it ran (a
judge that stopped early has fewer songs), its auc with the 95% interval over songs, paired,
the auc again on the songs every judge here has (from windows.csv, so judges that stopped at
different folds still compare on the same songs), and the auc after each epoch if it was
scored on the way (--eval-epochs). Aggregate only. Writes --out (markdown) and its .json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.results_table import judge_on, windows_of


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("judges", type=Path, nargs="+")
    ap.add_argument("--out", type=Path, default=None)
    return ap.parse_args(argv)


def fmt(v, spec: str = ".3f") -> str:
    return "" if v is None else format(v, spec)


def main(argv=None) -> int:
    a = parse_args(argv)
    judges = [j for j in a.judges if (j / "summary.json").exists() and (j / "windows.csv").exists()]
    for j in a.judges:
        if j not in judges:
            print(f"{j}: no summary.json and windows.csv, left out", file=sys.stderr)
    if not judges:
        return 2
    summaries = {j: json.loads((j / "summary.json").read_text()) for j in judges}
    windows = {j: windows_of(j) for j in judges}
    shared = set.intersection(*({k for k, c in w if c == "ai"} for w in windows.values()))
    rows = []
    for j in judges:
        s, args = summaries[j], summaries[j].get("args", {})
        ai = s.get("ai", {})
        again = judge_on(windows[j], "ai", shared) if shared else {}
        rows.append({"judge": j.name, "view": args.get("view", "full"), "sr": args.get("sr", "own"),
                     "audio": "off" if args.get("no_audio") else "on",
                     "folds": f"{s.get('folds_run', args.get('folds'))}/{s.get('folds', args.get('folds'))}",
                     "songs": ai.get("songs"), "auc": ai.get("auc"), "ci": ai.get("auc_ci"),
                     "paired": ai.get("paired"), "shared_auc": again.get("auc"),
                     "by_epoch": [(e["epoch"], e["auc"]) for e in s.get("by_epoch", [])],
                     "seconds": s.get("seconds")})
    lines = [f"| judge | view | SR | audio | folds | songs | auc [95%] | paired | "
             f"auc on the {len(shared)} shared songs | by epoch | minutes |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        ci = f" [{r['ci'][0]:.3f}, {r['ci'][1]:.3f}]" if r["ci"] else ""
        ep = ", ".join(f"{e}: {v:.3f}" for e, v in r["by_epoch"])
        mins = f"{r['seconds'] / 60:.0f}" if r["seconds"] else ""
        lines.append(f"| {r['judge']} | {r['view']} | {r['sr']} | {r['audio']} | {r['folds']} | "
                     f"{r['songs']} | {fmt(r['auc'])}{ci} | {fmt(r['paired'])} | "
                     f"{fmt(r['shared_auc'])} | {ep} | {mins} |")
    text = "\n".join(lines)
    print(text)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(text + "\n", encoding="utf-8")
        a.out.with_suffix(".json").write_text(json.dumps({"shared_songs": len(shared),
                                                          "judges": rows}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
