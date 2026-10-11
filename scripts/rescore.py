"""Score evaluate.py runs again from their saved charts, without sampling again.

    python scripts/rescore.py outputs/full-v1/eval_val_songs_continue_random_T128_s0 ...

For each run directory with charts.npz (evaluate.py saves it since 2026-10-02):
every generated chart, and the human chart from the token cache, is scored again
with patterns.summarize, holds.hold_stats, structure.phrase_repeats and
structure.ln_placement, so pattern, long-note and phrase columns added after the run
(e.g. lone_chord, bar_rhythm_repeat, rhythm_rep4, ln_scattered) appear, and
playability.guideline_checks for the chart's grade (chord_over, fast_jacks, anchor_over,
ln_short, release_close); with the log-Mel in the cache also structure.section_repeats
(section_rhythm, section_copy: the bars' audio similarity tells where the music comes
back), structure.alike_repeats (alike_rhythm, alike_counts_off: bars that sound the same)
and playability.chord_strength (--no-mel leaves those out). per_song.csv is updated
in place (the first rescore keeps the original as per_song.orig.csv) and summary.json
gets the new means. rho is left as it was.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from scripts.evaluate import PATTERN_KEYS, nan_mean, row_times
from src.data.mel import open_mel
from src.data.tokenizer import K, L
from src.evaluation.holds import hold_stats
from src.evaluation.patterns import summarize
from src.evaluation.playability import chord_strength, guideline_checks
from src.evaluation.structure import (
    BAR,
    alike_repeats,
    ln_placement,
    phrase_repeats,
    section_repeats,
    ssm_audio,
)


def scores(flat: np.ndarray) -> dict:
    p = summarize(flat, chance_seeds=3)
    return {**{k: p[k] for k in PATTERN_KEYS}, **hold_stats(flat),
            **phrase_repeats(flat, len(flat)), **ln_placement(flat, len(flat))}


def rescore(run: Path, cache: Path, mel: bool = True) -> int:
    charts_path, csv_path = run / "charts.npz", run / "per_song.csv"
    if not charts_path.exists():
        print(f"{run}: no charts.npz (runs before 2026-10-02 cannot be rescored)", file=sys.stderr)
        return 2
    charts = np.load(charts_path)
    store = open_mel(cache) if mel and (cache / "logmel").exists() else None
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        if r["key"] not in charts.files:
            print(f"{run}: {r['key']} missing from charts.npz", file=sys.stderr)
            return 2
        z = np.load(cache / "tokens" / f"{r['key']}.npz")
        n_cells = int(z["n_cells"])
        gen = charts[r["key"]].reshape(-1, K)[:n_cells]
        human = z["tokens"].reshape(-1, K)[:n_cells]
        r.update({k: v for k, v in scores(gen).items()})
        r.update({f"human_{k}": v for k, v in scores(human).items()})
        tps = [(float(t), float(bl)) for t, bl in z["timing_points"]]
        if r.get("grade"):
            times = row_times(tps, int(z["cell_offset"]), n_cells)
            r.update(guideline_checks(gen, times, r["grade"]))
            r.update({f"human_{k}": v for k, v in guideline_checks(human, times, r["grade"]).items()})
        if store is not None and store.has(r["key"]) and n_cells // BAR >= 2:
            from src.models.sampler import onset_strength
            frames = store.chart(r["key"], tps, int(z["cell_offset"]), (len(z["tokens"]) + 1) * L)
            s_audio = ssm_audio(frames, n_cells // BAR)
            r.update(section_repeats(gen, s_audio, n_cells))
            r.update({f"human_{k}": v for k, v in section_repeats(human, s_audio, n_cells).items()})
            r.update(alike_repeats(gen, s_audio, n_cells))
            r.update({f"human_{k}": v for k, v in alike_repeats(human, s_audio, n_cells).items()})
            strength = onset_strength(frames, n_cells)
            r["chord_strength"] = chord_strength(gen, strength)
            r["human_chord_strength"] = chord_strength(human, strength)
    backup = run / "per_song.orig.csv"
    if not backup.exists():
        shutil.copyfile(csv_path, backup)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    summary_path = run / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    for k in fields:
        vals = []
        for r in rows:
            try:
                vals.append(float(r[k]))
            except (TypeError, ValueError):
                break
        else:
            if vals:
                summary[f"mean_{k}"] = nan_mean(vals)
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"{run}: {len(rows)} songs rescored")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", type=Path, nargs="+")
    ap.add_argument("--cache", type=Path, default=Path("data/cache"))
    ap.add_argument("--no-mel", action="store_true",
                    help="leave out the columns that need the log-Mel (section_rhythm, "
                         "section_copy, chord_strength)")
    a = ap.parse_args(argv)
    return max(rescore(run, a.cache, mel=not a.no_mel) for run in a.runs)


if __name__ == "__main__":
    raise SystemExit(main())
