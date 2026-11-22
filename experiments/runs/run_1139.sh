#!/bin/sh
# 11/22 night (~45 min): how much of what the judge sees in the system's charts do the notes
# on new positions carry? run_1137 and run_1138 put what is left of the system's tell in the
# draw's rhythm, whose note starts land where no chart of the mapset has one 2.4 times as often
# as the human charts' (new_onsets 0.120 against 0.051). Before a pass is built for them, their
# ceiling, with the mapset's other charts as the oracle (scripts/set_positions.py): the system's
# charts (run_1134's mbh, the first 120 val songs) as they are (same, a control), the taps of
# the new rows taken out (drop), and moved to the nearest row within 1/4 beat another chart of
# the set uses (snap). Full-view judges as s1_mbh (full-v4's trunk, seed 1, the SR as shown,
# 2 epochs) on each, against s1_mbh. Why and the predictions: EXPERIMENTS 2026-11-22 (4).
# Writes under outputs/ only, nothing is deleted.
#   oracle  (~5 min) the three runs, three processes
#   judges  (~40 min) three judges
#   table   (seconds)
# All: for p in oracle judges table; do sh experiments/runs/run_1139.sh $p; done
set -ex
PHASE=${1:-oracle}
P=outputs/full-v4/best.pt
D=outputs/run_1105
J=outputs/judge
O=outputs/positions_1139
[ -f $D/mbh.path ] || { echo "no $D/mbh.path (run_1134: the system)"; exit 2; }
[ -f $J/s1_mbh/summary.json ] || { echo "no $J/s1_mbh (run_1134)"; exit 2; }
JUDGE="python scripts/judge.py --ckpt $P --seed 1 --sr view --epochs 2 --eval-epochs --limit 120"

case $PHASE in
oracle)
    [ -f $O/snap/summary.json ] && { echo "the runs are there"; exit 0; }
    mkdir -p $O
    for m in same drop snap; do
        python scripts/set_positions.py --run "$(cat $D/mbh.path)" --n 120 --mode $m \
            --out $O/$m > $O/$m.log 2>&1 &
    done
    wait
    for m in same drop snap; do
        [ -f $O/$m/summary.json ] || { echo "no $O/$m: see $O/$m.log"; exit 2; }
    done
    tail -n 1 $O/same.log $O/drop.log $O/snap.log
    ;;
judges)
    for m in same drop snap; do
        [ -f $O/$m/summary.json ] || { echo "no $O/$m: skipped"; continue; }
        [ -f $J/s1_o1139_$m/summary.json ] && { echo "s1_o1139_$m is there"; continue; }
        $JUDGE --ai $O/$m --name s1_o1139_$m
    done
    ;;
table)
    for m in same drop snap; do
        [ -f $J/s1_o1139_$m/summary.json ] || continue
        python scripts/judge_table.py $J/s1_mbh $J/s1_o1139_$m --out $D/results_1139_$m.md
    done
    python - "$(cat $D/mbh.path)" $O <<'EOF' | tee $D/results_1139_positions.md
import csv, sys
from pathlib import Path
import numpy as np
mbh, O = Path(sys.argv[1]), Path(sys.argv[2])
def rows(p):
    return list(csv.DictReader(open(p / "per_song.csv")))
runs = [(m, rows(O / m)) for m in ("same", "drop", "snap") if (O / m / "per_song.csv").exists()]
keys = {r["key"] for r in runs[0][1] if float(r["set_charts"] or 0) > 0}
human = {r["key"]: r for r in rows(mbh) if r["key"] in keys}
cols = ["new_onsets", "notes", "sr_gen", "motion_pred", "fast_jacks", "chord_share",
        "new_rows", "hold_rows", "moved", "dropped"]
def mean(rs, c):
    v = [float(r[c]) for r in rs if r.get(c) not in (None, "", "nan")]
    return np.mean(v) if v else float("nan")
def cell(x):
    return "" if np.isnan(x) else f"{x:.3f}"
print(f"| {len(keys)} songs with other charts in their set | " + " | ".join(cols) + " |")
print("|---" * (len(cols) + 1) + "|")
hr = list(human.values())
print("| human | " + " | ".join(cell(mean(hr, {"sr_gen": "sr_human_local"}.get(c, "human_" + c)))
                              for c in cols) + " |")
for m, rs in runs:
    rs = [r for r in rs if r["key"] in keys]
    print(f"| {m} | " + " | ".join(cell(mean(rs, c)) for c in cols) + " |")
print()
print("notes: note starts a song; new_rows, hold_rows: the onset rows on new positions, those that "
      "start a long note (left as they are); moved, dropped: rows a song.")
EOF
    ;;
*)
    echo "phases: oracle judges table"; exit 2
    ;;
esac
