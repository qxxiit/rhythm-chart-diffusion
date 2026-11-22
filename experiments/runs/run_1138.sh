#!/bin/sh
# 11/22 (~1 h 15 min): the system's lane passes judged by the lane-only judge. run_1137:
# full-v4's lanes at 1 on the human rhythm (rl1137_lt1) are nearly a mapper's (0.559), and the
# system's charts end with worse lanes; the jacks untangled turn into trills. Here the passes run
# on rl1137_lt1 as the system runs them (the flags of scripts/system.sh): uj (the jacks
# untangled), mb (the mirrored bars), lanes (both), and all (every pass of the system: rests,
# reuse, long notes, chords too). A full-view judge on each against the human charts, as
# run_1137's (full-v4's trunk, seed 1, the SR as shown, 2 epochs), against s1_l1137_lt1. uj, mb
# and lanes keep the onset rows (a chord may lose a tap the untangle cannot move), so their
# judges see little but the lanes; all changes the rhythm too. Why and the predictions: EXPERIMENTS 2026-11-22 (3). Writes under outputs/ only,
# nothing is deleted.
#   passes  (~25 min) the four runs, three processes each
#   judges  (~50 min) four judges
#   table   (seconds)
# All: for p in passes judges table; do sh experiments/runs/run_1138.sh $p; done
set -ex
PHASE=${1:-passes}
SHARDS=${SHARDS:-3}
P=outputs/full-v4/best.pt
D=outputs/run_1105
J=outputs/judge
[ -f $D/rl1137_lt1.path ] || { echo "no $D/rl1137_lt1.path (run_1137)"; exit 2; }
[ -f $J/s1_l1137_lt1/summary.json ] || { echo "no $J/s1_l1137_lt1 (run_1137)"; exit 2; }
R=$(cat $D/rl1137_lt1.path)
EV="python scripts/evaluate.py --ckpt $P --split val --n 120 --per-song --lane-guidance 2 \
--lane-temp 1 --stats oracle --from-charts $R"
JUDGE="python scripts/judge.py --ckpt $P --seed 1 --sr view --epochs 2 --eval-epochs --limit 120"
# The passes as scripts/system.sh has them (DECISIONS through 2026-11-20).
RESTS="--rest 1 --light 0.5 --light-alike --fill-holes 0.5 --rest-max-run 8"
REUSE="--carry-rhythm 0 --carry-keep --carry-holds 1 --copy-bias 0 --copy-min-lag 8 \
--copy-sections 4 --section-keep --section-back --alike 2 --alike-keep 4"
HOLDS="--hold-share oracle --hold-up --ln-tidy 0.5"
CHORDS="--strong-chords 1 --chord-cap 2"
JACKS="--untangle --untangle-move --untangle-grades --untangle-insane"
MIRROR="--mirror-bars --mirror-shares 0.034,0.049,0.062"
LAST="--snap-releases 2"
DENSE="--dense-taps 0.2 --dense-repeats"

passed() {               # NAME FLAGS...: SHARDS processes, merged; the run's path in $D/NAME.path
    name=$1
    shift
    [ -f $D/$name.path ] && { echo "$name is there"; return; }
    k=0
    while [ $k -lt $SHARDS ]; do
        $EV "$@" --shard $k/$SHARDS > $D/${name}_shard$k.log 2>&1 &
        k=$((k + 1))
    done
    wait
    run=$(sed -n 's/^run //p' $D/${name}_shard0.log | head -n 1)
    [ -n "$run" ] || { echo "no run line in $D/${name}_shard0.log"; exit 2; }
    python scripts/merge_shards.py "$run"
    echo "$run" > $D/$name.path
}

case $PHASE in
passes)
    passed ps1138_uj $JACKS
    passed ps1138_mb $MIRROR
    passed ps1138_lanes $JACKS $MIRROR
    passed ps1138_all $RESTS $REUSE $HOLDS $CHORDS $JACKS $MIRROR $LAST $DENSE
    ;;
judges)
    for m in uj mb lanes all; do
        [ -f $D/ps1138_$m.path ] || { echo "no ps1138_$m: skipped"; continue; }
        [ -f $J/s1_p1138_$m/summary.json ] && { echo "s1_p1138_$m is there"; continue; }
        $JUDGE --ai $(cat $D/ps1138_$m.path) --name s1_p1138_$m
    done
    ;;
table)
    for m in uj mb lanes all; do
        [ -f $J/s1_p1138_$m/summary.json ] || continue
        python scripts/judge_table.py $J/s1_l1137_lt1 $J/s1_p1138_$m --out $D/results_1138_$m.md
    done
    python - $D <<'EOF' | tee $D/results_1138_lanes.md
import csv, sys
from pathlib import Path
import numpy as np
D = Path(sys.argv[1])
runs = [("lt1", D / "rl1137_lt1.path")] + [(m, D / f"ps1138_{m}.path")
                                           for m in ("uj", "mb", "lanes", "all")]
runs = [(m, Path(p.read_text().strip())) for m, p in runs if p.exists()]
cols = ["motion_pred", "run_length", "fast_jacks", "move_jack", "move_trill", "move_stair",
        "move_leap", "chord_share", "new_onsets", "hold_beats"]
counts = ["untangled", "untangle_moved", "bars_mirrored"]
def mean(rows, c):
    v = [float(r[c]) for r in rows if r.get(c) not in (None, "", "nan")]
    return np.mean(v) if v else float("nan")
def cell(x):
    return "" if np.isnan(x) else f"{x:.3f}"
print("| the passes on lt1 (120 songs) | " + " | ".join(cols + counts) + " |")
print("|---" * (len(cols) + len(counts) + 1) + "|")
first = list(csv.DictReader(open(runs[0][1] / "per_song.csv")))
print("| human | " + " | ".join(cell(mean(first, "human_" + c)) for c in cols)
      + " |" * len(counts) + " |")
for m, run in runs:
    rows = list(csv.DictReader(open(run / "per_song.csv")))
    print(f"| {m} | " + " | ".join(cell(mean(rows, c)) for c in cols + counts) + " |")
print()
print("counts: per song, the jacks' chord taps untangled (and of those moved), the bars mirrored.")
EOF
    ;;
*)
    echo "phases: passes judges table"; exit 2
    ;;
esac
