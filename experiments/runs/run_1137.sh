#!/bin/sh
# 11/22 (~4 h 30 min): a judge of the lanes alone. run_1136: the judges hardly see lane
# patterns (one-hand trills, jumptrills not at all), learning the rhythm, where the AI charts
# differ most, instead. So here the rhythm is the human chart's: the first 120 val songs' human
# charts with their lanes drawn again by full-v4 as the system draws them (--relane: forward
# lanes, refine 2, lane guidance 2, the human chart's stats; rhythm, chord sizes and long notes
# stay), and a judge trained on them against the human charts can only tell them by the lanes.
# Three lane settings: lt1 (the system's, every row at temperature 1), lt07 (0.7: rejected on
# 11/11 by a judge that hardly sees lanes, though its pattern numbers came nearer the human
# charts), lu (each two beats' first onset row at 1, the rest at 0.5); and run_1136's human
# charts with 80% of their rows' lanes shuffled (lanes_0.8) for scale. Full-view judges from
# full-v4's trunk (seed 1, the SR as shown, 2 epochs). Why and the predictions: EXPERIMENTS
# 2026-11-22 (2). Writes under outputs/ only, nothing is deleted.
#   human   (~1 min) the human charts as a run to draw lanes on
#   relane  (~3 h 15 min) the three settings, three processes each
#   judges  (~50 min) four judges
#   table   (seconds)
# All: for p in human relane judges table; do sh experiments/runs/run_1137.sh $p; done
set -ex
PHASE=${1:-human}
SHARDS=${SHARDS:-3}
P=outputs/full-v4/best.pt
D=outputs/run_1105
J=outputs/judge
H=outputs/lanes_1137/human
RND=outputs/corrupt/val_1136/a/runs/lanes_0.8
[ -f $D/mbh.path ] || { echo "no $D/mbh.path (run_1134)"; exit 2; }
[ -f $RND/charts.npz ] || { echo "no $RND (run_1136)"; exit 2; }
EV="python scripts/evaluate.py --ckpt $P --split val --n 120 --per-song --lane-guidance 2 \
--stats oracle --sample-seed 0 --from-charts $H --relane --lanes forward --refine 2"
JUDGE="python scripts/judge.py --ckpt $P --seed 1 --sr view --epochs 2 --eval-epochs --limit 120"

relaned() {              # NAME FLAGS...: SHARDS processes, merged; the run's path in $D/NAME.path
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
human)
    [ -f $H/charts.npz ] && { echo "$H is there"; exit 0; }
    mkdir -p $H
    python - "$(cat $D/mbh.path)" $H <<'EOF'
import csv, sys
from pathlib import Path
import numpy as np
run, out = Path(sys.argv[1]), Path(sys.argv[2])
keys = [r["key"] for r in csv.DictReader(open(run / "per_song.csv"))][:120]
np.savez_compressed(out / "charts.npz",
                    **{k: np.load(f"data/cache/tokens/{k}.npz")["tokens"] for k in keys})
print(f"{len(keys)} human charts in {out}")
EOF
    ;;
relane)
    [ -f $H/charts.npz ] || { echo "no $H"; exit 2; }
    relaned rl1137_lt1 --lane-temp 1
    relaned rl1137_lt07 --lane-temp 0.7
    relaned rl1137_lu --lane-temp 1 --lane-inner-temp 0.5 --lane-unit 24
    ;;
judges)
    for m in lt1 lt07 lu; do
        [ -f $D/rl1137_$m.path ] || { echo "no rl1137_$m: skipped"; continue; }
        [ -f $J/s1_l1137_$m/summary.json ] && { echo "s1_l1137_$m is there"; continue; }
        $JUDGE --ai $(cat $D/rl1137_$m.path) --name s1_l1137_$m
    done
    [ -f $J/s1_l1137_rnd80/summary.json ] || $JUDGE --ai $RND --name s1_l1137_rnd80
    ;;
table)
    for m in lt07 lu rnd80; do
        [ -f $J/s1_l1137_$m/summary.json ] || continue
        python scripts/judge_table.py $J/s1_l1137_lt1 $J/s1_l1137_$m --out $D/results_1137_$m.md
    done
    python - $D $RND <<'EOF' | tee $D/results_1137_lanes.md
import csv, sys
from pathlib import Path
import numpy as np
D = Path(sys.argv[1])
runs = [(m, Path((D / f"rl1137_{m}.path").read_text().strip())) for m in ("lt1", "lt07", "lu")
        if (D / f"rl1137_{m}.path").exists()]
cols = ["motion_pred", "run_length", "fast_jacks", "move_jack", "move_trill", "move_stair",
        "move_leap", "chord_share"]
def mean(rows, c):
    v = [float(r[c]) for r in rows if r.get(c) not in (None, "", "nan")]
    return np.mean(v) if v else float("nan")
print("| lanes on the human rhythm (120 songs) | " + " | ".join(cols) + " |")
print("|---" * (len(cols) + 1) + "|")
first = list(csv.DictReader(open(runs[0][1] / "per_song.csv")))
print("| human | " + " | ".join(f"{mean(first, 'human_' + c):.3f}" for c in cols) + " |")
for m, run in runs:
    rows = list(csv.DictReader(open(run / "per_song.csv")))
    print(f"| {m} | " + " | ".join(f"{mean(rows, c):.3f}" for c in cols) + " |")
EOF
    ;;
*)
    echo "phases: human relane judges table"; exit 2
    ;;
esac
