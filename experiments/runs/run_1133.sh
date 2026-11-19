#!/bin/sh
# 11/20 night (~7 h): a model with six chart stats from its first step. run_1132's pat-v1 got
# the stair share and the long notes' length after full-v4's 60k steps (20k more, their
# embeddings from zero) and its draws hardly followed them (the stair share's correlation with
# the human chart +0.14 -> +0.19, full-v4's own draw +0.28; the long notes' length level).
# A stat a model has from the start it reads better (the trill share, +0.41). And the chord
# share brought to the human chart's by a pass (run_1130, run_1131) the judges found at once,
# so it becomes an input too. pat-v2: full-v4's recipe exactly (60k steps, lr 3e-4, warmup
# 1000, row masks 0.5), its three stats and move_stair, hold_beats, chord_share
# (data/chart_stats_v2.csv: build_chart_stats --more). First a probe of pat-v1: the human
# chart's stats with the stair share and the long notes' length set low (0.05, 0.5 beats) and
# high (0.35, 2 beats), 15 songs each: does it read them at all? Then a raw draw of pat-v2 on
# the first 120 val songs (the system's decoding, the human chart's stats as inputs; one draw
# a window, no choice, bias or pass) against full-v4's (run_1126's raw4), judged from both
# trunks (a judge favours its own trunk's model, run_1126): full-v4's (s1_) and pat-v2's (p2_),
# full and taps views. Why and the predictions: EXPERIMENTS 2026-11-20 (1). Writes under
# outputs/ (and data/chart_stats_v2.csv) only, nothing is deleted.
#   probe   (~25 min) pat-v1 with the two stats set low and high, 15 songs each
#   stats   (~5 min) data/chart_stats_v2.csv
#   train   (~3 h 30 min) outputs/pat-v2
#   draw    (~1 h 30 min) the raw draw of pat-v2, three processes
#   judges  (~1 h 25 min) 7 judges (s1_raw4 is run_1126's)
#   table   (seconds)
# All: for p in probe stats train draw judges table; do sh experiments/runs/run_1133.sh $p; done
set -ex
PHASE=${1:-probe}
SHARDS=${SHARDS:-3}
P=outputs/full-v4/best.pt
G1=outputs/pat-v1/best.pt
G=outputs/pat-v2/best.pt
D=outputs/run_1105
J=outputs/judge
CS=data/chart_stats_v2.csv
NAMES=hold_share,move_jack,move_trill,move_stair,hold_beats,chord_share
[ -f $D/raw4.path ] || { echo "no $D/raw4.path (run_1126: full-v4's raw draw)"; exit 2; }

RAW="--schedule cosine --lanes forward --refine 2 --sample-seed 0"
EV="python scripts/evaluate.py --per-song --lane-guidance 2 --lane-temp 1"
JUDGE="python scripts/judge.py --seed 1 --sr view --epochs 2 --eval-epochs --limit 120"

sharded() {              # NAME FLAGS...: SHARDS processes, merged; the run's path in $D/NAME.path
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

judged() {               # TRUNK(s1|p2) NAME VIEW RUN
    ck=$P
    [ $1 = p2 ] && ck=$G
    sfx=""
    [ $3 = full ] || sfx=_$3
    [ -f $J/$1_$2$sfx/summary.json ] && { echo "$1_$2$sfx is there"; return; }
    $JUDGE --ckpt $ck --ai $4 --view $3 --name $1_$2$sfx
}

case $PHASE in
probe)
    [ -f $G1 ] || { echo "no $G1 (run_1132): no probe"; exit 0; }
    sharded probe_lo --ckpt $G1 $RAW --n 15 --stats oracle,stair=0.05,lnlen=0.5
    sharded probe_hi --ckpt $G1 $RAW --n 15 --stats oracle,stair=0.35,lnlen=2
    ;;
stats)
    [ -f $CS ] && { echo "$CS is there"; exit 0; }
    python scripts/build_chart_stats.py --more --out $CS
    ;;
train)
    [ -f $G ] && { echo "$G is there: not trained again"; exit 0; }
    python scripts/train.py --chart-stats $CS --stats-names $NAMES --row-mask 0.5 \
        --steps 60000 --lr 3e-4 --warmup 1000 --val-every 2000 --run pat-v2 > $D/pat2_train.log 2>&1
    ;;
draw)
    [ -f $G ] || { echo "no $G: nothing to do"; exit 0; }
    sharded rawp2 --ckpt $G $RAW --n 120 --stats oracle
    ;;
judges)
    [ -f $D/rawp2.path ] || { echo "no rawp2: nothing to judge"; exit 0; }
    for t in s1 p2; do
        for m in raw4 rawp2; do
            judged $t $m full $(cat $D/$m.path)
            judged $t $m taps $(cat $D/$m.path)
        done
    done
    ;;
table)
    for t in s1 p2; do
        [ -f $J/${t}_rawp2_taps/summary.json ] || continue
        python scripts/judge_table.py $J/${t}_raw4 $J/${t}_rawp2 --out $D/results_1133_$t.md
        python scripts/judge_table.py $J/${t}_raw4_taps $J/${t}_rawp2_taps \
            --out $D/results_1133_${t}_taps.md
    done
    if [ -f $D/rawp2.path ]; then
        python scripts/diag_judge.py $(cat $D/raw4.path) $(cat $D/rawp2.path) \
            --out $D/diag_judge_1133.md
        python scripts/results_table.py $(cat $D/raw4.path) $(cat $D/rawp2.path) \
            --out $D/results_1133_runs.md
    fi
    for r in full-v4 pat-v2; do
        [ -f outputs/$r/val.csv ] && printf '%s ' $r && tail -n 1 outputs/$r/val.csv
    done | tee $D/results_1133_val.txt
    if [ -f $D/probe_hi.path ]; then         # the probe: the two stats in, out (the 15 songs)
        python - "$(cat $D/probe_lo.path)" "$(cat $D/probe_hi.path)" "$(cat $D/rawpat.path)" \
            <<'EOF' | tee $D/results_1133_probe.txt
import csv, sys
import numpy as np
runs = dict(zip(("set low", "set high", "human's"), sys.argv[1:]))
keys = [r["key"] for r in csv.DictReader(open(f"{sys.argv[1]}/per_song.csv"))]
for name, run in runs.items():
    rows = {r["key"]: r for r in csv.DictReader(open(f"{run}/per_song.csv"))}
    line = [name]
    for c in ("move_stair", "hold_beats", "move_trill", "hold_share"):
        v = np.array([float(rows[k][c] or "nan") for k in keys if k in rows])
        s = np.array([float(rows[k].get("stat_" + c) or "nan") for k in keys if k in rows])
        line.append(f"{c} in {np.nanmean(s):.3f} out {np.nanmean(v):.3f}")
    print(" | ".join(line))
EOF
    fi
    ;;
*)
    echo "phases: probe stats train draw judges table"; exit 2
    ;;
esac
