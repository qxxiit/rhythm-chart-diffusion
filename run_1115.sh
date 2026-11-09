#!/bin/sh
# 11/9 (~3 h): the lanes' and the long notes' settings on run_1112's chosen draw. What is left of
# that run's tell is the lanes (taps view +0.167 over the rhythm view) and the long notes (full
# +0.117 over taps), and choosing among lane versions did not carry over (run_1114), so the
# lever is how the passes draw them. Each setting runs on the chosen draw as saved (the lanes
# drawn again with --relane, or only F's passes), so nothing is drawn again, and is read by its
# own fresh judge, paired with run_1112's s1_sel on the same 120 songs. Why and the
# predictions: EXPERIMENTS 2026-11-09 (3). Writes under outputs/ only, nothing is deleted.
#   deaf   (~13 min) a taps-view fresh judge that hears silence, on run_1112's chosen run:
#          how much of the lanes' tell is the lanes against the music
#   rt1    (~25 min) F's passes with the releases of the long notes they add drawn at
#          temperature 1 instead of 0.5 (--release-temp 1: the long notes run 0.76 beats
#          against the human 0.94); its fresh judge
#   lg0    (~75 min) the lanes drawn again without guidance towards the chart stats
#          (--lane-guidance 0 instead of 2), then F's passes; its fresh judge
#   lt1    (~75 min) the lanes drawn again at temperature 1 instead of 0.5; its fresh judge
#   table  (seconds) each setting against run_1112, the diagnostics, the results table
#   nohu   (~25 min, not in the chain) F's passes without the long notes topped up to the
#          human share (no --hold-share oracle --hold-up): the model's own long notes only
# All: for p in deaf rt1 lg0 lt1 table; do sh run_1115.sh $p; done
set -ex
PHASE=${1:-deaf}
SHARDS=${SHARDS:-3}
N=${N:-120}
P=outputs/full-v4/best.pt
D=outputs/run_1105
[ -f $D/sel120.path ] && [ -f $D/self120.path ] || { echo "no run_1112 chosen draw"; exit 2; }
SEL=$(cat $D/sel120.path)
SELF=$(cat $D/self120.path)

RESTS="--rest 1 --light 0.5 --light-alike --fill-holes 0.5"
REUSE="--carry-rhythm 0 --carry-keep --carry-holds 1 --copy-bias 0 --copy-min-lag 8 \
--copy-sections 4 --section-keep --section-back --alike 2 --alike-keep 4"
HOLDS="--hold-share oracle --hold-up --ln-tidy 0.5"
CHORDS="--strong-chords 1 --chord-cap 2"
JACKS="--untangle --untangle-move --untangle-grades --untangle-insane"
RELANE="--relane --lanes forward --refine 2"
E="python scripts/evaluate.py --ckpt $P --per-song --lane-guidance 2 --n $N --stats oracle \
--from-charts $SEL"
FRESH="python scripts/judge.py --ckpt $P --seed 1 --sr view --epochs 2 --eval-epochs --limit $N"

# shards NAME ARGS...: the run on $SHARDS processes, merged; its directory in $D/NAME.path
shards() {
    name=$1
    shift
    k=0
    while [ $k -lt $SHARDS ]; do
        $E "$@" --shard $k/$SHARDS > $D/${name}_shard$k.log 2>&1 &
        k=$((k + 1))
    done
    wait
    run=$(sed -n 's/^run //p' $D/${name}_shard0.log | head -n 1)
    [ -n "$run" ] || { echo "no run line in $D/${name}_shard0.log"; exit 2; }
    python scripts/merge_shards.py "$run"
    echo "$run" > $D/$name.path
}

case $PHASE in
deaf)
    $FRESH --ai $SELF --view taps --no-audio --name s1_sel_taps_deaf
    ;;
rt1)
    shards rt1 $RESTS $REUSE $HOLDS --release-temp 1 $CHORDS $JACKS
    $FRESH --ai $(cat $D/rt1.path) --name s1_rt1
    ;;
nohu)
    shards nohu $RESTS $REUSE --ln-tidy 0.5 $CHORDS $JACKS
    $FRESH --ai $(cat $D/nohu.path) --name s1_nohu
    ;;
lg0)
    shards lg0 $RELANE --lane-guidance 0 $RESTS $REUSE $HOLDS $CHORDS $JACKS
    $FRESH --ai $(cat $D/lg0.path) --name s1_lg0
    ;;
lt1)
    shards lt1 $RELANE --lane-temp 1 $RESTS $REUSE $HOLDS $CHORDS $JACKS
    $FRESH --ai $(cat $D/lt1.path) --name s1_lt1
    ;;
table)
    J=outputs/judge
    [ -f $J/s1_sel_taps_deaf/summary.json ] && python scripts/judge_table.py \
        $J/s1_sel_rhythm $J/s1_sel_taps_deaf $J/s1_sel_taps --out $D/results_1115_deaf.md
    runs=""
    for s in rt1 lg0 lt1 nohu; do
        [ -f $J/s1_$s/summary.json ] || continue
        python scripts/judge_table.py $J/s1_sel $J/s1_$s --out $D/results_1115_$s.md
        python scripts/diag_judge.py $SELF $(cat $D/$s.path) --out $D/diag_judge_$s.md
        runs="$runs $(cat $D/$s.path)"
    done
    [ -n "$runs" ] && python scripts/results_table.py $SELF $runs --out $D/results_1115.md
    ;;
*)
    echo "phases: deaf rt1 lg0 lt1 table (nohu: not in the chain)"; exit 2
    ;;
esac
