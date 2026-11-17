#!/bin/sh
# 11/9 evening (~2 h 10 min): the long notes, drawn by the model. At lane temperature 1 the
# lanes' tell fell (taps view 0.701 -> 0.607, -0.095 [-0.165, -0.031]) but the full view
# hardly moved (0.818 -> 0.807): the long notes now carry it (full over taps +0.200). F's
# top-up adds 181 long notes a song, shorter than the model's own (0.52 against 0.63 beats),
# so here the model draws more of them itself (--hold-bias 0.5: about 1.6 times as many
# start, from 0.126 of the onsets towards the human 0.225) and the top-up fills only what is
# left; the lanes at temperature 1. run_1112's draw again otherwise (the windows chosen among
# 4 by sel_rhythm, the same seeds), then F's passes. Why and the predictions: EXPERIMENTS
# 2026-11-09 (5). Writes under outputs/ only, nothing is deleted.
#   draw    (~1 h 30 min) the 120 songs, three processes, under select_watch.py
#   passes  (~10 min) F's passes (their lanes at temperature 1 too)
#   judges  (~25 min) fresh full-view and taps-view judges (seed 1, --sr view, 2 epochs,
#           5 folds of the 120 songs), as s1_sel / s1_lt1 and their taps views
#   table   (seconds) the steps run_1112 -> lanes at 1 -> long notes drawn, both views
# All: for p in draw passes judges table; do sh experiments/runs/run_1116.sh $p; done
set -ex
PHASE=${1:-draw}
SHARDS=${SHARDS:-3}
N=${N:-120}
P=outputs/full-v4/best.pt
D=outputs/run_1105
SJ=outputs/judge/sel_rhythm
for f in self120.path lt1.path; do
    [ -f $D/$f ] || { echo "no $D/$f (run_1112, run_1115)"; exit 2; }
done
SELF=$(cat $D/self120.path)
LT1=$(cat $D/lt1.path)

RESTS="--rest 1 --light 0.5 --light-alike --fill-holes 0.5"
REUSE="--carry-rhythm 0 --carry-keep --carry-holds 1 --copy-bias 0 --copy-min-lag 8 \
--copy-sections 4 --section-keep --section-back --alike 2 --alike-keep 4"
HOLDS="--hold-share oracle --hold-up --ln-tidy 0.5"
CHORDS="--strong-chords 1 --chord-cap 2"
JACKS="--untangle --untangle-move --untangle-grades --untangle-insane"
BASE="--lanes forward --refine 2 --loud-bias 0.1 --onset-bias 1 --onset-taper 2.7,5,0.35"
E="python scripts/evaluate.py --ckpt $P --per-song --lane-guidance 2 --lane-temp 1 --n $N \
--stats oracle"
FRESH="python scripts/judge.py --ckpt $P --seed 1 --sr view --epochs 2 --eval-epochs --limit $N"

# merged NAME: the run of $D/NAME_shard*.log merged, its directory in $D/NAME.path
merged() {
    run=$(sed -n 's/^run //p' $D/${1}_shard0.log | head -n 1)
    [ -n "$run" ] || { echo "no run line in $D/${1}_shard0.log"; exit 2; }
    python scripts/merge_shards.py "$run"
    echo "$run" > $D/$1.path
}

case $PHASE in
draw)
    [ -f $SJ/models.json ] || { echo "no $SJ/models.json (run_1112 selj)"; exit 2; }
    pids=""
    k=0
    while [ $k -lt $SHARDS ]; do
        $E $BASE --schedule cosine --sample-seed 0 --select-judge $SJ --select-k 4 \
            --hold-bias 0.5 --shard $k/$SHARDS > $D/hb_shard$k.log 2>&1 &
        pids="$pids $!"
        k=$((k + 1))
    done
    logs=$(k=0; while [ $k -lt $SHARDS ]; do printf "%s " $D/hb_shard$k.log; k=$((k + 1)); done)
    python scripts/select_watch.py --logs $logs --judge $SJ --pids $pids --after 30 \
        --min-share 0.1 &
    wait
    if [ -f $D/hb_shard0.log.STOPPED ]; then
        cat $D/hb_shard0.log.STOPPED
        echo "stopped early: the later phases have nothing to do"; exit 0
    fi
    merged hb
    ;;
passes)
    [ -f $D/hb.path ] || { echo "no draw (stopped early?): nothing to do"; exit 0; }
    k=0
    while [ $k -lt $SHARDS ]; do
        $E --from-charts $(cat $D/hb.path) $RESTS $REUSE $HOLDS $CHORDS $JACKS \
            --shard $k/$SHARDS > $D/hbf_shard$k.log 2>&1 &
        k=$((k + 1))
    done
    wait
    merged hbf
    ;;
judges)
    [ -f $D/hbf.path ] || { echo "no run: nothing to do"; exit 0; }
    $FRESH --ai $(cat $D/hbf.path) --name s1_hb
    $FRESH --ai $(cat $D/hbf.path) --view taps --name s1_hb_taps
    ;;
table)
    [ -f $D/hbf.path ] || { echo "no run: nothing to do"; exit 0; }
    J=outputs/judge
    python scripts/judge_table.py $J/s1_sel $J/s1_lt1 $J/s1_hb --out $D/results_1116.md
    python scripts/judge_table.py $J/s1_sel_taps $J/s1_lt1_taps $J/s1_hb_taps \
        --out $D/results_1116_taps.md
    python scripts/diag_judge.py $SELF $LT1 $(cat $D/hbf.path) --out $D/diag_judge_hb.md
    python scripts/results_table.py $SELF $LT1 $(cat $D/hbf.path) --out $D/results_hb.md
    ;;
*)
    echo "phases: draw passes judges table"; exit 2
    ;;
esac
