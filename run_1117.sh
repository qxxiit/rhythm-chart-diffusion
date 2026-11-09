#!/bin/sh
# The windows chosen by a judge that sees the long notes (~2 h 30 min). run_1112's choosing
# judge saw the rhythm view (long notes as taps), and its choice carried over to what it saw
# (the rhythm view 0.664 -> 0.535) but not to the long notes, which now carry the tell. Here
# the choosing judge sees the holds view (judge.view_tokens "holds": the rhythm view's random
# lanes, every long note kept whole in its lane), so a window's draws are chosen
# by their rhythm and their long notes together; the lanes at temperature 1, the hold bias
# $HB (0.2 by default, run_1118's; HB=0 for none). Why: EXPERIMENTS 2026-11-09 (6); the
# predictions to be written before it runs.
# Writes under outputs/ only, nothing is deleted.
#   selj    (~25 min) the choosing judge sel_holds: the holds view of the cosine draw before
#           F's passes, the human chart's SR, all 5 folds kept
#   draw    (~1 h 30 min) the 120 songs, the windows chosen among 4 by it, under the watch
#   passes  (~10 min) F's passes (their lanes at temperature 1)
#   judges  (~25 min) fresh full-view and taps-view judges as run_1116's
#   table   (seconds) against run_1118 (s1_hb2) and run_1112 (s1_sel), both views
# All: for p in selj draw passes judges table; do sh run_1117.sh $p; done
set -ex
PHASE=${1:-selj}
SHARDS=${SHARDS:-3}
N=${N:-120}
HB=${HB:-0.2}
P=outputs/full-v4/best.pt
D=outputs/run_1105
COS=${COS:-$(cat $D/base_cos240.path)}
[ -d "$COS" ] || { echo "no COS=$COS (run_1108)"; exit 2; }
SJ=outputs/judge/sel_holds

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

merged() {
    run=$(sed -n 's/^run //p' $D/${1}_shard0.log | head -n 1)
    [ -n "$run" ] || { echo "no run line in $D/${1}_shard0.log"; exit 2; }
    python scripts/merge_shards.py "$run"
    echo "$run" > $D/$1.path
}

case $PHASE in
selj)
    python scripts/judge.py --ckpt $P --ai $COS --view holds --sr human --epochs 2 \
        --eval-epochs --save-models --name sel_holds
    ;;
draw)
    [ -f $SJ/models.json ] || { echo "no $SJ/models.json: run selj first"; exit 2; }
    bias=""
    [ "$HB" != "0" ] && bias="--hold-bias $HB"
    pids=""
    k=0
    while [ $k -lt $SHARDS ]; do
        $E $BASE --schedule cosine --sample-seed 0 --select-judge $SJ --select-k 4 $bias \
            --shard $k/$SHARDS > $D/sh_shard$k.log 2>&1 &
        pids="$pids $!"
        k=$((k + 1))
    done
    logs=$(k=0; while [ $k -lt $SHARDS ]; do printf "%s " $D/sh_shard$k.log; k=$((k + 1)); done)
    python scripts/select_watch.py --logs $logs --judge $SJ --pids $pids --after 30 \
        --min-share 0.1 &
    wait
    if [ -f $D/sh_shard0.log.STOPPED ]; then
        cat $D/sh_shard0.log.STOPPED
        echo "stopped early: the later phases have nothing to do"; exit 0
    fi
    merged sh
    ;;
passes)
    [ -f $D/sh.path ] || { echo "no draw (stopped early?): nothing to do"; exit 0; }
    k=0
    while [ $k -lt $SHARDS ]; do
        $E --from-charts $(cat $D/sh.path) $RESTS $REUSE $HOLDS $CHORDS $JACKS \
            --shard $k/$SHARDS > $D/shf_shard$k.log 2>&1 &
        k=$((k + 1))
    done
    wait
    merged shf
    ;;
judges)
    [ -f $D/shf.path ] || { echo "no run: nothing to do"; exit 0; }
    $FRESH --ai $(cat $D/shf.path) --name s1_sh
    $FRESH --ai $(cat $D/shf.path) --view taps --name s1_sh_taps
    ;;
table)
    [ -f $D/shf.path ] || { echo "no run: nothing to do"; exit 0; }
    J=outputs/judge
    python scripts/judge_table.py $J/s1_sel $J/s1_hb2 $J/s1_sh --out $D/results_1117.md
    python scripts/judge_table.py $J/s1_sel_taps $J/s1_hb2_taps $J/s1_sh_taps \
        --out $D/results_1117_taps.md
    python scripts/diag_judge.py $(cat $D/self120.path) $(cat $D/hb2f.path) $(cat $D/shf.path) \
        --out $D/diag_judge_sh.md
    python scripts/results_table.py $(cat $D/self120.path) $(cat $D/hb2f.path) \
        $(cat $D/shf.path) --out $D/results_sh.md
    ;;
*)
    echo "phases: selj draw passes judges table"; exit 2
    ;;
esac
