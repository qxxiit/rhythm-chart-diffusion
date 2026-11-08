#!/bin/sh
# 11/9 (~1.7 h, less if stopped): the lanes chosen by a judge too. run_1112's choice moved what
# its judge sees (the rhythm view: 0.664 -> 0.535) but the full view less (0.858 -> 0.818): the
# tell left is in the lanes and the long notes, which that judge never saw. Here the lane
# passes draw 4 versions of the lanes at once (one forward pass of 4 a row) and every span
# keeps the version a full-view judge finds most human; where the versions change, 2 beats
# on each side are drawn again with both sides in view, and the spans end mid-window, so the
# judges that score the charts see the seams (sampler.choose_lanes). The lane passes choose
# every lane from the rhythm, chord sizes and holds alone, so they run on run_1112's chosen
# draw as saved (evaluate.py --from-charts, sampler.relane_song): the songs are those of the
# run_1112 draw with lanes as its passes would have drawn them, and the hour of drawing is not
# done again. Why and the predictions: EXPERIMENTS 2026-11-09 (1). Writes under outputs/ only,
# nothing is deleted.
#   selj    (~25 min) the choosing judge: the full view of the cosine draw before F's passes,
#           the human chart's SR, all 5 folds kept
#   lanes   (~1 h) the 120 songs of run_1112's chosen draw, their lanes chosen so, then F's
#           passes, three processes; select_watch.py stops them after 30 songs if the
#           choosing moves under a tenth of the judge's gap
#   judges  (~13 min) a fresh judge as run_1112's s1_sel (seed 1, --sr view, all 5 folds of
#           the 120 songs) on the run with its lanes chosen
#   table   (seconds) the steps (cosine, windows chosen, lanes chosen too) and the diagnostics
# All: for p in selj lanes judges table; do sh run_1114.sh $p; done
set -ex
PHASE=${1:-selj}
SHARDS=${SHARDS:-3}
N=${N:-120}
P=outputs/full-v4/best.pt
D=outputs/run_1105
COS=${COS:-$(cat $D/base_cos240.path)}
COSF=${COSF:-$(cat $D/cos240.path)}
[ -d "$COS" ] && [ -d "$COSF" ] || { echo "no COS=$COS or COSF=$COSF (run_1108)"; exit 2; }
[ -f $D/sel120.path ] && [ -f $D/self120.path ] || { echo "no run_1112 chosen draw"; exit 2; }
SEL=$(cat $D/sel120.path)
TAG_F=_hu-oracle_rc0kh1_sc1_cs4kb_cp0l8_al2k4_lt0.5_fh0.5_rb1_li0.5a_ujmgi_cc2_st-oracle_lg2
LJ=outputs/judge/sel_lanes
# evaluate.py's name for it: the saved draw's, then _lsel4sel_lanes and F's tag
SELL=${SEL}_lsel4sel_lanes$TAG_F

RESTS="--rest 1 --light 0.5 --light-alike --fill-holes 0.5"
REUSE="--carry-rhythm 0 --carry-keep --carry-holds 1 --copy-bias 0 --copy-min-lag 8 \
--copy-sections 4 --section-keep --section-back --alike 2 --alike-keep 4"
HOLDS="--hold-share oracle --hold-up --ln-tidy 0.5"
CHORDS="--strong-chords 1 --chord-cap 2"
JACKS="--untangle --untangle-move --untangle-grades --untangle-insane"
LANES="--lanes forward --refine 2 --lane-select-judge $LJ --lane-select-k 4"
E="python scripts/evaluate.py --ckpt $P --per-song --lane-guidance 2"
FRESH="python scripts/judge.py --ckpt $P --seed 1 --sr view --epochs 2 --eval-epochs --limit $N"

case $PHASE in
selj)
    python scripts/judge.py --ckpt $P --ai $COS --view full --sr human --epochs 2 \
        --eval-epochs --save-models --name sel_lanes
    ;;
lanes)
    [ -f $LJ/models.json ] || { echo "no $LJ/models.json: run selj first"; exit 2; }
    pids=""
    k=0
    while [ $k -lt $SHARDS ]; do
        $E --n $N --stats oracle --from-charts $SEL $LANES $RESTS $REUSE $HOLDS $CHORDS \
            $JACKS --shard $k/$SHARDS > $D/lanes_shard$k.log 2>&1 &
        pids="$pids $!"
        k=$((k + 1))
    done
    logs=$(k=0; while [ $k -lt $SHARDS ]; do printf "%s " $D/lanes_shard$k.log; k=$((k + 1)); done)
    python scripts/select_watch.py --logs $logs --judge $LJ --pids $pids --after 30 \
        --min-share 0.1 --field lane &
    wait
    if [ -f $D/lanes_shard0.log.STOPPED ]; then
        cat $D/lanes_shard0.log.STOPPED
        echo "stopped early: the later phases have nothing to do"; exit 0
    fi
    python scripts/merge_shards.py $SELL
    echo $SELL > $D/sell120.path
    ;;
judges)
    [ -f $D/sell120.path ] || { echo "no run with its lanes chosen: nothing to do"; exit 0; }
    $FRESH --ai $(cat $D/sell120.path) --name s1_lanes
    ;;
control)
    # not in the chain: the rhythm view, which the lanes never reach (expected: no step)
    [ -f $D/sell120.path ] || { echo "no run with its lanes chosen: nothing to do"; exit 0; }
    $FRESH --ai $(cat $D/sell120.path) --view rhythm --name s1_lanes_rhythm
    python scripts/judge_table.py outputs/judge/s1_sel_rhythm outputs/judge/s1_lanes_rhythm \
        --out $D/results_1114_rhythm.md
    ;;
table)
    [ -f $D/sell120.path ] || { echo "no run with its lanes chosen: nothing to do"; exit 0; }
    python scripts/judge_table.py outputs/judge/s1_cos outputs/judge/s1_sel \
        outputs/judge/s1_lanes --out $D/results_1114.md
    python scripts/diag_judge.py $(cat $D/self120.path) $(cat $D/sell120.path) \
        --out $D/diag_judge_lanes.md
    python scripts/results_table.py $(cat $D/self120.path) $(cat $D/sell120.path) \
        --out $D/results_lanes.md
    ;;
*)
    echo "phases: selj lanes judges table (control: the rhythm view, not in the chain)"; exit 2
    ;;
esac
