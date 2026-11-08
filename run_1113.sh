#!/bin/sh
# Night of 11/8-11/9 (~4 h): run_1112's choosing with 8 draws a window, on all 240 val songs,
# scored by the standard judges. The 4 draws of run_1112 cost x1.06 of the plain draw (one
# forward pass of 4) and took its fresh judge from 0.858 to 0.818 on 120 songs; with 8 draws
# the choice reaches further, and 240 songs let the standard judge (the one the playtest gate
# reads) score it. Why and the predictions: EXPERIMENTS 2026-11-08 (6). The choosing judge is
# run_1112's (outputs/judge/sel_rhythm, every song held out by one fold). Writes under outputs/
# only, nothing is deleted.
#   select  (~3.5 h) the 240 songs drawn with --select-k 8 on three processes, under the watch
#   passes  (~20 min) F's passes on them
#   judges  (~20 min) fresh standard judges (seed 1, --sr view, 2 epochs, 2 folds) on the
#           chosen run and on the cosine run: the same 96 songs
#   table   (seconds) the step, the diagnostics judge and the results table
# All: for p in select passes judges table; do sh run_1113.sh $p; done
set -ex
PHASE=${1:-select}
SHARDS=${SHARDS:-3}
P=outputs/full-v4/best.pt
D=outputs/run_1105
COS=${COS:-$(cat $D/base_cos240.path)}
COSF=${COSF:-$(cat $D/cos240.path)}
[ -d "$COS" ] && [ -d "$COSF" ] || { echo "no COS=$COS or COSF=$COSF (run_1108)"; exit 2; }
TAG_F=_hu-oracle_rc0kh1_sc1_cs4kb_cp0l8_al2k4_lt0.5_fh0.5_rb1_li0.5a_ujmgi_cc2_st-oracle_lg2
SJ=outputs/judge/sel_rhythm
SEL=$(echo $COS | sed "s/_s0\$/_sel8sel_rhythm_s0/")
SELF=$SEL$TAG_F

RESTS="--rest 1 --light 0.5 --light-alike --fill-holes 0.5"
REUSE="--carry-rhythm 0 --carry-keep --carry-holds 1 --copy-bias 0 --copy-min-lag 8 \
--copy-sections 4 --section-keep --section-back --alike 2 --alike-keep 4"
HOLDS="--hold-share oracle --hold-up --ln-tidy 0.5"
CHORDS="--strong-chords 1 --chord-cap 2"
JACKS="--untangle --untangle-move --untangle-grades --untangle-insane"
BASE="--lanes forward --refine 2 --loud-bias 0.1 --onset-bias 1 --onset-taper 2.7,5,0.35"
E="python scripts/evaluate.py --ckpt $P --per-song --lane-guidance 2"
FRESH="python scripts/judge.py --ckpt $P --seed 1 --sr view --epochs 2 --eval-epochs --max-folds 2"

case $PHASE in
select)
    [ -f $SJ/models.json ] || { echo "no $SJ/models.json (run_1112 selj)"; exit 2; }
    pids=""
    k=0
    while [ $k -lt $SHARDS ]; do
        $E --n 0 --stats oracle $BASE --schedule cosine --sample-seed 0 \
            --select-judge $SJ --select-k 8 --shard $k/$SHARDS > $D/select8_shard$k.log 2>&1 &
        pids="$pids $!"
        k=$((k + 1))
    done
    logs=$(k=0; while [ $k -lt $SHARDS ]; do printf "%s " $D/select8_shard$k.log; k=$((k + 1)); done)
    python scripts/select_watch.py --logs $logs --judge $SJ --pids $pids --after 30 \
        --min-share 0.1 &
    wait
    if [ -f $D/select8_shard0.log.STOPPED ]; then
        cat $D/select8_shard0.log.STOPPED
        echo "stopped early: the later phases have nothing to do"; exit 0
    fi
    python scripts/merge_shards.py $SEL
    echo $SEL > $D/sel8.path
    ;;
passes)
    [ -f $D/sel8.path ] || { echo "no chosen draw (stopped early?): nothing to do"; exit 0; }
    k=0
    while [ $k -lt $SHARDS ]; do
        $E --n 0 --stats oracle --from-charts $(cat $D/sel8.path) $RESTS $REUSE $HOLDS \
            $CHORDS $JACKS --shard $k/$SHARDS &
        k=$((k + 1))
    done
    wait
    python scripts/merge_shards.py $SELF
    echo $SELF > $D/self8.path
    ;;
judges)
    [ -f $D/self8.path ] || { echo "no chosen run: nothing to do"; exit 0; }
    $FRESH --ai $COSF --name s1_cos240
    $FRESH --ai $(cat $D/self8.path) --name s1_sel8
    ;;
table)
    [ -f $D/self8.path ] || { echo "no chosen run: nothing to do"; exit 0; }
    python scripts/judge_table.py outputs/judge/s1_cos240 outputs/judge/s1_sel8 \
        --out $D/results_1113.md
    python scripts/diag_judge.py $COSF $(cat $D/self8.path) --out $D/diag_judge_sel8.md
    python scripts/results_table.py $COSF $(cat $D/self8.path) --out $D/results_sel8.md
    ;;
*)
    echo "phases: select passes judges table"; exit 2
    ;;
esac
