#!/bin/sh
# Overnight 11/8-11/9 (~4 h, less if stopped): charts chosen by a judge. The tell is spread over
# the 8-bar window (EXPERIMENTS 2026-11-08 (4)), so a judge that sees the window chooses: every
# window of the cosine draw is drawn 4 times (one forward pass of 4) and the song's held-out
# fold judge keeps the draw it finds most human (evaluate.py --select-judge). The judge that
# chooses never scores the result: fresh judges with another seed and fold split on the chosen
# run and on the cosine run, the same 120 songs (the first 120 val songs). Why and the
# predictions: EXPERIMENTS 2026-11-08 (5). Writes under outputs/ only, nothing is deleted.
#   selj    (~25 min) the choosing judge: the rhythm view of the cosine draw before F's passes
#           (what a window's drawing decides; the lane passes after it choose the lanes
#           again), the human chart's SR (what a draw is asked for), all 5 folds kept
#   select  (~2.5 h) the first 120 val songs drawn so, three processes; select_watch.py stops
#           them after 30 songs if the choosing moves under a tenth of the judge's gap
#   passes  (~10 min) F's passes on the chosen draw
#   judges  (~50 min) fresh judges (seed 1, all 5 folds of the 120 songs): full view and
#           rhythm view, on the chosen run and on the cosine run's same songs
#   table   (seconds) the steps (chosen minus cosine) and the diagnostics
# All: for p in selj select passes judges table; do sh run_1112.sh $p; done
set -ex
PHASE=${1:-selj}
SHARDS=${SHARDS:-3}
N=${N:-120}
P=outputs/full-v4/best.pt
D=outputs/run_1105
COS=${COS:-$(cat $D/base_cos240.path)}
COSF=${COSF:-$(cat $D/cos240.path)}
[ -d "$COS" ] && [ -d "$COSF" ] || { echo "no COS=$COS or COSF=$COSF (run_1108)"; exit 2; }
TAG_F=_hu-oracle_rc0kh1_sc1_cs4kb_cp0l8_al2k4_lt0.5_fh0.5_rb1_li0.5a_ujmgi_cc2_st-oracle_lg2
SJ=outputs/judge/sel_rhythm
# evaluate.py's name for the chosen draw: the cosine draw's with _sel4sel_rhythm before _s0
# and --n's songs
SEL=$(echo $COS | sed "s/_s0\$/_sel4sel_rhythm_s0/")
SELF=$SEL$TAG_F

RESTS="--rest 1 --light 0.5 --light-alike --fill-holes 0.5"
REUSE="--carry-rhythm 0 --carry-keep --carry-holds 1 --copy-bias 0 --copy-min-lag 8 \
--copy-sections 4 --section-keep --section-back --alike 2 --alike-keep 4"
HOLDS="--hold-share oracle --hold-up --ln-tidy 0.5"
CHORDS="--strong-chords 1 --chord-cap 2"
JACKS="--untangle --untangle-move --untangle-grades --untangle-insane"
BASE="--lanes forward --refine 2 --loud-bias 0.1 --onset-bias 1 --onset-taper 2.7,5,0.35"
E="python scripts/evaluate.py --ckpt $P --per-song --lane-guidance 2"
FRESH="python scripts/judge.py --ckpt $P --seed 1 --sr view --epochs 2 --eval-epochs --limit $N"

case $PHASE in
selj)
    python scripts/judge.py --ckpt $P --ai $COS --view rhythm --sr human --epochs 2 \
        --eval-epochs --save-models --name sel_rhythm
    ;;
select)
    [ -f $SJ/models.json ] || { echo "no $SJ/models.json: run selj first"; exit 2; }
    pids=""
    k=0
    while [ $k -lt $SHARDS ]; do
        $E --n $N --stats oracle $BASE --schedule cosine --sample-seed 0 \
            --select-judge $SJ --select-k 4 --shard $k/$SHARDS > $D/select_shard$k.log 2>&1 &
        pids="$pids $!"
        k=$((k + 1))
    done
    logs=$(k=0; while [ $k -lt $SHARDS ]; do printf "%s " $D/select_shard$k.log; k=$((k + 1)); done)
    python scripts/select_watch.py --logs $logs --judge $SJ --pids $pids --after 30 \
        --min-share 0.1 &
    wait
    if [ -f $D/select_shard0.log.STOPPED ]; then
        cat $D/select_shard0.log.STOPPED
        echo "stopped early: the later phases have nothing to do"; exit 0
    fi
    python scripts/merge_shards.py $SEL
    echo $SEL > $D/sel120.path
    ;;
passes)
    [ -f $D/sel120.path ] || { echo "no chosen draw (stopped early?): nothing to do"; exit 0; }
    k=0
    while [ $k -lt $SHARDS ]; do
        $E --n $N --stats oracle --from-charts $(cat $D/sel120.path) $RESTS $REUSE $HOLDS \
            $CHORDS $JACKS --shard $k/$SHARDS &
        k=$((k + 1))
    done
    wait
    python scripts/merge_shards.py $SELF
    echo $SELF > $D/self120.path
    ;;
judges)
    [ -f $D/self120.path ] || { echo "no chosen run: nothing to do"; exit 0; }
    $FRESH --ai $COSF --name s1_cos
    $FRESH --ai $(cat $D/self120.path) --name s1_sel
    $FRESH --ai $COSF --view rhythm --name s1_cos_rhythm
    $FRESH --ai $(cat $D/self120.path) --view rhythm --name s1_sel_rhythm
    ;;
table)
    [ -f $D/self120.path ] || { echo "no chosen run: nothing to do"; exit 0; }
    python scripts/judge_table.py outputs/judge/s1_cos outputs/judge/s1_sel \
        --out $D/results_1112.md
    python scripts/judge_table.py outputs/judge/s1_cos_rhythm outputs/judge/s1_sel_rhythm \
        --out $D/results_1112_rhythm.md
    python scripts/diag_judge.py $COSF $(cat $D/self120.path) --out $D/diag_judge_sel.md
    python scripts/results_table.py $COSF $(cat $D/self120.path) --songs-of \
        $(cat $D/self120.path) --out $D/results_sel.md
    ;;
*)
    echo "phases: selj select passes judges table"; exit 2
    ;;
esac
