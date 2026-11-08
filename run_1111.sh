#!/bin/sh
# From 11/8 night (~35 min): bars that sound alike, made the same rhythm more often. The cosine
# run's drawn charts (run_1108) given F's passes again with align_alike at 6 nats instead of 2,
# so a bar that sounds like others takes one of their rhythms more readily: when the music
# repeats, the AI gives the same rhythm less often than mappers do (0.354 against 0.419 of the
# pairs alike in sound inside 8 bars), most of the gap near-copies one or two rows apart. Does
# that move the judge? Why and the predictions: EXPERIMENTS 2026-11-08 (4). No drawing: the
# passes only; writes under outputs/ only.
#   alike   (~20 min) F's passes with --alike 6 on the cosine draw, three processes
#   judge   (~10 min) its own judge, as srv_full (--sr view, 2 epochs, the same 2 folds)
#   check   (seconds) the step against srv_full on the same songs; the diagnostics judge and
#           the results table against the cosine run
# All: for p in alike judge check; do sh run_1111.sh $p; done
set -ex
PHASE=${1:-alike}
SHARDS=${SHARDS:-3}
P=outputs/full-v4/best.pt
D=outputs/run_1105
COS=${COS:-$(cat $D/base_cos240.path)}
COSF=${COSF:-$(cat $D/cos240.path)}
[ -d "$COS" ] && [ -d "$COSF" ] || { echo "no COS=$COS or COSF=$COSF (run_1108)"; exit 2; }
TAG_F=_hu-oracle_rc0kh1_sc1_cs4kb_cp0l8_al2k4_lt0.5_fh0.5_rb1_li0.5a_ujmgi_cc2_st-oracle_lg2
A6=$COS$(echo $TAG_F | sed 's/_al2k4_/_al6k4_/')

RESTS="--rest 1 --light 0.5 --light-alike --fill-holes 0.5"
REUSE="--carry-rhythm 0 --carry-keep --carry-holds 1 --copy-bias 0 --copy-min-lag 8 \
--copy-sections 4 --section-keep --section-back --alike 6 --alike-keep 4"
HOLDS="--hold-share oracle --hold-up --ln-tidy 0.5"
CHORDS="--strong-chords 1 --chord-cap 2"
JACKS="--untangle --untangle-move --untangle-grades --untangle-insane"
E="python scripts/evaluate.py --ckpt $P --per-song --lane-guidance 2"

case $PHASE in
alike)
    k=0
    while [ $k -lt $SHARDS ]; do
        $E --n 0 --stats oracle --from-charts $COS $RESTS $REUSE $HOLDS $CHORDS $JACKS \
            --shard $k/$SHARDS &
        k=$((k + 1))
    done
    wait
    python scripts/merge_shards.py $A6
    echo $A6 > $D/alike6.path
    ;;
judge)
    python scripts/judge.py --ckpt $P --ai $(cat $D/alike6.path) --name srv_alike6 \
        --sr view --epochs 2 --eval-epochs --max-folds 2
    ;;
check)
    python scripts/judge_table.py outputs/judge/srv_full outputs/judge/srv_alike6 \
        --out $D/results_1111.md
    python scripts/diag_judge.py $COSF $(cat $D/alike6.path) --out $D/diag_judge_alike6.md
    python scripts/results_table.py $COSF $(cat $D/alike6.path) --out $D/results_alike6.md
    ;;
*)
    echo "phases: alike judge check"; exit 2
    ;;
esac
