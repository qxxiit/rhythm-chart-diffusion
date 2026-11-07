#!/bin/sh
# From 11/8: the cosine schedule (few cells while little is in view, the same 128 steps) on all
# 240 val songs with F's passes, judged by its own judge and by the diagnostics judge. Why and
# the predictions: EXPERIMENTS 2026-11-08. About 5 h on the MacBook. Writes under outputs/ only,
# nothing is deleted; --sample-seed 0 (the default seed) names the draw apart from the sweep's
# 60-song cosine run. The per_song.csv files hold the playtest songs too: read the tables.
#   cosine  (~4.3 h) the draw and F's passes on it
#   judge   (~45 min) its own judge
#   diag    (seconds) the diagnostics judge with F rescored
#   table   (seconds) results.md with it
# All four: for p in cosine judge diag table; do sh run_1108.sh $p; done
# SHARDS=N (default 3): the drawing and the passes run as N processes sharing the GPU, each on
# the songs whose index is K mod N (evaluate.py --shard; the charts are the one-process run's),
# then scripts/merge_shards.py puts them together. One process left the GPU mostly idle (11-08:
# 65 s a song, CPU 10%); SHARDS=1 is the old way.
set -ex
PHASE=${1:-cosine}
SHARDS=${SHARDS:-3}
P=outputs/full-v4/best.pt
T=${T:-outputs/full-v4/eval_val_songs_continue_random_T128_fwd_ref2t0.5_lbq0.1_og1_ot2.7-5-0.35_st-oracle_lg2}
TAG_F=_hu-oracle_rc0kh1_sc1_cs4kb_cp0l8_al2k4_lt0.5_fh0.5_rb1_li0.5a_ujmgi_cc2_st-oracle_lg2
F=${F:-$T$TAG_F}
D=outputs/run_1105
[ -d "$F" ] || { echo "no F=$F: give it, F=... sh run_1108.sh $PHASE"; exit 2; }
# the names evaluate.py gives these runs (as run_1107's ctrl: base_ref and sweep_ref)
COS=outputs/full-v4/eval_val_songs_continue_random_T128_cosine_fwd_ref2t0.5_lbq0.1_og1_ot2.7-5-0.35_st-oracle_lg2_s0
COSF=$COS$TAG_F

RESTS="--rest 1 --light 0.5 --light-alike --fill-holes 0.5"
REUSE="--carry-rhythm 0 --carry-keep --carry-holds 1 --copy-bias 0 --copy-min-lag 8 \
--copy-sections 4 --section-keep --section-back --alike 2 --alike-keep 4"
HOLDS="--hold-share oracle --hold-up --ln-tidy 0.5"
CHORDS="--strong-chords 1 --chord-cap 2"
JACKS="--untangle --untangle-move --untangle-grades --untangle-insane"
BASE="--lanes forward --refine 2 --loud-bias 0.1 --onset-bias 1 --onset-taper 2.7,5,0.35"
E="python scripts/evaluate.py --ckpt $P --per-song --lane-guidance 2"
J="python scripts/judge.py --ckpt $P"

case $PHASE in
cosine)
    k=0
    while [ $k -lt $SHARDS ]; do
        $E --n 0 --stats oracle $BASE --schedule cosine --sample-seed 0 --shard $k/$SHARDS &
        k=$((k + 1))
    done
    wait
    python scripts/merge_shards.py $COS
    echo $COS > $D/base_cos240.path
    k=0
    while [ $k -lt $SHARDS ]; do
        $E --n 0 --stats oracle --from-charts $COS $RESTS $REUSE $HOLDS $CHORDS $JACKS \
            --shard $k/$SHARDS &
        k=$((k + 1))
    done
    wait
    python scripts/merge_shards.py $COSF
    echo $COSF > $D/cos240.path
    ;;
judge)
    $J --ai $COSF --name cosine
    ;;
diag)
    [ -d "${F}_st-oracle" ] && FR=${F}_st-oracle || FR=$F
    python scripts/diag_judge.py $FR $COSF --out $D/diag_judge_cos240.md
    ;;
table)
    [ -d "${F}_st-oracle" ] && FR=${F}_st-oracle || FR=""
    judges=$(ls -d outputs/judge/F_1105 outputs/judge/F_ladder outputs/judge/M0 \
        outputs/judge/M1 outputs/judge/M3 outputs/judge/T outputs/judge/no_reuse \
        outputs/judge/cosine 2>/dev/null || true)
    runs="$F $FR $T $(cat $D/no_*.path $D/t3.path $D/m3.path $D/m0.path $D/m1.path 2>/dev/null || true) $COSF"
    python scripts/results_table.py $runs ${judges:+--judge $judges} --out $D/results.md
    ;;
*)
    echo "phases: cosine judge diag table"; exit 2
    ;;
esac
