#!/bin/sh
# From 11/7, one phase at a time: sh run_1107.sh <phase>, after patch 0062. Why and the
# predictions: EXPERIMENTS 2026-11-07, second entry. F's judge scores any run unlike F lower
# (M3 -0.83 though M3's own judge equals F's), the sweep had no control drawn with F's own
# settings, and the table put 60 songs beside 240. Every phase reads data/ and outputs/ and
# writes under outputs/ only; nothing is deleted, and --sample-seed 0 (the default seed) only
# names the new draws apart from the old directories. The per_song.csv files hold the playtest
# songs too: read the tables, not the rows (PT01-PT07, PT14, PT15 stay blind).
#   ctrl    (~1 h) F's settings drawn again on the sweep's 60 songs with today's code and the
#           same seed, and F's passes on it: what the six sweep runs are measured against
#   sweepj  (~40 min) F's judge again, scoring the control and the six sweep runs
#   diag    (seconds) the diagnostics judge on the sweep's 60 songs: the control, the sweep
#           (before and after F's passes) and F rescored
#   town    (~45 min) T's own judge: F's passes all off, judged by a judge that learned T
#   ladder  (~10 h) M0 and M1 drawn, their own judges, F's judge scoring M0, M1 and T
#   table   (seconds) results.md (240 songs) and results_sweep.md (the sweep's 60 songs)
set -ex
PHASE=${1:-ctrl}
P=outputs/full-v4/best.pt
T=${T:-outputs/full-v4/eval_val_songs_continue_random_T128_fwd_ref2t0.5_lbq0.1_og1_ot2.7-5-0.35_st-oracle_lg2}
TAG_F=_hu-oracle_rc0kh1_sc1_cs4kb_cp0l8_al2k4_lt0.5_fh0.5_rb1_li0.5a_ujmgi_cc2_st-oracle_lg2
F=${F:-$T$TAG_F}
D=outputs/run_1105
mkdir -p $D
[ -d "$T" ] || { echo "no T=$T: give the base directory, T=... sh run_1107.sh $PHASE"; exit 2; }
[ -d "$F" ] || { echo "no F=$F: give it, F=... sh run_1107.sh $PHASE"; exit 2; }

# the pass families of F (docs/EVALUATION.md), as in run_1105.sh
RESTS="--rest 1 --light 0.5 --light-alike --fill-holes 0.5"
REUSE="--carry-rhythm 0 --carry-keep --carry-holds 1 --copy-bias 0 --copy-min-lag 8 \
--copy-sections 4 --section-keep --section-back --alike 2 --alike-keep 4"
HOLDS="--hold-share oracle --hold-up --ln-tidy 0.5"
CHORDS="--strong-chords 1 --chord-cap 2"
JACKS="--untangle --untangle-move --untangle-grades --untangle-insane"
BASE="--lanes forward --refine 2 --loud-bias 0.1 --onset-bias 1 --onset-taper 2.7,5,0.35"
E="python scripts/evaluate.py --ckpt $P --per-song --lane-guidance 2"
J="python scripts/judge.py --ckpt $P"
latest() { ls -td outputs/full-v4/eval_*/ | head -1 | sed 's:/$::'; }
keep() { latest > $D/$1.path; cat $D/$1.path; }       # remember a run's directory by name
SWEEP="ref cosine steps256 cosinesteps256 lt0.7 lt0.85 lt1"   # run_1105.sh's names, ref first
paths() { for s in $SWEEP; do for p in "$@"; do cat $D/${p}_$s.path 2>/dev/null || true; done; done; }

case $PHASE in
ctrl)
    $E --n 60 --stats oracle $BASE --sample-seed 0; keep base_ref
    $E --n 60 --stats oracle --from-charts $(cat $D/base_ref.path) \
        $RESTS $REUSE $HOLDS $CHORDS $JACKS; keep sweep_ref
    ;;
sweepj)
    $J --ai $F --name F_sweep --score $(paths sweep)
    ;;
diag)
    [ -d "${F}_st-oracle" ] && FR=${F}_st-oracle || FR=""
    python scripts/diag_judge.py $(paths base sweep) $FR --out $D/diag_judge_sweep.md
    ;;
town)
    $J --ai $T --name T
    ;;
ladder)
    $E --n 0 --stats oracle --sample-seed 0; keep m0
    $E --n 0 --stats oracle --lanes forward --refine 2 --sample-seed 0; keep m1
    $J --ai $(cat $D/m0.path) --name M0
    $J --ai $(cat $D/m1.path) --name M1
    $J --ai $F --name F_ladder --score $(cat $D/m0.path) $(cat $D/m1.path) $T
    ;;
table)
    judges=$(ls -d outputs/judge/F_1105 outputs/judge/F_sweep outputs/judge/F_ladder \
        outputs/judge/M0 outputs/judge/M1 outputs/judge/M3 outputs/judge/T \
        outputs/judge/no_reuse 2>/dev/null || true)
    [ -d "${F}_st-oracle" ] && FR=${F}_st-oracle || FR=""
    runs="$F $FR $T $(cat $D/no_*.path $D/t3.path $D/m3.path $D/m0.path $D/m1.path 2>/dev/null || true)"
    python scripts/results_table.py $runs ${judges:+--judge $judges} --out $D/results.md
    [ -f $D/sweep_ref.path ] && python scripts/results_table.py $F $FR $(paths base sweep) \
        ${judges:+--judge $judges} --songs-of $(cat $D/sweep_ref.path) --out $D/results_sweep.md
    ;;
*)
    echo "phases: ctrl sweepj diag town ladder table"; exit 2
    ;;
esac
