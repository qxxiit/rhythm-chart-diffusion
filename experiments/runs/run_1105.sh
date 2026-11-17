#!/bin/sh
# From 11/5, one phase at a time: sh experiments/runs/run_1105.sh <phase>. Apply patches 0056-0060 first
# (git am outputs/patches/0056-*.patch ... 0060-*.patch). Why: docs/EVALUATION.md; the
# predictions: EXPERIMENTS 2026-11-05. Every phase reads data/ and outputs/ and writes under
# outputs/ only; nothing is deleted. The per_song.csv files hold the playtest songs too: read
# the tables, not the rows (PT01-PT07, PT14, PT15 stay blind).
#   meta      (~3 h) the checks before any number counts: the corruption test on the 240 val
#             human charts (and their runs for the judge), the probe's calibration with a bar
#             masked and with nothing in view, F's judge with shuffled labels, F's judge
#             scoring the corrupted charts, and what of it the diagnostics explain
#   families  (~30 min) F without each pass family, from T's saved charts
#   m3        (~3 h) M3 = F without oracle inputs (--stats sample, no hold share asked)
#   sweep     (~5 h) on the first 60 songs: cosine and 256 steps, then the lane temperature
#             (0.7, 0.85, 1), each with F's passes
#   score     (~1.5 h) one judge of F scoring families, m3 and sweep; M3's own judge
#   ladder    (~5 h) M0 and M1, their judges, and F's judge scoring M0, M1, T (the order check)
#   table     (seconds) the one results table of what exists, outputs/run_1105/results.md
set -ex
PHASE=${1:-meta}
P=outputs/full-v4/best.pt
# T: the base charts F is made from (T_ref); F: the default played on 10-12. Override T if
# its directory has another name: T=outputs/full-v4/eval_... sh experiments/runs/run_1105.sh meta
T=${T:-outputs/full-v4/eval_val_songs_continue_random_T128_fwd_ref2t0.5_lbq0.1_og1_ot2.7-5-0.35_st-oracle_lg2}
TAG_F=_hu-oracle_rc0kh1_sc1_cs4kb_cp0l8_al2k4_lt0.5_fh0.5_rb1_li0.5a_ujmgi_cc2_st-oracle_lg2
F=${F:-$T$TAG_F}
D=outputs/run_1105
mkdir -p $D
[ -d "$T" ] || { echo "no T=$T: give the base directory, T=... sh experiments/runs/run_1105.sh $PHASE"; exit 2; }
[ -d "$F" ] || { echo "no F=$F: give it, F=... sh experiments/runs/run_1105.sh $PHASE"; exit 2; }

# the pass families of F (docs/EVALUATION.md)
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

case $PHASE in
meta)
    python scripts/corrupt_test.py --split val --per-song --n 0 --write-runs
    python scripts/probe_bars.py --ckpt $P --stats oracle --out $D/probe_bars
    python scripts/probe_bars.py --ckpt $P --stats oracle --context none \
        --out $D/probe_bars_none
    $J --ai $F --name F_null --shuffle-labels
    $J --ai $F --name F_corrupt --score outputs/corrupt/val/runs/*
    python scripts/judge_vs_diag.py --judge outputs/judge/F_corrupt --run $F
    ;;
families)
    $E --n 0 --stats oracle --from-charts $T $REUSE $HOLDS $CHORDS $JACKS; keep no_rests
    $E --n 0 --stats oracle --from-charts $T $RESTS $HOLDS $CHORDS $JACKS; keep no_reuse
    $E --n 0 --stats oracle --from-charts $T $RESTS $REUSE $CHORDS $JACKS; keep no_holds
    $E --n 0 --stats oracle --from-charts $T $RESTS $REUSE $HOLDS $JACKS; keep no_chords
    $E --n 0 --stats oracle --from-charts $T $RESTS $REUSE $HOLDS $CHORDS; keep no_jacks
    ;;
m3)
    $E --n 0 --stats sample $BASE; keep t3
    $E --n 0 --stats sample --from-charts $(cat $D/t3.path) $RESTS $REUSE --ln-tidy 0.5 \
        $CHORDS $JACKS; keep m3
    ;;
sweep)
    for s in "--schedule cosine" "--steps 256" "--schedule cosine --steps 256" \
             "--lane-temp 0.7" "--lane-temp 0.85" "--lane-temp 1"; do
        name=$(echo "$s" | tr -d ' -' | sed 's/schedule//;s/lanetemp/lt/')
        $E --n 60 --stats oracle $BASE $s; keep base_$name
        $E --n 60 --stats oracle --from-charts $(cat $D/base_$name.path) \
            $RESTS $REUSE $HOLDS $CHORDS $JACKS; keep sweep_$name
    done
    ;;
score)
    $J --ai $F --name F_1105 --score $(cat $D/no_*.path $D/m3.path $D/sweep_*.path 2>/dev/null)
    $J --ai $(cat $D/m3.path) --name M3
    ;;
ladder)
    $E --n 0 --stats oracle; keep m0
    $E --n 0 --stats oracle --lanes forward --refine 2; keep m1
    $J --ai $(cat $D/m0.path) --name M0
    $J --ai $(cat $D/m1.path) --name M1
    $J --ai $F --name F_ladder --score $(cat $D/m0.path) $(cat $D/m1.path) $T
    ;;
table)
    runs="$F $T $(cat $D/*.path 2>/dev/null || true)"
    judges=$(ls -d outputs/judge/F_1105 outputs/judge/F_ladder outputs/judge/M0 \
        outputs/judge/M1 outputs/judge/M3 2>/dev/null || true)
    python scripts/results_table.py $runs ${judges:+--judge $judges} --out $D/results.md
    ;;
*)
    echo "phases: meta families m3 sweep score ladder table"; exit 2
    ;;
esac
