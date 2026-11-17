#!/bin/sh
# 11/11 night (~4 h 30 min, after run_1119): where the system's tell is left, and two of its
# settings. The system (run_1117's: the windows chosen among 4 by sel_holds, the hold bias 0.2,
# the lanes at temperature 1, F's passes) reads 0.743 on the full view and 0.699 on the taps
# view of the 120-song fresh judges; its choice by the long notes gave up some of the rhythm
# (the taps view 0.614 -> 0.699 against run_1118) and leans to more long notes than the human
# charts (0.271 against 0.225). All on the first 120 songs, read by fresh judges as s1_sh's.
# Why and the predictions: EXPERIMENTS 2026-11-11 (1). Writes under outputs/ only, nothing is
# deleted.
#   views  (~40 min) rhythm- and holds-view fresh judges on the system, a holds-view one on
#          run_1112
#   lt07   (~65 min) the system's draw with its lanes drawn again at temperature 0.7 (the
#          human lane statistics lie between 0.5 and 1), F's passes; full and taps judges
#   hb1    (~2 h 45 min) the system drawn with the hold bias 0.1, F's passes; full and taps
#   table  (seconds) each against the system, the views against run_1112's
# All: for p in views lt07 hb1 table; do sh experiments/runs/run_1120.sh $p; done
set -ex
PHASE=${1:-views}
SHARDS=${SHARDS:-3}
N=${N:-120}
P=outputs/full-v4/best.pt
D=outputs/run_1105
SJ=outputs/judge/sel_holds
for f in self120.path sh.path shf.path; do
    [ -f $D/$f ] || { echo "no $D/$f (run_1112, run_1117)"; exit 2; }
done
SELF=$(cat $D/self120.path)
SH=$(cat $D/sh.path)
SHF=$(cat $D/shf.path)

RESTS="--rest 1 --light 0.5 --light-alike --fill-holes 0.5"
REUSE="--carry-rhythm 0 --carry-keep --carry-holds 1 --copy-bias 0 --copy-min-lag 8 \
--copy-sections 4 --section-keep --section-back --alike 2 --alike-keep 4"
HOLDS="--hold-share oracle --hold-up --ln-tidy 0.5"
CHORDS="--strong-chords 1 --chord-cap 2"
JACKS="--untangle --untangle-move --untangle-grades --untangle-insane"
BASE="--lanes forward --refine 2 --loud-bias 0.1 --onset-bias 1 --onset-taper 2.7,5,0.35"
E="python scripts/evaluate.py --ckpt $P --per-song --lane-guidance 2 --n $N --stats oracle"
FRESH="python scripts/judge.py --ckpt $P --seed 1 --sr view --epochs 2 --eval-epochs --limit $N"

merged() {
    run=$(sed -n 's/^run //p' $D/${1}_shard0.log | head -n 1)
    [ -n "$run" ] || { echo "no run line in $D/${1}_shard0.log"; exit 2; }
    python scripts/merge_shards.py "$run"
    echo "$run" > $D/$1.path
}

case $PHASE in
views)
    $FRESH --ai $SHF --view rhythm --name s1_sh_rhythm
    $FRESH --ai $SHF --view holds --name s1_sh_holds
    $FRESH --ai $SELF --view holds --name s1_sel_holds
    ;;
lt07)
    k=0
    while [ $k -lt $SHARDS ]; do
        $E --lane-temp 0.7 --from-charts $SH --relane --lanes forward --refine 2 $RESTS \
            $REUSE $HOLDS $CHORDS $JACKS --shard $k/$SHARDS > $D/lt07_shard$k.log 2>&1 &
        k=$((k + 1))
    done
    wait
    merged lt07
    $FRESH --ai $(cat $D/lt07.path) --name s1_lt07
    $FRESH --ai $(cat $D/lt07.path) --view taps --name s1_lt07_taps
    ;;
hb1)
    [ -f $SJ/models.json ] || { echo "no $SJ/models.json (run_1117 selj)"; exit 2; }
    k=0
    while [ $k -lt $SHARDS ]; do
        $E --lane-temp 1 $BASE --schedule cosine --sample-seed 0 --select-judge $SJ \
            --select-k 4 --hold-bias 0.1 --shard $k/$SHARDS > $D/hb1_shard$k.log 2>&1 &
        k=$((k + 1))
    done
    wait
    merged hb1
    k=0
    while [ $k -lt $SHARDS ]; do
        $E --lane-temp 1 --from-charts $(cat $D/hb1.path) $RESTS $REUSE $HOLDS $CHORDS $JACKS \
            --shard $k/$SHARDS > $D/hb1f_shard$k.log 2>&1 &
        k=$((k + 1))
    done
    wait
    merged hb1f
    $FRESH --ai $(cat $D/hb1f.path) --name s1_hb1
    $FRESH --ai $(cat $D/hb1f.path) --view taps --name s1_hb1_taps
    ;;
table)
    J=outputs/judge
    for v in rhythm holds; do
        [ -f $J/s1_sh_$v/summary.json ] && [ -f $J/s1_sel_$v/summary.json ] && \
            python scripts/judge_table.py $J/s1_sel_$v $J/s1_sh_$v --out $D/results_1120_$v.md
    done
    runs=""
    for s in lt07 hb1; do
        [ -f $J/s1_$s/summary.json ] || continue
        python scripts/judge_table.py $J/s1_sh $J/s1_$s --out $D/results_1120_$s.md
        python scripts/judge_table.py $J/s1_sh_taps $J/s1_${s}_taps --out $D/results_1120_${s}_taps.md
        f=$s
        [ $s = hb1 ] && f=hb1f                    # the passed run (lt07's has its passes)
        runs="$runs $(cat $D/$f.path)"
    done
    [ -n "$runs" ] && python scripts/diag_judge.py $SHF $runs --out $D/diag_judge_1120.md
    [ -n "$runs" ] && python scripts/results_table.py $SHF $runs --out $D/results_1120.md
    ;;
*)
    echo "phases: views lt07 hb1 table"; exit 2
    ;;
esac
