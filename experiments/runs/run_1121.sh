#!/bin/sh
# 11/14 (~6 h 30 min): the windows chosen by the rhythm judge and the holds judge together. The
# system chooses by the holds view, and gave some of the rhythm away for the long notes (the
# rhythm view 0.535 -> 0.603, the taps view 0.614 -> 0.699 against run_1118). Here each window's
# 4 draws are scored by sel_rhythm and sel_holds, each over its own gap, summed
# (evaluate.py --select-judge J1 J2, src/models/select.combined_scorer); the rest as the system
# (the hold bias 0.2, the lanes at temperature 1, F's passes). On all 240 val songs, and read
# against the system by fresh 5-fold judges on the same 240 songs (paired), the full view and
# the rhythm view. Why and the predictions: EXPERIMENTS 2026-11-14 (1). Writes under outputs/
# only, nothing is deleted.
#   draw    (~4 h) the 240 songs, three processes
#   passes  (~25 min) F's passes
#   judges  (~2 h) fresh full- and rhythm-view judges (seed 1, --sr view, 2 epochs, 5 folds)
#           on the system's 240 songs and on this run's
#   table   (seconds) the steps, the diagnostics, the results table
# All: for p in draw passes judges table; do sh experiments/runs/run_1121.sh $p; done
set -ex
PHASE=${1:-draw}
SHARDS=${SHARDS:-3}
P=outputs/full-v4/best.pt
D=outputs/run_1105
for f in sysf.path; do
    [ -f $D/$f ] || { echo "no $D/$f (run_1119)"; exit 2; }
done
for j in sel_rhythm sel_holds; do
    [ -f outputs/judge/$j/models.json ] || { echo "no outputs/judge/$j (run_1112, run_1117)"; exit 2; }
done
SYSF=$(cat $D/sysf.path)

RESTS="--rest 1 --light 0.5 --light-alike --fill-holes 0.5"
REUSE="--carry-rhythm 0 --carry-keep --carry-holds 1 --copy-bias 0 --copy-min-lag 8 \
--copy-sections 4 --section-keep --section-back --alike 2 --alike-keep 4"
HOLDS="--hold-share oracle --hold-up --ln-tidy 0.5"
CHORDS="--strong-chords 1 --chord-cap 2"
JACKS="--untangle --untangle-move --untangle-grades --untangle-insane"
BASE="--lanes forward --refine 2 --loud-bias 0.1 --onset-bias 1 --onset-taper 2.7,5,0.35"
E="python scripts/evaluate.py --ckpt $P --per-song --lane-guidance 2 --lane-temp 1 --n 240 \
--stats oracle"
FRESH="python scripts/judge.py --ckpt $P --seed 1 --sr view --epochs 2 --eval-epochs"

merged() {
    run=$(sed -n 's/^run //p' $D/${1}_shard0.log | head -n 1)
    [ -n "$run" ] || { echo "no run line in $D/${1}_shard0.log"; exit 2; }
    python scripts/merge_shards.py "$run"
    echo "$run" > $D/$1.path
}

case $PHASE in
draw)
    k=0
    while [ $k -lt $SHARDS ]; do
        $E $BASE --schedule cosine --sample-seed 0 \
            --select-judge outputs/judge/sel_rhythm outputs/judge/sel_holds --select-k 4 \
            --hold-bias 0.2 --shard $k/$SHARDS > $D/two_shard$k.log 2>&1 &
        k=$((k + 1))
    done
    wait
    merged two
    ;;
passes)
    [ -f $D/two.path ] || { echo "no draw: nothing to do"; exit 0; }
    k=0
    while [ $k -lt $SHARDS ]; do
        $E --from-charts $(cat $D/two.path) $RESTS $REUSE $HOLDS $CHORDS $JACKS \
            --shard $k/$SHARDS > $D/twof_shard$k.log 2>&1 &
        k=$((k + 1))
    done
    wait
    merged twof
    ;;
judges)
    [ -f $D/twof.path ] || { echo "no run: nothing to do"; exit 0; }
    $FRESH --ai $SYSF --name s1_sys240f5
    $FRESH --ai $(cat $D/twof.path) --name s1_two240f5
    $FRESH --ai $SYSF --view rhythm --name s1_sys240f5_rhythm
    $FRESH --ai $(cat $D/twof.path) --view rhythm --name s1_two240f5_rhythm
    ;;
table)
    [ -f $D/twof.path ] || { echo "no run: nothing to do"; exit 0; }
    J=outputs/judge
    python scripts/judge_table.py $J/s1_sys240f5 $J/s1_two240f5 --out $D/results_1121.md
    python scripts/judge_table.py $J/s1_sys240f5_rhythm $J/s1_two240f5_rhythm \
        --out $D/results_1121_rhythm.md
    python scripts/diag_judge.py $SYSF $(cat $D/twof.path) --out $D/diag_judge_two.md
    python scripts/results_table.py $SYSF $(cat $D/twof.path) --out $D/results_two.md
    ;;
*)
    echo "phases: draw passes judges table"; exit 2
    ;;
esac
