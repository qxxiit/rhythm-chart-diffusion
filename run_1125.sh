#!/bin/sh
# 11/16 (~7 h 30 min): the stems learned from the start, then the beat answers made stricter.
# (1) run_1123's stems-v1 widened full-v4 with zero weights and trained 20k steps more: most
# of its val_ce gain was the training itself (the control ft-v1 0.0726, stems-v1 0.0721,
# full-v4 0.0740), and its charts were no better. Here stems-v2 is trained from the start on
# full-v4's recipe exactly (60k steps, lr 3e-4, warmup 1000, batch 16, row-mask 0.5, the chart
# stats, seed 0), --stems the one difference, so full-v4 is its control; then the system's
# draw with it on the first 120 val songs (the windows chosen among 4 by sel_holds, hold bias
# 0.2, lanes at temperature 1, F's passes) against the system there (s1_sh, s1_sh_rhythm).
# (2) run_1124's answers at bias 0 matched the human beat-to-beat rhythm and mirror rates but
# the rhythm judge liked them less (+0.044); here at bias -1 and -2 (an answer only where the
# model clearly prefers it), with the rest guard, against the rest guard alone (s1_rgf).
# Why and the predictions: EXPERIMENTS 2026-11-16 (1). Writes under outputs/ only, nothing
# is deleted.
#   train    (~3 h 30 min) outputs/stems-v2
#   draw     (~1 h 30 min) the 120 songs with it, three processes
#   passes   (~10 min) F's passes
#   judges   (~25 min) fresh full- and rhythm-view judges as s1_sh's
#   answers  (~1 h 50 min) F + the rest guard + --answer -1 / -2 on the system's draw (240),
#            fresh judges on each
#   table    (seconds)
# All: for p in train draw passes judges answers table; do sh run_1125.sh $p; done
set -ex
PHASE=${1:-train}
SHARDS=${SHARDS:-3}
P=outputs/full-v4/best.pt
G=outputs/stems-v2/best.pt
D=outputs/run_1105
SJ=outputs/judge/sel_holds
SHF=$(cat $D/shf.path)
SYS=$(cat $D/sys.path)

RESTS="--rest 1 --light 0.5 --light-alike --fill-holes 0.5"
REUSE="--carry-rhythm 0 --carry-keep --carry-holds 1 --copy-bias 0 --copy-min-lag 8 \
--copy-sections 4 --section-keep --section-back --alike 2 --alike-keep 4"
HOLDS="--hold-share oracle --hold-up --ln-tidy 0.5"
CHORDS="--strong-chords 1 --chord-cap 2"
JACKS="--untangle --untangle-move --untangle-grades --untangle-insane"
BASE="--lanes forward --refine 2 --loud-bias 0.1 --onset-bias 1 --onset-taper 2.7,5,0.35"
EV="python scripts/evaluate.py --per-song --lane-guidance 2 --lane-temp 1 --stats oracle"
FRESH="python scripts/judge.py --ckpt $P --seed 1 --sr view --epochs 2 --eval-epochs --limit 120"

merged() {
    run=$(sed -n 's/^run //p' $D/${1}_shard0.log | head -n 1)
    [ -n "$run" ] || { echo "no run line in $D/${1}_shard0.log"; exit 2; }
    python scripts/merge_shards.py "$run"
    echo "$run" > $D/$1.path
}

case $PHASE in
train)
    [ -f $G ] && { echo "$G is there: not trained again"; exit 0; }
    python scripts/train.py --stems --chart-stats data/chart_stats.csv --row-mask 0.5 \
        --steps 60000 --val-every 2000 --run stems-v2 > $D/stems2_train.log 2>&1
    ;;
draw)
    [ -f $G ] || { echo "no $G: nothing to do"; exit 0; }
    k=0
    while [ $k -lt $SHARDS ]; do
        $EV --ckpt $G --n 120 $BASE --schedule cosine --sample-seed 0 --select-judge $SJ \
            --select-k 4 --hold-bias 0.2 --shard $k/$SHARDS > $D/stm2_shard$k.log 2>&1 &
        k=$((k + 1))
    done
    wait
    merged stm2
    ;;
passes)
    [ -f $D/stm2.path ] || { echo "no draw: nothing to do"; exit 0; }
    k=0
    while [ $k -lt $SHARDS ]; do
        $EV --ckpt $G --n 120 --from-charts $(cat $D/stm2.path) $RESTS $REUSE $HOLDS $CHORDS \
            $JACKS --shard $k/$SHARDS > $D/stm2f_shard$k.log 2>&1 &
        k=$((k + 1))
    done
    wait
    merged stm2f
    ;;
judges)
    [ -f $D/stm2f.path ] || { echo "no run: nothing to do"; exit 0; }
    $FRESH --ai $(cat $D/stm2f.path) --name s1_stm2
    $FRESH --ai $(cat $D/stm2f.path) --view rhythm --name s1_stm2_rhythm
    ;;
answers)
    for b in 1 2; do
        name=ab${b}f
        k=0
        while [ $k -lt $SHARDS ]; do
            $EV --ckpt $P --n 240 --from-charts $SYS $RESTS $REUSE $HOLDS $CHORDS $JACKS \
                --rest-max-run 8 --answer -$b --shard $k/$SHARDS > $D/${name}_shard$k.log 2>&1 &
            k=$((k + 1))
        done
        wait
        merged $name
        $FRESH --ai $(cat $D/$name.path) --name s1_$name
        $FRESH --ai $(cat $D/$name.path) --view rhythm --name s1_${name}_rhythm
    done
    ;;
table)
    J=outputs/judge
    if [ -f $J/s1_stm2/summary.json ]; then
        python scripts/judge_table.py $J/s1_sh $J/s1_stm2 --out $D/results_1125_stems.md
        python scripts/judge_table.py $J/s1_sh_rhythm $J/s1_stm2_rhythm \
            --out $D/results_1125_stems_rhythm.md
        python scripts/diag_judge.py $SHF $(cat $D/stm2f.path) --out $D/diag_judge_stm2.md
        python scripts/results_table.py $SHF $(cat $D/stm2f.path) --out $D/results_stm2.md
    fi
    for b in 1 2; do
        [ -f $J/s1_ab${b}f/summary.json ] || continue
        python scripts/judge_table.py $J/s1_rgf $J/s1_ab${b}f --out $D/results_1125_ab$b.md
        python scripts/judge_table.py $J/s1_rgf_rhythm $J/s1_ab${b}f_rhythm \
            --out $D/results_1125_ab${b}_rhythm.md
    done
    for r in full-v4 ft-v1 stems-v1 stems-v2; do
        [ -f outputs/$r/val.csv ] && printf '%s ' $r && tail -n 1 outputs/$r/val.csv
    done | tee $D/results_1125_val.txt
    ;;
*)
    echo "phases: train draw passes judges answers table"; exit 2
    ;;
esac
