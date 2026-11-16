#!/bin/sh
# 11/17 (~3 h 30 min): the patterns followed. The pilot: the AI's patterns are hard to make
# out, so their repeats (mirrored, varied) do not show either. On the first 120 val songs the
# lane motion's predictability (patterns.motion_pred: stairs, trills and unnamed motifs that
# come back) is human 0.128, the system 0.085 (its lanes at temperature 1 since 11/10), the
# system of 11/09 at 0.5 0.139 with its patterns held twice as long as the human charts'
# (coverage 0.083, human 0.042). One temperature for every row moves between the two; here the
# first onset row of each unit is drawn at 1 and the later ones lower
# (sampler.unit_temperature): a figure picked freely, then followed. The lanes of the system's
# draw (run_1119's sys) drawn again (--relane: forward lanes, refine 2, lane guidance 2) with
# the rest guard and F's passes: lanes at 1 (the control), units of a beat at 0.5, units of
# two beats at 0.5; fresh full- and taps-view judges, each against the control. Why and the
# predictions: EXPERIMENTS 2026-11-17 (1). Writes under outputs/ only, nothing is deleted.
#   lanes   (~2 h 15 min) the three, three processes each
#   judges  (~1 h 15 min) fresh full- and taps-view judges on each
#   table   (seconds)
# All: for p in lanes judges table; do sh run_1127.sh $p; done
set -ex
PHASE=${1:-lanes}
SHARDS=${SHARDS:-3}
P=outputs/full-v4/best.pt
D=outputs/run_1105
J=outputs/judge
[ -f $D/sys.path ] || { echo "no $D/sys.path (run_1119)"; exit 2; }
SYS=$(cat $D/sys.path)

RESTS="--rest 1 --light 0.5 --light-alike --fill-holes 0.5 --rest-max-run 8"
REUSE="--carry-rhythm 0 --carry-keep --carry-holds 1 --copy-bias 0 --copy-min-lag 8 \
--copy-sections 4 --section-keep --section-back --alike 2 --alike-keep 4"
HOLDS="--hold-share oracle --hold-up --ln-tidy 0.5"
CHORDS="--strong-chords 1 --chord-cap 2"
JACKS="--untangle --untangle-move --untangle-grades --untangle-insane"
LANES="--relane --lanes forward --refine 2 --lane-temp 1"
EV="python scripts/evaluate.py --ckpt $P --per-song --lane-guidance 2 --stats oracle --n 120"
JUDGE="python scripts/judge.py --ckpt $P --seed 1 --sr view --epochs 2 --eval-epochs --limit 120"

merged() {
    run=$(sed -n 's/^run //p' $D/${1}_shard0.log | head -n 1)
    [ -n "$run" ] || { echo "no run line in $D/${1}_shard0.log"; exit 2; }
    python scripts/merge_shards.py "$run"
    echo "$run" > $D/$1.path
}

relaned() {             # NAME EXTRA...: the system's draw, its lanes drawn again with EXTRA
    name=$1
    shift
    [ -f $D/$name.path ] && { echo "$name is there"; return; }
    k=0
    while [ $k -lt $SHARDS ]; do
        $EV --from-charts $SYS $LANES "$@" $RESTS $REUSE $HOLDS $CHORDS $JACKS \
            --shard $k/$SHARDS > $D/${name}_shard$k.log 2>&1 &
        k=$((k + 1))
    done
    wait
    merged $name
}

case $PHASE in
lanes)
    relaned ul0
    relaned ul1 --lane-inner-temp 0.5 --lane-unit 12
    relaned ul2 --lane-inner-temp 0.5 --lane-unit 24
    ;;
judges)
    for name in ul0 ul1 ul2; do
        [ -f $D/$name.path ] || { echo "no $name: skipped"; continue; }
        [ -f $J/s1_${name}_taps/summary.json ] && { echo "s1_$name is there"; continue; }
        $JUDGE --ai $(cat $D/$name.path) --name s1_$name
        $JUDGE --ai $(cat $D/$name.path) --view taps --name s1_${name}_taps
    done
    ;;
table)
    for name in ul1 ul2; do
        [ -f $J/s1_${name}_taps/summary.json ] && [ -f $J/s1_ul0_taps/summary.json ] || continue
        python scripts/judge_table.py $J/s1_ul0 $J/s1_$name --out $D/results_1127_$name.md
        python scripts/judge_table.py $J/s1_ul0_taps $J/s1_${name}_taps \
            --out $D/results_1127_${name}_taps.md
    done
    runs=""
    for name in ul0 ul1 ul2; do
        [ -f $D/$name.path ] && runs="$runs $(cat $D/$name.path)"
    done
    [ -n "$runs" ] || exit 0
    python scripts/diag_judge.py $runs --out $D/diag_judge_1127.md
    python scripts/results_table.py $runs --out $D/results_1127_runs.md
    ;;
*)
    echo "phases: lanes judges table"; exit 2
    ;;
esac
