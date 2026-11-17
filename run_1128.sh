#!/bin/sh
# 11/17 (~2 h): short long notes where the chart is busy. The user (11/17): in hard charts the
# AI covers its busy parts with short long notes, repeated in one lane, where good human charts
# put taps on the rhythm. On the first 120 val songs at SR 3.5+ (busy: 4+ onsets a beat): in
# the songs whose human chart is mostly taps (25 of 59), a short long note (half a beat or
# less) followed in its lane within a beat by another is 0.6% of the busy notes, the system
# 3.1% (a quarter of those songs 5%+), the raw draw 1.1%: the hold bias and the passes. In the
# other songs the human charts do it (15.1%) as the system does (15.3%); the human chart's hold
# share tells the two kinds apart (under 0.15: 78% of the first, none of the second).
# --dense-taps T (sampler.dense_taps): in songs whose hold share target is under T, those long
# notes become taps. On the system's draw (run_1119's sys, 240 songs) with the rest guard and
# F's passes (= run_1124's rgf, the control): T 0.15 (the tap-chart songs), T 1 (every song);
# fresh full- and holds-view judges against rgf's. Why and the predictions: EXPERIMENTS
# 2026-11-17 (2). Writes under outputs/ only, nothing is deleted.
#   passes  (~50 min) the two, three processes each
#   judges  (~1 h 5 min) a holds-view judge on rgf; full- and holds-view judges on each
#   table   (seconds)
# All: for p in passes judges table; do sh run_1128.sh $p; done
set -ex
PHASE=${1:-passes}
SHARDS=${SHARDS:-3}
P=outputs/full-v4/best.pt
D=outputs/run_1105
J=outputs/judge
for f in sys.path rgf.path; do
    [ -f $D/$f ] || { echo "no $D/$f (run_1119 / run_1124)"; exit 2; }
done
SYS=$(cat $D/sys.path)

RESTS="--rest 1 --light 0.5 --light-alike --fill-holes 0.5 --rest-max-run 8"
REUSE="--carry-rhythm 0 --carry-keep --carry-holds 1 --copy-bias 0 --copy-min-lag 8 \
--copy-sections 4 --section-keep --section-back --alike 2 --alike-keep 4"
HOLDS="--hold-share oracle --hold-up --ln-tidy 0.5"
CHORDS="--strong-chords 1 --chord-cap 2"
JACKS="--untangle --untangle-move --untangle-grades --untangle-insane"
E="python scripts/evaluate.py --ckpt $P --per-song --lane-guidance 2 --lane-temp 1 --n 240 \
--stats oracle"
JUDGE="python scripts/judge.py --ckpt $P --seed 1 --sr view --epochs 2 --eval-epochs --limit 120"

merged() {
    run=$(sed -n 's/^run //p' $D/${1}_shard0.log | head -n 1)
    [ -n "$run" ] || { echo "no run line in $D/${1}_shard0.log"; exit 2; }
    python scripts/merge_shards.py "$run"
    echo "$run" > $D/$1.path
}

passes_with() {          # NAME EXTRA...: F's passes and the rest guard on the system's draw
    name=$1
    shift
    [ -f $D/$name.path ] && { echo "$name is there"; return; }
    k=0
    while [ $k -lt $SHARDS ]; do
        $E --from-charts $SYS $RESTS $REUSE $HOLDS $CHORDS $JACKS "$@" \
            --shard $k/$SHARDS > $D/${name}_shard$k.log 2>&1 &
        k=$((k + 1))
    done
    wait
    merged $name
}

judged() {               # NAME VIEW RUN
    sfx=""
    [ $2 = full ] || sfx=_$2
    [ -f $J/s1_$1$sfx/summary.json ] && { echo "s1_$1$sfx is there"; return; }
    if [ $2 = full ]; then
        $JUDGE --ai $3 --name s1_$1
    else
        $JUDGE --ai $3 --view $2 --name s1_$1$sfx
    fi
}

case $PHASE in
passes)
    passes_with dt15 --dense-taps 0.15
    passes_with dt100 --dense-taps 1
    ;;
judges)
    judged rgf holds $(cat $D/rgf.path)
    for name in dt15 dt100; do
        [ -f $D/$name.path ] || { echo "no $name: skipped"; continue; }
        judged $name full $(cat $D/$name.path)
        judged $name holds $(cat $D/$name.path)
    done
    ;;
table)
    for name in dt15 dt100; do
        [ -f $J/s1_${name}_holds/summary.json ] || continue
        python scripts/judge_table.py $J/s1_rgf $J/s1_$name --out $D/results_1128_$name.md
        python scripts/judge_table.py $J/s1_rgf_holds $J/s1_${name}_holds \
            --out $D/results_1128_${name}_holds.md
    done
    runs="$(cat $D/rgf.path)"
    for name in dt15 dt100; do
        [ -f $D/$name.path ] && runs="$runs $(cat $D/$name.path)"
    done
    python scripts/diag_judge.py $runs --out $D/diag_judge_1128.md
    python scripts/results_table.py $runs --out $D/results_1128_runs.md
    ;;
*)
    echo "phases: passes judges table"; exit 2
    ;;
esac
