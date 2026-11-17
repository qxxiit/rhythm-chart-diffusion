#!/bin/sh
# 11/16 night (~2 h 30 min, after run_1123): the pilot's two tells fixed in the passes, on
# the system's own draw. (1) --rest-max-run 8: the rest pass never leaves 8 beats with nothing
# at all around a bar it clears (it joined and lengthened the draw's empty stretches: songs
# with a fatal one 3% in the draw, 5% after F's passes). (2) --answer 0: every beat offered
# the rhythm of the one before it when the model scores it at least as well (the system's
# next beat takes the rhythm again 14.6% of the time, the human charts 25.0%), its lanes drawn
# again by the model (which knows the mirror). Each alone and both, as F's passes on the
# system's draw of the 240 val songs (run_1119's sys), full-v4 as the system; read on the
# first 120 by the 120-song fresh judges against the system there (s1_sh, s1_sh_rhythm).
# Why and the predictions: EXPERIMENTS 2026-11-15 (2). Writes under outputs/ only, nothing is
# deleted.
#   passes  (~1 h 15 min) the three, three processes each
#   judges  (~1 h 15 min) fresh full- and rhythm-view judges on each
#   table   (seconds)
# All: for p in passes judges table; do sh experiments/runs/run_1124.sh $p; done
set -ex
PHASE=${1:-passes}
SHARDS=${SHARDS:-3}
P=outputs/full-v4/best.pt
D=outputs/run_1105
for f in sys.path; do
    [ -f $D/$f ] || { echo "no $D/$f (run_1119)"; exit 2; }
done
SYS=$(cat $D/sys.path)

RESTS="--rest 1 --light 0.5 --light-alike --fill-holes 0.5"
REUSE="--carry-rhythm 0 --carry-keep --carry-holds 1 --copy-bias 0 --copy-min-lag 8 \
--copy-sections 4 --section-keep --section-back --alike 2 --alike-keep 4"
HOLDS="--hold-share oracle --hold-up --ln-tidy 0.5"
CHORDS="--strong-chords 1 --chord-cap 2"
JACKS="--untangle --untangle-move --untangle-grades --untangle-insane"
E="python scripts/evaluate.py --ckpt $P --per-song --lane-guidance 2 --lane-temp 1 --n 240 \
--stats oracle"
FRESH="python scripts/judge.py --ckpt $P --seed 1 --sr view --epochs 2 --eval-epochs --limit 120"

merged() {
    run=$(sed -n 's/^run //p' $D/${1}_shard0.log | head -n 1)
    [ -n "$run" ] || { echo "no run line in $D/${1}_shard0.log"; exit 2; }
    python scripts/merge_shards.py "$run"
    echo "$run" > $D/$1.path
}

passes_with() {          # NAME EXTRA...: F's passes on the system's draw with EXTRA
    name=$1
    shift
    k=0
    while [ $k -lt $SHARDS ]; do
        $E --from-charts $SYS $RESTS $REUSE $HOLDS $CHORDS $JACKS "$@" \
            --shard $k/$SHARDS > $D/${name}_shard$k.log 2>&1 &
        k=$((k + 1))
    done
    wait
    merged $name
}

case $PHASE in
passes)
    passes_with rgf --rest-max-run 8
    passes_with abf --answer 0
    passes_with rabf --rest-max-run 8 --answer 0
    ;;
judges)
    for name in rgf abf rabf; do
        [ -f $D/$name.path ] || { echo "no $name: skipped"; continue; }
        $FRESH --ai $(cat $D/$name.path) --name s1_$name
        $FRESH --ai $(cat $D/$name.path) --view rhythm --name s1_${name}_rhythm
    done
    ;;
table)
    J=outputs/judge
    runs=""
    for name in rgf abf rabf; do                  # each against the system
        [ -f $J/s1_$name/summary.json ] || continue
        python scripts/judge_table.py $J/s1_sh $J/s1_$name --out $D/results_1124_$name.md
        python scripts/judge_table.py $J/s1_sh_rhythm $J/s1_${name}_rhythm \
            --out $D/results_1124_${name}_rhythm.md
        runs="$runs $(cat $D/$name.path)"
    done
    python scripts/diag_judge.py $(cat $D/sysf.path) $runs --out $D/diag_judge_1124.md
    python scripts/results_table.py $(cat $D/sysf.path) $runs --out $D/results_1124_runs.md
    ;;
*)
    echo "phases: passes judges table"; exit 2
    ;;
esac
