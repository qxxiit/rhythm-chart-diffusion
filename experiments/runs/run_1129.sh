#!/bin/sh
# 11/17 evening (~2 h): only the repeats become taps. run_1128's --dense-taps 0.15 halved the
# tap-chart songs' short long note repeats (2.6% -> 1.3%, the human 0.6%) and left the LN
# charts alone, but the holds view leaned against it (+0.029 [-0.000, +0.062]): in the songs
# it touches it takes every short long note out of the busy parts, where the human charts keep
# a few (4.6% of their busy notes) and almost never repeat one in its lane (0.6%). Here
# --dense-repeats: only a short long note next to another in its lane within a beat becomes
# a tap, a lone one stays; the gate at 0.15 (as run_1128) and at 0.20 (85% of the tap-chart
# songs, 9% of the LN-chart songs). On the system's draw (240) with the rest guard and F's
# passes; fresh full- and holds-view judges against the control's (rgf, run_1128) and dt15's.
# Why and the predictions: EXPERIMENTS 2026-11-17 (3). Writes under outputs/ only, nothing is
# deleted.
#   passes  (~50 min) the two, three processes each
#   judges  (~50 min) full- and holds-view judges on each
#   table   (seconds)
# All: for p in passes judges table; do sh experiments/runs/run_1129.sh $p; done
set -ex
PHASE=${1:-passes}
SHARDS=${SHARDS:-3}
P=outputs/full-v4/best.pt
D=outputs/run_1105
J=outputs/judge
for f in sys.path rgf.path dt15.path; do
    [ -f $D/$f ] || { echo "no $D/$f (run_1119 / run_1124 / run_1128)"; exit 2; }
done
for n in s1_rgf s1_rgf_holds s1_dt15 s1_dt15_holds; do
    [ -f $J/$n/summary.json ] || { echo "no $J/$n (run_1124 / run_1128)"; exit 2; }
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
    passes_with rp15 --dense-taps 0.15 --dense-repeats
    passes_with rp20 --dense-taps 0.2 --dense-repeats
    ;;
judges)
    for name in rp15 rp20; do
        [ -f $D/$name.path ] || { echo "no $name: skipped"; continue; }
        judged $name full $(cat $D/$name.path)
        judged $name holds $(cat $D/$name.path)
    done
    ;;
table)
    for name in rp15 rp20; do
        [ -f $J/s1_${name}_holds/summary.json ] || continue
        for base in rgf dt15; do                     # against the control, and against dt15
            python scripts/judge_table.py $J/s1_$base $J/s1_$name \
                --out $D/results_1129_${name}_vs_$base.md
            python scripts/judge_table.py $J/s1_${base}_holds $J/s1_${name}_holds \
                --out $D/results_1129_${name}_vs_${base}_holds.md
        done
    done
    runs="$(cat $D/rgf.path) $(cat $D/dt15.path)"
    for name in rp15 rp20; do
        [ -f $D/$name.path ] && runs="$runs $(cat $D/$name.path)"
    done
    python scripts/diag_judge.py $runs --out $D/diag_judge_1129.md
    python scripts/results_table.py $runs --out $D/results_1129_runs.md
    ;;
*)
    echo "phases: passes judges table"; exit 2
    ;;
esac
