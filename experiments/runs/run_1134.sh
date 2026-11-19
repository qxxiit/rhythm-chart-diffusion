#!/bin/sh
# 11/20 (~2 h): the mirrored bars at the shares counted again, and the same bar again 4 bars
# later. run_1131's --mirror-bars joined the system with shares (4.4 / 6.0 / 6.8% at lags 1 / 2
# / 4) from a count not reproduced; counted again on the onsets the human charts mirror 3.4 /
# 4.9 / 6.2% and the system now 4.5 / 6.1 / 6.7%. And the human charts play a bar's lanes again
# exactly four bars later in 6.4% of the pairs with the same onset rows, the system in 2.9%.
# --mirror-shares 0.034,0.049,0.062 (mbh); --repeat-bars (sampler.mirror_bars with flip off at
# lag 4, REPEAT_SHARE, before the mirrors; rp4); both (mbhrp4). On the system's draw (240) with
# the system's passes (= run_1131's mb, the control, its judges s1_mb and s1_mb_taps); fresh
# full- and taps-view judges; the bar copies counted on every run and the human charts. Why and
# the predictions: EXPERIMENTS 2026-11-20 (2). Writes under outputs/ only, nothing is deleted.
#   passes  (~45 min) the three, three processes each
#   judges  (~1 h 15 min) full- and taps-view judges on each
#   table   (~5 min)
# All: for p in passes judges table; do sh experiments/runs/run_1134.sh $p; done
set -ex
PHASE=${1:-passes}
SHARDS=${SHARDS:-3}
P=outputs/full-v4/best.pt
D=outputs/run_1105
J=outputs/judge
for f in sys.path mb.path; do
    [ -f $D/$f ] || { echo "no $D/$f (run_1119 / run_1131)"; exit 2; }
done
[ -f $J/s1_mb_taps/summary.json ] || { echo "no $J/s1_mb_taps (run_1131)"; exit 2; }
SYS=$(cat $D/sys.path)

RESTS="--rest 1 --light 0.5 --light-alike --fill-holes 0.5 --rest-max-run 8"
REUSE="--carry-rhythm 0 --carry-keep --carry-holds 1 --copy-bias 0 --copy-min-lag 8 \
--copy-sections 4 --section-keep --section-back --alike 2 --alike-keep 4"
HOLDS="--hold-share oracle --hold-up --ln-tidy 0.5 --dense-taps 0.2 --dense-repeats"
CHORDS="--strong-chords 1 --chord-cap 2"
JACKS="--untangle --untangle-move --untangle-grades --untangle-insane --snap-releases 2"
E="python scripts/evaluate.py --ckpt $P --per-song --lane-guidance 2 --lane-temp 1 --n 240 \
--stats oracle"
JUDGE="python scripts/judge.py --ckpt $P --seed 1 --sr view --epochs 2 --eval-epochs --limit 120"

merged() {
    run=$(sed -n 's/^run //p' $D/${1}_shard0.log | head -n 1)
    [ -n "$run" ] || { echo "no run line in $D/${1}_shard0.log"; exit 2; }
    python scripts/merge_shards.py "$run"
    echo "$run" > $D/$1.path
}

passes_with() {          # NAME EXTRA...: the system's passes on its draw, and EXTRA
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
    $JUDGE --ai $3 --view $2 --name s1_$1$sfx
}

ARMS="mbh rp4 mbhrp4"
MBH="--mirror-bars --mirror-shares 0.034,0.049,0.062"
case $PHASE in
passes)
    passes_with mbh $MBH
    passes_with rp4 --mirror-bars --repeat-bars
    passes_with mbhrp4 $MBH --repeat-bars
    ;;
judges)
    for name in $ARMS; do
        [ -f $D/$name.path ] || { echo "no $name: skipped"; continue; }
        judged $name full $(cat $D/$name.path)
        judged $name taps $(cat $D/$name.path)
    done
    ;;
table)
    runs="$(cat $D/mb.path)"
    for name in $ARMS; do
        [ -f $J/s1_${name}_taps/summary.json ] || continue
        python scripts/judge_table.py $J/s1_mb $J/s1_$name --out $D/results_1134_$name.md
        python scripts/judge_table.py $J/s1_mb_taps $J/s1_${name}_taps \
            --out $D/results_1134_${name}_taps.md
        runs="$runs $(cat $D/$name.path)"
    done
    python scripts/diag_judge.py $runs --out $D/diag_judge_1134.md
    python scripts/results_table.py $runs --out $D/results_1134_runs.md
    python scripts/bar_copies.py $runs --out $D/results_1134_copies.md
    ;;
*)
    echo "phases: passes judges table"; exit 2
    ;;
esac
