#!/bin/sh
# 11/19 (~2 h 50 min): a softer chord share, and bars mirrored where the rhythm repeats. run_1130
# brought each song's chord share to its human chart's exactly and the judges were against it
# (full view +0.035): ranked strictly, the chords ended on the strongest rows far past the
# human charts (chord strength 1.131, human 0.829). Here half the way (--chord-share-step 0.5)
# in an order drawn by weight (--chord-share-temp 1). And the user's mirrored repeats: of the
# bar pairs 1, 2 and 4 bars apart with the same onset rows the human charts mirror the earlier
# bar exactly in 4.4%, 6.0%, 6.8%, the system in 2.1%, 2.5%, 2.8%; --mirror-bars
# (sampler.mirror_bars) makes up the difference, the pairs whose audio is most alike first.
# On the system's draw (240) with the system's passes (= run_1130's sn2, the control): the
# soft chords, the mirrored bars, both; fresh full- and taps-view judges against sn2's (a
# taps-view judge on sn2 first). Why and the predictions: EXPERIMENTS 2026-11-19 (1). Writes
# under outputs/ only, nothing is deleted.
#   passes  (~1 h 15 min) the three, three processes each
#   judges  (~1 h 30 min) a taps-view judge on sn2; full- and taps-view judges on each
#   table   (seconds)
# All: for p in passes judges table; do sh experiments/runs/run_1131.sh $p; done
set -ex
PHASE=${1:-passes}
SHARDS=${SHARDS:-3}
P=outputs/full-v4/best.pt
D=outputs/run_1105
J=outputs/judge
for f in sys.path sn2.path; do
    [ -f $D/$f ] || { echo "no $D/$f (run_1119 / run_1130)"; exit 2; }
done
[ -f $J/s1_sn2/summary.json ] || { echo "no $J/s1_sn2 (run_1130)"; exit 2; }
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
    if [ $2 = full ]; then
        $JUDGE --ai $3 --name s1_$1
    else
        $JUDGE --ai $3 --view $2 --name s1_$1$sfx
    fi
}

ARMS="chsoft mb chmb"
SOFT="--chord-share oracle --chord-share-step 0.5 --chord-share-temp 1"
case $PHASE in
passes)
    passes_with chsoft $SOFT
    passes_with mb --mirror-bars
    passes_with chmb $SOFT --mirror-bars
    ;;
judges)
    judged sn2 taps $(cat $D/sn2.path)
    for name in $ARMS; do
        [ -f $D/$name.path ] || { echo "no $name: skipped"; continue; }
        judged $name full $(cat $D/$name.path)
        judged $name taps $(cat $D/$name.path)
    done
    ;;
table)
    for name in $ARMS; do
        [ -f $J/s1_${name}_taps/summary.json ] || continue
        python scripts/judge_table.py $J/s1_sn2 $J/s1_$name --out $D/results_1131_$name.md
        python scripts/judge_table.py $J/s1_sn2_taps $J/s1_${name}_taps \
            --out $D/results_1131_${name}_taps.md
    done
    runs="$(cat $D/sn2.path)"
    for name in $ARMS; do
        [ -f $D/$name.path ] && runs="$runs $(cat $D/$name.path)"
    done
    python scripts/diag_judge.py $runs --out $D/diag_judge_1131.md
    python scripts/results_table.py $runs --out $D/results_1131_runs.md
    ;;
*)
    echo "phases: passes judges table"; exit 2
    ;;
esac
