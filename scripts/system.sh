#!/bin/sh
# The system as decided (DECISIONS through 2026-11-20, run_1134), on the songs of a split:
#   full-v4; every window drawn 4 times and chosen by the holds-view judge sel_holds (cosine
#   schedule, forward lanes at temperature 1, refine 2, lane guidance 2, the loud and onset
#   biases with the onset taper by SR, hold bias 0.2, the human chart's stats as inputs); then
#   F's passes with the rest guard (--rest-max-run 8), and in the songs whose human chart is
#   mostly taps a short long note repeated in its lane in a busy part made a tap
#   (--dense-taps 0.2 --dense-repeats), bars that repeat the rhythm of the bar 1, 2 or 4 bars
#   back made its mirror as often as the human charts do (--mirror-bars --mirror-shares
#   0.034,0.049,0.062), and last the releases moved to the next note within two cells
#   (--snap-releases 2).
# From the repository root:
#   sh scripts/system.sh [SPLIT] [N]     SPLIT val or test (val), N songs, 0 for all (120)
# Environment: SHARDS processes (3), CKPT (outputs/full-v4/best.pt), JUDGE (the choosing judge,
#   outputs/judge/sel_holds: judge.py --save-models), SEED (0), DENSE (the dense-taps flags,
#   "--dense-taps 0.2 --dense-repeats"), EXTRA (more flags for the passes, e.g. the lanes
#   drawn again in units: "--relane --lanes forward --refine 2 --lane-temp 1
#   --lane-inner-temp 0.5 --lane-unit 24").
# The draw and the final run are written where evaluate.py writes them (CKPT's directory); the
# final run's path is in outputs/system/SPLIT_nN.path, the logs beside it. 120 songs take
# about 1 h 30 min on the MacBook (M-series, MPS). Then, for a playtest pack:
#   python scripts/playtest_pack.py --single --from-run "$(cat outputs/system/test_n0.path)" ...
set -e
SPLIT=${1:-val}
N=${2:-120}
SHARDS=${SHARDS:-3}
CKPT=${CKPT:-outputs/full-v4/best.pt}
JUDGE=${JUDGE:-outputs/judge/sel_holds}
SEED=${SEED:-0}
DENSE=${DENSE:---dense-taps 0.2 --dense-repeats}
EXTRA=${EXTRA:-}
O=outputs/system
[ -f "$CKPT" ] || { echo "no $CKPT"; exit 2; }
[ -d "$JUDGE" ] || { echo "no $JUDGE (the choosing judge)"; exit 2; }
mkdir -p $O

EV="python scripts/evaluate.py --ckpt $CKPT --split $SPLIT --n $N --per-song --lane-guidance 2 \
--lane-temp 1 --stats oracle"
DRAW="--lanes forward --refine 2 --loud-bias 0.1 --onset-bias 1 --onset-taper 2.7,5,0.35 \
--schedule cosine --sample-seed $SEED --select-judge $JUDGE --select-k 4 --hold-bias 0.2"
RESTS="--rest 1 --light 0.5 --light-alike --fill-holes 0.5 --rest-max-run 8"
REUSE="--carry-rhythm 0 --carry-keep --carry-holds 1 --copy-bias 0 --copy-min-lag 8 \
--copy-sections 4 --section-keep --section-back --alike 2 --alike-keep 4"
HOLDS="--hold-share oracle --hold-up --ln-tidy 0.5"
CHORDS="--strong-chords 1 --chord-cap 2"
JACKS="--untangle --untangle-move --untangle-grades --untangle-insane"
MIRROR="--mirror-bars --mirror-shares 0.034,0.049,0.062"
LAST="--snap-releases 2"

sharded() {             # NAME FLAGS...: SHARDS processes, merged; the run's path in $O/NAME.path
    name=$1
    shift
    k=0
    while [ $k -lt $SHARDS ]; do
        $EV "$@" --shard $k/$SHARDS > $O/${name}_shard$k.log 2>&1 &
        k=$((k + 1))
    done
    wait
    run=$(sed -n 's/^run //p' $O/${name}_shard0.log | head -n 1)
    [ -n "$run" ] || { echo "no run line in $O/${name}_shard0.log"; exit 2; }
    python scripts/merge_shards.py "$run"
    echo "$run" > $O/$name.path
}

tag=${SPLIT}_n$N
sharded ${tag}_draw $DRAW
sharded $tag --from-charts "$(cat $O/${tag}_draw.path)" $RESTS $REUSE $HOLDS $CHORDS $JACKS \
    $MIRROR $LAST $DENSE $EXTRA
echo "the system's charts: $(cat $O/$tag.path)"
