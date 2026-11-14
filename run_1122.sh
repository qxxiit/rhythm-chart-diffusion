#!/bin/sh
# 11/14 night (~2 h 50 min, then the stems ~20 h): the rhythm drawn beat by beat, left to right
# (--order block), against the random order, then every audio file's stems. The pilot's
# tells, measured on the 240 val songs: the AI's next beat takes the rhythm of the one before
# half as often as the human charts (busy beat pairs: system draw 12.2%, after F's passes
# 14.6%, human 25.0%), so it answers a pattern with its mirror at about 40% of the human rate,
# though the model knows the mirror (top-1 77% at the first row where a human chart mirrors,
# 26% elsewhere: the mirror probe on 40 val charts). The random order opens ~9 cells a step
# anywhere in the window, so neighbouring beats are drawn without each other; block order
# opens a beat only after the one before it (autoregressive across beats, parallel within
# one). The windows' choice among 4 is not batched for block order yet (sample_windows: k
# calls, about 4x the time), so both draws here take one draw a window, the rest as the
# system (hold bias 0.2, lanes at temperature 1, F's passes), on the first 120 val songs, the
# same seeds; fresh 120-song judges on both (full and rhythm view, paired). Why and the
# predictions: EXPERIMENTS 2026-11-14 (2). Writes under outputs/ and data/cache/stems/ only,
# nothing is deleted.
#   draw    (~1 h 40 min) the 120 songs in random order, then in block order, three processes
#   passes  (~20 min) F's passes on both
#   judges  (~50 min) fresh full- and rhythm-view judges on both
#   table   (seconds) block against random, both views
#   stems   (~20 h) scripts/stems.py: the val and test audio first (~2 h), then train
# All: for p in draw passes judges table stems; do sh run_1122.sh $p; done
set -ex
PHASE=${1:-draw}
SHARDS=${SHARDS:-3}
N=${N:-120}
P=outputs/full-v4/best.pt
D=outputs/run_1105

RESTS="--rest 1 --light 0.5 --light-alike --fill-holes 0.5"
REUSE="--carry-rhythm 0 --carry-keep --carry-holds 1 --copy-bias 0 --copy-min-lag 8 \
--copy-sections 4 --section-keep --section-back --alike 2 --alike-keep 4"
HOLDS="--hold-share oracle --hold-up --ln-tidy 0.5"
CHORDS="--strong-chords 1 --chord-cap 2"
JACKS="--untangle --untangle-move --untangle-grades --untangle-insane"
BASE="--lanes forward --refine 2 --loud-bias 0.1 --onset-bias 1 --onset-taper 2.7,5,0.35"
E="python scripts/evaluate.py --ckpt $P --per-song --lane-guidance 2 --lane-temp 1 --n $N \
--stats oracle"
FRESH="python scripts/judge.py --ckpt $P --seed 1 --sr view --epochs 2 --eval-epochs --limit $N"

merged() {
    run=$(sed -n 's/^run //p' $D/${1}_shard0.log | head -n 1)
    [ -n "$run" ] || { echo "no run line in $D/${1}_shard0.log"; exit 2; }
    python scripts/merge_shards.py "$run"
    echo "$run" > $D/$1.path
}

case $PHASE in
draw)
    for o in random block; do
        name=rnd
        [ $o = block ] && name=blk
        k=0
        while [ $k -lt $SHARDS ]; do
            $E $BASE --order $o --schedule cosine --sample-seed 0 --hold-bias 0.2 \
                --shard $k/$SHARDS > $D/${name}_shard$k.log 2>&1 &
            k=$((k + 1))
        done
        wait
        merged $name
    done
    ;;
passes)
    for name in rnd blk; do
        [ -f $D/$name.path ] || { echo "no $name draw: skipped"; continue; }
        k=0
        while [ $k -lt $SHARDS ]; do
            $E --from-charts $(cat $D/$name.path) $RESTS $REUSE $HOLDS $CHORDS $JACKS \
                --shard $k/$SHARDS > $D/${name}f_shard$k.log 2>&1 &
            k=$((k + 1))
        done
        wait
        merged ${name}f
    done
    ;;
judges)
    for name in rnd blk; do
        [ -f $D/${name}f.path ] || { echo "no ${name}f: skipped"; continue; }
        $FRESH --ai $(cat $D/${name}f.path) --name s1_$name
        $FRESH --ai $(cat $D/${name}f.path) --view rhythm --name s1_${name}_rhythm
    done
    ;;
table)
    [ -f $D/blkf.path ] && [ -f $D/rndf.path ] || { echo "no runs: nothing to do"; exit 0; }
    J=outputs/judge
    python scripts/judge_table.py $J/s1_rnd $J/s1_blk --out $D/results_1122.md
    python scripts/judge_table.py $J/s1_rnd_rhythm $J/s1_blk_rhythm --out $D/results_1122_rhythm.md
    python scripts/diag_judge.py $(cat $D/rndf.path) $(cat $D/blkf.path) --out $D/diag_judge_blk.md
    python scripts/results_table.py $(cat $D/rndf.path) $(cat $D/blkf.path) --out $D/results_blk.md
    ;;
stems)
    PYTORCH_ENABLE_MPS_FALLBACK=1 python scripts/stems.py > $D/stems_1122.log 2>&1
    ;;
*)
    echo "phases: draw passes judges table stems"; exit 2
    ;;
esac
