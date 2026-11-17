#!/bin/sh
# 11/18 night (~3 h 30 min): the chords and the releases of the system put where the human
# chart has them. The user's blind play of 5 songs (11/17): the AI's chords are spread evenly
# where the human chart keeps them for the accents (zeke 0.35 against 0.12; Mirai the other
# way, 0.22 against 0.45), and its long notes end where nothing happens (Run*2 71% released on
# another lane's onset, the human 97%). Over the 240 val songs the chord share is off the
# human chart's by more than 0.10 in 40% of the songs (23% more, 17% fewer), and 0.689 of the
# releases fall on another lane's onset (human 0.759; snapping within 2 cells 0.731, 3 cells
# 0.820, measured on rp20's charts). Two passes that were never run: --chord-share oracle
# (sampler.chord_share_to: the weakest chords thinned or the strongest single notes given a
# mirrored tap, to the human chart's share) and --snap-releases N (sampler.snap_releases).
# On the system's draw (240) with the system's passes (= run_1129's rp20, the control):
# chords, releases within 2 and within 3 cells, chords and releases within 2; fresh full- and
# holds-view judges against rp20's. Why and the predictions: EXPERIMENTS 2026-11-18 (1).
# Writes under outputs/ only, nothing is deleted.
#   passes  (~1 h 40 min) the four, three processes each
#   judges  (~1 h 45 min) full- and holds-view judges on each
#   table   (seconds)
# All: for p in passes judges table; do sh experiments/runs/run_1130.sh $p; done
set -ex
PHASE=${1:-passes}
SHARDS=${SHARDS:-3}
P=outputs/full-v4/best.pt
D=outputs/run_1105
J=outputs/judge
for f in sys.path rp20.path; do
    [ -f $D/$f ] || { echo "no $D/$f (run_1119 / run_1129)"; exit 2; }
done
for n in s1_rp20 s1_rp20_holds; do
    [ -f $J/$n/summary.json ] || { echo "no $J/$n (run_1129)"; exit 2; }
done
SYS=$(cat $D/sys.path)

RESTS="--rest 1 --light 0.5 --light-alike --fill-holes 0.5 --rest-max-run 8"
REUSE="--carry-rhythm 0 --carry-keep --carry-holds 1 --copy-bias 0 --copy-min-lag 8 \
--copy-sections 4 --section-keep --section-back --alike 2 --alike-keep 4"
HOLDS="--hold-share oracle --hold-up --ln-tidy 0.5 --dense-taps 0.2 --dense-repeats"
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

ARMS="chs sn2 sn3 chsn2"
case $PHASE in
passes)
    passes_with chs --chord-share oracle
    passes_with sn2 --snap-releases 2
    passes_with sn3 --snap-releases 3
    passes_with chsn2 --chord-share oracle --snap-releases 2
    ;;
judges)
    for name in $ARMS; do
        [ -f $D/$name.path ] || { echo "no $name: skipped"; continue; }
        judged $name full $(cat $D/$name.path)
        judged $name holds $(cat $D/$name.path)
    done
    ;;
table)
    for name in $ARMS; do
        [ -f $J/s1_${name}_holds/summary.json ] || continue
        python scripts/judge_table.py $J/s1_rp20 $J/s1_$name --out $D/results_1130_$name.md
        python scripts/judge_table.py $J/s1_rp20_holds $J/s1_${name}_holds \
            --out $D/results_1130_${name}_holds.md
    done
    runs="$(cat $D/rp20.path)"
    for name in $ARMS; do
        [ -f $D/$name.path ] && runs="$runs $(cat $D/$name.path)"
    done
    python scripts/diag_judge.py $runs --out $D/diag_judge_1130.md
    python scripts/results_table.py $runs --out $D/results_1130_runs.md
    ;;
*)
    echo "phases: passes judges table"; exit 2
    ;;
esac
