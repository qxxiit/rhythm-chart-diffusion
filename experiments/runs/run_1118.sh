#!/bin/sh
# 11/9 night (~2 h 10 min): the long notes drawn by the model, at the human share. run_1116's
# --hold-bias 0.5 tripled them (0.126 -> 0.374 of the onsets before F's passes, the human
# 0.225): the long notes' own step fell (full over taps +0.200 -> +0.039) but the taps view rose
# (0.607 -> 0.766, +0.159 [+0.101, +0.223]) and the full view stayed (0.805). Here the bias the
# two runs put at about the human share (0.2: the share's log-odds rose 2.84 a unit of bias),
# the rest as run_1116 (the windows chosen among 4 by sel_rhythm, the same seeds, the lanes at
# temperature 1, F's passes, fresh full- and taps-view judges). Why and the predictions:
# EXPERIMENTS 2026-11-09 (6). Writes under outputs/ only, nothing is deleted.
#   draw    (~1 h 30 min) the 120 songs, three processes, under select_watch.py
#   passes  (~10 min) F's passes (their lanes at temperature 1 too)
#   judges  (~25 min) fresh full-view and taps-view judges as run_1116's
#   table   (seconds) the steps run_1112 -> lanes at 1 -> bias 0.2 -> bias 0.5, both views
# All: for p in draw passes judges table; do sh experiments/runs/run_1118.sh $p; done
set -ex
PHASE=${1:-draw}
SHARDS=${SHARDS:-3}
N=${N:-120}
HB=${HB:-0.2}
P=outputs/full-v4/best.pt
D=outputs/run_1105
SJ=outputs/judge/sel_rhythm
for f in self120.path lt1.path hbf.path; do
    [ -f $D/$f ] || { echo "no $D/$f (run_1112, run_1115, run_1116)"; exit 2; }
done
SELF=$(cat $D/self120.path)
LT1=$(cat $D/lt1.path)
HBF=$(cat $D/hbf.path)

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

# merged NAME: the run of $D/NAME_shard*.log merged, its directory in $D/NAME.path
merged() {
    run=$(sed -n 's/^run //p' $D/${1}_shard0.log | head -n 1)
    [ -n "$run" ] || { echo "no run line in $D/${1}_shard0.log"; exit 2; }
    python scripts/merge_shards.py "$run"
    echo "$run" > $D/$1.path
}

case $PHASE in
draw)
    [ -f $SJ/models.json ] || { echo "no $SJ/models.json (run_1112 selj)"; exit 2; }
    pids=""
    k=0
    while [ $k -lt $SHARDS ]; do
        $E $BASE --schedule cosine --sample-seed 0 --select-judge $SJ --select-k 4 \
            --hold-bias $HB --shard $k/$SHARDS > $D/hb2_shard$k.log 2>&1 &
        pids="$pids $!"
        k=$((k + 1))
    done
    logs=$(k=0; while [ $k -lt $SHARDS ]; do printf "%s " $D/hb2_shard$k.log; k=$((k + 1)); done)
    python scripts/select_watch.py --logs $logs --judge $SJ --pids $pids --after 30 \
        --min-share 0.1 &
    wait
    if [ -f $D/hb2_shard0.log.STOPPED ]; then
        cat $D/hb2_shard0.log.STOPPED
        echo "stopped early: the later phases have nothing to do"; exit 0
    fi
    merged hb2
    ;;
passes)
    [ -f $D/hb2.path ] || { echo "no draw (stopped early?): nothing to do"; exit 0; }
    k=0
    while [ $k -lt $SHARDS ]; do
        $E --from-charts $(cat $D/hb2.path) $RESTS $REUSE $HOLDS $CHORDS $JACKS \
            --shard $k/$SHARDS > $D/hb2f_shard$k.log 2>&1 &
        k=$((k + 1))
    done
    wait
    merged hb2f
    ;;
judges)
    [ -f $D/hb2f.path ] || { echo "no run: nothing to do"; exit 0; }
    $FRESH --ai $(cat $D/hb2f.path) --name s1_hb2
    $FRESH --ai $(cat $D/hb2f.path) --view taps --name s1_hb2_taps
    ;;
table)
    [ -f $D/hb2f.path ] || { echo "no run: nothing to do"; exit 0; }
    J=outputs/judge
    python scripts/judge_table.py $J/s1_sel $J/s1_lt1 $J/s1_hb2 $J/s1_hb --out $D/results_1118.md
    python scripts/judge_table.py $J/s1_sel_taps $J/s1_lt1_taps $J/s1_hb2_taps $J/s1_hb_taps \
        --out $D/results_1118_taps.md
    python scripts/diag_judge.py $SELF $LT1 $HBF $(cat $D/hb2f.path) --out $D/diag_judge_hb2.md
    python scripts/results_table.py $SELF $LT1 $HBF $(cat $D/hb2f.path) --out $D/results_hb2.md
    ;;
*)
    echo "phases: draw passes judges table"; exit 2
    ;;
esac
