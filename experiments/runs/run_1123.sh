#!/bin/sh
# 11/15 night (~4 h 30 min after the stems): full-v4 taught the stems. The fatal empty
# stretches (5% of the system's songs, all in the quieter part of their song, 13 of 18
# already empty in the draw) are where the mix gets quiet but a human chart goes on
# following something; the stems (scripts/stems.py: htdemucs's drums, bass, other and vocals,
# 16 bands each) let the model hear what. train.py --init full-v4 --stems: the same model,
# its audio input widened from 80 to 144 channels with zero weights (it starts as full-v4),
# 20k steps at lr 1e-4 on full-v4's recipe; then the system's draw with it (the windows
# chosen among 4 by sel_holds, hold bias 0.2, lanes at temperature 1, F's passes) on the
# first 120 val songs, read against the system there by the 120-song fresh judges (full
# view s1_sh, rhythm view s1_sh_rhythm; their trunk stays full-v4); last, the same 20k steps
# without the stems, to tell the stems from more training in val_ce. Why and the
# predictions: EXPERIMENTS 2026-11-15 (1). Writes under outputs/ only, nothing is deleted.
#   wait     until run_1122's stems are done (checks every 5 min)
#   train    (~1 h 15 min) outputs/stems-v1
#   draw     (~1 h 30 min) the 120 songs, three processes
#   passes   (~10 min) F's passes
#   judges   (~25 min) fresh full- and rhythm-view judges as s1_sh's
#   table    (seconds)
#   control  (~1 h 15 min) outputs/ft-v1: the same steps without the stems
#   vtable   (seconds) val_ce of full-v4, ft-v1, stems-v1
# All: for p in wait train draw passes judges table control vtable; do sh experiments/runs/run_1123.sh $p; done
set -ex
PHASE=${1:-wait}
SHARDS=${SHARDS:-3}
N=${N:-120}
P=outputs/full-v4/best.pt
G=outputs/stems-v1/best.pt
D=outputs/run_1105
SJ=outputs/judge/sel_holds
SHF=$(cat $D/shf.path)

RESTS="--rest 1 --light 0.5 --light-alike --fill-holes 0.5"
REUSE="--carry-rhythm 0 --carry-keep --carry-holds 1 --copy-bias 0 --copy-min-lag 8 \
--copy-sections 4 --section-keep --section-back --alike 2 --alike-keep 4"
HOLDS="--hold-share oracle --hold-up --ln-tidy 0.5"
CHORDS="--strong-chords 1 --chord-cap 2"
JACKS="--untangle --untangle-move --untangle-grades --untangle-insane"
BASE="--lanes forward --refine 2 --loud-bias 0.1 --onset-bias 1 --onset-taper 2.7,5,0.35"
E="python scripts/evaluate.py --ckpt $G --per-song --lane-guidance 2 --lane-temp 1 --n $N \
--stats oracle"
FRESH="python scripts/judge.py --ckpt $P --seed 1 --sr view --epochs 2 --eval-epochs --limit $N"
FT="python scripts/train.py --init $P --chart-stats data/chart_stats.csv --row-mask 0.5 \
--steps 20000 --lr 1e-4 --warmup 500 --val-every 2000"

merged() {
    run=$(sed -n 's/^run //p' $D/${1}_shard0.log | head -n 1)
    [ -n "$run" ] || { echo "no run line in $D/${1}_shard0.log"; exit 2; }
    python scripts/merge_shards.py "$run"
    echo "$run" > $D/$1.path
}

case $PHASE in
wait)
    while pgrep -f "scripts/stems.py" > /dev/null; do sleep 300; done
    python scripts/stems.py --limit 0          # how many have their stems
    ;;
train)
    [ -f $G ] && { echo "$G is there: not trained again"; exit 0; }
    $FT --stems --run stems-v1 > $D/stems_train.log 2>&1
    ;;
draw)
    [ -f $G ] || { echo "no $G: nothing to do"; exit 0; }
    k=0
    while [ $k -lt $SHARDS ]; do
        $E $BASE --schedule cosine --sample-seed 0 --select-judge $SJ --select-k 4 \
            --hold-bias 0.2 --shard $k/$SHARDS > $D/stm_shard$k.log 2>&1 &
        k=$((k + 1))
    done
    wait
    merged stm
    ;;
passes)
    [ -f $D/stm.path ] || { echo "no draw: nothing to do"; exit 0; }
    k=0
    while [ $k -lt $SHARDS ]; do
        $E --from-charts $(cat $D/stm.path) $RESTS $REUSE $HOLDS $CHORDS $JACKS \
            --shard $k/$SHARDS > $D/stmf_shard$k.log 2>&1 &
        k=$((k + 1))
    done
    wait
    merged stmf
    ;;
judges)
    [ -f $D/stmf.path ] || { echo "no run: nothing to do"; exit 0; }
    $FRESH --ai $(cat $D/stmf.path) --name s1_stm
    $FRESH --ai $(cat $D/stmf.path) --view rhythm --name s1_stm_rhythm
    ;;
table)
    [ -f $D/stmf.path ] || { echo "no run: nothing to do"; exit 0; }
    J=outputs/judge
    python scripts/judge_table.py $J/s1_sh $J/s1_stm --out $D/results_1123.md
    python scripts/judge_table.py $J/s1_sh_rhythm $J/s1_stm_rhythm --out $D/results_1123_rhythm.md
    python scripts/diag_judge.py $SHF $(cat $D/stmf.path) --out $D/diag_judge_stm.md
    python scripts/results_table.py $SHF $(cat $D/stmf.path) --out $D/results_stm.md
    ;;
control)
    [ -f outputs/ft-v1/best.pt ] && { echo "ft-v1 is there: not trained again"; exit 0; }
    $FT --run ft-v1 > $D/ft_train.log 2>&1
    ;;
vtable)
    for r in full-v4 ft-v1 stems-v1; do
        [ -f outputs/$r/val.csv ] && printf '%s ' $r && tail -n 1 outputs/$r/val.csv
    done | tee $D/results_1123_val.txt
    ;;
*)
    echo "phases: wait train draw passes judges table control vtable"; exit 2
    ;;
esac
