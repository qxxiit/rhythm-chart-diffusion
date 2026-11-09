#!/bin/sh
# 11/10 (~2 h 30 min): the new system on all 240 val songs, read by the standard judges and the
# playtest gate. run_1117's system (the windows chosen among 4 by sel_holds, the hold bias 0.2,
# the lanes at temperature 1, F's passes) took the 120-song fresh judge to 0.743 against
# run_1112's 0.818 (-0.075 [-0.137, -0.019]). The gate is read on the standard judges (240
# songs, seed 1, --sr view, 2 epochs, 2 folds: s1_cos240 0.911, s1_sel8 0.878), so the other
# 120 songs are drawn and passed the same way (evaluate.py --reuse: run_1117's 120 kept as
# they are, the seeds those of a 240-song run). Why and the predictions: EXPERIMENTS
# 2026-11-10 (1). Writes under outputs/ only (run_1117's directories grow to 240 songs, their
# first 120 as they were), nothing is deleted.
#   draw    (~2 h) the 120 songs not yet drawn, three processes
#   passes  (~12 min) F's passes on them
#   judges  (~20 min) the standard judge on the 240
#   table   (seconds) against the cosine run and run_1113's 8 draws on the songs they share
# All: for p in draw passes judges table; do sh run_1119.sh $p; done
set -ex
PHASE=${1:-draw}
SHARDS=${SHARDS:-3}
P=outputs/full-v4/best.pt
D=outputs/run_1105
SJ=outputs/judge/sel_holds
for f in sh.path shf.path; do
    [ -f $D/$f ] || { echo "no $D/$f (run_1117)"; exit 2; }
done

RESTS="--rest 1 --light 0.5 --light-alike --fill-holes 0.5"
REUSE="--carry-rhythm 0 --carry-keep --carry-holds 1 --copy-bias 0 --copy-min-lag 8 \
--copy-sections 4 --section-keep --section-back --alike 2 --alike-keep 4"
HOLDS="--hold-share oracle --hold-up --ln-tidy 0.5"
CHORDS="--strong-chords 1 --chord-cap 2"
JACKS="--untangle --untangle-move --untangle-grades --untangle-insane"
BASE="--lanes forward --refine 2 --loud-bias 0.1 --onset-bias 1 --onset-taper 2.7,5,0.35"
E="python scripts/evaluate.py --ckpt $P --per-song --lane-guidance 2 --lane-temp 1 --n 240 \
--stats oracle"

merged() {
    run=$(sed -n 's/^run //p' $D/${1}_shard0.log | head -n 1)
    [ -n "$run" ] || { echo "no run line in $D/${1}_shard0.log"; exit 2; }
    python scripts/merge_shards.py "$run"
    echo "$run" > $D/$1.path
}

case $PHASE in
draw)
    [ -f $SJ/models.json ] || { echo "no $SJ/models.json (run_1117 selj)"; exit 2; }
    k=0
    while [ $k -lt $SHARDS ]; do
        $E $BASE --schedule cosine --sample-seed 0 --select-judge $SJ --select-k 4 \
            --hold-bias 0.2 --reuse $(cat $D/sh.path) --shard $k/$SHARDS \
            > $D/sys_shard$k.log 2>&1 &
        k=$((k + 1))
    done
    wait
    merged sys
    ;;
passes)
    [ -f $D/sys.path ] || { echo "no draw: nothing to do"; exit 0; }
    k=0
    while [ $k -lt $SHARDS ]; do
        $E --from-charts $(cat $D/sys.path) $RESTS $REUSE $HOLDS $CHORDS $JACKS \
            --reuse $(cat $D/shf.path) --shard $k/$SHARDS > $D/sysf_shard$k.log 2>&1 &
        k=$((k + 1))
    done
    wait
    merged sysf
    ;;
judges)
    [ -f $D/sysf.path ] || { echo "no run: nothing to do"; exit 0; }
    python scripts/judge.py --ckpt $P --seed 1 --sr view --epochs 2 --eval-epochs --max-folds 2 \
        --ai $(cat $D/sysf.path) --name s1_sys240
    ;;
table)
    [ -f $D/sysf.path ] || { echo "no run: nothing to do"; exit 0; }
    J=outputs/judge
    python scripts/judge_table.py $J/s1_cos240 $J/s1_sel8 $J/s1_sys240 --out $D/results_1119.md
    python scripts/diag_judge.py $(cat $D/cos240.path) $(cat $D/self8.path) $(cat $D/sysf.path) \
        --out $D/diag_judge_sys240.md
    python scripts/results_table.py $(cat $D/cos240.path) $(cat $D/self8.path) \
        $(cat $D/sysf.path) --out $D/results_sys240.md
    ;;
*)
    echo "phases: draw passes judges table"; exit 2
    ;;
esac
