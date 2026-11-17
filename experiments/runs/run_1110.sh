#!/bin/sh
# From 11/8 night: run_1109's views again, each chart shown with the SR of the chart as shown
# (judge.py --sr view). run_1109 gave every chart the human chart's SR, the SR the run was asked
# for, and its full view told the charts apart at 0.944 where the cosine run's own judge does at
# 0.901 on the same 96 songs: the judge read "not as hard as it says" (EXPERIMENTS 2026-11-08
# (3)). Judges only, on the cosine run; writes under outputs/judge and outputs/run_1105.
#   taps    long notes as taps
#   rhythm  taps on lanes drawn at random (not run when taps leaves little)
#   deaf    rhythm without the audio (not run when rhythm leaves little)
#   full    everything: the views' reference
#   table   results_1110.md: the steps of the nest with their intervals (judge_table.py)
# All: for p in taps rhythm deaf full table; do sh experiments/runs/run_1110.sh $p; done
# Time: 2 epochs (run_1109: epoch 2 within 0.002 of epoch 3 in three judges of four, 0.008 in
# the fourth, the learning rate still at a quarter there), 2 folds unless the interval holds
# 0.65 (where a step is skipped); ~10 min a judge, ~45 min in all.
set -ex
PHASE=${1:-taps}
P=outputs/full-v4/best.pt
D=outputs/run_1105
COSF=${COSF:-$(cat $D/cos240.path)}
[ -d "$COSF" ] || { echo "no COSF=$COSF (run_1108's cosine run)"; exit 2; }
LITTLE=0.65
J="python scripts/judge.py --ckpt $P --ai $COSF --sr view --epochs 2 --eval-epochs --stop-bounds $LITTLE --min-folds 2"

# 0 when the judge's auc is LITTLE or more (something is left), not 0 when not or without it
left() {
    python -c "import json, sys; s = json.load(open('outputs/judge/$1/summary.json'))['ai']; \
print('$1 auc', round(s['auc'], 3)); sys.exit(0 if s['auc'] >= $LITTLE else 1)"
}

case $PHASE in
taps)
    $J --view taps --name srv_taps
    ;;
rhythm)
    if left srv_taps; then
        $J --view rhythm --name srv_rhythm
    else
        echo "taps leaves little: the rhythm view is not run"
    fi
    ;;
deaf)
    if [ -f outputs/judge/srv_rhythm/summary.json ] && left srv_rhythm; then
        $J --view rhythm --no-audio --name srv_deaf
    else
        echo "no rhythm view, or it leaves little: the deaf view is not run"
    fi
    ;;
full)
    $J --view full --name srv_full
    ;;
table)
    # in this order each row's step is one change: the human SR (run_1109) to the own SR, to
    # the SR as shown with 2 epochs, then one more removal a row
    python scripts/judge_table.py outputs/judge/view_full outputs/judge/cosine \
        outputs/judge/srv_full outputs/judge/srv_taps outputs/judge/srv_rhythm \
        outputs/judge/srv_deaf --out $D/results_1110.md
    ;;
*)
    echo "phases: taps rhythm deaf full table"; exit 2
    ;;
esac
