#!/bin/sh
# From 11/8 evening: what of the charts the judge reads. run_1108's cosine run (the base from
# now on, DECISIONS 2026-11-08) judged again with part of every chart taken away alike, so each
# judge can only use what is left (judge.py --view; every chart shown with the human chart's
# SR, which a view needs). Why and the predictions: EXPERIMENTS 2026-11-08 (2). Judges only, no
# drawing; writes under outputs/judge and outputs/run_1105, nothing is deleted.
#   taps    long notes as taps
#   rhythm  taps on lanes drawn at random: which rows have notes and how many
#           (not run when taps leaves little: it can only leave less)
#   deaf    rhythm without the audio (not run when rhythm leaves little)
#   full    everything, with the human chart's SR: the views' reference
#   table   results_1109.md (scripts/judge_table.py)
# All: for p in taps rhythm deaf full table; do sh run_1109.sh $p; done
# Time: each judge stops after 2 of its 5 folds once its auc and 95% interval over songs lie
# below 0.65, between 0.65 and 0.86, or above 0.86 (--stop-bounds; ~14 min a judge then, ~35
# without), and scores the held-out songs after every epoch (--eval-epochs, ~4% more) to learn
# whether a judge needs 3 epochs. About 1 h if every judge stops at 2 folds, 2.5 h at most.
set -ex
PHASE=${1:-taps}
P=outputs/full-v4/best.pt
D=outputs/run_1105
COSF=${COSF:-$(cat $D/cos240.path)}
[ -d "$COSF" ] || { echo "no COSF=$COSF (run_1108's cosine run)"; exit 2; }
LITTLE=0.65
J="python scripts/judge.py --ckpt $P --ai $COSF --sr human --eval-epochs --stop-bounds $LITTLE,0.86 --min-folds 2"

# 0 when the judge's auc is LITTLE or more (something is left), not 0 when not or without it
left() {
    python -c "import json, sys; s = json.load(open('outputs/judge/$1/summary.json'))['ai']; \
print('$1 auc', round(s['auc'], 3)); sys.exit(0 if s['auc'] >= $LITTLE else 1)"
}

case $PHASE in
taps)
    $J --view taps --name view_taps
    ;;
rhythm)
    if left view_taps; then
        $J --view rhythm --name view_rhythm
    else
        echo "taps leaves little: the rhythm view is not run"
    fi
    ;;
deaf)
    if [ -f outputs/judge/view_rhythm/summary.json ] && left view_rhythm; then
        $J --view rhythm --no-audio --name view_deaf
    else
        echo "no rhythm view, or it leaves little: the deaf view is not run"
    fi
    ;;
full)
    $J --view full --name view_full
    ;;
table)
    python scripts/judge_table.py outputs/judge/cosine outputs/judge/view_full \
        outputs/judge/view_taps outputs/judge/view_rhythm outputs/judge/view_deaf \
        --out $D/results_1109.md
    ;;
*)
    echo "phases: taps rhythm deaf full table"; exit 2
    ;;
esac
