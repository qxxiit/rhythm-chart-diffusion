#!/bin/sh
# 11/22 night (~1 h 30 min): does the judge see what players hate? On 11/05 F's judge took the
# human charts with their lanes shuffled for more human than the human charts (it had learned
# that the AI charts have few jacks), and every model comparison since has had each trunk's
# judge favour its own model. Here the human charts of the first 120 val songs are corrupted
# the ways players and modders call bad (src/evaluation/corrupt.py: off-sync taps, shuffled
# lanes, taps on nothing, jumptrills, minijacks, one-hand trills, chords off the beat, short
# long notes over busy parts, the bar before pasted again), each at three doses, and another
# human chart of the same song (another difficulty of its set on the same grid) is the
# control. Judges trained as now (full-v4's trunk, the system's charts, run_1134's mbh: seed 1,
# the SR as shown, 2 epochs, 120 songs), full and taps views, score them all. Why and the
# predictions: EXPERIMENTS 2026-11-22 (1). Writes under outputs/ only, nothing is deleted.
#   runs    (~15 min) the corrupted runs and the control, three processes
#   judges  (~1 h 10 min) the two judges, scoring all of them
#   table   (seconds)
# All: for p in runs judges table; do sh experiments/runs/run_1136.sh $p; done
set -ex
PHASE=${1:-runs}
P=outputs/full-v4/best.pt
D=outputs/run_1105
J=outputs/judge
C=outputs/corrupt/val_1136
[ -f $D/mbh.path ] || { echo "no $D/mbh.path (run_1134: the system)"; exit 2; }
CT="python scripts/corrupt_test.py --split val --per-song --n 120 --write-runs"
JUDGE="python scripts/judge.py --ckpt $P --seed 1 --sr view --epochs 2 --eval-epochs --limit 120"

case $PHASE in
runs)
    [ -f $C/c/summary.json ] && { echo "the runs are there"; exit 0; }
    mkdir -p $C
    $CT --other-human --corruptions timing lanes new --out $C/a > $C.a.log 2>&1 &
    $CT --corruptions jumptrill minijack trill --out $C/b > $C.b.log 2>&1 &
    $CT --corruptions weakchord lnflood copypaste --out $C/c > $C.c.log 2>&1 &
    wait
    tail -n 2 $C.a.log $C.b.log $C.c.log
    ;;
judges)
    runs=$(ls -d $C/a/runs/* $C/b/runs/* $C/c/runs/*)
    for v in full taps; do
        sfx=""
        [ $v = full ] || sfx=_$v
        [ -f $J/s1_c1136$sfx/summary.json ] && { echo "s1_c1136$sfx is there"; continue; }
        $JUDGE --ai $(cat $D/mbh.path) --view $v --score $runs --name s1_c1136$sfx
    done
    ;;
table)
    python - $J/s1_c1136 $J/s1_c1136_taps <<'EOF' | tee $D/results_1136.md
import json, re, sys
views = [json.load(open(f"{p}/summary.json")) for p in sys.argv[1:]]
def cell(s, k):
    b = s.get(k)
    if not b:
        return " | "
    return f"{b['paired']:.3f} | {b['songs_paired']:.3f}"
labels = [k for k in views[0]["labels"] if k != "ai"]
order = lambda k: (k != "other_human", re.sub(r"_[0-9.]+$", "", k), k)
print("| chart (the human chart corrupted, dose) | songs | full: paired | full: songs | "
      "taps: paired | taps: songs |")
print("|---|---|---|---|---|---|")
print(f"| the system (mbh) | {views[0]['ai']['songs']} | {cell(views[0], 'ai')} | {cell(views[1], 'ai')} |")
for k in sorted(labels, key=order):
    print(f"| {k} | {views[0][k]['songs']} | {cell(views[0], k)} | {cell(views[1], k)} |")
print()
print("paired: the share of windows scored above the human chart's at the same rows (0.5: "
      "no different from the human chart); songs: the same with a song's windows averaged.")
EOF
    ;;
*)
    echo "phases: runs judges table"; exit 2
    ;;
esac
