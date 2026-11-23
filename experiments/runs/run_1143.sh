#!/bin/sh
# 11/23 night (~6 h): the final results table (docs/EVALUATION.md: a row per system, the same
# columns) on the test split, which chose nothing: the system as frozen (scripts/system.sh,
# DECISIONS through 2026-11-23) on every test song (281, a chart each as evaluate.py
# --per-song picks), its draw before the passes (system.sh's first stage: chosen window by
# window, no passes), and another human chart of each song (another difficulty of its set on
# the same grid, corrupt_test.py --other-human) with the human charts moved off sync (timing)
# as the judge's controls. Full and taps judges as on val (full-v4's trunk, seed 1, the SR as
# shown, 2 epochs), now on all the songs. Aggregate only: the playtest's 16 songs are among
# them. Why and the predictions: EXPERIMENTS 2026-11-23 (4). Writes under outputs/ only,
# nothing is deleted.
#   system   (~3 h 45 min) scripts/system.sh test 0 (OUT=outputs/system_test)
#   control  (~20 min) the other human chart, and timing at three doses
#   judges   (~2 h) four judges: the system and the draw, full and taps views
#   table    (~5 min) results_1143*.md
# All: for p in system control judges table; do sh experiments/runs/run_1143.sh $p; done
set -ex
PHASE=${1:-system}
P=outputs/full-v4/best.pt
D=outputs/run_1105
J=outputs/judge
S=outputs/system_test
C=outputs/corrupt/test_1143
JUDGE="python scripts/judge.py --ckpt $P --seed 1 --sr view --epochs 2 --eval-epochs"

case $PHASE in
system)
    [ -f $S/test_n0.path ] && { echo "the system's test charts are there"; exit 0; }
    OUT=$S sh scripts/system.sh test 0
    ;;
control)
    [ -f $C/summary.json ] && { echo "the controls are there"; exit 0; }
    mkdir -p $C
    python scripts/corrupt_test.py --split test --per-song --n 0 --write-runs --other-human \
        --corruptions timing --out $C > $C.log 2>&1
    tail -n 2 $C.log
    ;;
judges)
    [ -f $S/test_n0.path ] || { echo "no $S/test_n0.path"; exit 2; }
    sys=$(cat $S/test_n0.path)
    draw=$(cat $S/test_n0_draw.path)
    runs=$(ls -d $C/runs/* 2>/dev/null || true)
    score=""
    [ -n "$runs" ] && score="--score $runs"
    [ -f $J/t1_sys/summary.json ] || $JUDGE --ai $sys $score --name t1_sys
    [ -f $J/t1_sys_taps/summary.json ] || $JUDGE --ai $sys --view taps $score --name t1_sys_taps
    [ -f $J/t1_draw/summary.json ] || $JUDGE --ai $draw --name t1_draw
    [ -f $J/t1_draw_taps/summary.json ] || $JUDGE --ai $draw --view taps --name t1_draw_taps
    ;;
table)
    sys=$(cat $S/test_n0.path)
    draw=$(cat $S/test_n0_draw.path)
    python scripts/diag_judge.py $sys $draw --out $D/diag_judge_1143.md
    python scripts/results_table.py $sys $draw --judge $J/t1_sys $J/t1_sys_taps $J/t1_draw \
        $J/t1_draw_taps --out $D/results_1143.md
    python scripts/judge_table.py $J/t1_draw $J/t1_sys --out $D/results_1143_full.md
    python scripts/judge_table.py $J/t1_draw_taps $J/t1_sys_taps --out $D/results_1143_taps.md
    python - $J/t1_sys $J/t1_sys_taps <<'EOF' | tee $D/results_1143_controls.md
import json, re, sys
views = [json.load(open(f"{p}/summary.json")) for p in sys.argv[1:]]
def cell(s, k):
    b = s.get(k)
    return " | " if not b else f"{b['paired']:.3f} | {b['songs_paired']:.3f}"
labels = [k for k in views[0]["labels"] if k != "ai"]
order = lambda k: (k != "other_human", re.sub(r"_[0-9.]+$", "", k), k)
print("| test charts | songs | full: paired | full: songs | taps: paired | taps: songs |")
print("|---|---|---|---|---|---|")
print(f"| the system | {views[0]['ai']['songs']} | {cell(views[0], 'ai')} | {cell(views[1], 'ai')} |")
for k in sorted(labels, key=order):
    print(f"| {k} | {views[0][k]['songs']} | {cell(views[0], k)} | {cell(views[1], k)} |")
print()
print("paired: the share of windows scored above the human chart's at the same rows (0.5: no "
      "different from the human chart); songs: the same with a song's windows averaged.")
EOF
    ;;
*)
    echo "phases: system control judges table"; exit 2
    ;;
esac
