#!/bin/sh
# 11/25 night (~4 h): the rest of the final table on the test split (run_1143's 281 songs).
# (1) The steps of the system, each by the judges: full-v4's draw as it comes (system.sh's draw
# flags without the choosing: one draw a window), against run_1143's draw (chosen among 4 by
# sel_holds) and the system (then the passes). (2) The same draw with every chunk drawn on its
# own (--mode independent: no continuation, holds closed at the chunk edges). (3) The judge's
# AUC on the system against the songs it learns from (the first 60, 120 and 200 of the 281):
# val's judges had 120, test's 281, and the table has to say which numbers compare. Judges as
# in run_1143 (full-v4's trunk, seed 1, the SR as shown, 2 epochs). Aggregate only: the
# playtest's 16 songs are among them. Why and the predictions: EXPERIMENTS 2026-11-25 (1).
# Needs run_1143's outputs. Writes under outputs/ only, nothing is deleted.
#   raw      (~50 min) the draw without the choosing (outputs/system_test/test_n0_raw.path)
#   rawj     (~55 min) its full and taps judges
#   curve    (~1 h 10 min) the system's judges on 60, 120, 200 songs (full), 120 (taps)
#   ind      (~50 min) the raw draw, every chunk on its own (test_n0_ind.path)
#   indj     (~25 min) its full judge
#   table    (~10 min) results_1144*.md
# All: for p in raw rawj curve ind indj table; do sh experiments/runs/run_1144.sh $p; done
set -ex
PHASE=${1:-raw}
P=outputs/full-v4/best.pt
D=outputs/run_1105
J=outputs/judge
S=outputs/system_test
SHARDS=${SHARDS:-3}
JUDGE="python scripts/judge.py --ckpt $P --seed 1 --sr view --epochs 2 --eval-epochs"
# system.sh's EV and DRAW (DECISIONS through 2026-11-23) without --select-judge / --select-k
EV="python scripts/evaluate.py --ckpt $P --split test --n 0 --per-song --lane-guidance 2 \
--lane-temp 1 --stats oracle"
RAW="--lanes forward --refine 2 --loud-bias 0.1 --onset-bias 1 --onset-taper 2.7,5,0.35 \
--schedule cosine --sample-seed 0 --hold-bias 0.2"

sharded() {             # NAME FLAGS...: as system.sh, the run's path in $S/NAME.path
    name=$1
    shift
    k=0
    while [ $k -lt $SHARDS ]; do
        $EV "$@" --shard $k/$SHARDS > $S/${name}_shard$k.log 2>&1 &
        k=$((k + 1))
    done
    wait
    run=$(sed -n 's/^run //p' $S/${name}_shard0.log | head -n 1)
    [ -n "$run" ] || { echo "no run line in $S/${name}_shard0.log"; exit 2; }
    python scripts/merge_shards.py "$run"
    echo "$run" > $S/$name.path
}

[ -f $S/test_n0.path ] || { echo "no $S/test_n0.path (run_1143 first)"; exit 2; }
sys=$(cat $S/test_n0.path)
draw=$(cat $S/test_n0_draw.path)

case $PHASE in
raw)
    [ -f $S/test_n0_raw.path ] && { echo "the raw draw is there"; exit 0; }
    sharded test_n0_raw $RAW
    ;;
rawj)
    raw=$(cat $S/test_n0_raw.path)
    [ -f $J/t1_raw/summary.json ] || $JUDGE --ai $raw --name t1_raw
    [ -f $J/t1_raw_taps/summary.json ] || $JUDGE --ai $raw --view taps --name t1_raw_taps
    ;;
curve)
    for n in 60 120 200; do
        [ -f $J/t1_sys_l$n/summary.json ] || $JUDGE --ai $sys --limit $n --name t1_sys_l$n
    done
    [ -f $J/t1_sys_taps_l120/summary.json ] \
        || $JUDGE --ai $sys --view taps --limit 120 --name t1_sys_taps_l120
    ;;
ind)
    [ -f $S/test_n0_ind.path ] && { echo "the independent draw is there"; exit 0; }
    sharded test_n0_ind $RAW --mode independent
    ;;
indj)
    ind=$(cat $S/test_n0_ind.path)
    [ -f $J/t1_ind/summary.json ] || $JUDGE --ai $ind --name t1_ind
    ;;
table)
    raw=$(cat $S/test_n0_raw.path)
    runs="$raw $draw $sys"
    judges="$J/t1_raw $J/t1_raw_taps $J/t1_draw $J/t1_draw_taps $J/t1_sys $J/t1_sys_taps"
    if [ -f $S/test_n0_ind.path ]; then
        runs="$raw $(cat $S/test_n0_ind.path) $draw $sys"
        [ -f $J/t1_ind/summary.json ] && judges="$judges $J/t1_ind"
    fi
    python scripts/diag_judge.py $runs --out $D/diag_judge_1144.md || true
    python scripts/results_table.py $runs --judge $judges --out $D/results_1144.md
    python scripts/judge_table.py $J/t1_raw $J/t1_draw $J/t1_sys --out $D/results_1144_full.md
    python scripts/judge_table.py $J/t1_raw_taps $J/t1_draw_taps $J/t1_sys_taps \
        --out $D/results_1144_taps.md
    [ -f $J/t1_ind/summary.json ] \
        && python scripts/judge_table.py $J/t1_raw $J/t1_ind --out $D/results_1144_ind.md
    python - $J <<'EOF' | tee $D/results_1144_curve.md
import json, sys
from pathlib import Path
J = Path(sys.argv[1])
print("| the system's judge | view | songs it learns from | AUC [95%] | paired | by epoch |")
print("|---|---|---|---|---|---|")
for name, view in [("t1_sys_l60", "full"), ("t1_sys_l120", "full"), ("t1_sys_l200", "full"),
                   ("t1_sys", "full"), ("t1_sys_taps_l120", "taps"), ("t1_sys_taps", "taps")]:
    f = J / name / "summary.json"
    if not f.exists():
        print(f"| {name} | {view} | | (not run) | | |")
        continue
    s = json.loads(f.read_text())
    b = s["ai"]
    ci = f" [{b['auc_ci'][0]:.3f}, {b['auc_ci'][1]:.3f}]" if "auc_ci" in b else ""
    ep = ", ".join(f"{e['auc']:.3f}" for e in s.get("by_epoch", []))
    print(f"| {name} | {view} | {b['songs']} | {b['auc']:.3f}{ci} | {b['paired']:.3f} | {ep} |")
print()
print("val's judges (run_1119, run_1134) learnt from 120 songs: compare them with the 120 rows.")
EOF
    ;;
*)
    echo "phases: raw rawj curve ind indj table"; exit 2
    ;;
esac
