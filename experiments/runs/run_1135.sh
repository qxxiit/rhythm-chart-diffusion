#!/bin/sh
# 11/21 (~4 h 30 min): the system drawn with pat-v2. run_1133's pat-v2 (six chart stats from its
# first step) follows the song's long-note length and chord share better than full-v4 in raw
# draws, the judges level; whether that holds through the system decides it. As full-v4's
# system was made: a draw of pat-v2 without the choice (the system's draw settings otherwise;
# the first 120 val songs), a holds-view judge on it from pat-v2's trunk with its fold models
# saved (sel_holds_p2, as sel_holds from full-v4's), then scripts/system.sh with CKPT pat-v2 and
# that judge (4 draws a window, the system's passes). Against the system (run_1134's mbh, its
# judges s1_mbh, s1_mbh_taps), judged from both trunks: full-v4's (s1_) and pat-v2's (p2_), full
# and taps views. Why and the predictions: EXPERIMENTS 2026-11-21 (1). Writes under outputs/
# only, nothing is deleted.
#   plain   (~1 h 30 min) pat-v2's draw without the choice, three processes
#   choose  (~15 min) the choosing judge sel_holds_p2
#   system  (~1 h 35 min) scripts/system.sh val 120 with pat-v2 (OUT=outputs/system_p2)
#   judges  (~1 h 15 min) 6 judges
#   table   (~5 min)
# All: for p in plain choose system judges table; do sh experiments/runs/run_1135.sh $p; done
set -ex
PHASE=${1:-plain}
SHARDS=${SHARDS:-3}
P=outputs/full-v4/best.pt
G=outputs/pat-v2/best.pt
D=outputs/run_1105
J=outputs/judge
SP=outputs/system_p2
[ -f $G ] || { echo "no $G (run_1133)"; exit 2; }
[ -f $D/mbh.path ] || { echo "no $D/mbh.path (run_1134: the system)"; exit 2; }
[ -f $J/s1_mbh_taps/summary.json ] || { echo "no $J/s1_mbh_taps (run_1134)"; exit 2; }

EV="python scripts/evaluate.py --ckpt $G --split val --n 120 --per-song --lane-guidance 2 \
--lane-temp 1 --stats oracle"
PLAIN="--lanes forward --refine 2 --loud-bias 0.1 --onset-bias 1 --onset-taper 2.7,5,0.35 \
--schedule cosine --sample-seed 0 --hold-bias 0.2"
JUDGE="python scripts/judge.py --seed 1 --sr view --epochs 2 --eval-epochs --limit 120"

judged() {               # TRUNK(s1|p2) NAME VIEW RUN
    ck=$P
    [ $1 = p2 ] && ck=$G
    sfx=""
    [ $3 = full ] || sfx=_$3
    [ -f $J/$1_$2$sfx/summary.json ] && { echo "$1_$2$sfx is there"; return; }
    $JUDGE --ckpt $ck --ai $4 --view $3 --name $1_$2$sfx
}

case $PHASE in
plain)
    [ -f $D/plainp2.path ] && { echo "plainp2 is there"; exit 0; }
    k=0
    while [ $k -lt $SHARDS ]; do
        $EV $PLAIN --shard $k/$SHARDS > $D/plainp2_shard$k.log 2>&1 &
        k=$((k + 1))
    done
    wait
    run=$(sed -n 's/^run //p' $D/plainp2_shard0.log | head -n 1)
    [ -n "$run" ] || { echo "no run line in $D/plainp2_shard0.log"; exit 2; }
    python scripts/merge_shards.py "$run"
    echo "$run" > $D/plainp2.path
    ;;
choose)
    [ -f $D/plainp2.path ] || { echo "no plainp2: nothing to train on"; exit 2; }
    [ -f $J/sel_holds_p2/summary.json ] && { echo "sel_holds_p2 is there"; exit 0; }
    python scripts/judge.py --ckpt $G --ai $(cat $D/plainp2.path) --view holds --sr human \
        --epochs 2 --eval-epochs --save-models --name sel_holds_p2
    ;;
system)
    [ -d $J/sel_holds_p2 ] || { echo "no $J/sel_holds_p2"; exit 2; }
    [ -f $D/sysp2.path ] && { echo "sysp2 is there"; exit 0; }
    OUT=$SP CKPT=$G JUDGE=$J/sel_holds_p2 sh scripts/system.sh val 120
    cp $SP/val_n120.path $D/sysp2.path
    ;;
judges)
    [ -f $D/sysp2.path ] || { echo "no sysp2: nothing to judge"; exit 0; }
    judged s1 sysp2 full $(cat $D/sysp2.path)
    judged s1 sysp2 taps $(cat $D/sysp2.path)
    for m in mbh sysp2; do
        judged p2 $m full $(cat $D/$m.path)
        judged p2 $m taps $(cat $D/$m.path)
    done
    ;;
table)
    for t in s1 p2; do
        [ -f $J/${t}_sysp2_taps/summary.json ] || continue
        python scripts/judge_table.py $J/${t}_mbh $J/${t}_sysp2 --out $D/results_1135_$t.md
        python scripts/judge_table.py $J/${t}_mbh_taps $J/${t}_sysp2_taps \
            --out $D/results_1135_${t}_taps.md
    done
    if [ -f $D/sysp2.path ]; then
        python scripts/diag_judge.py $(cat $D/mbh.path) $(cat $D/sysp2.path) \
            --out $D/diag_judge_1135.md
        python scripts/results_table.py $(cat $D/mbh.path) $(cat $D/sysp2.path) \
            --out $D/results_1135_runs.md
        python scripts/bar_copies.py $(cat $D/sysp2.path) $(cat $D/mbh.path) \
            --out $D/results_1135_copies.md
    fi
    ;;
*)
    echo "phases: plain choose system judges table"; exit 2
    ;;
esac
