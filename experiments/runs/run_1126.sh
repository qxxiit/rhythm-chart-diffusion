#!/bin/sh
# 11/16 evening (~5 h 15 min): is stems-v2 worse, or only worse through the system? run_1125:
# val_ce -1.7% against full-v4, but through the system the fresh judges +0.083 (full view) and
# +0.128 (rhythm view). Two things there were set on full-v4: (1) the judges themselves start
# from full-v4's trunk (judge.py --ckpt), which may see full-v4's own mistakes less; (2) the
# system's settings (hold bias 0.2, the loud and onset biases, the choosing judge sel_holds
# trained on full-v4's draws, F's passes). Here (1) the same two runs judged again from
# stems-v2's trunk, and (2) a draw of each model with the system's decoding only (cosine,
# forward lanes at temperature 1, refine 2, lane guidance 2, the oracle stats, one draw a
# window, nothing chosen, no bias, no pass) on the first 120 val songs, judged from both
# trunks. Why and the predictions: EXPERIMENTS 2026-11-16 (2). Writes under outputs/ only,
# nothing is deleted.
#   trunk      (~55 min) fresh full- and rhythm-view judges from stems-v2's trunk on the
#              system (s1_sh's run) and on stm2f (s1_stm2's)
#   raw        (~2 h 40 min) the two draws, three processes each
#   rawjudges  (~1 h 40 min) fresh judges on both draws, both views, both trunks
#   table      (seconds)
# All: for p in trunk raw rawjudges table; do sh experiments/runs/run_1126.sh $p; done
set -ex
PHASE=${1:-trunk}
SHARDS=${SHARDS:-3}
P=outputs/full-v4/best.pt
G=outputs/stems-v2/best.pt
D=outputs/run_1105
J=outputs/judge
for f in $D/shf.path $D/stm2f.path $G; do
    [ -f $f ] || { echo "no $f (run_1125)"; exit 2; }
done
SHF=$(cat $D/shf.path)
STM2F=$(cat $D/stm2f.path)

RAW="--schedule cosine --lanes forward --refine 2 --sample-seed 0"
EV="python scripts/evaluate.py --per-song --lane-guidance 2 --lane-temp 1 --stats oracle --n 120"
JUDGE="python scripts/judge.py --seed 1 --sr view --epochs 2 --eval-epochs --limit 120"

merged() {
    run=$(sed -n 's/^run //p' $D/${1}_shard0.log | head -n 1)
    [ -n "$run" ] || { echo "no run line in $D/${1}_shard0.log"; exit 2; }
    python scripts/merge_shards.py "$run"
    echo "$run" > $D/$1.path
}

judged() {               # PREFIX CKPT NAME RUN: fresh full- and rhythm-view judges from CKPT
    [ -f $J/$1_$3_rhythm/summary.json ] && { echo "$1_$3 is there"; return; }
    $JUDGE --ckpt $2 --ai $4 --name $1_$3
    $JUDGE --ckpt $2 --ai $4 --view rhythm --name $1_$3_rhythm
}

case $PHASE in
trunk)
    judged t2 $G sh $SHF
    judged t2 $G stm2 $STM2F
    ;;
raw)
    for m in 4 2; do
        [ -f $D/raw$m.path ] && { echo "raw$m is there"; continue; }
        ck=$P
        [ $m = 2 ] && ck=$G
        k=0
        while [ $k -lt $SHARDS ]; do
            $EV --ckpt $ck $RAW --shard $k/$SHARDS > $D/raw${m}_shard$k.log 2>&1 &
            k=$((k + 1))
        done
        wait
        merged raw$m
    done
    ;;
rawjudges)
    for m in 4 2; do
        [ -f $D/raw$m.path ] || { echo "no raw$m: skipped"; continue; }
        judged s1 $P raw$m $(cat $D/raw$m.path)
    done
    for m in 4 2; do
        [ -f $D/raw$m.path ] || continue
        judged t2 $G raw$m $(cat $D/raw$m.path)
    done
    ;;
table)
    pair() {             # A B OUT: B against A, both views, where both are there
        [ -f $J/$1/summary.json ] && [ -f $J/$2/summary.json ] || return 0
        python scripts/judge_table.py $J/$1 $J/$2 --out $D/$3.md
        [ -f $J/$2_rhythm/summary.json ] && \
            python scripts/judge_table.py $J/$1_rhythm $J/$2_rhythm --out $D/$3_rhythm.md
        return 0
    }
    pair t2_sh t2_stm2 results_1126_trunk
    pair s1_raw4 s1_raw2 results_1126_raw
    pair t2_raw4 t2_raw2 results_1126_raw_t2
    if [ -f $D/raw4.path ] && [ -f $D/raw2.path ]; then
        python scripts/diag_judge.py $(cat $D/raw4.path) $(cat $D/raw2.path) \
            --out $D/diag_judge_raw.md
        python scripts/results_table.py $(cat $D/raw4.path) $(cat $D/raw2.path) \
            --out $D/results_raw.md
    fi
    ;;
*)
    echo "phases: trunk raw rawjudges table"; exit 2
    ;;
esac
