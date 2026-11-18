#!/bin/sh
# 11/19 (~5 h 30 min): each song's own patterns as inputs. The user's blind play: a human
# chart keeps to its own patterns (one song stairs, another trills; long notes of its own
# length), the AI mixes the same everywhere. Over the 240 val songs a song's stair share
# follows its human chart's at 0.17 (trills 0.36, conditioned already) and varies half as
# much from song to song (SD 0.059 against 0.117); the long notes' length varies 0.42 beats
# against 0.72. chartstats.MORE (move_stair, hold_beats) added to full-v4's three chart stats:
# train.py --init full-v4 --stats-names (the new stats' embeddings start at zero), 20k steps
# at lr 1e-4 on full-v4's recipe, exactly as ft-v1 (run_1123's control: the same without the
# new stats); then a raw draw of each on the first 120 val songs (the system's decoding, the
# human chart's stats as inputs, one draw a window, no choice, bias or pass), fresh full- and
# taps-view judges, and the pattern numbers. Why and the predictions: EXPERIMENTS 2026-11-19
# (2). Writes under outputs/ (and data/chart_stats_more.csv) only, nothing is deleted.
#   stats   (~5 min) data/chart_stats_more.csv
#   train   (~1 h 15 min) outputs/pat-v1
#   draw    (~3 h 15 min) the raw draws of ft-v1 and pat-v1, three processes each
#   judges  (~55 min) full- and taps-view judges on both
#   table   (seconds)
# All: for p in stats train draw judges table; do sh experiments/runs/run_1132.sh $p; done
set -ex
PHASE=${1:-stats}
SHARDS=${SHARDS:-3}
P=outputs/full-v4/best.pt
F=outputs/ft-v1/best.pt
G=outputs/pat-v1/best.pt
D=outputs/run_1105
J=outputs/judge
CS=data/chart_stats_more.csv
[ -f $F ] || { echo "no $F (run_1123)"; exit 2; }

RAW="--schedule cosine --lanes forward --refine 2 --sample-seed 0"
EV="python scripts/evaluate.py --per-song --lane-guidance 2 --lane-temp 1 --stats oracle --n 120"
JUDGE="python scripts/judge.py --ckpt $P --seed 1 --sr view --epochs 2 --eval-epochs --limit 120"

merged() {
    run=$(sed -n 's/^run //p' $D/${1}_shard0.log | head -n 1)
    [ -n "$run" ] || { echo "no run line in $D/${1}_shard0.log"; exit 2; }
    python scripts/merge_shards.py "$run"
    echo "$run" > $D/$1.path
}

judged() {               # NAME VIEW RUN
    sfx=""
    [ $2 = full ] || sfx=_$2
    [ -f $J/s1_$1$sfx/summary.json ] && { echo "s1_$1$sfx is there"; return; }
    if [ $2 = full ]; then
        $JUDGE --ai $3 --name s1_$1
    else
        $JUDGE --ai $3 --view $2 --name s1_$1$sfx
    fi
}

case $PHASE in
stats)
    [ -f $CS ] && { echo "$CS is there"; exit 0; }
    python scripts/build_chart_stats.py --more --out $CS
    ;;
train)
    [ -f $G ] && { echo "$G is there: not trained again"; exit 0; }
    python scripts/train.py --init $P --chart-stats $CS \
        --stats-names hold_share,move_jack,move_trill,move_stair,hold_beats --row-mask 0.5 \
        --steps 20000 --lr 1e-4 --warmup 500 --val-every 2000 --run pat-v1 > $D/pat_train.log 2>&1
    ;;
draw)
    [ -f $G ] || { echo "no $G: nothing to do"; exit 0; }
    for m in ft pat; do
        [ -f $D/raw$m.path ] && { echo "raw$m is there"; continue; }
        ck=$F
        [ $m = pat ] && ck=$G
        k=0
        while [ $k -lt $SHARDS ]; do
            $EV --ckpt $ck $RAW --shard $k/$SHARDS > $D/raw${m}_shard$k.log 2>&1 &
            k=$((k + 1))
        done
        wait
        merged raw$m
    done
    ;;
judges)
    for m in ft pat; do
        [ -f $D/raw$m.path ] || { echo "no raw$m: skipped"; continue; }
        judged raw$m full $(cat $D/raw$m.path)
        judged raw$m taps $(cat $D/raw$m.path)
    done
    ;;
table)
    if [ -f $J/s1_rawpat_taps/summary.json ] && [ -f $J/s1_rawft_taps/summary.json ]; then
        python scripts/judge_table.py $J/s1_rawft $J/s1_rawpat --out $D/results_1132.md
        python scripts/judge_table.py $J/s1_rawft_taps $J/s1_rawpat_taps \
            --out $D/results_1132_taps.md
        python scripts/diag_judge.py $(cat $D/rawft.path) $(cat $D/rawpat.path) \
            --out $D/diag_judge_1132.md
        python scripts/results_table.py $(cat $D/rawft.path) $(cat $D/rawpat.path) \
            --out $D/results_1132_runs.md
    fi
    for r in full-v4 ft-v1 pat-v1; do
        [ -f outputs/$r/val.csv ] && printf '%s ' $r && tail -n 1 outputs/$r/val.csv
    done | tee $D/results_1132_val.txt
    ;;
*)
    echo "phases: stats train draw judges table"; exit 2
    ;;
esac
