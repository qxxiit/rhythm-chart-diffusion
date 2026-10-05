#!/bin/sh
# 10/6-10/7: a pairs playtest pack with full-v4 (ready first, ~30 min), then lane guidance
# towards the chart stats on 60 val songs (~2.5 h). Results: EXPERIMENTS 2026-10-06.
python scripts/playtest_pack.py --ckpt outputs/full-v4/best.pt --pairs --songs 16 --seed 1 \
    --settings random:128:continue:fwd+ref2+cp0+lbq0.1+st+lg1 \
               random:128:continue:fwd+ref2+cp0+lbq0.1+stsample+lg1
E="python scripts/evaluate.py --ckpt outputs/full-v4/best.pt --per-song --n 60 --lanes forward --refine 2 --stats oracle"
$E --lane-guidance 1
$E --lane-guidance 2
