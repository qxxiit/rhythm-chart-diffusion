#!/bin/sh
# 10/5-10/6: full-v4 (the chart's own long-note share, jack and trill rate as inputs, plus
# row masks), its evaluation, then the quiet-side loudness bias on full-v1. ~9 h on the MacBook.
python scripts/build_chart_stats.py &&
python scripts/train.py --steps 60000 --val-every 2000 --row-mask 0.5 \
    --chart-stats data/chart_stats.csv --run full-v4 &&
python scripts/evaluate.py --ckpt outputs/full-v4/best.pt --per-song --n 0 --lanes forward --refine 2 --stats oracle &&
python scripts/evaluate.py --ckpt outputs/full-v4/best.pt --per-song --n 60 --lanes forward --refine 2 &&
python scripts/evaluate.py --ckpt outputs/full-v4/best.pt --per-song --n 60 --lanes forward --refine 2 --stats sample
python scripts/evaluate.py --ckpt outputs/full-v1/best.pt --per-song --n 60 --lanes forward --refine 2 --loud-bias 0.1
