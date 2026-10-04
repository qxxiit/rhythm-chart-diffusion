#!/bin/sh
# 10/4-10/5: long-note passes and loudness bias on saved charts, then full-v3 (row masks).
B=outputs/full-v1/eval_val_songs_continue_random_T128_fwd_ref2t0.5
E="python scripts/evaluate.py --ckpt outputs/full-v1/best.pt --per-song"
$E --n 0 --from-charts $B --copy-bias 0
$E --n 0 --from-charts $B --refine-holds --copy-bias 0
$E --n 0 --from-charts $B --hold-share oracle --copy-bias 0
$E --n 0 --from-charts $B
$E --n 60 --lanes forward --refine 2 --loud-bias 0.1
python scripts/train.py --steps 60000 --val-every 2000 --row-mask 0.5 --run full-v3 &&
python scripts/pattern_probe.py --ckpt outputs/full-v3/best.pt &&
python scripts/evaluate.py --ckpt outputs/full-v3/best.pt --per-song --n 0 --lanes forward --refine 2
