#!/bin/sh
# Night of 10/5-10/6 (~5-6 h). 1) A pairs playtest pack with lane guidance 2: 16 new songs
# (seed 81: none from earlier packs), starts at once, needs no new code. 2) Waits up to 2 h for
# patch 0035 (onset-strength gate + quiet-section metrics) in outputs/patches, applies it, then
# 3) the quiet-section experiments on the first 60 val songs (full-v4, oracle stats, lane
# guidance 2), predictions in EXPERIMENTS 2026-10-06 (quiet sections).
P=outputs/full-v4/best.pt
python scripts/playtest_pack.py --ckpt $P --pairs --songs 16 --seed 81 --tag 1006g \
    --out outputs/full-v4/playtest_lg2 \
    --settings random:128:continue:fwd+ref2+cp0+lbq0.1+st+lg2 \
               random:128:continue:fwd+ref2+cp0+lbq0.1+stsample+lg2
i=0
while [ $i -lt 120 ] && ! ls outputs/patches/0035-*.patch >/dev/null 2>&1; do
    sleep 60; i=$((i + 1))
done
if ls outputs/patches/0035-*.patch >/dev/null 2>&1 && git am outputs/patches/0035-*.patch; then
    B=outputs/full-v4/eval_val_songs_continue_random_T128_fwd_ref2t0.5_st-oracle_lg2
    E="python scripts/evaluate.py --ckpt $P --per-song --n 60 --lanes forward --refine 2 --stats oracle --lane-guidance 2"
    python scripts/evaluate.py --ckpt $P --per-song --n 60 --from-charts $B   # new columns, reference
    $E --onset-bias 1
    $E --onset-bias 2
    $E --loud-bias 0.3
    $E --onset-bias 1 --loud-bias 0.3
else
    echo "patch 0035 not found or not applied: sampled styles with lane guidance 2 instead"
    python scripts/evaluate.py --ckpt $P --per-song --n 60 --lanes forward --refine 2 --stats sample --lane-guidance 2
fi
