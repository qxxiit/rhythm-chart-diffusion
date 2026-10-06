#!/bin/sh
# Night of 10/6-10/7 (~6.5 h). Apply patch 0036 first (git am outputs/patches/0036-*.patch).
# 1) A pairs playtest pack of the playable candidate (full-v4, lane guidance 2, onset gate 1,
#    quiet bars 0.1, copies): 16 songs none of the earlier packs had (--exclude), tag 1007.
# 2) The onset gate as a start penalty on the first 60 val songs (1, 1.5), against last
#    night's runs. 3) The candidate on all 240 val songs, against full-v4 oracle (rescored).
# Predictions: EXPERIMENTS 2026-10-06/07 (the onset gate and a fix).
P=outputs/full-v4/best.pt
EARLIER="outputs/full-v1/playtest/answers.csv outputs/full-v1/playtest_1002/answers.csv outputs/full-v4/playtest/answers.csv outputs/full-v4/playtest_lg2/answers.csv"
python scripts/playtest_pack.py --ckpt $P --pairs --songs 16 --seed 3 --tag 1007 \
    --exclude $EARLIER --out outputs/full-v4/playtest_gate \
    --settings random:128:continue:fwd+ref2+cp0+lbq0.1+st+lg2+og1 \
               random:128:continue:fwd+ref2+cp0+lbq0.1+stsample+lg2+og1
E="python scripts/evaluate.py --ckpt $P --per-song --lanes forward --refine 2 --stats oracle --lane-guidance 2"
$E --n 60 --onset-bias 1
$E --n 60 --onset-bias 1.5
python scripts/evaluate.py --ckpt $P --per-song --n 0 \
    --from-charts outputs/full-v4/eval_val_songs_continue_random_T128_fwd_ref2t0.5_st-oracle
$E --n 0 --onset-bias 1 --loud-bias 0.1
