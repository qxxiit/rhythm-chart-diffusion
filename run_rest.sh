#!/bin/sh
# Night of 10/7-10/8 (~5 h). Apply patch 0038 first (git am outputs/patches/0038-*.patch).
# Choices for a whole bar (EXPERIMENTS 2026-10-07 night; the predictions are there):
# 1) probe: with a bar masked, does the model expect the bars humans leave empty, and the
#    long-note bars? The human charts as the context, then the candidate's own charts.
# 2) the rest pass on the candidate's saved charts (240 val songs, no new sampling): 0.5, 1, 2
#    (and the candidate rescored, for the new bar columns).
# 3) the candidate sampled at the target SR + 0.15 (the gate leaves it 0.14 under), then the
#    rest pass at 1 on those charts. The comparisons are printed at the end.
set -x
P=outputs/full-v4/best.pt
C=outputs/full-v4/eval_val_songs_continue_random_T128_fwd_ref2t0.5_lbq0.1_og1_st-oracle_lg2
S=outputs/full-v4/eval_val_songs_continue_random_T128_fwd_ref2t0.5_lbq0.1_og1_so0.15_st-oracle_lg2
python scripts/probe_bars.py --ckpt $P --stats oracle
python scripts/probe_bars.py --ckpt $P --stats oracle --charts $C/charts.npz
python scripts/evaluate.py --ckpt $P --per-song --n 0 --from-charts $C
E="python scripts/evaluate.py --ckpt $P --per-song --n 0 --stats oracle"
for t in 0.5 1 2; do $E --from-charts $C --rest $t; done
$E --lanes forward --refine 2 --lane-guidance 2 --onset-bias 1 --loud-bias 0.1 --sr-offset 0.15
$E --from-charts $S --rest 1
python scripts/compare_runs.py ${C}_rescored ${C}_rb0.5_st-oracle ${C}_rb1_st-oracle \
    ${C}_rb2_st-oracle $S ${S}_rb1_st-oracle
