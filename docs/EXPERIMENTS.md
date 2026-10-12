# Experiment Log

> One entry per significant training run. Newest at top.
> Add results to `paper/` when ready for the paper.

| Date | Run ID | Phase | Model | Key Config | Val F1 | Test F1 | wandb | Notes |
|------|--------|-------|-------|------------|--------|---------|-------|-------|
| 2026-10-04 | full-v2 | 2 | D-32 + genre / mapper embeddings, 6.93M | full-v1's recipe, labels dropped 0.1 / 0.1, `--style data/style.csv`, MPS, ~3.5 h | F1@50 0.377 (240 val songs, fwd + ref2, own labels) | - | - | val CE 0.0732 with labels = 0.0732 without (full-v1 0.0717): the labels go unused; differences to full-v1 not attributable to them |
| 2026-10-04 | full-v1 + lane passes + bar copies | 2 | D-32, 6.87M | `--copy-bias 0` on the saved fwd + ref2 charts (no training) | F1@50 0.375 (+0.001) | - | - | whole-bar copies about 15% of bars (human 14.5%), rho_in 0.219 → 0.231 (human 0.228) |
| 2026-10-03 | full-v1 + lane passes | 2 | D-32, 6.87M, audio_add | sampler: random T128, then forward_lanes and refine_lanes 2 at lane temperature 0.5 (no training) | F1@50 0.373 (240 val songs) | - | - | pattern coverage 0.047 (human 0.042), runs 7.6 (7.5), motion_pred 0.122 (0.132); jacks a quarter of the human rate |
| 2026-09-29 | full-v1 | 2 | D-32, 6.87M, audio_add | all train songs (16,410 charts, 218,142 chunks), 60k steps (4.4 epochs), batch 16, MPS, ~3 h | F1@50 0.373 (240 val songs, random T128) | - | - | val CE 0.072; vs subset: F1 same, SR error 0.375→0.236, rho at the human level, pattern coverage still 2x chance; sampler fixed to random T128 |
| 2026-09-27 | subset600-v1 | 2 | D-32, 6.87M, audio_add | 600 train songs (2,110 charts, 28,004 chunks), 20k steps, batch 16, MPS, 1 h | F1@50 0.37 (240 val songs, every sampler) | - | - | val CE 0.087, no overfitting; sampler moves density 0.81-1.16; without audio onset CE +79% |
| 2026-09-26 | 20260926-060904 | 2 | D-32, 6.87M, audio_add | `--overfit 10`, fake mel, 3000 steps, batch 16, MPS | - | - | - | Plumbing check; sampler order matters (see Phase 2) |
| TBD  | -      | 1     | -     | -          | -      | -       | -     | First baseline run |

## How to log

After a meaningful run completes:

1. Add a row to the table above.
2. Use the wandb run name as Run ID.
3. Link to the wandb run page.
4. Brief notes: what was tried, observations, next steps.

For ablation tables, add a subsection below.

## Phase 1 Baseline Reproduction

_TBD — target Aug 11, 2026._

## Phase 2 Diffusion Initial

### 2026-09-26 · Overfit check and sampler order (fake mel)

Recorded after the fact: no prediction was written before this run.

Run `20260926-060904`: D-32 (6,869,204 parameters, audio_add on), `--overfit 10`
(10 train charts, 161 chunks), 3,000 steps, batch 16, fp32 on MPS at 94 chunks/s.
The mel is `cache.oracle_mel`, which encodes the answer: this checks the code,
not learning from audio. Training ended at loss 0.0078 (val masked CE 0.0072);
chunk 0 rebuilt from all-MASK matched every cell.

Evaluation on the same 10 charts, T = 32:

| mode | order | F1@50 | precision@50 | recall@50 | density | SR error |
|---|---|---|---|---|---|---|
| continue | random | 0.976 | 0.953 | 1.000 | 1.049 | 0.451 |
| independent | random | 0.983 | 0.967 | 0.999 | 1.034 | 0.282 |
| continue | confidence | 1.000 | 0.9995 | 1.000 | 1.000 | 0.008 |

Tokenizer ceiling: F1@20 0.9945 (continue + confidence reached 0.9941). With
confidence order, rho and the pattern numbers equal the human chart's.

Reading: with random order the sampler adds about 5% notes even when the model
knows the answer, and the extra notes raise SR by 0.45. Window positions (fixed
chunks in training, a window every 6 bars in continuation) explain about a
third of it; the order explains the rest. The sampler order is therefore a
first-order setting, not an ablation: fix it before the main comparison
(queue 9), and read SR error together with precision and density. Decide on
real mel, where confidence order may under-generate instead.

### 2026-09-27 · Audio alignment (data check, before training)

`check_alignment.py`, train, 200 songs, twice (random and `--cached-only`). Onset
strength peaks 18.2-18.9 ms after the note times for mp3 with a LAME tag and
for ogg, whose starts osu! finds by different rules; mp3 without a tag 17.4 /
20.5 ms (27 / 22 songs); the token grid as the Dataset reads it gives the same
(+20.7 / +21.7 ms). Decoding rule confirmed on real files; no global shift
(DECISIONS 2026-09-27, design doc §3.4). Open: whether untagged mp3s split
into two groups about 12 ms apart (needs the full cache).

### 2026-09-27 · Training (Try #1)

Prediction: Final validation CE expected to be around 0.2, since there are no deterministic answer to the given audio. Loss at @0.1 is expected to be quite lower than @0.9. Train loss and val CE will start going different ways around 2000 steps. Failure if val CE > 0.55.

Run `subset600-v1`: D-32, `preprocess_data.py --mel --splits train val
--train-groups 600` (2,110 train charts, 28,004 chunks; val 785 charts, 10,664
chunks), `train.py --steps 20000 --val-every 1000`, defaults otherwise (batch
16, lr 3e-4, warmup 1k, cosine, dropout 0, chunk windows). MPS at ~91 chunks/s,
about one hour, 11.4 epochs. Before it, `--overfit 10` on real mel reached loss
0.0001 and rebuilt chunk 0 exactly.

| step | val CE | @0.1 | @0.3 | @0.5 | @0.7 | @0.9 |
|---|---|---|---|---|---|---|
| 1,000 | 0.275 | 0.212 | 0.244 | 0.270 | 0.300 | 0.347 |
| 5,000 | 0.152 | 0.070 | 0.100 | 0.133 | 0.180 | 0.276 |
| 10,000 | 0.100 | 0.040 | 0.057 | 0.077 | 0.112 | 0.215 |
| 20,000 | 0.087 | 0.031 | 0.047 | 0.066 | 0.098 | 0.194 |

Against the prediction: val CE ended far below 0.2; @0.1 vs @0.9 as predicted;
no divergence at all (val improved to the last step, so more data should help);
far from the failure line. A model that only knows the class frequencies has
masked CE ~0.65. Train `masked_ce` (gamma ~ U(0,1)) and val CE (fixed ratios)
are not directly comparable.

### 2026-09-27 · Sampling settings (queue 9, first pass)

Questions written before the run: 밀도비가 1에 가장 가까운 설정은 어느 쪽일까? random에서 T를 128로 늘리면 여분 노트가 줄어들까?

`evaluate.py --per-song --n 0` on `subset600-v1/best.pt`: 240 val songs, one
chart each, continuation. Brackets: 95% bootstrap intervals over songs.

| order, steps | F1@50 | F1@50 any lane | precision / recall | density | SR error | rho_all | rho_in | pattern coverage (chance) |
|---|---|---|---|---|---|---|---|---|
| random, 32 | 0.366 [0.355, 0.378] | 0.718 | 0.349 / 0.390 | 1.163 [1.127, 1.204] | 0.538 | 0.156 | 0.136 | 0.0020 (0.0012) |
| confidence, 32 | 0.369 [0.354, 0.383] | 0.633 | 0.426 / 0.337 | 0.814 [0.782, 0.849] | 0.496 | 0.301 | 0.224 | 0.0147 (0.0115) |
| noisy tau 1, 32 | 0.368 [0.356, 0.381] | 0.716 | 0.360 / 0.382 | 1.102 [1.069, 1.142] | 0.389 | 0.151 | 0.132 | 0.0025 (0.0012) |
| random, 128 | 0.369 [0.357, 0.381] | 0.732 | 0.361 / 0.381 | 1.088 [1.059, 1.121] | 0.375 | 0.166 | 0.162 | 0.0047 (0.0023) |
| human charts | - | - | - | 1 | - | 0.234 | 0.228 | 0.0422 (0.0063) |

(random, 32: the means of F1@50 and any-lane F1 were cut from the log; shown is
the middle of the interval.) Tokenizer ceiling F1@20 0.996.

Answers: closest to density 1 is random with T = 128 (1.09); more steps cut the
extra notes from 16% to 9%, as the independent-sampling explanation predicts.

- F1 does not separate the settings; they trade precision against recall.
- SR: with random T = 128 the slope of generated SR on target SR is 0.97
  (r 0.96) with a bias of +0.20 to +0.33 in every grade, so s works and the
  error is the extra notes. Confidence compresses (slope 0.75, Expert and
  above -1.05).
- rho moves from 0.15 to 0.30 with the sampler alone, across the human level.
- Pattern coverage is 1.3-2.1x its chance level in every setting; human
  charts 6.7x. The main weakness.
- F1@50 by grade (random, 128): Easy 0.24, Normal 0.33, Hard 0.39, Insane 0.42,
  Expert and above 0.46.
- An earlier evaluation on the first 50 val charts (13 songs) read the same
  way (random density 1.23, confidence 0.88); per-song scoring replaced it.

Next: noisy with tau 0.3-0.5 and random with T = 256, to bracket density 1.

### 2026-09-28 · Training on the full set (full-v1)

No prediction was written before this run.

Run `full-v1`: `preprocess_data.py --mel` (all splits: 5,693 audio files,
18,143 charts indexed, 48 charts without audio, 27.8 GB), then `train.py
--steps 60000 --val-every 2000 --run full-v1`, defaults otherwise. 218,142 train
chunks from 16,410 charts (7.8x the subset), 10,664 val chunks. MPS at 86-89
chunks/s, about 3 h, 4.4 epochs.

| step | val CE | @0.1 | @0.3 | @0.5 | @0.7 | @0.9 | subset600-v1 |
|---|---|---|---|---|---|---|---|
| 2,000 | 0.231 | 0.157 | 0.192 | 0.224 | 0.262 | 0.322 | - |
| 10,000 | 0.100 | 0.040 | 0.057 | 0.077 | 0.113 | 0.215 | 0.100 |
| 20,000 | 0.086 | 0.030 | 0.046 | 0.065 | 0.097 | 0.193 | 0.087 |
| 40,000 | 0.076 | 0.023 | 0.038 | 0.056 | 0.087 | 0.174 | - |
| 60,000 | 0.072 | 0.020 | 0.035 | 0.052 | 0.083 | 0.168 | - |

Reading: at the same step the full run and the subset run have the same val
CE (0.100 / 0.086 vs 0.100 / 0.087), though the subset had seen each chunk
about six times by 20k and the full set less than twice. The gain after 20k
comes from 40k more steps without overfitting. This run alone cannot separate
"more data" from "more steps"; a subset run to 60k would (three hours on the
MacBook). Val CE was still falling slowly at the end (0.0722 → 0.0717 over the
last 6k steps, with the learning rate near zero).

### 2026-09-29 · Audio alignment on the full cache

`check_alignment.py --cached-only --n 2000`, train. Peak lag of onset strength
after the note times, median over songs (IQR):

| group | songs | median ms | IQR | >10 ms from the median |
|---|---|---|---|---|
| mp3 lame | 1,240 | 18.95 | 15.2-22.0 | 10.1% |
| vorbis | 494 | 19.13 | 15.6-21.8 | 8.9% |
| mp3 none | 242 | 18.72 | 15.0-22.6 | 12.0% |
| mp3 xing | 24 | 16.33 | 11.7-22.2 | 20.8% |
| all | 2,000 | 18.98 | 15.2-22.0 | 10.2% |

Token grid: +2.94 frames (+21.9 ms). Chunks faster than 215 BPM: 10.2%.

Answer to the open question: untagged mp3s do not split. Their distribution
has one mode and matches LAME files (median difference -0.2 ms, 95% interval
-1.3 to +0.8; KS p = 0.56); a group decoded 529 samples differently would sit
~12 ms away. Rule and no-shift decision stand (DECISIONS 2026-09-29).

### 2026-09-29 · Sampling settings (second pass, full-v1)

`evaluate.py --per-song --n 0` on `full-v1/best.pt`, the same 240 val songs
and difficulties as the first pass. Brackets: 95% bootstrap intervals.

| order, steps | F1@50 | F1@50 any lane | precision / recall | density | SR error | rho_all / rho_in | pattern coverage (chance) | run length |
|---|---|---|---|---|---|---|---|---|
| noisy tau 0.3, 32 | 0.364 [0.353, 0.375] | 0.727 | 0.366 / 0.367 | 1.030 [1.007, 1.054] | 0.276 [0.245, 0.309] | 0.207 / 0.205 | 0.0060 (0.0019) | 3.60 |
| noisy tau 0.5, 32 | 0.368 [0.356, 0.379] | 0.734 | 0.364 / 0.376 | 1.061 [1.036, 1.088] | 0.291 [0.260, 0.321] | 0.198 / 0.199 | 0.0056 (0.0017) | 3.53 |
| random, 128 | 0.373 [0.362, 0.384] | 0.761 | 0.367 / 0.382 | 1.065 [1.042, 1.090] | 0.236 [0.205, 0.269] | 0.207 / 0.215 | 0.0093 (0.0046) | 4.76 |
| subset600-v1, random, 128 | 0.369 [0.357, 0.381] | 0.732 | 0.361 / 0.381 | 1.088 [1.059, 1.121] | 0.375 [0.340, 0.411] | 0.166 / 0.162 | 0.0047 (0.0023) | 2.96 |
| human charts | - | - | - | 1 | - | 0.234 / 0.228 | 0.0422 (0.0063) | 7.49 |

Paired over songs (same charts): random 128 minus noisy 0.3: F1 +0.009
[+0.005, +0.012], |SR error| -0.040 [-0.069, -0.008]; minus noisy 0.5: F1
+0.005 [+0.001, +0.010], |SR error| -0.055 [-0.082, -0.028].

SR bias (generated minus target) by grade:

| setting | Easy (47) | Normal (31) | Hard (73) | Insane (66) | Expert (18) | Expert+ (5) | slope |
|---|---|---|---|---|---|---|---|
| noisy tau 0.3 | +0.22 | +0.16 | +0.12 | +0.01 | -0.28 | -1.07 | 0.87 |
| noisy tau 0.5 | +0.23 | +0.19 | +0.20 | +0.08 | -0.19 | -0.74 | 0.90 |
| random, 128 | +0.10 | +0.09 | +0.16 | +0.07 | -0.15 | -0.49 | 0.94 |
| subset, random, 128 | +0.33 | +0.27 | +0.33 | +0.24 | +0.20 | +0.19 | 0.97 |

Decision: random order, T = 128 is the default (DECISIONS 2026-09-29). Noisy
tau 0.3 is closest to density 1, but it gets there by adding notes to easy
charts and removing them from hard ones (the confidence order's compression,
weaker), so its SR error is higher. Random 128 still adds 6.5% [4.2, 9.0]
notes; T = 256 was not run.

Full set vs subset (random 128, paired): F1 +0.004 [+0.001, +0.008], the same
by grade; |SR error| -0.139 [-0.171, -0.108], the bias left the easy grades;
rho_in 0.162 → 0.215 (human 0.228); runs 2.96 → 4.76 events (human 7.49).
Pattern coverage doubled, but so did its chance level: 2.0x chance in both,
against 6.7x in human charts. More data and steps fixed difficulty control and
repetition structure, not pattern clarity. F1@50 any lane (0.76) against F1@50
(0.37): the timing is mostly right and the lane choice is where charts differ.

### 2026-09-30 · Sampling settings (third pass: T = 256)

Questions written before the run: does opening half as many cells per step
bring the density ratio closer to 1, and what happens to SR error and pattern
clarity?

`evaluate.py --per-song --n 0 --order random --steps 256` on `full-v1/best.pt`,
the same 240 songs and charts as the T = 128 row of the second pass.

| steps | F1@50 | density | SR error | rho_in | pattern coverage (chance) | coverage / chance | run length |
|---|---|---|---|---|---|---|---|
| 128 | 0.373 [0.362, 0.384] | 1.065 [1.042, 1.090] | 0.236 [0.205, 0.269] | 0.215 | 0.0093 (0.0046) | 2.0 | 4.76 |
| 256 | 0.372 [0.361, 0.383] | 1.051 [1.028, 1.076] | 0.257 [0.225, 0.291] | 0.211 | 0.0117 (0.0042) | 2.8 | 5.14 |
| human | - | 1 | - | 0.228 | 0.0422 (0.0063) | 6.7 | 7.49 |

Paired over songs (256 minus 128): F1 -0.001 [-0.005, +0.003]; |SR error|
+0.021 [-0.007, +0.048]; density -0.014 [-0.030, +0.001]; rho_in -0.003
[-0.021, +0.015]; run length +0.39 [-0.20, +0.98]; pattern coverage +0.0023
[+0.0004, +0.0043] with its chance level unchanged (-0.0004 [-0.0011,
+0.0004]); coverage over chance 1.35x [1.09, 1.70]. SR bias by grade moved
down (Easy +0.10 → +0.07, Expert -0.15 → -0.23, Expert+ -0.49 → -0.65).

Answers: the density hardly moves. From T = 32 to 128 more steps removed
extra notes (subset: 16% → 9%), but past 128 the remaining ~5% do not come
from opening many cells at once, so they are more likely the model's own.
SR error does not improve.
Pattern coverage rose, the first setting that moved it, but the gain is
uneven: 43% of songs rose and 42% fell, the top 10% of songs carry more than
the whole net gain, and it came from Easy and Hard (Insane and above flat).
One sampled chart per song cannot tell that from sampling noise.

Decision: the default stays T = 128 (no gain in density or SR, twice the
cost). Next: a replicate of T = 128 with other sampler draws
(`--sample-seed 1`) gives the noise level; if T = 256's gain is well outside
it, T = 512 shows the trend.

### 2026-10-01 · Patterns in a playtest song (PT08, xi - Rebellion Trigger)

One val song from the first playtest pack: A, B, C are the human chart (put
through the tokenizer), random T = 128 and confidence T = 128, labels hidden
when played (which is which inferred from the numbers; answers.csv has it).

| chart | notes | coverage (chance) | run length | motion_pred |
|---|---|---|---|---|
| A (human) | 2,130 | 0.098 (0.011) | 10.4 | 0.138 |
| C (random, 128) | 1,924 | 0.017 (0.001) | 6.3 | 0.072 |
| B (confidence, 128) | 1,187 | 0.021 (0.006) | 8.0 | 0.066 |

- The human chart runs triplet stairs (1-2-3-4) and alternates mirrored
  chords (1+4 / 2+3); the random-order chart moves to a neighbouring lane a
  little less often (48% of single-note moves against 57%) and does not keep a
  direction or come back to a motif.
- B leaves bars 14-48 empty (35 bars, 1 note) while the audio is as loud as
  elsewhere and A and C fill them: the confidence order fixes EMPTY first, and
  continuation hands the empty bars on as context; a loud section after the
  bar-48 break ends it.
- Players read the AI charts as having almost no patterns, more than the
  strict coverage shows; motion_pred (DECISIONS 2026-10-01) halves for the AI
  charts.
- Long notes: the human chart has none; C has 202 (10.5% of onsets), mostly 2,
  3 or 6 cells long (21, 49 and 39 of them), B has 10. Most human charts do
  have long notes (next entry: 93% of train charts, median 15% of onsets), and
  this one is among the 10% of Insane charts without any: the model cannot
  tell such songs from the rest. Whether it has too many in general is for
  the per-song comparison (evaluate's hold_share against human_hold_share).
  The short holds come from the sampler closing holds cell by cell
  (DECISIONS 2026-10-01, long notes).

Next: `pattern_probe.py` (does the model know the lanes when the rest of the
chart is visible?) and `evaluate.py --refine 2` at lane temperatures 0.5 and 1
against T = 128 alone, with the replicate (`--sample-seed 1`) for the noise;
`hold_stats.py` for the human long-note levels behind the clean-up defaults.

### 2026-10-01 · Long notes in human charts (data check)

`hold_stats.py`, train split, from the token cache (cells of 1/12 beat);
`docs/_stats/hold_stats.csv`.

| grade | charts | without long notes | long-note share p50 / p90 | length p10 / p50 (cells) | under 3 cells | next press 1 / 2 / 3-5 cells after release |
|---|---|---|---|---|---|---|
| Easy | 4,122 | 5.2% | 0.15 / 0.35 | 6 / 12 | 0.3% | 0.01% / 0.13% / 1.1% |
| Normal | 3,138 | 5.3% | 0.15 / 0.39 | 5 / 6 | 0.6% | 0.03% / 0.35% / 6.1% |
| Hard | 5,314 | 7.1% | 0.15 / 0.44 | 3 / 6 | 2.0% | 0.08% / 1.0% / 20% |
| Insane | 3,167 | 10.4% | 0.13 / 0.48 | 3 / 6 | 5.1% | 0.34% / 3.0% / 36% |
| Expert | 565 | 11.3% | 0.16 / 0.55 | 3 / 4 | 8.2% | 1.0% / 6.4% / 46% |
| Expert+ | 148 | 10.1% | 0.40 / 0.63 | 3 / 5 | 5.1% | 0.6% / 7.8% / 59% |
| all | 16,454 | 7.1% | 0.15 / 0.43 | 3 / 6 | 3.5% | 0.28% / 2.4% / 27% |

- Long notes are the rule, not the exception: 93% of charts have some, a
  median 15% of onsets, and the share spreads widely (p90 0.43).
- Short long notes and quick re-presses grow with the grade: at Easy almost
  nothing under half a beat or pressed again within 5 cells; from Hard up a
  quarter beat (3 cells) is the common short length and a press 3-5 cells after
  a release is frequent. One or two cells stay rare everywhere (under 3 cells
  8% at most; next press 1 cell later at most 1%).
- Clean-up rules by target SR from this table (sampler.HOLD_RULES, DECISIONS
  2026-10-01): min_hold 6 / 5 / 3 cells and release gap 5 / 2 / 2 for Easy /
  Normal / Hard and up.

### 2026-10-02 · Lane choice: probe, refinement and a replicate (full-v1)

Questions written before the runs (2026-10-01): does the model know which lanes a
human chart uses when the rest of the chart is visible, and does re-choosing the
lanes after sampling (`refine_lanes`) raise pattern clarity? How far do two runs
of one setting differ by chance?

`pattern_probe.py` on 800 val chunks (tap-only rows with 1-3 taps; all four lanes
of every 8th such row hidden at a time; the exact lane set scored):

| context | 1 tap | 2 taps | 3 taps | all |
|---|---|---|---|---|
| full (rest of the chunk visible) | 0.800 | 0.744 | 0.793 | 0.782 |
| thin50 (half of the rest hidden) | 0.611 | 0.561 | 0.617 | 0.596 |
| thin90 (90% hidden) | 0.366 | 0.306 | 0.426 | 0.352 |
| chance | 0.250 | 0.167 | 0.250 | 0.225 |
| previous onset row's lanes | 0.025 | 0.028 | 0.032 | 0.026 |
| best of 8 (oracle: one of the last 8 events) | 0.781 | 0.475 | 0.395 | 0.664 |

34,619 rows per context. `evaluate.py --per-song --n 0`, random T = 128, the same
240 val songs and charts as the 09-29 and 09-30 entries (`overnight_1001.log`;
brackets: 95% bootstrap intervals):

| setting | F1@50 | SR error | density | rho_in | coverage (chance) | / chance | run length | motion_pred |
|---|---|---|---|---|---|---|---|---|
| T128 (baseline, hold clean-up of 0021) | 0.373 [0.362, 0.384] | 0.237 [0.206, 0.270] | 1.065 | 0.213 | 0.0093 (0.0046) | 2.0 | 4.76 | 0.050 [0.047, 0.052] |
| + refine 2, lane temp 0.5 | 0.375 [0.363, 0.386] | 0.229 [0.198, 0.263] | 1.065 | 0.208 | 0.0194 (0.0046) | 4.2 | 5.75 | 0.091 [0.087, 0.094] |
| + refine 2, lane temp 1 | 0.374 [0.363, 0.386] | 0.229 [0.199, 0.262] | 1.065 | 0.210 | 0.0158 (0.0046) | 3.4 | 5.72 | 0.074 [0.071, 0.078] |
| T128, replicate (`--sample-seed 1`) | 0.371 [0.360, 0.382] | 0.234 [0.205, 0.265] | 1.063 | 0.205 | 0.0096 (0.0040) | 2.4 | 4.74 | 0.050 [0.048, 0.053] |
| human charts | - | - | 1 | 0.228 | 0.0422 (0.0063) | 6.7 | 7.49 | 0.132 |

Coverage intervals: baseline [0.0077, 0.0110], refine at 0.5 [0.0160, 0.0231].
Refinement moves only the taps of rows that have a lane choice, so onsets, density
and holds are those of the baseline (F1@50 any lane 0.761 in all three).

Readings:
1. The model knows lane patterns: with the rest of the chunk visible it picks the
   human lane set for 78% of tap rows, 3.5x chance and above an oracle that knows
   which of the last 8 events the pattern repeats (66%); for chords far above it
   (0.74 / 0.79 against 0.48 / 0.40).
2. The knowledge needs context: 60% with half of the chunk hidden, 35% with 90%
   hidden. In the random order a cell opens after a uniform share of the others,
   so half of all lane choices are made with less than half of the chart open,
   the first ones close to chance, and later choices follow those.
3. Choosing the lanes again with the whole chart open doubles pattern clarity:
   coverage 2.0x → 4.2x its chance level, motion_pred 0.050 → 0.091 (half of the
   gap to the human 0.132), runs 4.8 → 5.8 events (a third of the gap), at the same
   F1, density and rho; SR error 0.008 lower. Lane temperature 0.5 beats 1.
4. Noise: the replicate moves coverage by 0.0003 and motion_pred by 0.0005, 30-80x
   less than refinement; breaks_per_100 doubles between replicates (0.010 / 0.020),
   too noisy to compare settings; rho_in and rho_all move by about 0.01.
5. Long notes, the first full read after 0020-0021: generated charts have fewer
   (hold share 0.154 against the human 0.218) and shorter ones (0.77 against 1.00
   beats), none shorter than the grade's minimum and no quick re-presses (human
   2.7% and 1.9%). PT08's surplus was that song, not the rule.

Decision: pattern clarity is worked on in the sampler first (DECISIONS 2026-10-02);
retraining for it (row masks, a rhythm channel) waits until the lane passes are
used up.

Next (0022), questions and decision rules written before the runs:
- `pattern_probe.py` again, with one hidden row at a time: how close is
  `past+rhythm` (earlier lanes as the human chart has them, later lanes hidden,
  later empty rows visible: what the new left-to-right lane pass gives the model)
  to `full (1 row)`? Close means lanes can be chosen left to right without the
  later lanes that sampling drew with little context.
- `--lanes forward --refine 2` against `--refine 2` (paired, `compare_runs.py`):
  if motion_pred and coverage over chance rise with F1 and SR error unchanged, it
  becomes the setting for playtest charts. `--lanes forward` alone tells how much is
  the pass itself.
- `--refine 4`: does refinement saturate after 2 sweeps?
- `--spread` (same cost, cells of one beat in different steps) and `--order block`
  (beats left to right): is the problem parallel sampling within a beat, or the
  order in which lanes are chosen?
- `--n 20 --steps 0` (one cell per forward pass, the first 20 songs; paired with
  the same 20 songs of the baseline): if patterns stay near T128's, parallel
  sampling is not the cause and the order is.
- For every setting, coverage by kind (`coverage_p1` jacks, `_p2` trills, `_p3plus`
  longer motifs) against the human charts: a lane pass must not make everything a
  trill. Cost per song is in the new `passes` and `seconds` columns.

### 2026-10-02 · Lane passes on PT08 (playtest check, one song)

`playtest_pack.py --keys 2ba50d694ea9c976 --settings random:128
random:128:continue:ref2 random:128:continue:fwd+ref2` (xi - Rebellion Trigger,
Insane 4.47; `outputs/full-v1/playtest_1002`). The three AI charts share one
sampled rhythm (2,017 notes; human 2,130), so they differ only in lanes. Played
blind: patterns clearly more distinct with the lane passes, some good stretches
and some poor ones, and mostly stairs. Measured on the decoded charts
(`patterns.summarize`, move kinds defined below before tonight's runs):

| chart | stair | trill | irregular | jack | 1-4 leap | chords | motion_pred | coverage (chance) | runs by period |
|---|---|---|---|---|---|---|---|---|---|
| human (B) | 0.25 | 0.23 | 0.52 | 1.1% | 5.0% | 35% | 0.138 | 0.098 (0.011) | 2, 3, 4, 6, 8 |
| T128 (D) | 0.16 | 0.20 | 0.65 | 2.1% | 11.9% | 26% | 0.089 | 0.005 (0.005) | 4 |
| ref2 (C) | 0.18 | 0.12 | 0.69 | 0.5% | 12.2% | 26% | 0.124 | 0.000 (0.005) | - |
| fwd + ref2 (A) | 0.20 | 0.09 | 0.70 | 0.5% | 13.3% | 26% | 0.155 | 0.018 (0.005) | 2, 3, 4 |

Stair and trill: among two single-note moves in a row that both change lane, the
second repeats the first step (stairs, rolls) or goes straight back (trills,
bounces). Jack and leap: among single-note moves.

- "Mostly stairs" is not more stairs than the human chart (0.20 against 0.25) but
  stairs left almost alone: trills fell from 0.20 to 0.09 (human 0.23) and jacks
  from 2.1% to 0.5%, so the stair-to-trill ratio went from 0.8 to 2.2 (human 1.1).
  Mode-seeking at lane temperature 0.5 is the first suspect: squaring the lane
  probabilities favours the commonest continuation.
- Lane motion became more predictable than the human chart's (motion_pred 0.155
  against 0.138) while exact repetition stayed at a fifth of it: predictable but
  narrow. The human chart's back-and-forth stairs (period 6) and 8-event motifs
  are missing.
- 1-4 leaps stay at 12-13% (human 5%), and the rhythm has fewer chords (26%
  against 35%), which the lane passes cannot change: patterns built on chords
  (the mirrored 1+4 / 2+3 alternation) have no room.

One song, one draw: tonight's 240 songs carry the same numbers (`evaluate.py`
now records move_stair, move_trill, move_jack, move_leap and chord_share, and
saves the generated tokens as charts.npz). Queue, in order: the probe; T128 and
ref2 again (for the move kinds); fwd + ref2; fwd at lane temperature 1 + ref2 at
0.5 (`--forward-temp 1`: motifs chosen at the model's own spread, polished at
0.5); ref4; fwd alone. Spread, block and the sequential ceiling move to the next
night.

### 2026-10-02/03 · Lane passes on 240 val songs (overnight)

`pattern_probe.py` (one hidden row per chunk, rows 96+, 5,400 rows per context):

| context | 1 tap | 2 taps | 3 taps | all |
|---|---|---|---|---|
| full (1 row) | 0.795 | 0.742 | 0.782 | 0.779 |
| past+rhythm | 0.593 | 0.588 | 0.715 | 0.599 |
| past | 0.563 | 0.582 | 0.718 | 0.578 |

Hiding the later lanes costs 0.18 (to the thin50 level); the later empty rows
add only 0.02; single taps lose the most (0.80 → 0.59), 3-tap chords little.
Human lanes are chosen with the later notes in view, so the left-to-right pass
alone chooses each lane at about the 0.6 level: it is a starting point for
`refine_lanes`, not a replacement (the decision rule of the 10-02 entry).

`evaluate.py --per-song --n 0`, random T128, the same 240 songs and charts; paired
differences against T128 (`compare_runs.py`), 95% intervals:

| setting | coverage (/ chance) | run length | motion_pred | stair | trill | jack | leap | SR error | s / song |
|---|---|---|---|---|---|---|---|---|---|
| T128 (`_s0`, same charts as 10-01) | 0.0093 (2.0) | 4.76 | 0.050 | 0.160 | 0.206 | 0.047 | 0.127 | 0.237 | 31 |
| ref2 at 0.5 (`_s0`) | 0.0194 (4.2) | 5.75 | 0.091 | 0.184 | 0.139 | 0.012 | 0.122 | 0.229 | 35 |
| fwd + ref2 at 0.5 | **0.0465 (10.2)** | **7.61** | **0.122** | 0.192 | 0.112 | 0.011 | 0.143 | **0.226** | 48 |
| fwd at 1 + ref2 at 0.5 | 0.0267 (5.8) | 6.59 | 0.099 | 0.187 | 0.128 | 0.012 | 0.133 | 0.227 | 48 |
| fwd + ref2 at 1 | 0.0223 (4.9) | 6.19 | 0.082 | 0.183 | 0.143 | 0.021 | 0.136 | 0.230 | 48 |
| human charts | 0.0422 (6.7) | 7.49 | 0.132 | 0.211 | 0.153 | 0.044 | 0.122 | - | - |

fwd + ref2 at 0.5 against T128: coverage +0.037 [+0.031, +0.043] (up in 81% of
songs, down in 6%), run length +2.85 [+2.30, +3.37], motion_pred +0.072 [+0.068,
+0.076] (up in every song), SR error -0.010 [-0.017, -0.003], F1@50 +0.001
[-0.002, +0.003], rho_in +0.007 [-0.001, +0.014]. Against ref2 alone it more than
doubles coverage and adds 0.031 motion_pred. Gains hold in every grade (largest
in coverage for Easy, in motion_pred for Insane and up).

Readings:
1. Pattern clarity reached the human level: coverage 0.047 against 0.042, runs
   7.6 against 7.5 events, motion_pred 92% of the human value, F1 unchanged and
   SR error lower. Coverage over chance is above the human ratio only because the
   sampled rhythm has a lower chance level.
2. Temperature is not the variety lever: lane temperature 1 for the forward pass
   (or both passes) gives back little (trills 0.11 → 0.13-0.14, jacks 0.011 →
   0.012-0.021) and loses much of the clarity.
3. Variety is what is left. Jacks: the sampled charts have the human rate (0.047),
   refinement alone takes it to 0.012, a quarter of the human 0.044, and the
   forward pass adds nothing to that. Trill moves fall below the human rate
   (0.11 against 0.15) while exact trill runs reach twice it (coverage_p2 0.018
   against 0.010): short back-and-forth disappears, a started trill runs long.
4. Cost: the lane passes take forward passes from 2,665 to 4,187 per song (+57%);
   one 240-song evaluation is 2 h (T128) or 3.2 h (with both passes) on the MacBook.

Next: `pattern_probe.py` now splits single-tap rows by the human move and compares
the human jack rate with the model's pick (`jack rate`): whether the model
under-predicts jacks with the whole chart visible (then the lane passes inherit
it, and the fix is in the scoring or training) or the passes lose them some
other way. Playtests (Oct 7) with fwd + ref2 at 0.5.

Answer (`pattern_probe.py`, 800 val chunks, single taps after a single tap at
most a beat earlier, full context): exact lane 0.124 where the human chart
jacks (516 rows), against 0.858 / 0.778 / 0.729 for moves of 1 / 2 / 3 lanes;
the human rows repeat the lane 3.6% of the time, the model's best lane 0.7%. The
model under-picks jacks with the whole chart in view and the lane passes inherit
it. Whether its probabilities are low (the model) or only rarely the largest (the
low temperature) is the next probe line (`jack prob` at lane temperature 1 and
0.5); meanwhile `--jack-bias` adds a log-score bonus for repeating the previous
onset's lanes in both passes, to be set so that the jack rate meets the human one.

### 2026-10-03 · Jack bias (60 val songs)

`evaluate.py --per-song --n 60 --lanes forward --refine 2 --jack-bias 1` (and 2),
paired against fwd + ref2 at 0.5 on the same 60 songs (`compare_runs.py`):

| jack bias | jack | trill | stair | motion_pred | coverage | F1@50 | SR bias |
|---|---|---|---|---|---|---|---|
| 0 | 0.010 | 0.112 | 0.191 | 0.124 | 0.052 | 0.366 | +0.041 |
| 1 | 0.022 (+0.012 [+0.007, +0.018]) | 0.118 | 0.195 | 0.099 (-0.025 [-0.031, -0.019]) | 0.038 | 0.371 | +0.051 |
| 2 | 0.178 | 0.167 | 0.163 | 0.042 | 0.030 | 0.368 | +0.168 |
| human | 0.055 | 0.160 | 0.193 | 0.129 | 0.046 | - | - |

A bonus of 1 doubles the jacks (still 40% of the human rate) and costs a fifth of
motion_pred; 2 puts jacks at three times the human rate, breaks the patterns and
raises SR by 0.13 (jacks are hard). The jack shortfall is not one number away: no
default bias; a playtest variant at most.

### 2026-10-03 · Why trills break and bars do not repeat (240 val songs, rescored)

Two complaints from the 10-03 playtests: long trills (fjfjfjfj... for bars) do not
come, and pairs of phrases do not come back (a1 b1 c1 a2 b2 c2). New columns in
`patterns.summarize`, computed for the saved charts with `scripts/rescore.py` (no
new sampling): `lone_chord` (of the events inside a single-note stream, the share
that are chords), `bar_rhythm_repeat` (bars with 4+ onsets whose rhythm, onset
cells and chord sizes, equals an earlier bar's) and `bar_lane_repeat` (of those,
the share whose lanes also equal such a bar's, as they are or mirrored). Same 240
songs and charts as the 10-02/03 entry.

| | T128 | ref2 at 0.5 | fwd + ref2 at 0.5 | human |
|---|---|---|---|---|
| lone_chord | 0.343 | 0.343 | 0.343 | 0.413 |
| bar_rhythm_repeat | 0.122 | 0.122 | 0.122 | 0.435 |
| bar_lane_repeat | 0.012 | 0.027 | 0.111 | 0.334 |
| bars copied whole (product) | 0.1% | 0.3% | 1.4% | 14.5% |

Single-note streams (consecutive single notes at one gap of at most a beat), share
of all events in streams of at least 8 / 16 / 32 notes: sampled rhythm 4.9% / 1.7% /
0.5%, human 7.8% / 3.6% / 2.1%. How streams of 8+ end: the next event a chord at
the same gap 41% (human 61%), one note missing (twice the gap) 28% (12.5%), a
faster note 20% (15%). Inside those streams, of two single-note moves in a row:
trill 0.079 and stair 0.194 with fwd + ref2 (human 0.135 / 0.165; T128 0.187 /
0.139). Strict single-note trill runs (period 2) hold 0.11% of events (human 0.32%),
runs of 8+ 0.03% (0.20%); jumptrills 1.26% (0.66%).

Human charts, all 240 val songs, 26,675 bars with 4+ onsets: 16.9% copy an earlier
bar exactly or mirrored, 27.5% more repeat its rhythm only. The source is the
earlier bar with the most similar audio (`structure.ssm_audio`) 71% of the time,
among the 3 most similar 85%, the 5 most similar 90%; it lies a median of 16 bars
back (within 2 bars: 15%). By the audio similarity of the most similar earlier
bar, the share of bars that copy some earlier bar: below 0.3 3%, 0.5-0.6 11%,
0.7-0.8 20%, 0.9 and up 42%.

Readings:
1. Chords do not cut the streams (the AI puts fewer chords in streams than the
   human charts). Long trills fail twice: the sampled rhythm drops single notes
   inside long streams (a run of independent draws; one miss ends the stream),
   and on the streams that remain the lane passes choose stairs over trills.
2. Whole-bar repetition is the largest gap measured so far, ten times: the
   rhythm itself repeats a quarter as often as in human charts, so lane
   refinement alone cannot close it.
3. Where humans copy is predictable from the audio (top-3 similar earlier bars),
   which a sampler move can use without training: `sampler.copy_bars` (next entry).

### 2026-10-03 · Bar copies (`sampler.copy_bars`), first check (12 val songs, CPU)

After the lane passes, every bar is offered a copy of an earlier bar: sources are
the 3 earlier bars with the most similar audio, at least 0.5 (human copies: top-1
71%, top-3 85%), with 4+ onset rows; the whole bar is copied (rhythm, lanes, long
notes), as it is or mirrored. One forward pass with the bar MASK scores both: row
by row, log P(that many onsets) under the model's cells (`onset_count_logp`).
The first source in order of similarity whose score + `copy_bias` reaches the
bar's own replaces it; its onset count must be within 15% of the bar's. Bars with
a long note across an edge are left alone. Cost: one pass per bar with a source
(about a quarter of the bars, under 1% of the passes of a song).

Check: the copy pass on the saved fwd + ref2 charts of 12 val songs (`copy_calib`,
full-v1 on CPU), against the same charts without it. F1 here is a token-level
proxy (onsets within 2 cells of a human onset in the same lane), not F1@50.

| copy_bias | bars copied | notes | F1 proxy | bar_rhythm_repeat | bar_lane_repeat | run length |
|---|---|---|---|---|---|---|
| none | - | 2180 | 0.437 | 0.061 | 0.084 | 6.67 |
| 0 | 13.9% | -0.9% | -0.004 [-0.008, -0.000] | 0.188 | 0.750 | 6.51 |
| 4 | 20.3% | -0.6% | -0.002 [-0.007, +0.002] | 0.249 | 0.862 | 5.75 |
| 10 | 22.3% | -0.4% | -0.003 [-0.008, +0.002] | 0.266 | 0.882 | 5.45 |
| human (240 songs) | - | - | - | 0.435 | 0.334 | 7.49 |

Two scoring rules came first and were dropped: sum log p over the bar's cells,
best source taken (23 songs: 21% of bars copied, notes -6%, F1 proxy -0.009), and
the onset-count score with the best source taken (notes -5% at bias 0, -12% at
4). With the bar MASK the model spreads a sure note over the four lanes, and the
likeliest version is the one without the uncertain notes; taking the first
source by audio similarity and guarding the onset count keeps the density.

Readings: at bias 0 whole-bar copies go from 0.5% to 14% of bars (human 14.5%)
at nearly the same notes and F1; higher biases copy more and shorten the runs
(a copy does not continue the pattern across its edges). The rhythm repeats in
19% of bars (human 43%): the rest of the human repetition is rhythm only, with
new lanes. To check on 240 songs without sampling again: `evaluate.py
--from-charts <fwd + ref2 run> --copy-bias 0` (and 4), minutes instead of hours.
Off by default until then.

Result, 240 val songs (2026-10-04, `evaluate.py --from-charts`, full-v1 on MPS, 36
forward passes and 0.4 s per song), paired against the fwd + ref2 charts:

| | fwd + ref2 | + copies, bias 0 | + copies, bias 4 | human |
|---|---|---|---|---|
| F1@50 | 0.373 | 0.375 (+0.001 [+0.000, +0.002]) | 0.375 (+0.001 [+0.000, +0.003]) | - |
| density ratio | 1.065 | 1.061 | 1.062 | 1 |
| SR error | 0.226 | 0.227 (+0.000 [-0.005, +0.005]) | 0.229 | - |
| rho_in | 0.219 | **0.231** (+0.012 [+0.006, +0.018]) | 0.237 | 0.228 |
| bar_rhythm_repeat | 0.122 | 0.228 | 0.295 | 0.435 |
| bar_lane_repeat | 0.111 | 0.659 | 0.765 | 0.334 |
| whole-bar copies (product) | 1.4% | **15.0%** | 22.6% | 14.5% |
| coverage | 0.047 | 0.055 | 0.058 | 0.042 |
| motion_pred | 0.122 | 0.128 | 0.137 | 0.132 |
| trill / jack | 0.112 / 0.011 | 0.116 / 0.013 | 0.119 / 0.013 | 0.153 / 0.044 |

Readings: bias 0 brings whole-bar repetition to the human rate and rho_in, which
measures whether the chart repeats where the audio does, to the human level, with
F1 slightly up and SR unchanged; bias 4 overshoots both (23% of bars copied, rho_in
and motion_pred above the human charts). The rhythm alone still repeats half as
often as in human charts (rhythm repeats with new lanes are not made). Decision:
playable charts copy at bias 0 (DECISIONS 2026-10-04).

### 2026-10-03 · full-v2: genre and mapper as inputs (prediction, written before the run)

Why: mappers differ (1,146 made the train charts; 215 with 20+ train charts made
74% of them) and the model's pattern rate is wrong by genre (pop: AI 0.047 vs human
0.022). One model learns every style and the style-free distribution; sampling
chooses.

Setup: `build_style.py` -> data/style.csv (genre per set, mapper = the beatmap's
user_id). Vocab from the train charts: 13 genres; the 215 mappers with 20+ charts,
"other", and "no label". 555 of the 784 kept val charts (71%) are by a mapper in
the vocab. Model: full-v1's D-32 plus a genre and a mapper embedding added to the
s and b condition (+59k parameters, 6.93M). Condition dropout: both labels dropped
with probability 0.1, then each with 0.1 (no label at all for 11% of chunks).
Recipe exactly as full-v1 (60k steps, batch 16, lr 3e-4, fixed chunks, seed 0),
from scratch: `train.py --steps 60000 --val-every 2000 --style data/style.csv
--run full-v2`. Validation: val_ce with each chart's labels (best.pt), val_ce_null
with none.

Evaluation plan (240 val songs, fwd + ref2 at 0.5, compare_runs.py against
full-v1's fwd + ref2): `--style oracle` (each chart's own genre and mapper) and
`--style none`, `--by-genre`; then a pairs playtest (v1 against v2-oracle).

Predictions:
1. val CE: val_ce ends at 0.066-0.070 (full-v1: 0.072); val_ce_null within 0.002
   of full-v1. Style mostly decides which lanes and how dense, which the masked CE
   at high mask ratios carries, so the gain is largest at mask 0.7-0.9.
2. `--style none` vs full-v1: no difference beyond noise (F1@50 within 0.003,
   coverage within 0.005): dropout costs nothing.
3. `--style oracle` vs full-v1: F1@50 +0.005 to +0.015, SR error -0.01 to -0.03,
   density ratio closer to 1. The per-song distance to the human chart's pattern
   numbers (|coverage - human|, |motion_pred - human|, |move_stair - human|) falls
   by 10-30%.
4. By genre: the over-patterned genres (pop, anime, rock) move at least half way
   to their human coverage with `--style oracle`; electronic and video game change
   little.
5. The gains in 3 sit with the charts whose mapper is in the vocab (71%); "other"
   charts gain only through the genre.
6. Guidance (`--style-guidance 1-2`) widens the gap between styles but costs F1
   (-0.005 or more) at twice the passes; not a default.
7. The gaps of the 10-03 entry stay: bar_rhythm_repeat within 0.02 of full-v1's
   0.122 and as few long single-note streams. They come from how the chart is
   sampled, not from whose style it imitates.
How it could fail: the model ignores the labels (val_ce = val_ce_null within
0.001), because the audio already tells the genre and a mapper's charts are few;
or a mapper stands for a difficulty band, and SR error moves instead of the
patterns. Expected time: about 3.5 h on the MacBook (validation runs twice).

Result (2026-10-04, `overnight_1004.log`): 6,928,340 parameters; 12,079 of 16,410
train charts by a mapper in the vocab.

| step | val CE (own labels) | val CE (no labels) | @0.1 | @0.5 | @0.9 | full-v1 val CE |
|---|---|---|---|---|---|---|
| 2,000 | 0.2293 | 0.2293 | 0.154 | 0.222 | 0.320 | 0.231 |
| 20,000 | 0.0884 | 0.0885 | 0.030 | 0.066 | 0.199 | 0.086 |
| 40,000 | 0.0764 | 0.0763 | 0.023 | 0.056 | 0.176 | 0.076 |
| 60,000 | 0.0732 | 0.0732 | 0.021 | 0.054 | 0.171 | 0.0717 |

The labels change nothing in val CE at any point of the run. Probe on 71 val
windows (full-v2, CPU): with the window fully MASK the own labels move the cell
distributions by KL 0.001 nats per cell from no labels (an SR change of 0.3 moves
them by 0.002) and lower CE by 1.9% for charts by a known mapper; at mask 0.9 by
0.0005 and 0.2%, at 0.5 by nothing. The learned embeddings (norm about 0.8) are
small next to the SR condition (about 10). The model reads the style from the
visible part of the chart; labels matter only while almost nothing is visible.

`evaluate.py --lanes forward --refine 2 --style oracle`, 240 songs, paired against
full-v1's fwd + ref2: F1@50 +0.004 [+0.001, +0.008] (any lane +0.004), density
1.077 (+0.012), SR error +0.017 [-0.010, +0.043], SR bias +0.052 [+0.022, +0.085],
rho_in -0.013 [-0.026, +0.000], coverage 0.039 (-0.008), motion_pred 0.107 (-0.014
[-0.018, -0.011]), stair 0.174 (-0.018), trill 0.137 (+0.025 [+0.018, +0.032]),
long-note share 0.186 (+0.032; human 0.218), bar_rhythm_repeat 0.105 (-0.016).

Predictions: 1 wrong (val CE 0.0732, not 0.066-0.070, and equal without labels:
the stated failure mode). 3 wrong (F1 +0.004 is below the range, SR error went up,
the per-song distance to the human pattern numbers did not fall: coverage -7%
[-19%, +5%], motion_pred +6%). 4 wrong: coverage fell in every genre, also where
it was below the human level (electronic, video game), so pop's and rock's move
towards their human level is a global shift. 5 reversed (charts by "other" mappers
changed more than charts by known ones). 7 right (bar_rhythm_repeat within 0.02).
2 and 6 not run.

Readings: the style inputs as built go unused, so the differences above cannot be
credited to them; they are what a second training run with other random draws
gives (val CE 2% higher, more trills and long notes, less predictable motion, SR
higher). The size of run-to-run differences was not known before and is now a
reference for any retraining. full-v1 stays the default model. If style is tried
again, the label has to matter where it can: classifier-free guidance on full-v2
(cheap: `--style-guidance 3` on 60 songs), or conditioning in every block (adaLN)
with more training on high mask ratios; a run with `--style none` would show what
the labels do at generation at all.

### 2026-10-04 · Blind pairs playtest and what it found (16 songs; 240 val songs measured)

`playtest_pack.py --pairs --songs 16 --settings random:128:continue:fwd+ref2+cp0`
(full-v1; val songs, SR 1.5-4.5). One player (a rhythm-game player) picked the human
chart in 16 of 16 pairs: AI taken for human 0/16, 95% Wilson interval [0%, 19%].

Comments, and what the 240 val songs say (fwd + ref2 + copies against the human charts):

| comment | measured | |
|---|---|---|
| dense chord streams a little better | - | |
| no single-lane runs (ddddffff), no minijacks (11332244) | single notes in same-lane runs of 4+: 0% (human 0.41%); AABB minijacks 0.15 per 1,000 single notes (2.28) | confirmed |
| no simple repeated patterns (1313131324242424) | single-note trill runs of 8+: a seventh of human (10-03) | confirmed |
| long-note releases vague | releases on a row where another lane has an onset: 68% (77%) | confirmed |
| no intent in long-note placement | lengths and snaps as human (median 0.5 beat both; release snaps within 1 pp); the amount per song is not: charts without long notes 0.4% (5.0%), long-note share p75 / p90 0.21 / 0.29 (0.31 / 0.47) | the amount per song |
| difficulty flat over the song | 40 songs: bar density on bar loudness slope 0.20 (0.22); quietest bars (z < -1.5) 17% more notes than human; SR 4+: light bars 6.5% (8.2%), first 8 bars at 0.73 of the mean (0.65) | confirmed, moderate |
| not concise; sometimes only the beat would be better | onsets on the beat / half beat 32.0% / 25.6% (33.7% / 27.2%); 1/12 positions 9.1% (6.9%) | small |
| rightmost lane a little much | lane shares within 0.3 pp of human; outer lanes 0.3 pp more, 1-4 leaps 0.141 (0.122) | not as such |

Hypothesis "charts with many short long notes spoil the model", train split: 67% of all
long notes are at most half a beat (28% at most a quarter); 3,331 charts (20%) have
short long notes as 20%+ of their onsets, by 824 mappers; the top 10% of charts hold
55% of the short long notes. A common style, not a few bad charts: dropping them would
drop a fifth of the data. What the model gets wrong is choosing the style per song: it
mixes long notes into every song at about the same rate (table above).

Long notes decided again with the whole chart in view (`sampler.refine_holds`), first
check on 8 val songs (saved fwd + ref2 charts, full-v1 on CPU):

| | long-note share | beats | released on an onset row | ln_f1 vs human |
|---|---|---|---|---|
| sampled | 0.257 | 0.82 | 0.584 | 0.312 |
| scoring whole paths (dropped) | 0.16 (-40%) | - | - | - |
| refine_holds (p(START) vs p(TAP), then p(END)) | 0.232 | 0.76 | 0.564 | 0.266 |
| refine_holds, the human chart's share | 0.190 | 0.84 | 0.550 | 0.304 |
| human | 0.189 | 1.33 | 0.513 | - |

ln_f1: of the onsets both charts have, F1 of the AI's long notes against the human's.
Asking the model again with full context does not place long notes more like the
human charts or release them more on other notes: the model's own preference is the
limit, not the context it had when sampling. The share option gives each song its
amount (and lets a user ask for rice or a long-note chart).

Questions and predictions for the runs of 10-04/05, written before them:
1. `--from-charts <fwd + ref2> --refine-holds --copy-bias 0` and `--hold-share oracle
   --copy-bias 0` (240 songs): releases on onset rows within 0.02 of the copies-only run
   (0.68) in both; ln_f1 no higher than the copies-only run; with the oracle share the
   per-song long-note share equals the human one and its distribution (no-LN charts,
   p90) follows; F1@50 unchanged (onsets do not move).
2. `--loud-bias 0.1` (first 60 songs, paired with fwd + ref2): loud_slope up by 0.015 or
   more (towards the human 0.22), light_bars up, density ratio and SR error within 0.02,
   F1@50 within 0.005.
3. full-v3, the full-v1 recipe with `--row-mask 0.5` (half of the chunks masked by whole
   rows). The lane passes ask for whole rows with everything else visible, which cell-by-
   cell masking almost never shows the model (all four cells of a row masked at mask
   ratio 0.1: 1 in 10,000), and with the other lanes of a row visible the lane of a single
   note is given away; jacks are where that matters most (the model picks the human
   lane on 12% of human jack rows, against 73-86% for other moves). Predictions:
   val CE (cell by cell) at most 0.0745 (full-v1 0.0717); probe full (1 row) 0.82 or more
   (0.78), jack rows 0.25 or more (0.12); 240 songs with fwd + ref2: move_jack 0.02 or more
   (0.011; human 0.044), move_trill 0.13 or more (0.112; 0.153), same-lane runs twice as
   common; F1@50 within 0.005, SR error within 0.02. How it could fail: whole-row masking
   costs the rhythm (cell CE up more than 0.005, F1 down), or jacks stay rare because the
   model is too small for them rather than shown too few.

### 2026-10-05 · Long-note passes, loudness bias, full-v3, and why the style is averaged (240 val songs)

The runs of 10-04/05 (`run_1005.sh`), each paired song by song with `compare_runs.py`:
`--from-charts` on full-v1's saved fwd + ref2 charts (copies only, `--refine-holds`,
`--hold-share oracle`, all with `--copy-bias 0`, and rescored as they are);
`--loud-bias 0.1` sampled again on the first 60 songs; full-v3 trained and evaluated
with fwd + ref2 on all 240 songs.

**1. Long notes decided again** (predictions 1 of 10-04)

| | long-note share | no-LN charts | share p90 | beats | released on an onset row | ln_f1 | chance at the same counts | over chance |
|---|---|---|---|---|---|---|---|---|
| copies only (cp0) | 0.156 | 0.4% | 0.29 | 0.77 | 0.680 | 0.214 | 0.156 | +0.059 |
| `--refine-holds` + cp0 | 0.147 | 0% | 0.27 | 0.71 | 0.680 | 0.211 | 0.157 | +0.054 |
| `--hold-share oracle` + cp0 | 0.219 | 4.6% | 0.47 | 0.77 | 0.694 | 0.307 | 0.239 | +0.068 |
| human | 0.218 | 5.0% | 0.47 | 1.00 | 0.759 | - | - | - |

Chance: the F1 of placing the same number of long notes at random among the onsets
both charts share, 2ab / (n (a + b)) per song. F1@50 moves by at most 0.0004.
- refine_holds alone: no change in releases (+0.000) or ln_f1 (-0.003), long notes
  0.07 beats shorter, and every chart gets long notes (no-LN 0.4% → 0%, p90 0.29 →
  0.27): it averages the share further. As predicted; not a default.
- the human share: the per-song distribution follows by construction, releases on
  onset rows +0.017 [+0.004, +0.030] (predicted within 0.02: at the edge), ln_f1 +0.093
  [+0.071, +0.114] (predicted no higher: wrong as a number). But chance rises with the
  amount (0.156 → 0.239); over chance only +0.059 → +0.068. What the share fixes is the
  amount. Placement stays near chance.
- The model does know placement in context: refine_holds at the human share on the
  **human** chart (everything else human; the first 18 of the val songs with cached audio
  and long notes, full-v1, CPU) gives ln_f1 0.863 (median 0.887, range 0.67-0.96) against
  chance 0.240, and 80% of the long notes both keep end on the same cell. It fits long notes to the long notes around them. In its own charts
  those are its own, so its layout agrees with the human one hardly more than chance.

**2. Loudness bias 0.1** (first 60 songs, against fwd + ref2 on the same songs; prediction 2)

| | loud_slope | light bars | density ratio | SR bias | SR error | F1@50 |
|---|---|---|---|---|---|---|
| fwd + ref2 | 0.168 | 0.049 | 1.082 | +0.041 | 0.145 | 0.366 |
| + loud bias 0.1 | 0.200 (+0.032 [+0.022, +0.044]) | 0.062 (+0.012 [+0.004, +0.021]) | 1.100 | +0.134 | 0.194 (+0.049 [+0.008, +0.101]) | 0.370 |
| human | 0.191 | 0.070 | 1 | - | - | - |

Dynamics as predicted (slope past the human level of these songs), density within 0.02
(+0.018), F1 +0.003. Missed: SR error +0.049 (predicted within 0.02), SR bias +0.09. On
the 10 of the 60 songs with audio cached here, by the bar's loudness z: the quietest
bars (z < -1.5) lose 1.9 notes per bar (9.3 → 7.3; human 6.5), the louder bars (z > 0.5)
gain 0.3 (19.3 → 19.5; human 18.1). The loud side adds notes where the sampler is
already denser than human, and SR follows the hardest stretches. → `loud_side="quiet"`.

**3. full-v3: row masks** (prediction 3)

| | full-v1 | full-v3 | predicted | |
|---|---|---|---|---|
| val CE (cell by cell) | 0.0717 | 0.0734 | ≤ 0.0745 | hit |
| probe: full, one row | 0.779 | 0.800 | ≥ 0.82 | miss |
| probe: jack rows | 0.124 | 0.269 | ≥ 0.25 | hit |
| 240 songs, move_jack | 0.011 | 0.009 (-0.001 [-0.003, -0.000]) | ≥ 0.02 | miss |
| move_trill | 0.112 | 0.116 (+0.004, n.s.) | ≥ 0.13 | miss |
| single notes in same-lane runs of 2 / 3+ | 1.40% / 0.04% | 1.22% / 0.05% | twice as many | miss |
| F1@50 | 0.373 | 0.380 (+0.006 [+0.002, +0.010]) | within 0.005 | better |
| SR error | 0.226 | 0.223 (-0.003) | within 0.02 | hit |

Also: chord share 0.331 → 0.352 (human 0.372), lone chords 0.343 → 0.366 (0.413),
long-note share 0.154 → 0.187 (0.218), ln_f1 +0.013; against: bar rhythm repeats
0.122 → 0.099 (0.435), light bars 0.050 → 0.044 (0.063), density +2%. full-v2 differed
from full-v1 by about as much (10-04), so these are partly run-to-run. The model picks
the human lane on more human jack rows, and the generated charts have no more jacks:
the failure is neither of the two written down (rhythm cost, model too small).

**Why the jacks do not come, and what "no intent" means** (240 val songs; train split)

Sampled without lane passes, the charts have the human jack rate (move_jack 0.047, human
0.044); the lane passes take it to a quarter (ref2 at temperature 0.5: 0.012; fwd + ref2
at 1: 0.021). Single notes in same-lane runs and AABB minijacks (per 1,000 single notes):

| | runs of 2 | 3 | 4+ | AABB | single-note trill runs of 8+ (share of events) |
|---|---|---|---|---|---|
| sampled | 5.65% | 0.31% | 0.05% | 1.04 | 0.01% |
| ref2 at 0.5 | 1.69% | 0.05% | 0.00% | 0.17 | 0.02% |
| fwd + ref2 at 0.5 | 1.40% | 0.04% | 0.00% | 0.16 | 0.03% |
| fwd + ref2 at 1 | 2.83% | 0.12% | 0.01% | 0.32 | 0.00% |
| full-v3 fwd + ref2 | 1.22% | 0.04% | 0.01% | 0.15 | 0.01% |
| human | 3.97% | 0.39% | 0.48% | 2.28 | 0.20% |

The larger gap is between songs. Human jack rates are skewed: 30% of the songs have
under 1% jacks and the top tenth of the songs hold 39% of all jack moves (p90 0.109); the
sampled charts put jacks into every song at about the same rate (p25-p90 0.033-0.073).
The same holds for every style number except chords. Per song against the human chart
(`compare_runs.py` now prints this; correlation over songs, and SD over songs relative
to the human charts'):

| | long-note share | jacks | trills | stairs | chords |
|---|---|---|---|---|---|
| sampled | 0.12 / 0.50 | 0.22 / 0.38 | 0.04 / 0.36 | 0.11 / 0.30 | 0.71 / 0.87 |
| fwd + ref2 | 0.12 / 0.50 | 0.01 / 0.21 | -0.01 / 0.60 | 0.13 / 0.44 | 0.71 / 0.87 |
| full-v3 fwd + ref2 | 0.13 / 0.50 | 0.02 / 0.18 | 0.00 / 0.58 | 0.04 / 0.49 | 0.73 / 0.92 |

Does the song decide them? Two human charts of the same music, SR within 0.5 of each
other (train split, all kept charts; jacks as move_jack, trills as three single notes in
a row within a beat each going there and back):

| pairs | n | long-note share | jacks | trills | chords | long-note placement F1 (chance) |
|---|---|---|---|---|---|---|
| same audio, same mapper | 1,219 | 0.73 | 0.47 | 0.42 | 0.70 | 0.588 (0.236) |
| same audio, another mapper of the mapset | 1,189 | 0.60 | 0.38 | 0.33 | 0.66 | 0.460 (0.244) |
| same song (artist - title), another mapset | 1,194 | 0.25 | 0.31 | 0.20 | 0.71 | - |
| the AI (fwd + ref2) and the human chart | 240 | 0.12 | 0.01 | -0.01 | 0.71 | 0.214 (0.156) |

Genre × SR (1-star bins) explain 7% of the variance of the long-note share and 4% of jacks and
trills (6,000 train charts). Jacks are not cued by the sound either: of two consecutive
single notes within a beat, the human charts put the second in the same lane 3-6% of
the time whatever the audio similarity of the two onsets (40 songs, cosine bins from
below 0.5 to above 0.95).

Readings:
1. Long-note amount, jack rate and trill rate are mostly a choice of the mapset and the
   mapper: another mapset's chart of the same song predicts them at 0.2-0.3. Chords
   follow the music and the SR (0.71, and the AI matches that). The AI does not make
   the choice: it mixes every style into every song at an average rate, with half the
   human spread. That is what "no intent in the long notes" and "no ddddffff" say.
2. Placement follows the choice. Given a committed layout around it (the human chart),
   the model places long notes at F1 0.86 (chance 0.24); in its own uncommitted charts, barely
   above chance. Re-deciding afterwards (refine_holds) fixes the amount, not the layout.
3. The lane passes at temperature 0.5 remove the jacks the sampler had (0.047 → 0.011)
   and with them the little song dependence they carried (0.22 → 0.01).
4. full-v3's row masks cost nothing measurable and help rhythm and chords slightly; the
   default model stays full-v1, and the next run keeps the row masks.

Next, the choice as an input: full-v4 = full-v3 + the chart's own long-note share, jack
rate and trill rate as inputs (`src/data/chartstats.py`, `train.py --chart-stats`): 8
quantile buckets of the train charts each, one embedding table per stat summed into the
s/b condition, dropped in training (all with 0.15, then each with 0.15). Sampling takes
them from the user (`--stats ln=0.4,jack=0.05`), from a random human train chart of about
the same SR (`--stats sample`), from the human chart being scored (`--stats oracle`,
evaluation only, as the SR is), or none. Unlike the mapper labels, these are fixed by
the answer itself, so a model that ignores them pays for it at every high mask ratio.
On the 16,454 train charts (`build_chart_stats.py`, 24 s): long-note share median 0.150
(p10 0.008, p90 0.428), jacks 0.024 (0.000, 0.113; more than an eighth have none, which
get bucket 0 to themselves), trills 0.136 (0.045, 0.282).

Predictions for the runs of 10-05/06, written before them:
1. full-v4 training: val_ce (own stats) at least 0.001 below val_ce_null at the end;
   val_ce_null within 0.002 of full-v3 (0.0734).
2. full-v4 `--stats oracle`, fwd + ref2, 240 songs, against full-v3 fwd + ref2: per-song
   correlation with the human chart, long-note share 0.6 or more (0.13), jacks 0.4 or
   more (0.02), trills 0.3 or more (0.00); SD over songs relative to human 0.8 or more
   for the long-note share (0.50), 0.5 or more for jacks (0.18); means: move_jack 0.02 or
   more (0.009), move_trill 0.13 or more (0.116); ln_f1 over chance +0.10 or more
   (+0.053); same-lane runs of 3+ and AABB twice full-v3's; F1@50 within 0.005 and SR
   error within 0.02 of full-v3.
3. `--stats none` (60 songs): as full-v3 on the same songs within run-to-run size (F1
   within 0.01, long-note share within 0.03).
4. `--stats sample` (60 songs): SD over songs relative to human 0.8 or more for the
   long-note share and 0.5 or more for jacks; correlation with the human chart near 0
   (another human's choice, not this song's); F1 within 0.01 of `--stats none`.
5. `--loud-bias 0.1` quiet side only (full-v1, 60 songs, against fwd + ref2): light bars
   +0.008 or more, loud_slope +0.015 or more, SR bias within ±0.03 of fwd + ref2's
   (+0.04), density down 1-4%, F1@50 within 0.005.
How 2 could fail: the stats are used (prediction 1 holds) but the lane passes at
temperature 0.5 squash the jacks again, so correlation rises while the jack mean stays
under 0.02; or, like the mapper labels, they are ignored (val_ce = val_ce_null).

### 2026-10-06 · full-v4: chart stats as inputs; the quiet-side loudness bias (240 / 60 val songs)

`run_1006.sh`: `build_chart_stats.py` (18,191 charts, 24 s); full-v4 = the full-v3 recipe
(row masks 0.5) + `--chart-stats` (60,000 steps, 3.3 h); evaluated with fwd + ref2 on the
240 val songs with `--stats oracle`, and on the first 60 with none and with `sample`;
full-v1 fwd + ref2 with `--loud-bias 0.1` on the quiet side (first 60). Paired song by
song with `compare_runs.py` against full-v3 fwd + ref2 (or the run named).

**Predictions of 10-05, checked**

| # | prediction | result | |
|---|---|---|---|
| 1 | val_ce (own stats) ≥ 0.001 below val_ce_null | 0.0740 vs 0.0741 (gap 0.0001 at every check from 8k steps) | miss |
| 1 | val_ce_null within 0.002 of full-v3 (0.0734) | 0.0741 | hit |
| 2 | per-song correlation with the human chart: long-note share ≥ 0.6 | 0.75 [0.68, 0.81] (full-v3 0.13) | hit |
| 2 | jacks ≥ 0.4 | 0.42 [0.09, 0.65] (0.02) | hit, barely |
| 2 | trills ≥ 0.3 | 0.07 [-0.05, 0.21] (0.00) | miss |
| 2 | SD over songs / human: long-note share ≥ 0.8, jacks ≥ 0.5 | 0.71 (0.50), 0.34 (0.18) | miss, miss |
| 2 | move_jack ≥ 0.02, move_trill ≥ 0.13 | 0.0098 (0.0092), 0.132 (+0.017 [+0.008, +0.025]) | miss, hit |
| 2 | ln_f1 over chance ≥ +0.10 | +0.062 (0.250 vs chance 0.188; full-v3 +0.053) | miss |
| 2 | same-lane runs of 3+ and AABB twice full-v3's | 0.09% (0.05%), AABB 0.20 (0.15) per 1,000 | miss |
| 2 | F1@50 within 0.005, SR error within 0.02 | -0.001, +0.019 | hit |
| 3 | `--stats none` as full-v3 (F1 within 0.01, long-note share within 0.03) | F1 -0.004, share -0.027 | hit |
| 4 | `--stats sample`: SD / human ≥ 0.8 (long notes), ≥ 0.5 (jacks) | 0.95, 0.17 | hit, miss |
| 4 | correlation with the human chart near 0; F1 within 0.01 of none | 0.09 / 0.16; -0.010 [-0.018, -0.003] | hit, at the edge |
| 5 | quiet-side loud bias: light bars ≥ +0.008, loud_slope ≥ +0.015 | +0.013 [+0.004, +0.021] (0.062; human 0.070), +0.025 [+0.015, +0.037] (0.193; human 0.191) | hit, hit |
| 5 | SR bias within ±0.03, density -1 to -4%, F1 within 0.005 | +0.010, -1.2%, +0.001 | hit |

The val CE gap says nothing here: every validation ratio leaves enough of the chart in
view to read its style from (the mapper labels looked the same, 10-04), while generation,
which starts from nothing, follows the stats (below).

**How far each stat moves the charts** (the oracle run: the AI's value by the bucket of
the value it was given, 240 songs)

| bucket | long-note share: asked → AI | jacks: asked → AI | trills: asked → AI |
|---|---|---|---|
| 0 | 0.004 → 0.013 | 0.000 → 0.007 | 0.022 → 0.156 |
| 2 | 0.082 → 0.084 | 0.009 → 0.007 | 0.098 → 0.114 |
| 4 | 0.178 → 0.162 | 0.031 → 0.008 | 0.149 → 0.128 |
| 6 | 0.326 → 0.262 | 0.074 → 0.009 | 0.228 → 0.148 |
| 7 | 0.537 → 0.337 | 0.154 → 0.020 | 0.324 → 0.144 |

Long notes follow up to about 0.2 and fall short above it. Jacks and trills hardly move.

**Where the stats get lost** (CPU, one 8-bar window from the middle of each of 10 val
songs; the same window and seed with all three stats low (long notes 0.03, jacks 0,
trills 0.06) or high (0.40, 0.11, 0.28); counts pooled over the songs)

| | long-note share low / high | jacks low / high | trills low / high |
|---|---|---|---|
| sampled (no lane passes) | 0.022 / 0.454 | 0.027 / 0.069 | 0.149 / 0.219 |
| fwd + ref2 | 0.022 / 0.454 | 0.000 / 0.114 | 0.140 / 0.170 |
| fwd + ref2, guidance 2 throughout (`--style-guidance 2`) | 0.001 / 0.688 | 0.000 / 0.060 | 0.032 / 0.210 |
| fwd + ref2, guidance 1 in the lane passes only (`--lane-guidance 1`) | 0.022 / 0.454 | 0.000 / 0.136 | 0.076 / 0.195 |
| fwd + ref2, guidance 2 in the lane passes only | 0.022 / 0.454 | 0.000 / 0.170 | 0.099 / 0.246 |
| human (the same windows) | 0.191 | 0.030 | 0.070 |

296-317 single-note moves and 140-200 move pairs per cell.

Sampling follows all three (long notes 0.02 vs 0.45, trills 0.15 vs 0.22). The lane
passes keep the long notes (they keep the rhythm and the holds) and halve what is left
of the trill difference (0.03). Guidance throughout overshoots the long notes (0.001 /
0.69 for 0.03 / 0.40). Guidance in the lane passes only leaves the long notes as sampled
and brings the trill difference back (+0.12 at 1, +0.15 at 2). Jacks separate in these
windows even without guidance, unlike in the full songs; with all three stats high
together, part of it may come with the long notes (repeated long notes in one lane).

Readings:
1. The chart stats work for what sampling decides: the long-note share follows the
   chart's value (correlation 0.75, a human spread with `sample`), as the SR does. For
   long notes the AI now has an amount per song; where they go is still near chance
   (+0.06 over chance at the right amount).
2. Jacks and trills are decided again by the lane passes, which ask with the rest of the
   chart in view; there the context outweighs the stats, and the passes at temperature
   0.5 pick the likeliest lanes. Guidance only in the lane passes
   (`generate_song(lane_guidance=)`, `--lane-guidance`: logits c + w (c - u) towards the
   stats, u without them) restores the trill difference in windows without touching the
   long notes. To be checked on whole songs.
3. The quiet-side loudness bias does what the two-sided one did for the dynamics without
   the SR: playable charts use it from now on (`generate.py`, `sample.py`:
   `--loud-bias 0.1`, 0 turns it off).

Next (`run_1007.sh`): a pairs playtest pack with full-v4 for the 10-07 meeting
(16 val songs, `--seed 1`: none of the 10-04 songs; settings in turn: fwd + ref2 + copies
+ quiet-side loudness + lane guidance 1, with the human chart's stats (`st`) or a random
train chart's of that SR (`stsample`)), then full-v4 `--stats oracle --lane-guidance 1`
and `2` on the first 60 songs against `--stats oracle` alone on the same songs
(correlation with the human chart: long notes 0.79, jacks 0.60 [-0.10, 0.84], trills 0.00;
move_jack 0.012, move_trill 0.142, motion_pred 0.123, F1@50 0.373).

Predictions, written before the runs:
1. Lane guidance 1: per-song correlation of trills with the human chart 0.3 or more
   (0.00); move_jack 0.015 or more; long-note share and its correlation within 0.01 /
   0.05 (they are decided before the passes); F1@50 within 0.005, SR error within 0.02;
   motion_pred down by at most 0.01; forward passes 1.4-1.7 times.
2. Lane guidance 2: trills 0.35 or more; motion_pred down by at most 0.02, F1 within 0.01.
3. Playtest (one player, forced choice): AI taken for human 1-4 of 16 (0 of 16 on
   10-04); fewer comments on long notes without intent; jacks and minijacks still named.
How it could fail: guidance drags the lanes to the stat at the cost of the patterns
the passes were for (motion_pred and coverage down by more than 0.02), or whole songs
behave unlike the windows (trills stay flat).

### 2026-10-06/07 · Lane guidance towards the chart stats (full-v4, first 60 val songs)

`run_1007.sh`: full-v4 fwd + ref2 `--stats oracle` with `--lane-guidance 1` and `2`,
paired with `--stats oracle` alone on the same 60 songs. The playtest pack of that run
(16 songs, lane guidance 1) waits for its ratings.

| | no guidance | lane guidance 1 | lane guidance 2 | human |
|---|---|---|---|---|
| trills, per-song correlation with the human chart | 0.00 | 0.13 [-0.16, 0.40] | 0.32 [0.02, 0.57] | - |
| jacks, correlation | 0.60 | 0.60 | 0.60 [0.21, 0.82] | - |
| move_jack | 0.012 | 0.019 (+0.008 [+0.001, +0.017]) | 0.034 (+0.023 [+0.007, +0.042]) | 0.055 |
| move_trill | 0.142 | 0.147 | 0.162 (+0.020 [+0.004, +0.038]) | 0.160 |
| jacks asked 0.124 (buckets 6-7, 21 songs) → got | 0.019 | 0.040 | 0.085 | - |
| jacks asked 0.004 (buckets 0-2, 20 songs) → got | 0.010 | 0.011 | 0.009 | - |
| trills asked 0.299 / 0.074 (buckets 6-7 / 0-2) → got | 0.163 / 0.144 | 0.184 / 0.136 | 0.248 / 0.133 | - |
| single notes in same-lane runs of 3 / 4+ | 0.05% / 0.17% | 0.33% / 0.47% | 0.81% / 0.87% | 0.61% / 1.01% |
| AABB minijacks per 1,000 single notes | 0.39 | 1.61 | 2.41 | 3.08 |
| motion_pred | 0.123 | 0.123 | 0.122 | 0.129 |
| F1@50 / SR bias | 0.373 / +0.064 | 0.375 / +0.077 | 0.373 / +0.097 (+0.033 [+0.004, +0.073]) | - |
| forward passes | 4,051 | 5,444 | 5,444 | - |

Long-note share, chords and rhythm are the same in all three (decided before the passes).
Predictions: 1 (lane guidance 1) trills ≥ 0.3 miss (0.13), move_jack ≥ 0.015 hit, long
notes unchanged hit, F1 / SR hit, motion_pred hit, passes 1.4-1.7x miss (1.34x); 2 (lane
guidance 2) trills ≥ 0.35 miss narrowly (0.32), motion_pred and F1 hit.

Readings:
1. With guidance 2 in the lane passes the jacks come back where the chart asks for them
   and only there: jack-heavy songs 0.019 → 0.085, jack-free ones stay at 0.009.
   Same-lane runs and AABB minijacks reach the human rate (2.41 vs 3.08 per 1,000),
   which no sampler setting had done (jack bias 1: 0.89, at the cost of motion_pred).
   motion_pred and F1 do not move; SR rises by 0.03 (jacks are hard).
2. Trills follow halfway: songs asking for many get more (0.16 → 0.25), songs asking for
   few do not get fewer (0.13 for 0.07): below the model's habit the stat does not pull.
3. Next default candidate for playable charts with full-v4: fwd + ref2 + lane guidance 2
   + copies + quiet bars, the stats from `--stats sample` or the user. To be playtested
   against the lane-guidance-1 pack of 10-06.

### 2026-10-06 · Playtest of full-v4 (10 of 16 songs) and the quiet sections

The pairs pack of `run_1007.sh` (full-v4, fwd + ref2 + copies + quiet bars + lane guidance
1; the human chart's stats (`st`) and a random train chart's (`stsample`) in turn; 16 val
songs, seed 1), one player, 10 songs played: the human chart found in 10 of 10 (AI taken for
human 0 of 10, 95% Wilson [0%, 28%]). Confidence (1-5) 5, 5, 2, 4, 3, 4, 4, 4 on the 8
songs rated; with the human chart's stats 3.5 on average (4 songs), with a random chart's
4.25 (4). On PT13 the player would have taken the AI chart (st) for human if it had been
alone ("the long-note intent improved, confusing"); the human chart gave itself away by
its committed long-note style.

Comments, and what the numbers say:
- "off the beat in quiet parts, fatal", "notes where the song is really quiet (34-44 s),
  where the human chart is empty": confirmed and large. Of the AI onset rows with no human
  onset row within one cell (1/12 beat), by the bar's loudness z (40 val songs, full-v4
  oracle): z < -1 30%, -1..0 20%, 0..1 14%, > 1 6%. Bars inside the song that the human
  chart leaves empty (1.2% of the bars, 55% of them at loudness z < -1.5 against 8% of all
  bars): the AI puts notes in 83% (full-v1 fwd + ref2) to 92% (full-v4) of them; the
  quiet-side loudness bias 0.1 does not change that (84%). Odd snaps (1/6, 1/12) in quiet
  bars 14% (human 10%).
- "the long-note intent improved but is not there yet; long notes cut by taps (PT14)", "the
  speed of the song not handled (완급)", "steady-rhythm mash (PT15)", "phrases come in 4 / 8
  bars": not measured yet.

Where the off-rhythm notes land (40 val songs; the audio's onset strength per row: spectral
flux of the song-standardized log-Mel, the largest of the row's frames and of the next
row's, as audio onsets peak about 18 ms after note times; ranked within the song):

| onset rows | below the song's median onset strength | in its bottom quarter |
|---|---|---|
| human | 26% | 9% |
| AI, on the human rhythm (within a cell) | 22% | 8% |
| AI, off it | 62% | 31% |

(The previous row instead of the next one separates them less: 60% / 29% against 33% /
14%.) The AI's off-rhythm onsets sit where nothing in the music starts; the model's own
audio reading has not kept them out there.

`sampler.onset_gate` / `--onset-bias X`: while sampling, an EMPTY bias per row of
X * clip((0.5 - q) / 0.4, 0, 1), q the share of the song's rows with weaker onsets: 0 at
the median and above, X at the 10th percentile and below (silence included); it adds to the
loudness bias. New columns (`structure.quiet_rhythm`, also in `compare_runs.py`):
off_rhythm_quiet, off_rhythm_rest and empty_bar_fill as above.

Predictions for the night of 10-06 (`run_quiet.sh`; full-v4, fwd + ref2, oracle stats, lane
guidance 2, the first 60 val songs, against the same without the gate, rescored for the new
columns; that one is expected near off_rhythm_quiet 0.30, off_rhythm_rest 0.15,
empty_bar_fill 0.90):
1. `--onset-bias 1`: off_rhythm_quiet down by 0.03 or more, off_rhythm_rest by 0.02 or
   more, empty_bar_fill by 0.10 or more; F1@50 up by 0.005 or more (precision); density
   down 3-8%, SR bias down 0.05-0.2; motion_pred, move_jack, move_trill within 0.01.
2. `--onset-bias 2`: about twice that: off_rhythm_quiet down 0.06, F1 +0.01, density
   down 6-15%.
3. `--loud-bias 0.3` (quiet side): light bars up 0.02 or more, empty_bar_fill down 0.10 or
   more, off_rhythm_quiet down 0.02 or more (fewer notes there); density down 2-5%, SR bias
   within 0.05.
4. Both (1 and 3): the gains add up roughly; density down 5-12%.
How it could fail: the notes taken away come back elsewhere in the same window (the SR
condition holds the density), so the off-rhythm shares stay; or the gate thins every
quiet part alike and recall falls more than precision rises (F1 down).
Also in the run: a second pairs pack with lane guidance 2 (16 new songs, seed 81, tag
1006g).

### 2026-10-06/07 · The onset gate and a stronger quiet-bar bias (60 val songs), and a fix

`run_quiet.sh`, full-v4 fwd + ref2, oracle stats, lane guidance 2, first 60 val songs,
paired with the same run without them (rescored for the new columns). `--onset-bias` was
then an EMPTY bias (`_ob` directories).

| | reference | onset gate 1 | gate 2 | quiet bars 0.3 | gate 1 + quiet 0.3 | human |
|---|---|---|---|---|---|---|
| off the human rhythm, quiet bars | 0.293 | 0.256 (-0.036 [-0.058, -0.017]) | 0.236 (-0.057) | 0.225 (-0.067) | 0.202 (-0.090) | - |
| off the human rhythm, other bars | 0.204 | 0.161 (-0.043 [-0.051, -0.035]) | 0.139 (-0.064) | 0.193 (-0.010) | 0.151 (-0.053) | - |
| human-empty bars filled | 0.907 | 0.689 (-0.22) | 0.531 (-0.38) | 0.735 (-0.17) | 0.600 (-0.31) | - |
| F1@50 | 0.373 | 0.377 (+0.004 [-0.002, +0.010]) | 0.379 (+0.006 [+0.000, +0.012]) | 0.372 | 0.373 | - |
| density ratio | 1.114 | 1.028 | 0.982 | 1.044 | 0.972 | 1 |
| SR bias | +0.097 | -0.091 | -0.157 | +0.035 | -0.113 | - |
| long-note share | 0.157 | **0.098** | **0.074** | 0.145 | 0.088 | 0.190 |
| light bars / loud_slope | 0.046 / 0.167 | 0.065 / 0.184 | 0.075 / 0.189 | 0.077 / **0.247** | 0.088 / 0.259 | 0.070 / 0.191 |
| motion_pred / move_jack | 0.122 / 0.034 | 0.130 / 0.053 | 0.131 / 0.048 | 0.120 / 0.034 | 0.129 / 0.048 | 0.129 / 0.055 |
| chord share | 0.353 | 0.374 | 0.396 | 0.348 | 0.372 | 0.367 |

Predictions: 1 (gate 1) off-rhythm in quiet bars and elsewhere, empty bars: hit (-0.036,
-0.043, -0.22); F1 +0.005: miss (+0.004); density -3 to -8%: miss (-8.6%, but from 11%
over the human count to 3%); SR bias -0.05 to -0.2: hit (-0.19); motion_pred within 0.01:
hit; jacks and trills within 0.01: miss (+0.019, +0.016, towards human). 2 (gate 2):
off-rhythm -0.06 hit (-0.057), F1 +0.01 miss (+0.006), density hit. 3 (quiet bars 0.3):
light bars, empty bars, off-rhythm in quiet bars hit; density -2 to -5% and SR bias within
0.05 miss (-7%, -0.06); it overshoots the dynamics (loud_slope 0.247 against 0.191). 4:
the gains add up (off-rhythm in quiet bars -0.090), density miss (-14%).

Readings:
1. The gate does what it was for, in quiet and loud parts alike (the loudness bias only
   in quiet parts), with F1 a little up, density from 11% over the human count to 3%,
   and pattern numbers (motion_pred, jacks, chord share) moving towards human.
2. It also cost a third of the long notes (0.157 → 0.098; ln_f1 -0.069), not predicted:
   as an EMPTY bias it acts on every cell of a weak-onset row, and the body of a long
   note sits on weak onsets by nature (nothing starts while it is held). Fix: the gate is
   now a penalty on starting a note (TAP and HOLD_START) only; a body or release decides
   as before (`_open_cells(row_onset=)`, `sample_window(row_onset_bias=)`; the tag is
   `_og` now, playtest `ogX`).
3. Quiet bars at 0.3 overshoot the dynamics; the playable default stays at 0.1.

Predictions for the night of 10-06/07 (`run_gate.sh`), written before it:
1. Gate 1 as a start penalty (same 60 songs, against the reference): the off-rhythm and
   empty-bar gains within 0.02 / 0.02 / 0.05 of the EMPTY form's (-0.036 / -0.043 /
   -0.22); long-note share within 0.02 of the reference (0.157; the EMPTY form 0.098) and
   ln_f1 within 0.02; density down 5-10%; F1 up 0.003 or more.
2. Gate 1.5: off-rhythm gains between gate 1's and gate 2's; long notes within 0.03.
3. The candidate for playable charts on all 240 songs (oracle stats, lane guidance 2,
   gate 1, quiet bars 0.1), against full-v4 oracle without them (rescored): off-rhythm in
   quiet bars 0.25 or less, human-empty bars filled 0.75 or less, move_jack 0.04 or more,
   per-song correlation of the long-note share 0.7 or more, F1 within 0.005, SR error
   within 0.03.
4. A pairs pack of that candidate (16 songs none of the earlier packs had, tag 1007):
   AI taken for human 1-4 of 16; fewer comments on quiet parts.

### 2026-10-07 · The onset gate on note starts only; the candidate on 240 val songs

`run_gate.sh` (predictions in the 10-06/07 entry). Full-v4 fwd + ref2, oracle stats.

The gate as a start penalty, first 60 val songs, lane guidance 2, against the same without
it (the EMPTY form of 10-06 alongside):

| | reference | gate 1, EMPTY form | **gate 1, starts** | gate 1.5, starts |
|---|---|---|---|---|
| off the human rhythm, quiet bars | 0.293 | 0.256 | **0.258** (-0.034 [-0.052, -0.019]) | 0.257 |
| off the human rhythm, other bars | 0.204 | 0.161 | **0.167** (-0.036 [-0.045, -0.028]) | 0.157 |
| human-empty bars filled | 0.907 | 0.689 | **0.837** (-0.070 [-0.172, -0.007]) | 0.826 |
| long-note share (human 0.190) | 0.157 | 0.098 | **0.172** (+0.014) | 0.168 |
| long-note beats / released on an onset row | 0.78 / 0.70 | 0.79 / 0.70 | 0.82 / 0.68 | 0.83 / 0.68 |
| ln_f1 | 0.224 | 0.161 | 0.231 | 0.231 |
| F1@50 | 0.373 | 0.377 | 0.376 (+0.003 [-0.002, +0.008]) | 0.378 |
| density ratio / SR bias | 1.114 / +0.097 | 1.028 / -0.091 | 1.041 / -0.022 | 1.030 / -0.062 |
| motion_pred / chord share | 0.122 / 0.353 | 0.130 / 0.374 | 0.115 / 0.375 | 0.113 / 0.385 |

Predictions: 1 off-rhythm gains within 0.02 of the EMPTY form's hit (-0.034 / -0.036
against -0.036 / -0.043); empty bars within 0.05 miss (-0.070 against -0.22); long notes
within 0.02 and ln_f1 hit (+0.014, +0.007); density -5 to -10% hit (-7%); F1 +0.003 hit
(+0.0031). 2 (gate 1.5) off-rhythm in quiet bars between gates 1 and 2 miss (as gate 1),
elsewhere hit, long notes hit; SR error +0.046 [+0.003, +0.090].

The candidate for playable charts (lane guidance 2, gate 1, quiet bars 0.1) on all 240
val songs, against full-v4 oracle alone (rescored):

| | full-v4 oracle | candidate | human |
|---|---|---|---|
| off the human rhythm, quiet / other bars | 0.288 / 0.178 | **0.231 / 0.142** | - |
| human-empty bars filled | 0.865 | 0.778 | - |
| move_jack / move_trill | 0.010 / 0.132 | **0.033** / 0.173 | 0.044 / 0.153 |
| chord share / lone chords | 0.353 / 0.358 | 0.381 / 0.391 | 0.372 / 0.413 |
| loud_slope / light bars | 0.171 / 0.046 | **0.200 / 0.063** | 0.194 / 0.063 |
| per-song correlation with the human chart: long notes / jacks / trills | 0.75 / 0.42 / 0.07 | **0.77 / 0.62 / 0.36** | - |
| F1@50 / F1@50 any lane | 0.378 / 0.763 | 0.381 / 0.765 | - |
| density ratio / SR bias / SR error | 1.075 / +0.020 / 0.243 | 0.984 / **-0.138** / 0.260 | 1 |
| motion_pred / move_stair | 0.122 / 0.179 | 0.111 / 0.167 | 0.132 / 0.211 |
| released on an onset row | 0.696 | 0.667 | 0.759 |
| seconds per song (MacBook) | - | 60 | - |

Predictions 3: off-rhythm in quiet bars 0.25 or less hit (0.231); empty bars 0.75 or less
miss (0.778); move_jack 0.04 or more miss (0.033); long-note correlation 0.7 or more hit
(0.77); F1 within 0.005 hit (+0.003); SR error within 0.03 hit (+0.017). Prediction 4 (the
playtest of the pack, `outputs/full-v4/playtest_gate`, tag 1007) waits for ratings.

Readings:
1. As a start penalty the gate keeps the rhythm gains and the long notes (more of them,
   and longer: a held note is not cut where nothing starts). It no longer empties the
   bars humans leave empty: that part came from cutting long-note bodies, so silence
   needs its own rule (e.g. a strong start penalty where the audio is near its minimum).
2. Gate 1.5 is no better than 1 and misses the SR more: 1 stays.
3. The candidate moves most numbers onto the human level at once (dynamics, light bars,
   chords, jacks towards it, per-song style) with F1 a little up; it costs 0.14 SR under
   the target (2% fewer notes than human against 7% more before), motion_pred -0.01 and
   releases on onset rows -0.03.

### 2026-10-07 night · Choices for a whole bar: rests, long-note bars, the SR offset (predictions)

Three gaps the playtests named, measured on the 240 val songs (human charts against the
candidate, full-v4 + lane guidance 2 + onset gate 1 + quiet bars 0.1), all choices a chart
makes for a whole bar:

| | human | candidate | before the gate |
|---|---|---|---|
| bars inside a song without a note start (rest bars) | 0.85% | (the candidate fills 77% of the human ones) | 88% filled |
| bars with 4+ starts, 80%+ of them long notes (ln bars) | 7.5% | 3.8% | 3.5% |
| bars with 8+ onset rows, evenly spaced (steady bars) | 36.5% | 20.1% | 18.7% |

The audio does not find the rests. Their bars are quieter in the middle (loudness z -1.34
against 0.29, 11.6 dB under the song's loud bars against 2.6) but spread wide: a quarter
are above z -0.42, and humans fill 94% of the bars at z < -2. ROC AUC for a human rest:
level in dB 0.84, loudness z 0.84, the bar's strongest onset 0.71. A rule on loudness that
finds half the rests (z < -1.5) flags 7.6% of the bars, 5% of them rests, and takes 4.6% of
the human notes: the queue's silence rule is dropped. Each of the three is a joint choice
of a whole bar (192 cells kept empty; most starts long; no hole in a stream), which
cell-by-cell sampling seldom makes even where every cell leans that way, as with the lanes
(2026-10-02/03): ask the model about the whole bar, with the rest of the chart in view.

`run_rest.sh` (patch 0038):

1. `scripts/probe_bars.py`: for every bar inside each val song, the bar MASK and the rest of
   the chart in view (the human chart; then the candidate's charts), the expected note
   starts (sum of p(TAP) + p(HOLD_START) over the bar's cells) and long-note starts.
2. `--rest T` (`sampler.rest_bars`, tag `_rb`): after all other passes, every bar whose
   expected starts, asked that way, fall below T is left empty (its taps and the long notes
   that start in it). On the candidate's saved charts (`--from-charts`, no new sampling)
   at T 0.5, 1, 2.
3. `--sr-offset 0.15` (tag `_so`): the candidate sampled at the target SR + 0.15 against the
   gate's -0.14; then the rest pass at 1 on those charts.
4. New columns for both charts: `rest_bars`, `ln_bars`, `steady_bars` (structure.bar_kinds).

Predictions:

1. Probe, human context: rest AUC of the expected starts 0.93 or more (the audio 0.84); at
   T = 1, 0.5-1.5% of the bars below, 35% or more of them human rests, half of the rests
   found. Candidate context: AUC 0.88 or more, precision at T = 1 of 25% or more.
2. Probe, ln bars (busy bars): AUC of the expected long-note share 0.90 or more, the chart's
   long-note share alone (the chart-stat input) about 0.80.
3. Rest pass, T = 1: the candidate's rest bars to 0.5-1.5% (human 0.85%), human-empty bars
   filled 0.78 → 0.60 or less, F1@50 within 0.002, density -1% or less, the other numbers
   unchanged. T = 0.5 about half of that; T = 2 rest bars 1.5-3%, fill 0.45 or less, F1
   -0.002 to -0.005 (bars humans fill get emptied).
4. SR offset 0.15: SR bias -0.14 → within ±0.05; density 0.98 → 1.04-1.08; off the human
   rhythm in quiet bars 0.231 → 0.245 or less; F1 within ±0.004; jacks, trills, long notes,
   dynamics within their noise (±0.005, ±0.01 for the slope).

Failure modes: the model spreads its expectation over the masked bar and the expected
starts of human rests are not far below the filled bars' (AUC under 0.9): rests are then
not a bar-level choice the model knows, and a rest pass would empty bars humans fill. Or
the model knows the rests next to human context but not next to its own charts.

**Results** (10-08, `outputs/overnight_rest.log`; the candidate-context probe was written
to `probe_bars_eval_..._lbq0.{csv,json}`: `Path.with_suffix` cut the name at its last dot)

Probe, 240 val songs, 29,436 bars inside the songs, 0.85% of them human rests:

| | human context | candidate context |
|---|---|---|
| rest: AUC of the expected starts (audio: dB 0.84, loudness z 0.84) | **0.986** | 0.938 |
| T = 0.5: bars below / of them human rests / of the rests below | 0.61% / 73% / 53% | 0.78% / 19% / 18% |
| T = 1 | 1.60% / 36% / 68% | 1.21% / 19% / 28% |
| T = 2 | 3.20% / 23% / 87% | 2.73% / 16% / 50% |
| ln bars (busy bars): AUC of the expected long-note share (the chart's share alone 0.80) | **0.954** | 0.749 |

The rest pass on the candidate's saved charts (paired, 240 songs):

| | candidate | T 0.5 | **T 1** | T 2 | human |
|---|---|---|---|---|---|
| rest bars | 0.61% | 0.64% | **0.87%** | 1.70% | 0.75% |
| human-empty bars filled | 0.778 | 0.705 | **0.644** | 0.416 | - |
| off the human rhythm, quiet bars | 0.231 | 0.226 | 0.223 | 0.209 | - |
| density ratio | 0.984 | 0.983 | 0.981 | 0.972 | 1 |
| F1@50 / rho_in | 0.381 / 0.206 | 0.381 / 0.207 | 0.381 / 0.212 | 0.381 / 0.222 | - / 0.228 |
| light bars / loud_slope | 0.063 / 0.200 | 0.061 / 0.202 | 0.058 / 0.204 | 0.050 / 0.211 | 0.063 / 0.194 |

At T = 1 the pass empties 191 bars (0.8 per song): 34% are human rests, and in 68% the
human chart has at most 2 starts (median 1); the candidate had a median of 1 start there.

The SR offset 0.15 (sampled again, 240 songs): SR bias -0.138 → -0.026 [+0.087, +0.138],
density 0.98 → 1.04, F1@50 +0.003, off the human rhythm in quiet bars unchanged but +0.011
elsewhere, chords +0.015 (away from human), steady bars -0.016, trills' per-song correlation
0.36 → 0.46; SR error 0.260 → 0.293. By grade (SR bias, full-v4 without the gate /
candidate / +0.15):

| grade | SR bias | onset rows below the song's median onset strength: human / no gate / candidate |
|---|---|---|
| Easy (47) | +0.10 / +0.07 / +0.21 | 0.055 / 0.075 / 0.055 |
| Normal (32) | +0.08 / +0.05 / +0.16 | 0.088 / 0.117 / 0.087 |
| Hard (72) | +0.10 / -0.05 / +0.05 | 0.157 / 0.205 / 0.142 |
| Insane (66) | -0.03 / -0.27 / -0.17 | 0.244 / 0.280 / 0.189 |
| Expert (18) | -0.19 / -0.62 / -0.46 | 0.287 / 0.326 / 0.223 |
| Expert+ (5) | -0.83 / -1.10 / -1.13 | 0.313 / 0.306 / 0.215 |

Predictions: 1 hit (0.986; at T = 1 1.6% of the bars, a little over 0.5-1.5%; 36% human
rests; 68% of them found); candidate context AUC hit (0.938), precision miss (19% against
25%). 2 hit (0.954; the chart's share 0.80). 3 at T = 1: rest bars hit, filled bars miss
(0.644 against 0.60), F1 and density hit; T 0.5 about half hit; T 2 rest bars and fill hit,
F1 miss the other way (-0.0004 against -0.002 to -0.005). 4 SR bias hit, density hit
narrowly (1.037), off-rhythm in quiet bars and F1 hit, trills just miss (-0.006); chords
+0.015 not foreseen.

Readings:
1. The model knows where humans rest and where they hold a whole bar, when the rest of
   the chart is a human one (AUC 0.99 and 0.95, far above the audio and the chart's
   style). Next to its own charts it finds the rests less precisely (0.94; 19% of the bars
   under T 0.5 are human rests against 73%) and the long-note bars hardly (0.75, under the
   chart-level share): the context decides, and its own context tells less.
2. The rest pass at 1 brings the rest bars to the human count with nothing else lost: the
   bars it empties held about one note in both charts.
3. A flat SR offset is the wrong cure. The gate costs SR from Hard up only (density
   Insane 0.92, Expert 0.88), where humans stream through weak onsets: their onset rows
   below the median onset strength rise from 5.5% (Easy) to 29-31% (Expert, Expert+); the
   gate at 1 matches that up to Normal and cuts too much above (19-22% at Insane+), and
   steady bars fall short most there (Insane 0.17 against 0.32, Expert 0.11 against 0.26).
   The offset adds notes everywhere instead: Easy and Normal overshoot.

### 2026-10-08 · The gate tapered by SR, stream holes, rests and long-note bars without a chart (predictions)

From the 10-07 night: the gate costs SR from Hard up, where humans start a quarter of
their notes on weak onsets (streams), and the sampled streams miss notes (steady bars 28%
against 41% per song). And the model finds human rests and long-note bars only next to a
human chart. `run_taper.sh` (patch 0039):

1. `probe_bars.py --context none`: the whole chart MASK, so audio, SR and stats alone.
   New: AUCs within songs (`*_within`: pairs of bars of one song only; the chart-level
   long-note share is no help there).
2. `--fill-holes T` (`sampler.fill_holes`, tag `_fh`): a row where an evenly spaced stream
   (1/4, 1/3, 1/2 or 1 beat) misses exactly one note gets a tap when the model, with the
   row's free lanes MASK and the chart around it, starts a note there with probability T
   or more. On the candidate's charts with the rest pass (240 songs, no new sampling) at
   0.3 and 0.5.
3. `--onset-taper 2.7,5,0.35` (`sampler.taper_factor`, tag `_ot`): the gate at full strength
   up to SR 2.7, falling linearly to 0.35 at SR 5 and above (the strength the per-grade
   weak-onset shares ask for if the share moves linearly between gate 0 and 1). The
   candidate with the rest pass otherwise, sampled again on the 161 val songs at SR 2.7
   or above (`--sr-min 2.7`, which keeps the full run's seeds); then the holes at 0.5.
4. New column `weak_onsets` (and `human_`): the share of onset rows below the song's median
   onset strength; `compare_runs --by-grade` shows it with the density and steady bars.

Predictions:

1. No chart: rest AUC 0.90 (0.88 within songs; the audio alone 0.84); long-note bars 0.85
   (0.75 within songs). Below 0.8 within songs, deciding them before sampling is out.
2. Holes at 0.5: 3-6 filled per song; steady bars 0.28 → 0.31; breaks per 100 0.14 → 0.10;
   density +0.5 to 1%; F1@50 +0.001; SR +0.02. At 0.3: half again as many, steady bars
   0.32, F1 within ±0.002, SR +0.03.
3. Taper, by grade against the candidate with rests (per-song means, which sit 0.01-0.02
   off the pooled shares of 10-07): SR bias Hard -0.05 → -0.01 (±0.05), Insane -0.27 →
   -0.12 (±0.07), Expert -0.62 → -0.38 (±0.15); weak onsets Hard → 0.16, Insane → 0.24,
   Expert → 0.28 (human about 0.16 / 0.24 / 0.29); density Insane 0.92 → 0.99, Expert 0.88
   → 0.95; steady bars Insane 0.17 → 0.21, Expert 0.11 → 0.15; off the human rhythm in
   other bars Insane 0.10 → 0.13 (the cost); in quiet bars +0.02 or less; F1 within ±0.004.
4. Holes on the tapered charts: steady bars +0.02, SR +0.02 more.

**Results (`run_taper.sh`, night of 10-08)**

| # | prediction | result | |
|---|---|---|---|
| 1 | no chart: rest AUC 0.90 (0.88 within songs; audio 0.84) | 0.917 (0.936 within; audio 0.841) | hit |
| 1 | long-note bars 0.85 (0.75 within songs) | 0.820 (0.699 within) | miss, under the 0.8 bar |
| 2 | holes 0.5: 3-6 per song | 11.3 | miss |
| 2 | steady bars 0.28 → 0.31, breaks per 100 0.14 → 0.10 | 0.281 → 0.308; 0.142 → 0.133 | hit, miss |
| 2 | density +0.5 to 1%, F1 +0.001, SR +0.02 | +0.77%, +0.0003, +0.011 | hit, miss, miss |
| 2 | holes 0.3: half again as many, steady 0.32, F1 ±0.002, SR +0.03 | 16.4 (1.45x), 0.317, +0.0002, +0.017 | hit, hit, hit, miss |
| 3 | taper, SR bias Hard -0.05 → -0.01, Insane -0.27 → -0.12 (±0.07), Expert -0.62 → -0.38 (±0.15) | -0.010, -0.145, -0.281 | hit, hit, hit |
| 3 | weak onsets Hard 0.16, Insane 0.24, Expert 0.28 (human 0.14 / 0.23 / 0.30) | 0.132 → 0.144, 0.179 → 0.219, 0.224 → 0.277 | miss, miss (both at the human share), hit |
| 3 | density Insane 0.92 → 0.99, Expert 0.88 → 0.95 | 0.969, 0.962 | miss, hit |
| 3 | steady bars Insane 0.17 → 0.21, Expert 0.11 → 0.15 | 0.159, 0.104 | miss, miss |
| 3 | off the human rhythm elsewhere Insane 0.10 → 0.13; quiet bars +0.02 or less; F1 ±0.004 | 0.125; Hard +0.008, Insane +0.026, Expert +0.019; -0.0017 | hit, miss (Insane), hit |
| 4 | holes on the tapered charts: steady +0.02, SR +0.02 | +0.024, +0.011 | hit, miss |

Per-song means on the 161 songs at SR 2.7+ (candidate with rests → tapered → tapered with holes):

| grade (n) | SR bias | density | weak onsets (human) | steady bars (human) | off rhythm, other bars | off rhythm, quiet bars |
|---|---|---|---|---|---|---|
| Hard (72) | -0.050 → -0.010 → -0.000 | 0.990 → 1.001 → 1.007 | 0.132 → 0.144 → 0.146 (0.143) | 0.275 → 0.263 → 0.288 (0.458) | 0.128 → 0.135 | 0.185 → 0.193 |
| Insane (66) | -0.269 → -0.145 → -0.133 | 0.918 → 0.969 → 0.975 | 0.179 → 0.219 → 0.221 (0.226) | 0.168 → 0.159 → 0.185 (0.323) | 0.102 → 0.125 | 0.239 → 0.265 |
| Expert (18) | -0.620 → -0.281 → -0.265 | 0.874 → 0.962 → 0.966 | 0.224 → 0.277 → 0.279 (0.295) | 0.107 → 0.104 → 0.118 (0.259) | 0.093 → 0.117 | 0.202 → 0.221 |
| Expert+ (5) | -1.100 → -0.937 → -0.932 | 0.849 → 0.908 → 0.913 | 0.213 → 0.261 → 0.263 (0.288) | 0.146 → 0.157 → 0.180 (0.307) | 0.090 → 0.106 | 0.289 → 0.277 |

What it says:

- The taper does what it was for: the hard grades get their notes back where humans put them
  (weak-onset starts at the human share in Hard, Insane and Expert) and most of the SR back.
  The notes come back on weak onsets, so off the human rhythm rises with them (+0.02); that is
  the human level of weak-onset starts, not noise. F1 is unchanged. Adopted (DECISIONS 10-09).
- The returned notes do not make streams: steady bars do not move with the taper. The holes do
  (+0.02 to +0.03 in every grade), at 11 notes a song and +0.01 SR; but breaks in long streams
  stay at 0.13 per 100 (human 0.05): a stream that misses two notes in a row, or changes
  spacing, is not a hole. Adopted at 0.5 (DECISIONS 10-09); steady bars are still 0.31 against
  0.41.
- Expert+ (5 songs) stays -0.9 SR under: the taper's floor 0.35 is not the problem there
  (density 0.91); those charts need more notes than the model starts at any gate.
- Without a chart in view the model still ranks the human rests well (AUC 0.92, 0.94 within
  songs), but its expected starts there are high: under 1 expected start it catches a fifth
  of them (0.68 next to a human chart). Long-note bars within a song only at 0.70. Planning
  bars before sampling is out; the rest pass after sampling stays (DECISIONS 10-09).

### 2026-10-08 night · The path of training on the loss surface (notebook 6.5 at 6.9M parameters; predictions)

The deep-learning class of 10-08 drew optimizer steps on the contours of a two-parameter
loss (UDL notebook 6.5: gradient descent too slow with a small step and zigzagging with a
large one, the normalized gradient bouncing at the end, Adam getting there). The same
picture for our training: `train.py --snapshot-every` saves the parameters along the run
(no random numbers drawn, so the run is the same run), and `fit_viz.py` draws the path on
its PCA plane (Li et al. 2018: the top two principal directions of the offsets θ_i − θ_T),
with the loss measured on a grid of that plane (val_ce's recipe on fixed train chunks).
What the plane leaves out is reported with it: the share of the path's variance, and at
every snapshot the loss there against the loss at its projection. Also the straight line
from θ_0 to θ_T, the distance from θ_0, and for AdamW Adam's step per coordinate against
the coordinate's gradient scale (√v̂, eqs. 6.13-6.17). `run_fit.sh` (patch 0040), started
after `run_taper.sh` ends:

1. Four 3,000-step runs from full-v4's start (seed 0, its data and inputs, weight decay 0,
   warmup 300 then cosine): AdamW lr 3e-4 (the recipe's), SGD lr 0.05 and 0.5 (no momentum;
   at 0.5 a step is about as long as Adam's, at 0.05 a tenth), and the normalized gradient
   (AdamW with betas 0 0: every coordinate moves lr in the sign of its gradient, the
   notebook's eq. 6.14) at lr 3e-4. Snapshots at 20, 40, 100 and every 200 steps;
   `fit_viz.py` on the four (grid 21 × 21, 32 chunks), each on its own plane.
2. fit-v4: the full-v4 recipe again (60,000 steps, 3.3 h) with snapshots at 100, 200, 500
   and every 1,000 steps; `fit_viz.py` on it (grid 25 × 25, 64 chunks), with full-v4's
   best.pt marked on the plane.

Predictions:

1. fit-v4's plane: PC1 0.55-0.80 of the path's variance, PC1 + PC2 ≥ 0.75. The path one
   long sweep along PC1 in the first 10k steps (where val_ce goes 0.23 → 0.10), then a slow
   bend into the end, the dots bunching as the cosine takes the learning rate to 0.
   ||θ_T − θ_0|| 20-80, the path 1.3-2.5 times as long.
2. The plane is a caricature in the middle: at the median snapshot the loss at the
   projection is 0.01-0.05 above the loss at the snapshot (the off-plane part of the path
   does real work); within 0.005 over the last 10k steps. The grid's minimum within 0.003
   of the end's loss, within a tenth of the span from the end.
3. The straight line from θ_0 to θ_T: the loss falls all the way (no barrier), minimum at
   α 0.95-1.1.
4. Adam a quarter in (step 15k): the gradient scale spans ≥ 3 decades over the sampled
   coordinates (5-95%), Adam's step ≤ 1.5 decades, its median 0.05-0.3 lr (well under 1:
   the gradient's sign changes from batch to batch and the momentum averages it out). The
   median step ≥ 0.4 lr in the first 500 steps, ≤ 0.15 lr in the second half.
5. fit-v4 against full-v4: val_ce at 60k within ±0.001 of 0.0740, but not the same numbers
   (sums run in parallel on MPS, so the runs drift apart); full-v4's end ≥ 0.2 ||θ_T − θ_0||
   from fit-v4's, nearly all of it off the plane.
6. The four at 3,000 steps (loss on the 32 chunks): Adam 0.18-0.24; the normalized gradient
   as fast over the first 200 steps, then stuck 0.05-0.15 above Adam (the notebook's
   bouncing: every coordinate keeps moving lr, until the cosine stops it); SGD 0.5
   0.30-0.45; SGD 0.05 above 0.40 (the class frequencies per cell and little more).
   Distance from the start: the normalized gradient ≥ 3 × Adam's, SGD 0.05 under a tenth of
   it. The normalized run's step per coordinate is 1 lr exactly (a check of the plot).

**Results (`run_fit.sh`, night of 10-08)** — figures: `outputs/fit-v4/fit_viz/`, `outputs/opt-compare/`

| # | prediction | result | |
|---|---|---|---|
| 1 | PC1 0.55-0.80, PC1 + PC2 ≥ 0.75 | 0.782 + 0.112 = 0.894 | hit |
| 1 | one sweep along PC1 in 10k steps, then a slow bend; dots bunching at the end | an arc: PC1 falls steadily all run (110 → 65 by 10k → 0), PC2 rises to 35 by 16k and comes back; bunching yes | half |
| 1 | ‖θ_T − θ_0‖ 20-80, path 1.3-2.5 times as long | 112, path 526 (4.7x) | miss, miss |
| 2 | gap (plane − snapshot) +0.01-0.05 at the median snapshot | median \|gap\| 0.003; +0.04 to +0.06 only from 1k to 9k steps | miss |
| 2 | within 0.005 over the last 10k; grid minimum within 0.003 of the end, within a tenth of the span | max 0.0013; 0.0899 = the end's, 1.9 from it (the path spans 110) | hit, hit |
| 3 | straight line monotone, minimum at α 0.95-1.1 | monotone, 1.00 | hit |
| 4 | at 15k: gradient scale ≥ 3 decades, step ≤ 1.5, median step 0.05-0.3 lr | 2.0, 1.8, 0.129 | miss, miss, hit |
| 4 | median step ≥ 0.4 lr in the first 500 steps, ≤ 0.15 in the second half | 0.12-0.15 from step 100 on (opt-adam: 0.86 at 20, 0.34 at 40, 0.13 at 100); 0.14 | miss, hit |
| 5 | fit-v4 val_ce at 60k within ±0.001 of full-v4's 0.0740, not the same numbers | 0.0737; equal to 5 decimals at 2k and 4k, apart from 6k (0.14007 vs 0.13983) | hit, hit |
| 5 | full-v4's end ≥ 0.2 ‖θ_T − θ_0‖ away, nearly all off the plane | 50.8 (0.45x), 47.9 off the plane | hit |
| 6 | at 3k: Adam 0.18-0.24; SGD 0.5 0.30-0.45; SGD 0.05 above 0.40 | 0.241, 0.306, 0.487 | at the edge, hit, hit |
| 6 | normalized gradient as fast for 200 steps, then 0.05-0.15 above Adam | 0.510 vs 0.497 at 200; 0.258 (+0.017) | hit, miss |
| 6 | distance: normalized ≥ 3x Adam's, SGD 0.05 under a tenth | 33.9 vs 28.6 (1.19x); 2.2 (0.08x) | miss, hit |
| 6 | normalized run's step exactly 1 lr | 1.000 at every snapshot | hit |

What the pictures say:

- **The path is low-dimensional, the parameters are not.** 89% of the 60k-step path lies in
  one plane of the 6.9M-dimensional space, and on that plane it is a smooth arc into a long
  valley, much like the notebook's figure. But the plane only describes the path: the start
  sits in the plane at loss 0.44 while the real start is at 1.75 (its random part is off the
  plane), and from 1k to 9k steps, the fast phase, the plane misses 0.04-0.06 of loss. After
  10k the plane and the real loss agree to 0.005.
- **The start-to-end line has a shelf.** On the straight line from θ_0 to θ_T the loss falls
  to 0.65 by a fifth of the way, stays there to 0.4, then falls to 0.09. 0.6 is the "class
  frequencies" level: where plain SGD at 0.05 stops (0.49 after 3k steps) and where every run
  is at step 100. No barrier, but a flat stretch the optimizer has to cross.
- **Adam's normalization, measured.** Median gradient scale differs 80x between parameter
  groups (cross-attention 7e-6, output head 6e-4), the median step is 0.12-0.14 lr in every
  group. And the step is 0.13 lr, not 1: the gradient's sign changes from batch to batch, so
  the momentum averages most of it away, from step 100 to the end, whatever the learning rate.
  2.5% of the sampled coordinates have a gradient scale under ε = 1e-8: all of them the key
  part of the attention biases (median 6e-12). Softmax ignores a constant added to every
  key's score, so their true gradient is 0; there Adam moves by g / ε, i.e. not at all, as
  it should.
- **The notebook's four, on our model.** From the same start for 3,000 steps: SGD 0.05 stalls
  at the class frequencies (0.49), SGD 0.5 gets to 0.31 with bumps (step 200 above step 100,
  1,800 above 1,600), the normalized gradient follows Adam to within 0.02 (0.26 vs 0.24),
  Adam 0.24. The normalized gradient moves each coordinate 7x as far per step (1 lr against
  0.13) yet ends only 1.2x as far from the start: most of its motion goes back and forth,
  the notebook's bouncing in 6.9M dimensions; Adam's momentum takes that out, and ends a
  little lower.
- **Seed noise floor.** The same seed on the Mac's GPU gives the same run for 4k steps, then
  two runs whose ends are 51 apart (0.45 of the whole path's span, nearly all off the plane)
  and equally good (val_ce 0.0737 vs 0.0740, 0.0899 vs 0.0897 on the 64 chunks). Differences
  between our full-v* runs under about 0.0005 val_ce are noise.

### 2026-10-10 · Rhythms carried over, long notes among taps, the long-note share from the audio (predictions)

The playtest of 10-07 said the charts have no bite ("치는 맛이 없다"): the same pattern does
not go on for long; the first bars sometimes feel human, then it feels like an AI; long notes
mixed into taps unlike a human; too fond of holding a long note while tapping; the tap-heavy
charts that feel human are bland. Measured on the 240 val songs (the candidate with holes and
rests, `${C}_fh0.5_rb1`, against the human charts; per-song means, new columns in
`structure.py`):

1. **A rhythm does not come back.** Of the bars with 4+ onset rows, the share with the same
   onset rows as the bar 1 / 2 / 4 / 8 / 16 before (`rhythm_rep*`): human 0.27 / 0.31 / 0.29
   / 0.23 / 0.18, the candidate 0.14 / 0.15 / 0.12 / 0.09 / 0.07: half, at every distance.
   When a human bar repeats the rhythm of the bar before, its lanes change 94% of the time
   (`lane_rep1` 0.06; the candidate 0.03). The gap is the same in every 8-bar window of the
   song (human 0.25-0.32, the candidate 0.09-0.14 from the 2nd window to the 7th), so "human
   at first, then AI" is not a drift: the first phrase cannot show a missing repeat, the next
   ones do. The bar copies of 10-04 (on in `generate.py`) repeat whole bars, lanes included
   (bar_lane_repeat 0.66 against 0.33); the rhythm with new lanes is what is missing.
2. **Long notes are scattered.** Of the long notes humans start, 48% are in bars where at least
   half the starts are long notes and 19% among taps (under a quarter); the candidate 34% and
   29%, and its long notes are shorter (0.75 against 1.00 beats). Over the whole chart it
   holds a long note under taps less than humans (2% of tap bars against 3%), so "too fond of
   holding" is about where, not how much: short long notes dropped into runs of taps.
3. **Not the cause:** chord accents (chords on strong beats 0.58 against 0.61, rank gap 0.16
   both), drift along the song (F1 and density flat by position).

Tonight (`run_carry.sh`, patch 0042), all on the candidate's saved charts with the holes (0.5)
and the rest pass (1) after, as in the reference, no new sampling:

1. `--carry-rhythm B` (`sampler.carry_rhythm`, tag `_rc`): after the lane passes, in time order,
   a bar of taps with 4+ onset rows may take the rhythm (onset rows and chord sizes) of the bar
   1, 2, 4 or 8 before, if that bar is taps too, has other onset rows and an onset count
   within 15%. With the bar MASK and the chart around it the model scores a rhythm row by row
   (log P(that many onsets), as `copy_bars`); the first source by audio similarity whose
   score plus B reaches the bar's own gives its rhythm, and the bar's lanes are chosen again
   with the chart in view (the lane passes' model, guidance 2, temperature 0.5). B = -2, 0, 2.
2. The bar copies alone (`--copy-bias 0`, the generate.py default) and after the carry at 0.
3. `--ln-tidy X` (`sampler.tidy_holds`, tag `_lt`): in bars with 4+ note starts of which under
   a quarter are long notes, a long note shorter than X beats becomes a tap (onsets stay).
   X = 1 and 0.5; and 1 after the carry at 0.
4. `style_from_audio.py`: below (no model; already run).

Smoke check (7 val songs, 5 of them tap charts, carry on the saved charts without the holes
and rests, full-v4 on CPU): bias -2 / 0 / 2 carried 10% / 17% / 23% of the bars; rhythm_rep1
0.21 → 0.27 / 0.33 / 0.38 (human 0.40), rep4 0.19 → 0.26 / 0.30 / 0.34 (0.31); notes -0.5%;
no grammar violations. ln-tidy 1: 13 long notes a song, among taps 0.41 → 0.11 (human 0.22).

Predictions (240 songs, against the reference `${C}_fh0.5_rb1` rescored):

1. Carry at 0: 9-15% of the bars carried (~12-18 a song; long-note charts have few bars of
   taps); rhythm_rep1/2/4/8/16 0.14 / 0.15 / 0.12 / 0.09 / 0.07 → 0.21-0.25 / 0.22-0.26 /
   0.18-0.22 / 0.12-0.15 / 0.07-0.08; lane_rep1 0.03 → 0.04-0.07 (human 0.06);
   bar_rhythm_repeat 0.12 → 0.20-0.26 (0.43); steady bars 0.31 → 0.32-0.34; rho_in +0.005 to
   +0.015; F1@50 within ±0.003; density -0.3% to -0.8%; SR -0.01 to -0.04; the long-note
   columns unchanged (bars with long notes are left alone).
2. Bias -2: half the carries of 0 or fewer, rhythm_rep1 0.18-0.21. Bias 2: 14-20% of bars,
   rhythm_rep1/2/4 0.25-0.30 / 0.26-0.31 / 0.21-0.26 (at the human level or a little under),
   F1 -0.002 to -0.008.
3. Copies alone: about 15% of bars (as on 10-04); rhythm_rep1..8 +0.02 to +0.05 each,
   lane_rep1 0.03 → 0.25-0.45 (whole bars: the lanes repeat with the rhythm); F1 ±0.002.
   Carry then copies: rhythm_rep1/2/4 0.24-0.29 / 0.25-0.30 / 0.21-0.26, lane_rep1 0.15-0.35.
4. Tidy 1: 8-15 long notes a song made taps; hold_share 0.172 → 0.155-0.165 (human 0.218,
   the cost); hold_beats 0.75 → 0.80-0.86 (1.00); ln_scattered 0.29 → 0.10-0.15 (0.19:
   overshoots) and ln_sections 0.34 → 0.38-0.42 (0.48); ln_f1 +0.00 to +0.02; F1 and the
   rhythm columns unchanged (onsets stay); SR 0 to -0.03. Tidy 0.5: a third to a half of
   those changes, ln_scattered 0.17-0.22.

**Already measured: the long-note share from the audio** (`style_from_audio.py`, 777 val
charts, 269 tap charts under 0.1, 167 long-note charts at 0.3 or more; train charts within 0.3
SR). For a new song `--stats sample` takes the stats of a random train chart of the SR:

| guess for the val chart | r (long-note share) | mae | tap chart given ≥ 0.3 | long-note chart given ≤ 0.1 | AUC long-note vs tap |
|---|---|---|---|---|---|
| a random train chart of the SR (`--stats sample`) | 0.02 | 0.180 | 0.22 | 0.36 | 0.51 |
| the mean of the train charts of the SR | 0.16 | 0.130 | 0.01 | 0.00 | 0.60 |
| the mean of the 10 / 30 songs with the nearest audio | 0.25 / 0.26 | 0.126 / 0.126 | 0.08 / 0.04 | 0.02 / 0.00 | 0.66 / 0.69 |
| a chart of one of the 10 nearest songs | 0.10 | 0.167 | 0.20 | 0.31 | 0.55 |
| the mean of the 30 nearest, the tempo as a feature too | 0.25 | 0.125 | 0.06 | 0.01 | 0.68 |

(Audio: the per-band mean and std of the log-Mel, 16 principal components. The same within
±0.03 AUC with a 0.5 SR window, 8 or 32 components, 5 or 50 neighbours. Jacks and trills
r ≤ 0.25 for every guess.) The audio of similar songs says as much about the long-note share
as another mapset's chart of the same song (r 0.25 on 10-05): it is the mapper's choice. A
random draw (as now) is a coin: a third of the long-note charts come out tap charts and a fifth
of the tap charts long-note charts; a draw among the nearest songs is barely better. The means
rank well (AUC 0.69) but are never a tap chart or a long-note chart: a medium share, the mix
of long notes and taps the playtest found least human. So for a new song the long-note share
is a choice to give the user (`--stats ln=0` / `ln=0.4`), or a tap version and a long-note
version side by side.

**Results (`run_carry.sh`, night of 10-10)** — all against `${C}_fh0.5_rb1` (rescored), 240 songs

| # | prediction | result | |
|---|---|---|---|
| 1 | carry 0: 9-15% of the bars (~12-18 a song) | 10.0% (12.3) | hit |
| 1 | rhythm_rep1/2/4/8/16 0.21-0.25 / 0.22-0.26 / 0.18-0.22 / 0.12-0.15 / 0.07-0.08 | 0.193 / 0.201 / 0.170 / 0.122 / 0.082 | miss ×3 (two thirds of the rise), hit, miss (over) |
| 1 | lane_rep1 0.04-0.07; bar_rhythm_repeat 0.20-0.26; steady bars 0.32-0.34 | 0.056; 0.207; 0.326 | hit, hit, hit |
| 1 | rho_in +0.005 to +0.015; F1 ±0.003; density -0.3 to -0.8%; SR -0.01 to -0.04; long-note columns unchanged | +0.008; +0.0000; -0.36%; -0.013; unchanged | hit ×5 |
| 2 | bias -2: half the carries of 0 or fewer; rhythm_rep1 0.18-0.21 | 0.51x (6.3 a song); 0.165 | hit, miss |
| 2 | bias 2: 14-20% of bars; rhythm_rep1/2/4 0.25-0.30 / 0.26-0.31 / 0.21-0.26; F1 -0.002 to -0.008 | 14.5%; 0.219 / 0.222 / 0.187; -0.0015 [-0.0026, -0.0004] | hit, miss ×3, miss (less) |
| 3 | copies alone: about 15% of bars; rhythm_rep1..8 +0.02 to +0.05 each; lane_rep1 0.25-0.45; F1 ±0.002 | 12.9%; +0.035 / +0.035 / +0.035 / +0.018; 0.286; +0.0001 | hit; hit ×3, miss; hit; hit |
| 3 | carry then copies: rhythm_rep1/2/4 0.24-0.29 / 0.25-0.30 / 0.21-0.26; lane_rep1 0.15-0.35 | 0.210 / 0.221 / 0.187; 0.277 | miss ×3, hit |
| 4 | tidy 1: 8-15 long notes a song; hold_share 0.155-0.165; hold_beats 0.80-0.86 | 43.4; 0.148; 0.822 | miss, miss, hit |
| 4 | ln_scattered 0.10-0.15; ln_sections 0.38-0.42; ln_f1 +0.00 to +0.02 | 0.111; 0.405; -0.022 [-0.027, -0.017] | hit, hit, miss (down) |
| 4 | F1 and the rhythm columns unchanged; SR 0 to -0.03 | unchanged; -0.008 | hit, hit |
| 4 | tidy 0.5: a third to a half of those changes; ln_scattered 0.17-0.22 | 0.26-0.45x (19.6 a song); 0.241 | hit, miss |

Per-song means (rhythm repeats 1 / 2 / 4 / 8 / 16 bars back; whole-bar copies by the distance
of the nearest earlier copy, `copy_near` 1-2 / `copy_mid` 3-8 / `copy_far` 9+, columns added in
patch 0043 and computed on the saved charts):

| | bars changed a song | rhythm_rep1 / 2 / 4 / 8 / 16 | lane_rep1 | copies near / mid / far | F1@50 | SR bias | chords |
|---|---|---|---|---|---|---|---|
| reference (holes, rests) | - | 0.138 / 0.146 / 0.122 / 0.092 / 0.070 | 0.029 | 0.011 / 0.007 / 0.005 | 0.3812 | -0.127 | 0.378 |
| carry -2 | 6.3 | 0.165 / 0.175 / 0.145 / 0.106 / 0.076 | 0.040 | | 0.3812 | -0.136 | 0.383 |
| carry 0 | 12.3 | 0.193 / 0.201 / 0.170 / 0.122 / 0.082 | 0.056 | 0.025 / 0.015 / 0.005 | 0.3812 | -0.140 | 0.386 |
| carry 2 | 17.9 | 0.219 / 0.222 / 0.187 / 0.139 / 0.089 | 0.071 | 0.032 / 0.020 / 0.007 | 0.3797 | -0.143 | 0.387 |
| copies 0 | 16.0 | 0.173 / 0.181 / 0.157 / 0.110 / 0.077 | 0.286 | 0.085 / 0.037 / 0.019 | 0.3813 | -0.136 | 0.383 |
| carry 0, copies 0 | 12.3 + 16.2 | 0.210 / 0.221 / 0.187 / 0.131 / 0.088 | 0.277 | 0.100 / 0.042 / 0.017 | 0.3817 | -0.145 | 0.389 |
| human | | 0.266 / 0.312 / 0.293 / 0.233 / 0.178 | 0.064 | 0.036 / 0.055 / 0.068 | | | 0.372 |

By the human chart's long-note share (tap charts under 0.1, 80 songs; mixed, 97; long-note
charts 0.3 or more, 63): bars of taps 71% / 37% / 13% of the bars; carried at 0, 19% / 7% /
2%; rhythm_rep1 0.147 / 0.161 / 0.089 → 0.254 / 0.201 / 0.100 at 0 and 0.301 / 0.226 / 0.102
at 2 (human 0.305 / 0.255 / 0.231). Every grade gains (+0.04 to +0.07 at 0, Insane and Expert+
the most). Over songs the carry at 2 stays inside the human spread (p10 / 50 / 90 0.03 / 0.18 /
0.50 against 0.04 / 0.20 / 0.56; the reference 0.00 / 0.10 / 0.34).

What it says:

- **The carry does what it was for, at no cost at bias 0** (F1, the long notes and the rests
  untouched; steady bars +0.02, rho_in +0.008: the rhythm now repeats where the audio does).
  At bias 2 the tap charts repeat their rhythms as often as human ones, at -0.0015 F1. The
  lanes of a carried bar change as in human charts (lane_rep1 0.06-0.07, human 0.06). It
  barely reaches the long-note charts: only bars of taps take part, 13% of their bars.
  Adopted at 2 for playable charts (DECISIONS 10-11).
- **The carried rhythms are a little sparser.** Against the bars' own rhythms they have 9%
  fewer onset rows and 8% more chord rows (notes -3%): the masked-bar score likes the rhythm
  without the uncertain notes, as the copies did before the density guard (10-03). Hence
  chords +0.008 to +0.010 (0.378 → 0.387; human 0.372), density -0.4%, SR -0.015. Small next to
  the gain; a guard on the onset rows would take it out.
- **The copies copy the wrong bars.** Humans copy whole bars mostly from far back (a section
  that comes back: copy_far 0.068) and seldom the bar just before (0.036), whose lanes change
  94% of the time. copy_bars takes the bar with the most similar audio, usually the adjacent
  one, which also fits the context best: near copies at 2.4x the human rate, far ones at a
  quarter, lane_rep1 4.5x. With the carry at 2 the near ones are at the human rate without
  copies. Next: copies only from further back (`--copy-min-lag`, 2026-10-11).
- **Tidying takes out long notes humans have too.** The removed ones sat on a human long note
  12% of the time (the kept ones 16%) and on a human tap 26% (20%): worse than the rest, but
  not wrong enough to raise ln_f1. Human long notes among taps are short as well (62% under a
  beat, 25% under half; the AI's 82% / 37%); the AI has 1.4x as many of them (12,849 against
  9,316). Removing every short one overshoots (0.11 against 0.19) and takes the share further
  under the human (0.148 against 0.218); under half a beat is the measured middle (0.24, share
  -0.008, ln_f1 -0.004). Not a default yet; tried with the carry and later copies on 10-11.
- `style_from_audio.py` on the Mac gave the numbers above to the digit (seeded); its BLAS
  warnings were Accelerate's (as fit_viz's on 10-08) and are silenced in patch 0043.

### 2026-10-11 · Copies from further back; the candidate on a tapered base (predictions)

From 10-10: with the carry at 2 the bar just before is copied (lanes too) at the human rate
without any copies, while the copies triple it and leave the far copies, a section that comes
back, at a quarter of the human rate. `copy_bars(min_lag=N)` (`--copy-min-lag N`, tag
`_cp<bias>l<N>`; playtest `cp0l4`) takes its sources only from N or more bars back; the rest
of the rule is unchanged (top 3 by audio similarity, 0.5 or more, the onset count within 15%,
the first whose rhythm scores within the bias). With the static checks alone (no model), the
bars that have a source fall from 26% of the bars (1 back or more) to 21% (4) and 16% (8), and
the first source is 9+ bars back for 30% / 47% / 75% of them; at 1 the model took about half,
mostly the near ones. New columns (`phrase_repeats`, also rescored into the 10-10 runs):
`copy_near` / `copy_mid` / `copy_far`, the busy bars whose nearest earlier copy (lanes as they
are or mirrored) is 1-2 / 3-8 / 9+ bars back (human 0.036 / 0.055 / 0.068).

`run_copies.sh` (patch 0043):

1. On the candidate's saved charts as on 10-10 (`$C`, holes 0.5 and rests 1 after), against
   the carry at 2 alone: the carry at 2 with copies 0 from 1 back (the generate.py default
   now), from 4 back, from 8 back; the carry at 0 with copies from 4 back; the carry at 2,
   copies from 4 back and long notes among taps under half a beat made taps.
2. A new base, the candidate's sampling with the taper (`--onset-taper 2.7,5,0.35`, no passes
   after the lane passes) on all 240 val songs: `$T`. Under SR 2.7 the taper is off and the
   seeds are the same, so those 79 charts should come out as in `$C`.
3. On `$T`: holes and rests (the reference), and the candidate of item 1's last line; by grade.

Predictions:

1. Carry 2, copies from 1 back: 15-17 copies a song; rhythm_rep1 0.225-0.24; lane_rep1
   0.25-0.30; copy_near / mid / far 0.09-0.11 / 0.04-0.05 / 0.015-0.02; F1 within ±0.002 of
   the carry alone (0.3797).
2. Carry 2, copies from 4 back: 7-11 copies a song; copy_near 0.026-0.036 (human 0.036),
   copy_mid 0.045-0.07 (0.055), copy_far 0.025-0.045 (0.068); lane_rep1 0.06-0.09 (0.064);
   bar_lane_repeat 0.30-0.45 (0.33); rhythm_rep1 / 2 within +0.01 of the carry alone (0.219 /
   0.222), rhythm_rep4 / 8 / 16 +0.01 to +0.03 each; rho_in 0.223-0.235 (0.228); F1 within
   ±0.002; breaks per 100 and run length within ±0.02 / ±0.3 of the carry alone (the copies
   from 1 back moved them by -0.04 / +0.5); SR within ±0.01.
3. From 8 back: 5-8 copies a song; copy_mid 0.025-0.035, copy_far 0.03-0.05; copy_near as in 2;
   rhythm_rep8 and rep16 +0.01 to +0.02.
4. Carry 0, copies from 4 back: copy_near 0.022-0.03, copy_mid and far as in 2; rhythm_rep1
   0.195-0.21; F1 within ±0.002 of the reference (0.3812).
5. Carry 2, copies from 4 back, tidy 0.5: the copy and rhythm columns as in 2; 18-21 long
   notes a song made taps; hold_share -0.007 to -0.009, ln_scattered -0.04 to -0.05,
   ln_sections +0.015 to +0.02, ln_f1 -0.002 to -0.006 (as tidy 0.5 alone).
6. `$T` against `$C` (no passes after the lane passes, all 240 songs): the 79 songs under SR
   2.7 identical (every column); over all songs SR bias +0.05 to +0.10, density +2 to +4%, F1
   within ±0.003, weak onsets +0.01 to +0.02 (the taper, as on 10-08).
7. The candidate on `$T` against holes and rests on `$T`, by grade: rhythm_rep1 +0.06 to +0.10
   in every grade from Normal up (Easy +0.04 to +0.08); copy_far +0.02 to +0.04; SR -0.01 to
   -0.03 in every grade; F1 -0.001 to -0.004; ln_scattered -0.03 to -0.06.

**Results (`run_copies.sh`, night of 10-10/11)** — items 1-5 against the carry at 2 alone
(`${C}_rc2_fh0.5_rb1`; 0.3797 F1, rhythm_rep1 0.219, copies near / mid / far 0.032 / 0.020 /
0.007), 6 against `$C`, 7 by grade; 240 songs

| # | prediction | result | |
|---|---|---|---|
| 1 | copies from 1 back: 15-17 a song; rhythm_rep1 0.225-0.24; lane_rep1 0.25-0.30 | 17.4; 0.236; 0.300 | miss (just over), hit, hit |
| 1 | copy near / mid / far 0.09-0.11 / 0.04-0.05 / 0.015-0.02; F1 ±0.002 | 0.114 / 0.048 / 0.017; 0.3802 | miss (just over), hit, hit; hit |
| 2 | from 4 back: 7-11 a song; copy_near 0.026-0.036; copy_mid 0.045-0.07; copy_far 0.025-0.045 | 11.8; 0.044; 0.069; 0.020 | miss (over), miss (over), hit, miss (under) |
| 2 | lane_rep1 0.06-0.09; bar_lane_repeat 0.30-0.45; rhythm_rep1 / 2 within +0.01 | 0.102; 0.438; +0.005 / +0.006 | miss (over), hit, hit |
| 2 | rhythm_rep4 / 8 / 16 +0.01 to +0.03; rho_in 0.223-0.235 | +0.014 / +0.010 / +0.007; 0.214 | hit, hit, miss; miss (down) |
| 2 | F1 ±0.002; breaks and run length ±0.02 / ±0.3; SR ±0.01 | 0.0000; +0.007 / -0.02; -0.004 | hit, hit, hit |
| 3 | from 8 back: 5-8 a song; copy_mid 0.025-0.035; copy_far 0.03-0.05; copy_near as in 2 | 7.4; 0.039; 0.024; 0.037 | hit, miss, miss, miss (at the human 0.036) |
| 3 | rhythm_rep8 and rep16 +0.01 to +0.02 | +0.008 / +0.006 | miss, miss |
| 4 | carry 0, copies from 4 back: copy_near 0.022-0.03; mid, far as in 2; rhythm_rep1 0.195-0.21; F1 ±0.002 of 0.3812 | 0.036; 0.060, 0.019; 0.198; 0.3813 | miss, hit, miss; hit; hit |
| 5 | with tidy 0.5: the copy and rhythm columns as in 2; 18-21 a song; share -0.007 to -0.009; ln_scattered -0.04 to -0.05; ln_sections +0.015 to +0.02; ln_f1 -0.002 to -0.006 | the same; 19.3; -0.008; -0.045; +0.018; -0.0035 | hit ×6 |
| 6 | `$T` against `$C`: the 79 songs under SR 2.7 identical; SR bias +0.05 to +0.10; density +2 to +4%; F1 ±0.003 | identical; +0.075; +2.5%; -0.0012 | hit ×4 |
| 6 | weak onsets +0.01 to +0.02 | +0.019 (`$T` with holes and rests against `${C}_fh0.5_rb1`; `$C` has no column) | hit |
| 7 | rhythm_rep1 +0.06 to +0.10 from Normal up (Easy +0.04 to +0.08) | Normal +0.076, Hard +0.075, Insane +0.101, Expert +0.096, Expert+ +0.106; Easy +0.093 | hit ×3, miss ×3 (a little over) |
| 7 | copy_far +0.02 to +0.04 | +0.016 (by grade +0.004 to +0.026) | miss |
| 7 | SR -0.01 to -0.03 in every grade | Easy -0.002, Normal -0.010, Hard -0.013, Insane -0.040, Expert -0.064, Expert+ -0.059 | hit ×2, miss ×4 |
| 7 | F1 -0.001 to -0.004; ln_scattered -0.03 to -0.06 | -0.0014; -0.048 | hit, hit |

Copies by distance, all after the carry at 2 (per-song means; human 0.036 / 0.055 / 0.068):

| copies | a song | near / mid / far | lane_rep1 (0.064) | bar_lane_repeat (0.334) | run length (7.49) | rho_in (0.228) | F1 |
|---|---|---|---|---|---|---|---|
| none | - | 0.032 / 0.020 / 0.007 | 0.071 | 0.189 | 7.68 | 0.2225 | 0.3797 |
| from 1 back | 17.4 | 0.114 / 0.048 / 0.017 | 0.300 | 0.566 | 8.50 | 0.2144 | 0.3802 |
| from 4 back | 11.8 | 0.044 / 0.069 / 0.020 | 0.102 | 0.438 | 7.67 | 0.2136 | 0.3797 |
| from 8 back | 7.4 | 0.037 / 0.039 / 0.024 | 0.085 | 0.342 | 7.50 | 0.2177 | 0.3802 |

What it says:

- **Copies from 8 back are the human mix.** From 4 back the copies still repeat the bar just
  before (0.044): in a passage of one repeated pattern the bar 4 back equals the bar 1 back.
  From 8 back the near copies, the whole-bar repeats (bar_lane_repeat) and the run length are
  at the human level; the far copies triple but stay at a third of the human rate (0.024
  against 0.068): at bias 0 the model takes few far sources, so a chorus does not come back as
  a whole. Every copy setting costs some rho_in against the carry alone (a whole bar from
  elsewhere breaks the local fit the carry gave); from 8 back the least. Adopted, from 8 back
  (DECISIONS 10-11).
- **The carry at 2 was too much, where it acts.** The new columns `tap_rep1` / `ln_rep1` split
  rhythm_rep1 by the kind of bar (patch 0044, computed on the saved charts): human 0.273 /
  0.235, the reference 0.188 / 0.087, the carry at 0 0.289 / 0.089, at 2 0.337 / 0.090, the
  candidate on `$T` 0.324 / 0.090. The carry only touches bars of taps, and there the carry at
  0 is at the human rate (per song in tap charts 0.319 against 0.314, at 2 0.381); the mean
  over all bars looked short of the human only because the bars with long notes repeat their
  rhythm at a third of the human rate and the carry never touches them. The carry's repeats
  also grow with the song's length (correlation with the bar count +0.25 to +0.30; human
  +0.03). Back to bias 0 (DECISIONS 10-11, replacing the row on 2); the long-note bars next.
- **The candidate costs SR in the hard grades.** On `$T` it takes Insane from -0.133 to -0.173
  and Expert/Expert+ from -0.41 to -0.47: a third of the taper's gain in Insane, a fifth in
  Expert. On `$C` the carry at 2
  alone takes -0.027 (Insane) and -0.029 (Expert+) at -0.44% notes, the copies -0.008 to
  -0.010, tidy 0.5 -0.003 to -0.007: mostly the carry's sparser rhythms in dense bars.
- **The taper base is sound:** the songs under SR 2.7 came out note for note as in `$C`, and
  the taper moved SR as on 10-08 (+0.075 over all songs). `$T` is the base for what follows.

### 2026-10-11 · Long-note bars in the rhythm carry (predictions)

The bars that start long notes repeat the rhythm of the bar before at a third of the human
rate (`ln_rep1` 0.09 against 0.24), whatever the carry does. In the human charts they are
more of the busy bars than the bars of taps (12,639 against 10,472 that follow a busy bar),
and 71% of them have a long note across an edge. Humans change the lanes of a repeated
long-note rhythm 96% of the time (pooled: rhythm 24%, lanes too 4% of those), much as for
taps (31%, 8%).

`carry_rhythm(holds=B)` (`--carry-holds B`, tag `h<B>`; playtest `rcAhB`): a bar that starts
long notes may take the rhythm of a bar 1, 2, 4 or 8 before that starts long notes too, at a
bias B of its own, scored as before (the bar's own long notes taken out of the context, so
their tails past the bar do not show). The source's notes, long notes with their lengths,
replace those that started in the bar; long notes coming in from the bar before stay, and so
does everything in the next bar. Their lanes move as a whole: of the 24 lane permutations under
which every note fits (its cells empty, the release gaps kept), the one the model likes (sum
log p over the bar's cells), never the source's own lanes or their mirror
(`sampler._place_pattern`). `carry_rhythm(keep_notes=True)` (`--carry-keep`, tag `k`) leaves
out sources with fewer notes than the bar: the carried rhythms were sparser (10-10), and that
is what costs SR in dense charts. New columns `tap_rep1` / `ln_rep1`, and in `--by-grade`.

Smoke check (6 val songs with long notes, their saved charts, the carry alone, full-v4 on CPU;
human ln_rep1 0.25, the charts 0.05): the long-note bars at 0 took 24 sources a song and
repeated the bar before 0.34 of the time (over the human), at -2 (taps at -2 too) 12.5 and
0.24, at 0 with keep_notes 13 and 0.17. Before the source's own lanes and their mirror were
ruled out, the permutation the model liked was often the source's (lane_rep1 0.16, near
copies 10x the human rate); after, lane_rep1 0.02-0.04 (human 0.06). Notes -1.4% / -0.9% /
+1.0%; no grammar violations. Hence a bias of its own for the long-note bars.

`run_holds.sh` (patch 0044), on `$T` with copies from 8 back, tidy 0.5, holes 0.5 and rests 1
(the defaults now), against holes and rests on `$T`, by grade:

1. The carry at 0, bars of taps only (the playable default now).
2. The same, keep_notes.
3. Bars with long notes carried too, at -2, -1 and 0 (`--carry-holds`; taps at 0).
4. At -1 with keep_notes.

Predictions (`$T` with holes and rests: rhythm_rep1 0.135, tap_rep1 0.183, ln_rep1 0.087,
lane_rep1 0.031, copies 0.011 / 0.006 / 0.005, F1 0.3800, SR bias -0.052, hold_share 0.169):

1. Carry at 0: tap_rep1 0.27-0.30 (human 0.273), ln_rep1 0.088-0.095, rhythm_rep1 0.19-0.21;
   copies near / mid / far 0.025-0.035 / 0.02-0.035 / 0.015-0.025; lane_rep1 0.05-0.08; F1
   0 to -0.002; SR -0.01 to -0.02 over all, Insane -0.015 to -0.035, Expert/Expert+ -0.02 to
   -0.045 (about half the cost of last night's candidate); ln_scattered -0.04 to -0.05,
   hold_share -0.007 to -0.009 (the tidy).
2. keep_notes: 30-50% fewer bars carried; tap_rep1 0.24-0.28; density at least that of 1
   + 0.2%; SR over all -0.005 to -0.015, Insane -0.01 to -0.025.
3. Long-note bars at -2: 4-10 of them carried a song; ln_rep1 0.15-0.20 (human 0.235);
   lane_rep1 at most 0.08; copy_near +0.005 to +0.015 over 1; hold_share within ±0.005 and
   hold_beats within ±0.03 of 1; F1 -0.001 to -0.004, ln_f1 -0.003 to -0.015, SR -0.005 to
   -0.02, all against 1. At -1: ln_rep1 0.18-0.25; at 0: 0.22-0.30 (over the human
   possible).
4. At -1 with keep_notes: ln_rep1 0.12-0.18; density at least that of 1.

**Results (`run_holds.sh`, afternoon of 10-11)** — against holes and rests on `$T`
(paired means as `compare_runs` prints them), item 3 also against 1; 240 songs

| # | prediction | result | |
|---|---|---|---|
| 1 | carry 0: tap_rep1 0.27-0.30; ln_rep1 0.088-0.095; rhythm_rep1 0.19-0.21 | 0.290; 0.0885; 0.199 | hit ×3 |
| 1 | copies near / mid / far 0.025-0.035 / 0.02-0.035 / 0.015-0.025; lane_rep1 0.05-0.08 | 0.031 / 0.031 / 0.023; 0.078 | hit ×4 |
| 1 | F1 0 to -0.002; SR -0.01 to -0.02 over all | +0.0005; -0.023 | miss (a little up), miss (a little over) |
| 1 | SR Insane -0.015 to -0.035, Expert / Expert+ -0.02 to -0.045 | -0.035; -0.057 / -0.079 | hit (edge), miss ×2 |
| 1 | ln_scattered -0.04 to -0.05; hold_share -0.007 to -0.009 | -0.049; -0.0086 | hit, hit |
| 2 | keep_notes: 30-50% fewer bars carried; tap_rep1 0.24-0.28 | 7.6 against 13.5 a song (-44%); 0.242 | hit, hit |
| 2 | density at least 1's + 0.2%; SR over all -0.005 to -0.015, Insane -0.01 to -0.025 | +0.6% over 1; -0.002, +0.001 | hit; miss, miss (no cost at all) |
| 3 | at -2: 4-10 long-note bars carried a song; ln_rep1 0.15-0.20; lane_rep1 at most 0.08 | 13.4; 0.181; 0.073 | miss (over), hit, hit |
| 3 | copy_near +0.005 to +0.015 over 1; hold_share ±0.005 and hold_beats ±0.03 of 1 | +0.006; +0.0003, +0.003 | hit ×3 |
| 3 | against 1: F1 -0.001 to -0.004; ln_f1 -0.003 to -0.015; SR -0.005 to -0.02 | +0.0013; -0.0011; -0.020 | miss (up), miss (smaller), hit (edge) |
| 3 | ln_rep1 at -1 0.18-0.25; at 0 0.22-0.30 | 0.206; 0.241 | hit, hit |
| 4 | at -1 with keep_notes: ln_rep1 0.12-0.18; density at least 1's | 0.149; 1.018 against 1.009 | hit, hit |

All runs (human: tap_rep1 0.273, ln_rep1 0.235, lane_rep1 0.064, chords 0.372):

| run | carried a song (long-note bars) | tap_rep1 | ln_rep1 | lane_rep1 | chords | SR bias (Insane / Expert) | density | F1 |
|---|---|---|---|---|---|---|---|---|
| holes and rests on `$T` | - | 0.183 | 0.087 | 0.031 | 0.366 | -0.052 (-0.133 / -0.265) | 1.014 | 0.3800 |
| carry 0 | 13.5 | 0.290 | 0.089 | 0.078 | 0.377 | -0.075 (-0.168 / -0.323) | 1.009 | 0.3805 |
| carry 0, keep | 7.6 | 0.242 | 0.088 | 0.067 | 0.374 | -0.054 (-0.132 / -0.278) | 1.015 | 0.3813 |
| long notes at -2 | 27.0 (13.4) | 0.305 | 0.181 | 0.073 | 0.388 | -0.095 (-0.181 / -0.369) | 1.005 | 0.3818 |
| at -1 | 30.2 (16.7) | 0.305 | 0.206 | 0.073 | 0.389 | -0.098 (-0.187 / -0.369) | 1.004 | 0.3818 |
| at 0 | 34.2 (20.7) | 0.307 | 0.241 | 0.078 | 0.391 | -0.102 (-0.201 / -0.379) | 1.003 | 0.3815 |
| keep, at -1 | 16.2 (8.6) | 0.252 | 0.149 | 0.053 | 0.381 | -0.051 (-0.124 / -0.283) | 1.018 | 0.3824 |

What it says:

- **keep_notes takes the carry's SR cost away.** The carry at 0 cost 0.023 SR over all, 0.035
  in Insane and 0.057 in Expert; with keep_notes 0.002, none and 0.013, the density goes up
  instead of down, and F1 rises (+0.0013). It carries 44% fewer bars (37-48% in every grade),
  and tap_rep1 0.242 closes two thirds of the gap to the human instead of overshooting it.
  By grade it lands on the human in Easy (0.364 against 0.364) and Hard (0.25 against 0.24)
  and stays under in Insane (0.19 against 0.29) and Expert (0.12 against 0.23), as the carry
  without keep_notes did (0.25, 0.17): the dense grades want more than bias 0 gives.
- **The long-note bars take sources readily, and without keep_notes they cost SR.** At -2 /
  -1 / 0, 13.4 / 16.7 / 20.7 of them a song, the bar before repeated 0.181 / 0.206 / 0.241 of
  the time (human 0.235), F1 up (+0.0013 over the carry alone) and ln_f1 hardly moved
  (-0.001 to -0.003); but the SR fell 0.020-0.026 more than with the carry alone (Hard -0.05,
  Expert -0.10 against the reference), the density 0.4-0.6%, and the chords rose to
  0.388-0.391 (in long-note songs 0.42-0.43 against the human 0.38). tap_rep1 rose 0.015
  too, likely carried long-note bars that the tidy then made bars of taps.
- **With keep_notes at -1 the SR stays and F1 is the best of the six runs.** 8.6 long-note bars a
  song, ln_rep1 0.149, tap_rep1 0.252, lane_rep1 0.053 (human 0.064), SR +0.0005 over the
  reference (Expert -0.017), density +0.4%, F1 +0.0023, steady bars +0.038, rho_in +0.007.
  The one cost is the chords (0.381 against 0.366; human 0.372). Adopted with keep_notes as
  the default (DECISIONS 10-11).
- **The chords come from fewer onset rows.** Of the bars the carry changed under keep_notes
  (copies left out), 45% of the bars of taps and 53% of the long-note bars got fewer onset
  rows than they had, with +0.6 / +0.8 chord rows and +0.6 notes a bar: in the carried bars of
  taps the chord rows went from 0.32 to 0.38 of the onset rows. The score sums log P(onset
  count) row by row, and every onset row costs where the model is unsure where the notes
  go: it likes the notes in fewer, fuller rows, as it liked fewer notes before keep_notes.
  Next: no source with fewer onset rows either, and higher biases now that the sources are
  fewer.

### 2026-10-11 night · The carry's biases under keep_notes; sources with no fewer onset rows (predictions)

From the afternoon: keep_notes takes the carry's SR cost away, so the biases can go up where
the repeats are still short of the human: the bars of taps at 0.252 (human 0.273; Insane and
Expert well under), the long-note bars at 0.149 (0.235). The chords are the one cost (0.381
against 0.372), from sources with fewer onset rows. `carry_rhythm(keep_rows=True)`
(`--carry-rows`, tag `r` after `k`; playtest `rc0krh-1`) leaves those out as well.

`run_keep.sh` (patch 0045), on `$T` with the default passes (copies from 8 back, tidy 0.5,
holes 0.5, rests 1) and keep_notes, against holes and rests on `$T` and against the default
(`rc0kh-1`), by grade:

1. Bars of taps at 1 and at 2 (long-note bars at -1).
2. Long-note bars at 1 and at 3 (taps at 0).
3. Both raised: taps at 2, long-note bars at 1.
4. keep_rows: at the default, and with both raised as in 3.

Predictions (the default: 16.2 bars carried a song, 8.6 of them long-note bars; tap_rep1
0.252, ln_rep1 0.149, lane_rep1 0.053, chords 0.381, SR bias -0.051, density 1.018, F1 0.3824;
paired means against the reference):

1. Taps at 1: 8.5-10 bars of taps carried a song (7.6), tap_rep1 0.26-0.275; at 2: 9.5-12,
   tap_rep1 0.27-0.29, by grade Easy 0.38-0.41 (human 0.364), Insane 0.22-0.25 (0.286),
   Expert 0.18-0.22 (0.231); SR within ±0.01 of the default (Expert ±0.03), chords +0.002
   to +0.006, F1 within ±0.001, ln_rep1 within ±0.005.
2. Long-note bars at 1: 11-14 of them carried a song, ln_rep1 0.17-0.20; at 3: 13-18,
   ln_rep1 0.20-0.24, chords +0.004 to +0.012, SR 0 to -0.015, ln_f1 0 to -0.004,
   hold_share within ±0.003 of the default; tap_rep1 within ±0.015.
3. Taps at 2 and long-note bars at 1: tap_rep1 as at 2 in 1 and ln_rep1 as at 1 in 2 (each
   within ±0.01: they act on different bars); chords 0.385-0.395; SR within ±0.015 of the
   default.
4. keep_rows at the default: 9.5-12.5 bars carried a song (-25 to -40%), tap_rep1
   0.225-0.245, ln_rep1 0.115-0.14, chords 0.368-0.376; SR within ±0.01, density within
   ±0.3% and F1 within ±0.001 of the default. With both raised, against 3: chords -0.008 to
   -0.015, tap_rep1 and ln_rep1 -0.01 to -0.03 each.

**Results (`run_keep.sh`, evening of 10-11)** — paired means against holes and rests on `$T`
where the prediction is a level, against the default (`rc0kh-1`) where it is a change; bars
carried and the by-grade means are per-song means; 240 songs

| # | prediction | result | |
|---|---|---|---|
| 1 | taps at 1: 8.5-10 bars of taps carried a song; tap_rep1 0.26-0.275 | 10.2; 0.266 | miss (just over), hit |
| 1 | taps at 2: 9.5-12; tap_rep1 0.27-0.29 | 12.2; 0.274 | miss (just over), hit |
| 1 | at 2 by grade: Easy 0.38-0.41, Insane 0.22-0.25, Expert 0.18-0.22 | 0.383, 0.221, 0.180 | hit ×3 (the last two at the edge) |
| 1 | SR within ±0.01 of the default (Expert ±0.03) | +0.004 / +0.007 (Expert +0.002 / +0.015) | hit ×4 |
| 1 | chords +0.002 to +0.006; F1 ±0.001; ln_rep1 ±0.005 | +0.0016 / +0.0025; -0.0002 / -0.0002; +0.002 / +0.002 | miss (just under), hit; hit ×4 |
| 2 | long-note bars at 1: 11-14 of them carried a song; ln_rep1 0.17-0.20 | 14.8; 0.195 | miss (just over), hit |
| 2 | at 3: 13-18; ln_rep1 0.20-0.24 | 19.4; 0.218 | miss (over), hit |
| 2 | at 3 against the default: chords +0.004 to +0.012; SR 0 to -0.015 | +0.0033; +0.011 | miss (just under), miss (up, not down) |
| 2 | ln_f1 0 to -0.004; hold_share ±0.003; tap_rep1 ±0.015 | -0.0021; -0.0004; +0.0065 | hit ×3 |
| 3 | taps at 2, long-note bars at 1: tap_rep1 as taps at 2 ±0.01, ln_rep1 as long-note bars at 1 ±0.01 | 0.279 (0.274); 0.196 (0.195) | hit, hit |
| 3 | chords 0.385-0.395; SR within ±0.015 of the default | 0.387; +0.011 | hit, hit |
| 4 | keep_rows at the default: 9.5-12.5 carried a song; tap_rep1 0.225-0.245; ln_rep1 0.115-0.14 | 8.9 (-45%); 0.2249; 0.120 | miss (fewer), miss (at the edge), hit |
| 4 | chords 0.368-0.376; SR ±0.01; density ±0.3%; F1 ±0.001 | 0.368 (0.3677); -0.004; -0.17%; -0.0018 | miss (at the edge), hit, hit, miss |
| 4 | with both raised, against 3: chords -0.008 to -0.015; tap_rep1, ln_rep1 -0.01 to -0.03 | -0.023; -0.036, -0.043 | miss ×3 (more) |

All runs (human: tap_rep1 0.273, ln_rep1 0.235, lane_rep1 0.064, chords 0.372;
section_rhythm, rescored: 0.547; SR bias per song with Insane / Expert):

| run | carried a song (long-note bars) | tap_rep1 | ln_rep1 | lane_rep1 | chords | SR bias (Insane / Expert) | density | F1 | section_rhythm |
|---|---|---|---|---|---|---|---|---|---|
| holes and rests on `$T` | - | 0.183 | 0.087 | 0.031 | 0.366 | -0.052 (-0.133 / -0.265) | 1.014 | 0.3800 | 0.122 |
| default (taps 0, long-note bars -1) | 16.2 (8.6) | 0.252 | 0.149 | 0.053 | 0.381 | -0.051 (-0.124 / -0.283) | 1.018 | 0.3824 | 0.189 |
| taps at 1 | 18.9 (8.6) | 0.266 | 0.150 | 0.063 | 0.383 | -0.047 (-0.117 / -0.281) | 1.020 | 0.3821 | 0.195 |
| taps at 2 | 20.9 (8.6) | 0.274 | 0.151 | 0.066 | 0.384 | -0.044 (-0.115 / -0.268) | 1.021 | 0.3822 | 0.197 |
| long-note bars at 1 | 22.4 (14.8) | 0.256 | 0.195 | 0.059 | 0.384 | -0.046 (-0.125 / -0.259) | 1.021 | 0.3820 | 0.195 |
| long-note bars at 3 | 27.0 (19.4) | 0.257 | 0.218 | 0.060 | 0.385 | -0.041 (-0.123 / -0.242) | 1.023 | 0.3813 | 0.199 |
| taps at 2, long-note bars at 1 | 27.1 (14.8) | 0.279 | 0.196 | 0.068 | 0.387 | -0.041 (-0.117 / -0.243) | 1.024 | 0.3819 | 0.204 |
| keep_rows | 8.9 (4.3) | 0.225 | 0.120 | 0.054 | 0.368 | -0.056 (-0.136 / -0.284) | 1.016 | 0.3806 | 0.177 |
| keep_rows, taps at 2, long-note bars at 1 | 17.5 (9.0) | 0.243 | 0.153 | 0.063 | 0.364 | -0.047 (-0.126 / -0.254) | 1.021 | 0.3793 | 0.186 |

What it says:

- **Raising the bias for bars of taps reaches the human tap_rep1 over all songs, but in the
  wrong songs.** At 1 / 2 tap_rep1 is 0.266 / 0.274 (human 0.273), F1 unchanged, SR +0.004
  / +0.007 against the default. By grade the easy charts go over the human (at 2: Easy
  0.383, Normal 0.265, Hard 0.270 against 0.364 / 0.234 / 0.243) while Insane and Expert
  stay under per song (0.221 / 0.180 against 0.286 / 0.231). Pooled over bar pairs,
  Insane and up are near the human already (at 2: 0.278 against 0.289; the default
  0.248): the carry's repeats gather in long songs. Against the audio
  (`outputs/feel/repeat_vs_audio.py`), the default repeats the bar before as often as
  humans where the two bars sound unalike (cosine of their log-Mel under 0.2: 0.145 against
  0.145) and a little less where they sound alike (0.8 and over: 0.45 against 0.51); at 2
  the unlike bars go over (0.173). Players praise repetition where the music repeats and
  blame it elsewhere (EXPERIMENTS 2026-10-12): bias 0 stays.
- **The long-note bars take more sources than predicted, and the SR goes up.** At 1 / 3,
  14.8 / 19.4 of them a song, ln_rep1 0.195 / 0.218 (human 0.235), SR +0.005 / +0.011
  against the default (keep_notes keeps the sources at least as full), F1 -0.0004 / -0.0010,
  chords +0.003. At 3 Easy goes over (0.327 against 0.279). Against the audio, at 1 the
  long-note bars repeat the bar before more at every similarity (0.10 where unalike, 0.29
  where alike; human 0.13, 0.42; the default 0.08, 0.25). Adopted at 1 (DECISIONS 10-12).
- **The two biases act on different bars:** taps at 2 with long-note bars at 1 give each
  what it gives alone (tap_rep1 0.279, ln_rep1 0.196).
- **keep_rows takes the chords to the human level, at a price.** At the default the chords
  fall to 0.368 (human 0.372; long-note songs 0.3755 against 0.3750, tap songs 0.352
  against 0.367), but 45% of the carried bars go, the repeats fall back (tap_rep1 0.225,
  ln_rep1 0.120, rhythm_rep1 0.175) and F1 drops 0.0018. Not adopted; the chord excess is
  in the long-note songs only (0.390 against 0.375; tap songs 0.365 against 0.367).
- **None of the carry settings brings a section back.** Rescored with the new
  `section_rhythm` (EXPERIMENTS 2026-10-12), every run gives 0.18-0.20 against the human
  0.547: the carry looks at the bars just before, not at where the music was.

### 2026-10-12 · What players praise and blame, measured; sections that come back, jacks in chords, chords over the cap (predictions)

What makes a 4K chart "good to play" (치기 좋은 채보, 치는 맛) and what gets it panned: a survey
of the osu!mania ranking criteria and mapping guides, osu!, Etterna, FFR and Quaver forums and
contest judging, Japanese and Korean rhythm-game communities, and the papers on generated
charts (the UGRP project's `claude/chart_quality_research_1011.md`; Reddit could not be read,
the Korean side is DC Inside search results). In short:

- Praise sits on two axes. The music: notes on sounds you can hear, the same sound with the
  same pattern every time, chords on the strong sounds, the sounds chosen rather than all of
  them charted. The hands: stressful and restful patterns in turn, intensity that rises and
  falls with the song. Repetition is praised where the music repeats (a motif, a chorus that
  comes back) and blamed elsewhere.
- Blame falls on four things and sync: "random" patterning (no audible link between sound
  and pattern), difficulty the music does not justify (spikes, chords and streams of notes
  for their own sake), boredom (repetition the music does not ask for, flat load), and
  patterns that tangle the hands (fast one-finger jacks, one-hand trills, long anchors,
  hammering under a held long note). Being hard is not blamed as such.
- Long notes are welcome on sustained sounds, released where the sound ends.
- osu! has banned generative tools from Ranked since 2025-07-24; Korean players panned the
  charts O2Jam was said to have made with AI (September-October 2026) for sync, unnatural
  patterns and no play-testing.

The evaluation already measures several of these against the audio: notes on weak onsets
(`weak_onsets`), density following loudness (`loud_slope`), the chart repeating where the
audio repeats bar to bar (`rho_*`), all near the human. New columns, also in `rescore.py`:

- `section_rhythm` / `section_copy` (`structure.section_repeats`): `structure.section_runs`
  finds the stretches of 4+ bars whose audio comes back 8+ bars later (cosine 0.5 or more
  with the bar `lag` before; the longest first, a bar in one stretch at most): 5.0 a song,
  9.7 bars each, 40% of the bars. Of their bars with 4+ onset rows, the share with the
  rhythm of the bar where the music was before, and with its lanes too (as they are or
  mirrored).
- `chord_over`, `fast_jacks`, `anchor_over`, `ln_short`, `release_close`
  (`playability.guideline_checks`): the ranking criteria's 4K guidelines for the chart's
  grade, in time at the guidelines' 180 BPM: rows holding more notes than Easy / Normal /
  Hard allow (2 / 2 / 3, long notes held counting, a double on the other hand from a hold
  fine in Normal); notes per 100 back in their lane within 1/4 beat; anchors over 2 notes
  (Normal) or 4 (Hard) per 100 onset rows; long notes under 1 / 1/2 / 1/4 beat; releases
  within 1 / 1/2 beat of the one before (Easy / Normal).
- `chord_strength` (`playability.chord_strength`): the onset strength (z within the song)
  under chords less that under single notes: chords on the strong sounds.

Per song (240 val songs; the new default is long-note bars at 1, below; by grade where the
guideline is):

| column | human | holes and rests on `$T` | the new default |
|---|---|---|---|
| section_rhythm / section_copy (208 songs) | 0.547 / 0.244 | 0.122 / 0.002 | 0.195 / 0.069 |
| chord_strength | 0.829 | 0.802 | 0.763 |
| chord_over Easy / Normal / Hard | 0.0049 / 0.0284 / 0.0098 | 0.0079 / 0.0250 / 0.0076 | 0.0094 / 0.0265 / 0.0073 |
| fast_jacks (per 100 notes) Hard / Insane / Expert | 0.04 / 0.39 / 1.82 | 0.12 / 0.80 / 3.25 | 0.14 / 0.74 / 3.34 |
| anchor_over Normal / Hard | 0.020 / 0.004 | 0 / 0.002 | 0 / 0.001 |
| ln_short Easy / Normal / Hard | 0.29 / 0.14 / 0.05 | 0.41 / 0.14 / 0.04 | 0.41 / 0.14 / 0.03 |
| release_close Easy / Normal | 0.18 / 0.07 | 0.15 / 0.06 | 0.15 / 0.07 |

What stands out against what players say:

- **Sections do not come back.** Where the music of 4+ bars returns, humans bring its rhythm
  back in 55% of the bars (its notes in 24%), the default in 19% (7%). Humans copy sections,
  not bars: of their bars that copy a bar 3+ back, 26% do it alone and the rest in runs at
  one lag (11% in runs of 16+), the default's 70% alone; per stretch, humans bring most of it
  back or none (37% of the stretches 75%+ of their bars, 27% none), the default seldom much
  (5% 75%+, 49% none). Bar by bar the earlier bar seldom qualifies: of the human far copies,
  the default's two bars pass copy_bars' checks in 14% (onset counts within 15% in 48%, no
  long note across an edge in 47%), and as a carry source in 18%. The audio finds the
  source: the human source of a copy 9+ back is the bar most like it in sound 8+ back 57% of
  the time (78% among the 3 most alike).
- **Jacks inside chords.** The default has 3-4 times the human fast jacks in Hard and twice
  from Insane up. Pooled, the human charts' in Hard are 70% two single notes 1/4 beat apart;
  the default's have a chord at one end in 87% and are 1/6 beat apart in half (from Insane
  up: 17% at 1/6 beat, human 6%): the minijacks players blame, made by chords.
- **Chords lean off the strong sounds a little** (0.763 against 0.829, the carry -0.04), and
  rows over the chord cap are twice the human's in Easy; in Normal and Hard humans pass the
  cap as often (long notes held counting).
- Long notes under a beat in Easy are 41% against 29% (`hold_rules` lets Easy holds go to
  half a beat); the human charts break the guideline too, and the long-note share is low
  already, so this is left as it is.
- Anchors, hand bias and taps under a held long note are at or under the human level.

Three passes, all off by default, on trial:

1. `copy_sections` (`--copy-sections B`, `--section-keep`, `--section-lanes`; tag
   `_cs<B>[k][n]`, playtest `cs<B>[k][n]`), after the carry and before the copies: each
   stretch, in time order and in pieces of at most 16 bars and at most its lag, takes the
   notes that start in the bars `lag` before, long notes with their lengths, as they are or
   mirrored for the whole piece (whichever the model likes, sum log p over the cells, of the
   two that fit around the long notes coming in and the notes after). The two must have as
   many notes within 15% (one a bar); `--section-keep`: not from fewer. The rhythm decides
   bar by bar as in copy_bars (with the bar MASK, log P(onset count) row by row, the earlier
   bar's rhythm against its own), summed over the piece: taken if the sum + B x bars >= 0.
   `--section-lanes`: the bars of taps of a piece taken get their lanes chosen again, as
   the carry's.
2. `untangle_jacks` (`--untangle`, tag `_uj`), after the rests: a note that takes its lane
   again within 1/6 beat at 180 BPM (56 ms) of the note before there, where one of the two
   rows is a chord: the chord's tap goes (the later one's if both are chords).
3. `guideline_cap` (`--chord-cap`, tag `_cc`), last: in Easy, Normal and Hard, rows over the
   cap keep the taps the lane model likes most (one forward pass a row, the row's taps MASK).

Smoke check (the 10 val songs with many stretches of 10-11, the default of 10-11's saved
charts, each pass alone, full-v4 on CPU; on these songs human section_rhythm 0.70 and
section_copy 0.32, the charts 0.23 and 0.05):

- Sections: the model prefers each piece's own rhythm (per bar, median -1.2 nats, quartiles
  -4.2 / -0.2), and how much it prefers it does not follow where humans repeat (correlation
  -0.06 over 37 pieces): the bias sets how much comes back, the score keeps out pieces that
  fit badly (-10 to -25 nats a bar). Without the 15% check an almost empty stretch scored 3
  nats a bar over a full one. At 0 / 2 / 4: 6.5 / 10.3 / 14.4 bars a song, section_rhythm
  0.39 / 0.51 / 0.66, section_copy 0.24 / 0.40 / 0.58, SR -0.023 / -0.026 / -0.019; with
  `--section-keep` at 2 / 4: 2.9 / 5.2 bars, 0.33 / 0.43, 0.16 / 0.29, notes +0.3% / +1.1%,
  SR +0.002 / +0.007; with new lanes at 4: section_copy 0.26 (the lane model often picks the
  lanes the earlier bars had). About 20 forward passes a song.
- `--untangle`: 1.2 taps a song, fast_jacks 0.52 -> 0.46 per 100 notes (Expert 2.25 -> 2.00),
  SR -0.021 (Expert -0.064: jacks weigh in the SR), chords -0.0006.
- `--chord-cap`: 10.1 rows a song, most in one Normal long-note chart; chord_over 0.023 ->
  0.002, notes -0.8%, SR -0.022 (Normal -0.11: taps under held long notes went), chords
  -0.006, chord_strength -0.014.
- No grammar violations.

`run_feel.sh` (patch 0046), on `$T` with the new default passes (taps 0, long-note bars at 1,
copies from 8 back, tidy 0.5, holes 0.5, rests 1):

0. The new columns rescored into holes and rests on `$T`, the old default and the new one.
1. Sections with `--section-keep` at 2 and at 4, and at 4 with new lanes.
2. `--untangle` alone, `--chord-cap` alone.
3. All three: sections at 4 with `--section-keep`, `--untangle`, `--chord-cap`.

Compared against holes and rests on `$T` and against the new default, by grade.

Predictions (against the new default: section_rhythm 0.195, section_copy 0.069, copy_far
0.026, fast_jacks 0.54, chord_strength 0.763, chords 0.384, SR bias -0.046, density 1.021,
F1 0.3820; paired means):

1. Sections at 2 with keep: 1.5-3 bars a song; section_rhythm 0.24-0.29, section_copy
   0.11-0.16. At 4: 3-5 bars; section_rhythm 0.28-0.36, section_copy 0.16-0.26; copy_far
   +0.01 to +0.03, rhythm_rep8 +0.01 to +0.03; density +0.3% to +0.8%, SR 0 to +0.012, F1
   -0.003 to +0.001; tap_rep1, ln_rep1 and lane_rep1 within ±0.01. With new lanes:
   section_rhythm as at 4 ±0.01, section_copy 0.01-0.04 under it.
2. `--untangle`: 1-3 taps a song; fast_jacks -0.04 to -0.10 (Hard to 0.05-0.10); density
   -0.05% to -0.15%; SR -0.005 to -0.02 (Expert -0.02 to -0.06); F1 within ±0.0005;
   chords -0.0005 to -0.002.
3. `--chord-cap`: 1-4 rows a song; chord_over Easy and Hard under 0.002, Normal under 0.005;
   density -0.1% to -0.5%; SR -0.003 to -0.015 over all (Normal -0.02 to -0.08); chords
   -0.001 to -0.005; chord_strength -0.005 to -0.02; F1 within ±0.001.
4. All three: each column as with its pass alone (they touch different notes), SR the sum
   of the three ±0.01.

**Results (`run_feel.sh`, night of 10-11 to 10-12)** — paired means against the new default
(`rc0kh1`); bars, taps and rows a song and the by-grade means are per-song means; 240 songs

| # | prediction | result | |
|---|---|---|---|
| 1 | sections at 2 with keep: 1.5-3 bars a song; section_rhythm 0.24-0.29; section_copy 0.11-0.16 | 4.0; 0.240; 0.128 | miss (more), hit (at the edge), hit |
| 1 | at 4: 3-5 bars; section_rhythm 0.28-0.36; section_copy 0.16-0.26 | 6.1; 0.274; 0.165 | miss (more), miss (just under), hit |
| 1 | at 4: copy_far +0.01 to +0.03; rhythm_rep8 +0.01 to +0.03 | +0.012; +0.014 | hit, hit |
| 1 | at 4: density +0.3% to +0.8%; SR 0 to +0.012; F1 -0.003 to +0.001 | +0.30%; +0.0055; -0.0002 | hit (at the edge), hit, hit |
| 1 | at 4: tap_rep1, ln_rep1, lane_rep1 within ±0.01 | +0.001, -0.009, +0.000 | hit ×3 |
| 1 | new lanes: section_rhythm as at 4 ±0.01; section_copy 0.01-0.04 under it | +0.002; +0.001 | hit, miss (the lanes came out as they were) |
| 2 | `--untangle`: 1-3 taps a song; fast_jacks -0.04 to -0.10, Hard to 0.05-0.10 | 2.2; -0.067, Hard 0.121 | hit, hit, miss |
| 2 | density -0.05% to -0.15%; SR -0.005 to -0.02 (Expert -0.02 to -0.06) | -0.07%; -0.016 (Expert -0.137) | hit, hit (miss) |
| 2 | F1 within ±0.0005; chords -0.0005 to -0.002 | -0.0000; -0.0007 | hit, hit |
| 3 | `--chord-cap`: 1-4 rows a song; chord_over Easy and Hard under 0.002, Normal under 0.005 | 4.2; 0.0009 and 0.0011, 0.0006 | miss (just over), hit ×3 |
| 3 | density -0.1% to -0.5%; SR -0.003 to -0.015 (Normal -0.02 to -0.08) | -0.47%; -0.0175 (Normal -0.055) | hit, miss (just over) (hit) |
| 3 | chords -0.001 to -0.005; chord_strength -0.005 to -0.02; F1 within ±0.001 | -0.0012; -0.0005; -0.0013 | hit, miss (hardly moved), miss (just over) |
| 4 | all three: each column as with its pass alone; SR the sum of the three ±0.01 | section_rhythm 0.273, chord_over 0.0010, fast_jacks -0.058 (alone -0.067, sections +0.012); -0.027 (sum -0.028) | hit, hit |

All runs (per song; human: section_rhythm 0.547, section_copy 0.244, copy_far 0.068,
bar_lane_repeat 0.334, chord_strength 0.829; fast_jacks Hard / Insane / Expert 0.04 / 0.39 /
1.82; chord_over Easy / Normal / Hard 0.0049 / 0.0284 / 0.0098):

| run | section bars a song | section_rhythm / copy | copy_far | bar_lane_repeat | fast_jacks H / I / E | chord_over E / N / H | chord_strength | SR bias (Easy / Normal / Insane / Expert) | density | F1 | passes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| holes and rests on `$T` | - | 0.122 / 0.002 | 0.005 | 0.135 | 0.12 / 0.80 / 3.25 | 0.0079 / 0.0250 / 0.0076 | 0.802 | -0.052 (+0.079 / +0.066 / -0.133 / -0.265) | 1.014 | 0.3800 | 170 |
| the new default (`rc0kh1`) | - | 0.195 / 0.069 | 0.026 | 0.280 | 0.14 / 0.74 / 3.34 | 0.0094 / 0.0265 / 0.0073 | 0.763 | -0.046 (+0.094 / +0.074 / -0.125 / -0.259) | 1.021 | 0.3820 | 334 |
| sections at 2, keep | 4.0 | 0.244 / 0.133 | 0.032 | 0.313 | 0.14 / 0.75 / 3.34 | 0.0094 / 0.0269 / 0.0073 | 0.762 | -0.044 (+0.095 / +0.076 / -0.121 / -0.258) | 1.023 | 0.3821 | 346 |
| sections at 4, keep | 6.1 | 0.277 / 0.169 | 0.038 | 0.336 | 0.14 / 0.78 / 3.34 | 0.0094 / 0.0270 / 0.0073 | 0.748 | -0.041 (+0.099 / +0.077 / -0.116 / -0.256) | 1.024 | 0.3817 | 346 |
| sections at 4, keep, new lanes | 6.1 | 0.279 / 0.170 | 0.038 | 0.334 | 0.14 / 0.77 / 3.34 | 0.0094 / 0.0270 / 0.0073 | 0.748 | -0.041 (+0.099 / +0.077 / -0.116 / -0.263) | 1.024 | 0.3818 | 374 |
| `--untangle` | - | 0.195 / 0.068 | 0.026 | 0.280 | 0.12 / 0.68 / 2.79 | 0.0094 / 0.0265 / 0.0073 | 0.764 | -0.062 (+0.094 / +0.073 / -0.138 / -0.396) | 1.020 | 0.3820 | 334 |
| `--chord-cap` | - | 0.195 / 0.068 | 0.026 | 0.278 | 0.14 / 0.74 / 3.34 | 0.0009 / 0.0006 / 0.0011 | 0.763 | -0.064 (+0.078 / +0.019 / -0.125 / -0.259) | 1.016 | 0.3807 | 342 |
| all three | 6.1 | 0.277 / 0.167 | 0.038 | 0.334 | 0.12 / 0.72 / 2.80 | 0.0009 / 0.0006 / 0.0011 | 0.749 | -0.074 (+0.084 / +0.021 / -0.129 / -0.392) | 1.018 | 0.3805 | 354 |

What it says:

- **Sections come back where the carry could not, for nothing.** At 4 with keep, 6.1 bars a
  song take the notes of where their music was: section_rhythm 0.195 -> 0.274 and
  section_copy 0.069 -> 0.165 (paired), bar_lane_repeat at the human level (0.336 against
  0.334), copy_far 0.026 -> 0.038; the SR nearer the target (+0.0055), F1 as it was
  (-0.0002, CI -0.0008 to +0.0003), notes +0.3%, 12 forward passes a song. The costs are
  small: chord_strength -0.004 (CI -0.009 to +0.000), fast jacks +0.012 (Insane +0.04). By
  grade the easy charts gain most: Easy section_copy reaches the human (0.335 against
  0.331; section_rhythm 0.47 against 0.65), Normal to Expert stay at about half the human
  (section_rhythm 0.32 / 0.25 / 0.22 / 0.21 against 0.64 / 0.58 / 0.49 / 0.39). Adopted
  (DECISIONS 10-12).
- **What keeps the rest out** (40 val songs, the default's charts, the pass at 4 with keep
  bar by bar, `smoke_diag.py`): of the stretches' bars (48 a song), the 15% note-count check
  leaves out 32%, keep_notes 25% (the earlier stretch has fewer notes), pieces where neither
  the notes nor their mirror fit around the long notes 18%, the score at 4 11%; 12% are
  taken. Humans repeat the rhythm in 72% of the bars of the pieces taken (the score agrees
  with them), in 55% of those keep_notes leaves out, in 34-38% of the others. Those
  keep_notes leaves out are the ones to win: next.
- **New lanes change nothing:** the lane model picks the lanes the earlier bars had
  (section_copy +0.001), for 28 more passes a song. Not adopted.
- **Untangling at 1/6 beat takes the jacks out where the SR can least afford it:** 2.2 taps
  a song, of them 19 a song in Expert and 11 in Expert+, whose SR is short already (Expert
  -0.26 -> -0.40). In Hard only 0.46 a song: fast jacks 0.14 -> 0.12 (human 0.04), the Hard
  excess being 1/4 beat apart. Not adopted as it is; by grade, moving the tap, next.
- **The chord cap is right in Easy only.** 4.2 rows a song (Easy 4.0, Normal 14.3, Hard
  5.0) cut to the cap. Easy: rows over the cap 0.0094 -> 0.0009 (human 0.0049), SR bias
  +0.094 -> +0.078 (nearer), notes -0.6%, F1 -0.001. Normal: SR +0.074 -> +0.019 and
  density 1.025 -> 1.006, both nearer the human, but F1 -0.005 for rows humans pass the cap
  in as often (0.028), and chord_strength -0.006; Hard: SR -0.003 -> -0.027 (further), F1
  -0.001. chord_strength over all hardly moves (-0.0005; predicted -0.005 to -0.02).
  Adopted for Easy (`--chord-cap 2`, DECISIONS 10-12).
- **The three add up** (SR -0.027 against the sum -0.028; each column as alone).

### 2026-10-12 · Sections either way; jack taps moved, by grade (predictions)

From the night: the section copies are adopted at 4 with keep (DECISIONS 10-12), and what
keeps most of the remaining stretches out is that the later one has more notes (25% of the
stretches' bars; humans repeat the rhythm in 55% of those). The jack pass took notes out
of Expert, where they cost SR, and too few out of Hard, where they are 1/4 beat apart.

1. `copy_sections(back=True)` (`--section-back`, tag `b` after `k`; playtest `csBkb`):
   where the later stretch has more notes than the bars lag before, those earlier bars take
   its notes instead, with the same checks (the 15% check, the fit, the score of the
   earlier bars, masked, with the later rhythm against their own; implies keep_notes), so
   the two agree with no notes lost. Bars a piece taken wrote or read are not written
   backwards again, so a copy made earlier in the song stays a copy. STATS `section_back`
   (column with `--section-back`).
2. `untangle_jacks(move=True)` (`--untangle-move`, tag `m`) and the window by grade
   (`--untangle-grades`, tag `g`; `untangle_window`: the guidelines' 1/4 beat up to Hard,
   1/6 in Insane, none from Expert up): the chord's tap goes to another lane of its row
   where it makes no jack within the window (the cell EMPTY, past the release gap; of those
   the lane whose nearest note is furthest away), and out only if there is none; the chord
   and the note count stay. STATS `untangle_moved` (column with `--untangle-move`).
3. `--chord-cap` takes an SR: the cap in charts under it (`guideline_cap(below=)`, tag
   `_cc<sr>`; playtest `ccX`, `cc` alone 4). generate.py and sample.py cap under 2 (Easy) by
   default and copy sections at 4 with keep (`--no-sections`, `--chord-cap 0` turn them off).

Smoke check (40 val songs: the 10 of 10-11 with many stretches and 30 drawn by grade; the
pass alone on the new default's saved charts of 10-11, full-v4 on CPU; human section_rhythm
0.574, section_copy 0.281 on these):

- Sections either way at 4: 11.5 bars a song (5.6 of them backwards) against 5.9 one way;
  section_rhythm 0.185 -> 0.408 (one way 0.317), section_copy 0.073 -> 0.356 (0.234; the
  human 0.281: humans keep the lanes in half the bars whose rhythm comes back, the copies
  in nearly all); notes +0.8% (+0.45%), SR +0.006 (+0.004), chord_strength -0.018
  (-0.012), fast jacks +0.016 (+0.000); 21 forward passes a song (11). At 8: 15.5 bars (7.9
  backwards), section_rhythm 0.471, section_copy 0.427, notes +1.05%, SR +0.0135,
  chord_strength -0.025, fast jacks +0.023; one way at 8: 7.6 bars, 0.348 / 0.265.
- Jack taps: at 1/6 beat everywhere 0.85 a song, SR -0.008 (Expert -0.019, Expert+ -0.057);
  moving, 47% of them move, SR -0.006 (Expert -0.011); by grade and moving 0.47 a song (64%
  moved; Hard 1.0, Insane 1.0, from Expert up none), fast jacks Hard 0.07 -> 0.02, Insane
  0.34 -> 0.32, SR -0.004 (Hard -0.010, Insane -0.003); by grade without moving SR -0.006
  (Hard -0.016).
- No grammar violations.

`run_back.sh` (patch 0047), on `$T` with the new default passes (taps 0, long-note bars at 1,
sections at 4 with keep, copies from 8 back, tidy 0.5, holes 0.5, rests 1, the chord cap in
Easy), each against the new default and by grade:

0. The new default.
1. Sections either way at 4 and at 8.
2. Jack taps moved, by grade.
3. Both: sections either way at 4, jack taps moved by grade.

Predictions (the new default: from sections at 4 with keep and the Easy rows of the chord
cap, section_rhythm 0.274, section_copy 0.165, copy_far 0.038, fast_jacks 0.55 (Hard 0.145,
Insane 0.78, Expert 3.34), chord_strength 0.748, SR bias -0.044, density 1.023, F1 0.3815;
paired means):

0. The new default against sections at 4 with keep of 10-12: only the Easy charts change;
   rows over the cap in Easy 0.0094 -> under 0.002, Easy SR -0.01 to -0.02, Easy F1 -0.0005
   to -0.002.
1. Either way at 4: 10-13 bars a song (4.5-6.5 backwards); section_rhythm 0.31-0.35,
   section_copy 0.21-0.27; copy_far +0.005 to +0.015; notes +0.2% to +0.6%; SR +0.001 to
   +0.008; F1 -0.001 to +0.001; chord_strength 0 to -0.01; fast_jacks +0.005 to +0.03;
   8-14 more passes a song. At 8: 14-18 bars (6-9 backwards); section_rhythm 0.33-0.39,
   section_copy 0.24-0.31; notes +0.4% to +1.0%; SR +0.004 to +0.015; F1 -0.002 to +0.001;
   chord_strength -0.005 to -0.02; fast_jacks +0.01 to +0.04.
2. Jack taps moved, by grade: 0.5-1.0 a song (Hard 0.8-1.5, Insane 1.2-1.8, from Expert up
   none), 55-75% of them moved; fast_jacks -0.03 to -0.06 over all, Hard to 0.03-0.08
   (human 0.04), Insane 0.70-0.76, Expert as it is; SR -0.001 to -0.005 (Hard -0.003 to
   -0.012, Insane -0.003 to -0.010, Expert 0); notes within -0.02%; F1 and chords within
   ±0.0005.
3. Both: the section columns as either way at 4 (±0.01), fast_jacks in Hard and Insane as
   with the jack taps moved (±0.02), SR the sum of the two ±0.005.

**Results (`run_back.sh`, early morning of 10-12)** — paired means against the new default
(sections at 4 with keep, the chord cap in Easy) where the prediction is a change; bars, taps
and the by-grade means are per-song means; 240 songs

| # | prediction | result | |
|---|---|---|---|
| 0 | only the Easy charts change; rows over the cap in Easy under 0.002; Easy SR -0.01 to -0.02; Easy F1 -0.0005 to -0.002 | the rest as they were; 0.0009; -0.015; -0.0011 | hit ×4 |
| 1 | either way at 4: 10-13 bars a song (4.5-6.5 backwards) | 10.5 (4.3) | hit, miss (just under) |
| 1 | section_rhythm 0.31-0.35, section_copy 0.21-0.27 | 0.346, 0.245 | hit, hit |
| 1 | copy_far +0.005 to +0.015; notes +0.2% to +0.6%; SR +0.001 to +0.008; F1 -0.001 to +0.001 | +0.013; +0.26%; +0.0013 (CI -0.002 to +0.005); +0.0005 | hit ×4 |
| 1 | chord_strength 0 to -0.01; fast_jacks +0.005 to +0.03; 8-14 more passes a song | -0.002; +0.002; +9.6 | hit, miss (hardly moved), hit |
| 1 | at 8: 14-18 bars (6-9 backwards); section_rhythm 0.33-0.39, section_copy 0.24-0.31 | 15.7 (6.6); 0.430, 0.336 | hit, hit, miss (more), miss (more) |
| 1 | at 8: notes +0.4% to +1.0%; SR +0.004 to +0.015; F1 -0.002 to +0.001 | +0.52%; +0.0046; +0.0006 | hit ×3 (SR at the edge) |
| 1 | at 8: chord_strength -0.005 to -0.02; fast_jacks +0.01 to +0.04 | -0.004; +0.003 | miss (less), miss (hardly moved) |
| 2 | jack taps moved by grade: 0.5-1.0 a song (Hard 0.8-1.5, Insane 1.2-1.8, from Expert up none); 55-75% moved | 1.05 (2.07, 1.50, none); 48% (Hard 38%, Insane 63%) | miss (just over), miss, hit, hit; miss (fewer) |
| 2 | fast_jacks -0.03 to -0.06; Hard to 0.03-0.08; Insane 0.70-0.76; Expert as it is | -0.047; 0.036 (human 0.039); 0.732; as it is | hit ×4 |
| 2 | SR -0.001 to -0.005 (Hard -0.003 to -0.012, Insane -0.003 to -0.010, Expert 0) | -0.0049 (-0.010, -0.006, 0) | hit ×4 (over all at the edge) |
| 2 | notes within -0.02%; F1 and chords within ±0.0005 | -0.03%; +0.0000, -0.0002 | miss (just over), hit, hit |
| 3 | both: the section columns as either way at 4 ±0.01; fast_jacks Hard and Insane as moved ±0.02; SR the sum ±0.005 | 0.346 / 0.244; 0.036 / 0.736; -0.0041 (sum -0.0036) | hit ×3 |

All runs (per song; human: section_rhythm 0.547, section_copy 0.244, bar_lane_repeat 0.334,
copy_far 0.068, chord_strength 0.829, chords 0.372; fast_jacks Hard / Insane / Expert 0.04 /
0.39 / 1.82):

| run | section bars a song (backwards) | section_rhythm / copy | bar_lane_repeat | copy_far | fast_jacks H / I / E | chord_strength | chords | SR bias (Easy / Normal / Hard / Insane / Expert) | density | F1 | passes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| the new default (sections at 4, keep; cap in Easy) | 6.1 | 0.277 / 0.169 | 0.335 | 0.038 | 0.14 / 0.78 / 3.34 | 0.748 | 0.386 | -0.044 (+0.084 / +0.077 / +0.002 / -0.116 / -0.256) | 1.023 | 0.3815 | 347 |
| sections either way at 4 | 10.5 (4.3) | 0.346 / 0.245 | 0.385 | 0.051 | 0.15 / 0.78 / 3.35 | 0.746 | 0.389 | -0.043 (+0.086 / +0.077 / +0.002 / -0.114 / -0.253) | 1.025 | 0.3820 | 357 |
| sections either way at 8 | 15.7 (6.6) | 0.430 / 0.336 | 0.440 | 0.067 | 0.15 / 0.79 / 3.33 | 0.744 | 0.389 | -0.039 (+0.090 / +0.079 / +0.010 / -0.116 / -0.245) | 1.028 | 0.3822 | 357 |
| jack taps moved, by grade | 6.1 | 0.277 / 0.168 | 0.333 | 0.038 | 0.04 / 0.73 / 3.34 | 0.749 | 0.385 | -0.049 (+0.084 / +0.074 / -0.008 / -0.122 / -0.256) | 1.022 | 0.3816 | 347 |
| both (either way at 4, jacks moved) | 10.5 (4.3) | 0.346 / 0.244 | 0.383 | 0.050 | 0.04 / 0.74 / 3.35 | 0.747 | 0.388 | -0.048 (+0.086 / +0.075 / -0.009 / -0.121 / -0.253) | 1.025 | 0.3821 | 357 |

What it says:

- **Sections either way at 4 bring the exact copies to the human level.** section_copy 0.169
  -> 0.245 (human 0.244), section_rhythm 0.277 -> 0.346 (0.547), copy_far 0.038 -> 0.051
  (0.068), with F1 +0.0005, SR +0.0013 (not significant), notes +0.26%, 10 more passes a
  song; chord_strength and fast jacks do not move (-0.002, +0.002, both within noise).
  bar_lane_repeat goes over the human (0.385 against 0.334). By grade: section_rhythm Easy
  0.57 (human 0.65), Normal 0.35 (0.64), Hard 0.33 (0.58), Insane 0.28 (0.49), Expert 0.28
  (0.39); section_copy over the human in Easy (0.44 against 0.33) and Expert (0.22 against
  0.18). Adopted (DECISIONS 10-12).
- **At 8 the copies run past the human.** section_rhythm 0.430 (Easy at the human 0.648), but
  section_copy 0.336 against 0.244 (Easy 0.51 against 0.33, Hard 0.33 against 0.22) and
  bar_lane_repeat 0.440 against 0.334: humans keep the lanes in 45% of the bars whose rhythm
  comes back, these copies in 78%, the "copy pasted with zero variation" players blame. Not
  adopted: more of the rhythm without more exact copies is the next question.
- **Jack taps moved by grade bring Hard to the human level.** Fast jacks in Hard 0.145 ->
  0.036 (human 0.039), Insane 0.78 -> 0.73 (0.39), Normal 0.022 -> 0.007 (0), Expert as it
  was; 1.05 taps a song (Hard 2.1, Insane 1.5), half of them moved (Hard 38%, Insane 63%);
  SR -0.005 (Hard -0.010, Insane -0.006), F1 and chords as they were. Adopted.
- **The two add up**, each column as with its pass alone.

### 2026-10-12 · Bars that sound the same; chords on the strong sounds (predictions)

The charts of the default were rated by what players praise and blame (the UGRP project's
`claude/chart_quality_research_1011.md`; the rating, `claude/chart_eval_1012.md`, 240 val
songs, human against the default of run_feel): notes on audible sounds (weak onsets 0.151
against 0.152), density following the music (rank correlation of notes and loudness per bar
0.52 against 0.53, the densest bars as loud as the human's) and the load's ups and downs
(0.30 against 0.31) are at the human level, and so are the hands but for chord jacks (fast
one-hand trills, anchors, taps under a held long note on the same hand at or under the
human). Furthest from the human:

- **The same sound, the same pattern.** Of the pairs of bars whose audio is nearly the same
  (ssm_audio cosine 0.8 or more, both with 4+ onset rows, any distance), humans give 62% the
  same rhythm, the default 29% (the raw samples 17%), and 19% of them note counts more than a
  quarter apart (human 8.6%; Normal 18% against 5%), which the mapping guide calls wrong
  ("different amounts of notes for the same sounds"). The copies leave those alone: they ask
  for counts within 15%; the carry looks only 1, 2, 4 and 8 bars back. In a bar whose count
  is off the median of the bars that sound like it by over a quarter, the human count is
  nearer that median twice as often as the sampled one (`alike_consensus.py`).
- **Chords on the strong sounds.** 0.75 against 0.83, the raw samples 0.80: the carry and the
  copies bring the chord sizes of the bars they copy. Chords on the weakest fifth of the onset
  rows 14% (human 11%); in beats with a chord and a single note, the chord on the stronger
  onset 79% (82%).
- Easy is over-charted (notes +12%, SR +0.08 with the cap): fewer gaps of 2 beats between
  notes (2.8% of the gaps against 4.2%) and fewer light bars (0.037 against 0.055); left for later.
- The "breaks" (a pattern with one note off, then on again) the rating counted at 3.3 times
  the human are 0.13 a song (human 0.04): not worth a pass.

New columns (evaluate.py, rescore.py, compare_runs.py): `alike_rhythm`, `alike_counts_off`
(`structure.alike_repeats`): of the pairs of bars that sound nearly the same, those with the
same rhythm and those with note counts a quarter apart.

Two passes, off by default, on trial:

1. `align_alike` (`--alike B`, `--alike-keep`; tag `_al<B>[k]`, playtest `alB[k]`), after the
   copies: a bar with 4+ onset rows that sounds like others (cosine 0.8+) but shares the
   rhythm of none of them takes the rhythm of one of them whose note count is within a
   quarter of the group's median (the bar's own included), the most alike first, of its kind
   (taps / long notes), when the model scores it within B nats of its own (as the carry); the
   lanes are chosen again (a long-note bar: its lanes permuted). `--alike-keep`: not from
   fewer notes. A bar so aligned is a match for the bars after it.
2. `strong_chords` (`--strong-chords Z`; tag `_sc<Z>` after the carry's, playtest `scZ`), right
   after the carry: in each beat a chord on an onset Z (z of the onset strength over the
   chart's onset rows) weaker than a single note's gives one of its taps to that note, in a
   lane free there with no note within 1/4 beat (at 180 BPM); the notes, the chords and the
   rhythm stay. Done before the copies so that they keep it.

Smoke checks (full-v4 on CPU):

- On the default's saved charts (run_back's both, 40 val songs of 10-12), the passes alone:
  `--alike 2` aligns 6.5 bars a song: the alike pairs' same rhythm 0.26 -> 0.48 (human 0.66
  on these songs), counts a quarter apart 0.20 -> 0.15 (0.08), section_rhythm 0.35 -> 0.43
  (0.57), section_copy -0.002, notes -0.7%, SR -0.010 (Normal -0.019), chord_strength -0.003;
  at 0 / 4: 5.2 / 7.1 bars, 0.42 / 0.50. With `--alike-keep` at 2: 3.0 bars, 0.34, 0.19,
  section_rhythm 0.37, notes +0.3%, SR +0.003, fast jacks +0.010. Only the bars off their
  group's median (the first version): 0.8 bars, 0.28, 0.16.
- `strong_chords` after the rests at Z 1 (0.5): 9.4 (20.6) taps a song, chord_strength 0.757
  -> 0.833 (0.869; human 0.849), section_copy -0.022 (-0.046).
- On the whole default pipeline from the raw samples (11 val songs, 2 a grade): after the
  rests chord_strength +0.077 and section_copy -0.036; right after the carry +0.067,
  section_copy +0.002, the alike pairs' same rhythm +0.017, 12 taps a song: right after the
  carry it is.
- No grammar violations.

`run_alike.sh` (patch 0048), on `$T` with the new default passes (taps 0, long-note bars at 1,
sections either way at 4 with keep, copies from 8 back, tidy 0.5, holes 0.5, rests 1, jack taps
moved by grade, the chord cap in Easy), against the new default (run_back's both, rescored for
the new columns), by grade:

1. `--alike 2`; `--alike 2 --alike-keep`.
2. `--strong-chords 1`.
3. `--alike 2` with `--strong-chords 1`.

Predictions (the new default: alike_rhythm 0.30-0.34, alike_counts_off 0.17-0.20 per song
from the rating of 10-12, section_rhythm 0.346, section_copy 0.244, chord_strength 0.747,
chords 0.388, SR bias -0.048, density 1.025, F1 0.3821; paired means):

1. `--alike 2`: 4-8 bars a song; alike_rhythm +0.12 to +0.22, alike_counts_off -0.03 to
   -0.06; section_rhythm +0.04 to +0.09, section_copy within ±0.01; notes -0.3% to -0.8%, SR
   -0.004 to -0.014; F1 within ±0.001; chord_strength 0 to -0.008; 30-50 more passes a song.
   With keep: 2-4 bars; alike_rhythm +0.05 to +0.10, alike_counts_off -0.01 to -0.03,
   section_rhythm +0.01 to +0.04; notes +0.1% to +0.5%, SR 0 to +0.006; fast_jacks 0 to
   +0.015.
2. `--strong-chords 1`: 8-14 taps a song; chord_strength +0.05 to +0.08 (to 0.80-0.83); chords
   within ±0.002, notes within ±0.05%, section_copy within ±0.01, alike_rhythm 0 to +0.03; F1
   within ±0.001, SR within ±0.005, fast_jacks 0 to +0.015.
3. Both: each column as with its pass alone (alike_rhythm and chord_strength each within
   0.02 of it), SR the sum ±0.005.

**Results (`run_alike.sh`, morning of 10-12)** — paired means against the new default (run_back's
both, rescored for the alike columns); bars and taps a song are per-song means; 240 songs

| # | prediction | result | |
|---|---|---|---|
| 1 | `--alike 2`: 4-8 bars a song; alike_rhythm +0.12 to +0.22, alike_counts_off -0.03 to -0.06 | 7.6; +0.217 (0.322 -> 0.539, human 0.622); -0.061 (0.183 -> 0.122, human 0.086) | hit, hit (at the edge), miss (just over) |
| 1 | section_rhythm +0.04 to +0.09, section_copy within ±0.01 | +0.063, -0.003 | hit, hit |
| 1 | notes -0.3% to -0.8%, SR -0.004 to -0.014, F1 within ±0.001 | -0.42%; -0.019 (Easy -0.009, Normal -0.009, Hard -0.016, Insane -0.027, Expert -0.056); +0.0002 | hit, miss (more), hit |
| 1 | chord_strength 0 to -0.008; 30-50 more passes a song | -0.002; +42 | hit, hit |
| 1 | with keep: 2-4 bars; alike_rhythm +0.05 to +0.10, alike_counts_off -0.01 to -0.03, section_rhythm +0.01 to +0.04 | 3.8; +0.093, -0.018, +0.021 | hit ×4 |
| 1 | with keep: notes +0.1% to +0.5%, SR 0 to +0.006, fast_jacks 0 to +0.015 | +0.37%, +0.003, +0.007 | hit ×3 |
| 2 | `--strong-chords 1`: 8-14 taps a song; chord_strength +0.05 to +0.08 (to 0.80-0.83) | 10.3; +0.076 (0.747 -> 0.822, human 0.829) | hit, hit |
| 2 | chords within ±0.002, notes within ±0.05%, section_copy within ±0.01, alike_rhythm 0 to +0.03 | +0.0003, -0.00%, +0.0002, +0.0005 | hit ×4 |
| 2 | F1 within ±0.001, SR within ±0.005, fast_jacks 0 to +0.015 | +0.0000, +0.001, -0.010 | hit, hit, miss (down) |
| 3 | both: alike_rhythm and chord_strength within 0.02 of each alone; SR the sum ±0.005 | 0.543 (alone 0.539), 0.819 (0.822); -0.020 (sum -0.018) | hit ×3 |

All runs (per song; human: alike_rhythm 0.622, alike_counts_off 0.086, section_rhythm 0.547,
section_copy 0.244, bar_lane_repeat 0.334, chord_strength 0.829, chords 0.372):

| run | aligned bars / strong taps a song | alike_rhythm / counts_off | section_rhythm / copy | bar_lane_repeat | chord_strength | chords | SR bias (Easy / Normal / Hard / Insane / Expert) | density | F1 | passes |
|---|---|---|---|---|---|---|---|---|---|---|
| the default (run_back's both) | - | 0.322 / 0.183 | 0.346 / 0.244 | 0.383 | 0.747 | 0.388 | -0.048 (+0.086 / +0.075 / -0.009 / -0.121 / -0.253) | 1.025 | 0.3821 | 357 |
| alike 2 | 7.6 | 0.539 / 0.122 | 0.409 / 0.241 | 0.354 | 0.745 | 0.391 | -0.067 (+0.078 / +0.066 / -0.024 / -0.148 / -0.309) | 1.021 | 0.3822 | 399 |
| alike 2, keep | 3.8 | 0.415 / 0.165 | 0.367 / 0.241 | 0.366 | 0.747 | 0.390 | -0.045 (+0.086 / +0.076 / -0.005 / -0.115 / -0.249) | 1.029 | 0.3827 | 382 |
| strong chords 1 | 10.3 | 0.323 / 0.183 | 0.346 / 0.244 | 0.395 | 0.822 | 0.389 | -0.047 (+0.086 / +0.081 / -0.007 / -0.119 / -0.255) | 1.025 | 0.3821 | 357 |
| both (alike 2, strong chords 1) | 7.7 / 10.3 | 0.543 / 0.121 | 0.409 / 0.241 | 0.359 | 0.819 | 0.391 | -0.068 (+0.079 / +0.068 / -0.026 / -0.148 / -0.311) | 1.020 | 0.3822 | 400 |

alike_rhythm by grade (Easy / Normal / Hard / Insane / Expert; human 0.71 / 0.66 / 0.67 / 0.59 /
0.45): the default 0.42 / 0.42 / 0.33 / 0.27 / 0.24, alike 2 0.61 / 0.66 / 0.56 / 0.48 / 0.44,
with keep 0.50 / 0.48 / 0.44 / 0.37 / 0.33. chord_strength with strong chords: Easy 0.51 (human
0.56), Normal 0.77 (0.74), Hard 0.94 (0.91), Insane 0.93 (0.95), Expert 0.82 (0.83).

What it says:

- **Bars that sound alike now mostly share a rhythm.** At bias 2, 7.6 bars a song take the
  rhythm of a bar that sounds like them: alike_rhythm 0.32 -> 0.54 (human 0.62; Normal and
  Expert at the human level), counts a quarter apart 0.18 -> 0.12 (0.086), section_rhythm
  +0.06, bar_lane_repeat back towards the human (0.383 -> 0.354; 0.334), F1 and chords as they
  were. The cost is notes (-0.4%) and SR (-0.019), most in the hard grades (Insane -0.027,
  Expert -0.056), which are under the target already; in Easy and Normal, over it, the SR
  moves towards it. With keep (no source with fewer notes) half the bars and 43% of the gain,
  SR +0.003. So keep where the chart is already easier than asked, from SR 4 (Insane up).
  Assembled per song from the two runs (alike 2 under SR 4, with keep from 4): alike_rhythm
  0.489, counts_off 0.139, section_rhythm 0.394, SR bias -0.054 (Hard -0.024, Insane -0.115,
  Expert -0.249), density 1.024, F1 0.3824; keep from 2.7 instead gives 0.451 / 0.152 (Hard 0.44
  against 0.56) for Hard's SR at -0.005 instead of -0.024. Adopted at 2 with keep from SR 4
  (DECISIONS 10-12).
- **Chords on the strong sounds at the human level.** At 1, 10 taps a song move (Easy 1.9,
  Hard 10.8, Expert 22): chord_strength 0.747 -> 0.822 (human 0.829), and nothing else moves:
  notes, chords, F1, SR, the copies' and the alike columns as they were; fast jacks -0.010 (the
  tap goes to a lane with no note within 1/4 beat), bar_lane_repeat +0.012, motion_pred
  -0.0025. Adopted.
- **The two add up**, each column as with its pass alone and the SR the sum.

### 2026-10-12 · Long notes up to the share asked; light bars in Easy and Normal (predictions)

From run_alike: chords on the strong sounds (`--strong-chords 1`) and bars that sound alike at
bias 2 with keep from SR 4 (`--alike 2 --alike-keep 4`) are the default (DECISIONS 10-12). Of
the rating of 10-12, two gaps are left that a pass can reach:

- **Long notes.** With the chart stats the model follows the long-note share asked for up to
  about 0.2 and falls short above it (10-05); in the default's charts, by the human chart's
  share (240 val songs): 0.2-0.35 asked 0.261, got 0.196; 0.35-0.5: 0.431, 0.299; 0.5 and
  more: 0.710, 0.326 (13 songs). Over all 0.156 against 0.218, and 0.78 beats long against
  1.00. For a new song the share is asked for (DECISIONS 10-11): a long-note chart asked at 0.4
  comes out near 0.3.
- **Easy is busier than the human charts** (notes +11%, SR +0.08 after the cap), not in fewer
  rests (half-bars without a note 5.3% against 4.3%: the sampled ones fall in the quiet parts,
  loudness z under -1: 25% against 14%) but in fewer light bars: humans leave 13% of the bars
  inside an Easy chart with one or two onset rows (per song; pooled, one row 9.4% of the bars
  against 3.0%), the single rows on the downbeat (84%) and in runs (the bars around them hold
  1.2 rows), at any loudness; gaps of 2+ beats between onset rows 24 a song against 16, ending
  on a downbeat 48% against 35%. Normal: light bars 3.8% against 2.7%. The audio hardly tells
  those bars (AUC 0.55-0.60 for loudness and onsets); the model, with the bar masked and its
  own chart around, at 0.77 in Easy and 0.86 in Normal (0.96 / 0.97 with the human chart
  around; 15 val songs).
- The guideline on runs ("avoid using more than 5 consecutive 1/2 notes" in Easy, 1/4 in
  Normal), new column `run_over` (notes past the 5th per 100 onset rows, in the song's beats
  with the ranking criteria's Scaling BPM): human Easy 11.7, the default 15.0 (raw samples
  14.8); Normal 0.73 against 1.34.

New column `run_over` (playability.guideline_checks; evaluate.py, rescore.py, compare_runs.py);
compare_runs' by-grade table adds run_over, light_bars and hold_share.

Two passes, off by default, on trial:

1. `refine_holds(only_more=True)` (`--hold-share X --hold-up`; evaluate.py `--hold-share oracle
   --hold-up`, tag `_hu-<share>` in place of `_hr`; generate.py and sample.py `--hold-up` with
   `--hold-share` or the ln of `--stats`; playtest `hu`, `huX`), first of the passes after the
   lane passes: the long notes as sampled stay, and taps become long notes, those where the
   model, asked as refine_holds asks (every other onset of a lane masked from it to the next
   one, the rest of the chart in view), puts the most of p(START) against p(TAP), as many as
   the share asks beyond those already there, none shorter than the guidelines' shortest for
   the grade (Easy 1 beat, Normal 1/2, Hard 1/4, at 180 BPM) or the hold rules; released where
   the model puts the end (temperature 0.5). STATS / column `more_holds`.
2. `thin_bars` (`--light P`, tag `_li<P>` after the rests', playtest `liP`), after rest_bars,
   in charts under SR 2.7 only (Easy, Normal): a bar with 3+ onset rows that the model, with
   the bar masked and the chart around it, gives a chance over P of holding one or two onset
   rows keeps the one or two (the likelier count) it likes most, and the notes starting on its
   other rows go (long notes with them). A thinned bar is the chart around the bars after it,
   so light bars can follow each other. STATS / column `thinned_bars`.

Smoke checks (full-v4 on CPU, the 40 val songs of 10-12; the human level of these songs in
brackets):

- `--hold-up` alone on the default's saved charts (run_back's both), the share asked the human
  chart's: 165 taps a song made long notes (Easy 39, Expert 579), hold_share 0.146 -> 0.219
  (0.207); by the human chart's share, 0.35 and more 0.291 -> 0.497 (asked 0.496), 0.2-0.35
  0.206 -> 0.265 (0.260), under 0.2 near as it was (charts over the share keep theirs); SR
  +0.039 (0.35 and more +0.127; Expert +0.15, Easy +0.035), F1 as it was (the onsets stay),
  hold_beats -0.02 (1.26), long notes under the grade's shortest 0.166 -> 0.117 (0.115;
  without the guidelines' minimum for the new ones +0.017 instead), releases within the
  grade's spacing (Easy, Normal) +0.022 (0.121), rows over the Easy cap +0.006 (before the
  cap pass).
- `--light` alone on the same charts: at 0.5 Easy 6.0 bars a song thinned, notes -3.2%, bars
  with one or two onset rows 0.097 -> 0.172 (0.103 on these 8 songs: no gap here), F1 -0.0034,
  SR -0.003; Normal 4.4 bars, notes -1.3%, 0.045 -> 0.078 (0.108), F1 +0.0005, SR 0;
  section_copy -0.010, section_rhythm -0.015, chord_strength -0.005 over all (copies thinned
  apart). At 0.3: Easy 13.1 bars, notes -9.1%, 0.097 -> 0.286, F1 -0.010, SR -0.038, run_over
  25.8 -> 18.4 (20.0); Normal 10.9 bars, notes -2.5%, 0.045 -> 0.118, F1 +0.0003, SR 0. The
  thinned bars hardly move the SR, which the densest stretches set.
- The whole new default from the raw samples with and without `--hold-up`, 5 long-note songs
  (asked 0.46-0.61; Normal to Expert run before the new long notes got the guidelines'
  minimum, which changes little from Normal up): Easy 0.17 -> 0.38 and 0.39 -> 0.44, Normal
  0.39 -> 0.53, Hard 0.40 -> 0.55, Expert 0.29 -> 0.59; SR +0.04 to +0.15 where the chart was under the target (Normal +0.10,
  Expert +0.15), Easy 0 and -0.06; F1 -0.010 to +0.016; Easy releases within a beat of the
  last 0.09 -> 0.26 and 0.16 -> 0.19; 230-740 more passes a song.
- No grammar violations.

`run_space.sh` (patch 0049), on `$T`, 240 val songs:

0. The new default: run_alike's default passes with strong chords 1 and alike 2 with keep
   from SR 4; against run_alike's default.
1. The new default with `--hold-share oracle --hold-up` (the share asked: the human chart's,
   as the stats).
2. `--light 0.5` and `--light 0.3` on the new default's charts (thin_bars comes after the
   rests; the passes after it, the jacks and the chord cap, only take notes out, so on the
   finished charts it is the same but for those).

Predictions (paired means):

0. The new default against run_alike's default (assembled from run_alike, 10-12 above):
   alike_rhythm 0.48-0.50, alike_counts_off 0.13-0.15, section_rhythm 0.385-0.400,
   chord_strength 0.815-0.825, SR bias -0.050 to -0.058 (Hard -0.020 to -0.028, Insane -0.110
   to -0.120), density 1.022-1.026, F1 0.3820-0.3828, 25-35 more passes a song.
1. `--hold-up`: hold_share 0.156 -> 0.19-0.22 (human 0.218); by the human chart's share,
   0.35-0.5 -> 0.37-0.45, 0.5 and more -> 0.50-0.65, under 0.1 within ±0.01; more_holds
   100-220 a song; SR +0.01 to +0.04 (where the share is 0.35 or more +0.05 to +0.15); F1
   within ±0.003; ln_f1 +0.02 to +0.06; hold_beats -0.05 to +0.05; ln_short -0.01 to -0.05;
   release_close +0.01 to +0.04; chord_over in Normal and Hard 0 to +0.005; 150-350 more
   passes a song.
2. `--light 0.5`: Easy 2-8 bars a song thinned, Normal 1-6; notes Easy -1% to -4%, Normal
   -0.5% to -2%; bars with one or two onset rows Easy 0.073 -> 0.09-0.13 (human 0.129),
   Normal 0.027 -> 0.035-0.05 (0.038); light_bars Easy +0.005 to +0.02, Normal +0.01 to +0.03;
   SR Easy 0 to -0.015, Normal 0 to -0.01; F1 Easy -0.001 to -0.005, Normal ±0.002;
   section_copy (Easy, Normal) -0.005 to -0.02. At 0.3: Easy 6-15 bars, notes -4% to -10%,
   one or two rows 0.14-0.25 (past the human), SR -0.02 to -0.05, F1 -0.004 to -0.012,
   run_over -2 to -4.

**Results (`run_space.sh`, morning of 10-12)** — paired means; bars, taps and the by-grade means
are per-song means; 240 songs

| # | prediction | result | |
|---|---|---|---|
| 0 | the new default: alike_rhythm 0.48-0.50, alike_counts_off 0.13-0.15, section_rhythm 0.385-0.400, chord_strength 0.815-0.825 | 0.493, 0.139, 0.396, 0.821 | hit ×4 |
| 0 | SR bias -0.050 to -0.058 (Hard -0.020 to -0.028, Insane -0.110 to -0.120), density 1.022-1.026, F1 0.3820-0.3828, 25-35 more passes | -0.054 (-0.026, -0.114), 1.024, 0.3825, +29 | hit ×7 |
| 1 | `--hold-up`: hold_share 0.19-0.22; by the human chart's share 0.35-0.5 -> 0.37-0.45, 0.5+ -> 0.50-0.65, under 0.1 within ±0.01 | 0.220; 0.417 (human 0.431), 0.693 (0.710), +0.010 | hit (at the edge), hit, miss (more: to the human), hit (at the edge) |
| 1 | more_holds 100-220 a song; SR +0.01 to +0.04 (share 0.35+: +0.05 to +0.15); F1 within ±0.003 | 107; +0.018 (+0.061); +0.0003 | hit ×3 |
| 1 | ln_f1 +0.02 to +0.06; hold_beats -0.05 to +0.05; ln_short -0.01 to -0.05; release_close +0.01 to +0.04 | +0.064; -0.014; -0.053; +0.019 | miss (just over), hit, miss (just over), hit |
| 1 | chord_over in Normal and Hard 0 to +0.005; 150-350 more passes | +0.006, +0.001 (Easy +0.012); +177 | miss (Normal), hit; hit |
| 2 | `--light 0.5`: thinned Easy 2-8, Normal 1-6 bars a song; notes Easy -1% to -4%, Normal -0.5% to -2% | 4.4, 1.6; -2.2%, -0.5% | hit ×4 (Normal at the edge) |
| 2 | bars with one or two onset rows Easy 0.09-0.13 (human 0.129), Normal 0.035-0.05 (0.038); light_bars Easy +0.005 to +0.02, Normal +0.01 to +0.03 | 0.125, 0.044; +0.014, +0.011 | hit ×4 |
| 2 | SR Easy 0 to -0.015, Normal 0 to -0.01; F1 Easy -0.001 to -0.005, Normal ±0.002; section_copy (Easy, Normal) -0.005 to -0.02 | -0.005, 0.000; -0.0005, +0.0004; -0.012, +0.001 | hit, hit, miss (less), hit, hit, miss (none) |
| 2 | at 0.3: Easy 6-15 bars, notes -4% to -10%, one or two rows 0.14-0.25, SR -0.02 to -0.05, F1 -0.004 to -0.012, run_over -2 to -4 | 16.4, -7.9%, 0.246, -0.021, -0.003, -1.1 | miss (more), hit, hit, hit, miss (less), miss (less) |

All runs (per song; human: hold_share 0.218, long notes in long-note bars / among taps 0.483 /
0.191, ln_short 0.146, alike_rhythm 0.622, section_rhythm 0.547, section_copy 0.244,
chord_strength 0.829; Easy / Normal bars with one or two onset rows 0.129 / 0.038):

| run | taps made long / bars thinned a song | hold_share | hold_beats | ln_sections / scattered | ln_short | ln_f1 | Easy / Normal 1-2-row bars | alike / section_rhythm / copy | chord_strength | SR bias (Easy / Normal / Hard / Insane / Expert) | density | F1 | passes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| run_alike's default | - | 0.157 | 0.776 | 0.345 / 0.262 | 0.177 | 0.233 | 0.073 / 0.027 | 0.322 / 0.346 / 0.244 | 0.747 | -0.048 (+0.086 / +0.075 / -0.009 / -0.121 / -0.253) | 1.025 | 0.3821 | 357 |
| the new default (strong chords 1, alike 2 keep from 4) | - | 0.156 | 0.776 | 0.346 / 0.262 | 0.176 | 0.233 | 0.073 / 0.027 | 0.493 / 0.396 / 0.242 | 0.821 | -0.054 (+0.079 / +0.068 / -0.026 / -0.114 / -0.252) | 1.024 | 0.3825 | 386 |
| + hold-up | 107 | 0.220 | 0.765 | 0.458 / 0.192 | 0.122 | 0.295 | 0.073 / 0.027 | 0.495 / 0.390 / 0.232 | 0.818 | -0.036 (+0.100 / +0.094 / -0.014 / -0.096 / -0.228) | 1.022 | 0.3828 | 563 |
| + light 0.5 | 1.1 (Easy 4.4, Normal 1.6) | 0.156 | 0.774 | 0.343 / 0.265 | 0.177 | 0.233 | 0.125 / 0.044 | 0.493 / 0.391 / 0.236 | 0.818 | -0.055 (+0.074 / +0.068 / -0.026 / -0.114 / -0.252) | 1.018 | 0.3825 | - |
| + light 0.3 | 3.7 (16.4, 4.0) | 0.155 | 0.780 | 0.346 / 0.263 | 0.173 | 0.233 | 0.246 / 0.070 | 0.491 / 0.389 / 0.235 | 0.816 | -0.058 (+0.058 / +0.068 / -0.026 / -0.114 / -0.252) | 1.005 | 0.3820 | - |

By the human chart's long-note share, hold-up (the new default -> with it; human): 0.35-0.5 (37
songs) hold_share 0.298 -> 0.417 (0.431), long notes in long-note bars 0.49 -> 0.67 (0.68),
among taps 0.124 -> 0.046 (0.049), 0.58 -> 0.56 beats (0.63), SR bias -0.248 -> -0.221; 0.5 and
more (13) 0.325 -> 0.693 (0.710), 0.49 -> 0.91 (0.91), 0.086 -> 0.010 (0.011), 0.57 -> 0.61 beats
(0.61), ln_f1 0.47 -> 0.72, SR bias -0.262 -> -0.105; under 0.1 (80) +0.010. By grade, SR +0.021
Easy, +0.026 Normal, +0.012 Hard, +0.017 Insane, +0.024 Expert; rows over the chord cap Easy
0.001 -> 0.013 (human 0.005), Normal 0.028 -> 0.034 (0.028); section_copy Easy -0.034, Normal
-0.022, Expert -0.029; Expert alike_rhythm -0.033, section_rhythm -0.037.

What it says:

- **The new default came out as assembled** from run_alike: every number in its range.
- **Long notes as asked.** The long-note share now follows the one asked for at every level:
  long-note charts asked for 0.71 get 0.69 instead of 0.33, and their long notes sit as the
  human ones do (in long-note bars 0.91 against 0.91, among taps 0.010 against 0.011, 0.61 beats
  long against 0.61); over all songs long notes in long-note bars 0.346 -> 0.458 (0.483), among
  taps 0.262 -> 0.192 (0.191), ln_f1 +0.064, long notes under the grade's shortest 0.176 -> 0.122
  (0.146), F1 as it was, and the SR nearer the target from Hard up. Costs: Easy and Normal SR
  +0.02 (over the target already), rows over the chord cap in Easy 0.001 -> 0.013 (the cap pass
  cuts taps, not the long notes held through a row), section copies -0.01 (Expert -0.03, and
  alike_rhythm -0.03 there: the taps made long notes differ between repeats), motion_pred
  -0.0025, 177 more passes a song. Adopted where a share is asked (generate.py `--stats ln=X`),
  with a rule against the cap (DECISIONS 10-12).
- **Light bars at 0.5 bring Easy to the human level**: bars with one or two onset rows 0.073 ->
  0.125 (0.129), light bars 0.037 -> 0.051 (0.056), notes -2.2%, SR -0.005, F1 -0.0005; in Normal
  0.027 -> 0.044 (0.038) and light bars a little over the human (0.057 against 0.045), F1 +0.0004.
  The SR hardly moves: the densest stretches set it. Cost in Easy: section_rhythm -0.017 and
  section_copy -0.012 (still over the human, 0.41 against 0.33), chord_strength -0.012: a bar is
  thinned and its repeats not. At 0.3 past the human (0.246, notes -7.9%). Adopted at 0.5; the
  repeats thinned alike on trial.

### 2026-10-12 · Repeats thinned alike; jack windows in Insane and Expert (predictions)

From run_space: long notes up to the share asked (`--hold-up`) and light bars at 0.5 (`--light
0.5`) are the default (DECISIONS 10-12). The long notes added are now not held through a row at
the grade's chord cap (refine_holds' only_more; the cap pass cuts taps only, and with the
hold-up rows over the cap in Easy went 0.001 -> 0.013, human 0.005).

Measured since (240 val songs):

- The section columns count only pairs of bars with 4+ onset rows, so a bar thinned leaves
  them, and with it most of what light bars seemed to cost (`section_light.py`): counting every
  pair of bars of a stretch that comes back with a note in it, the same rhythm in Easy 0.540
  (the default of run_space), 0.541 with light bars at 0.5 (human 0.580). But of the pairs with
  a light bar in them, 14% have the same rhythm (human 31%; Normal 5% against 40%): a bar is
  thinned and the bar it comes back as is not.
- Jacks: in Insane the default's fast jacks are 0.73 per 100 notes against the human 0.39, and
  a chord is on one side of 66% of them (human 51%; single then chord 26% against 9%); in
  Expert 2.7 against 2.3 (with Expert+), 18% of them 1/6 beat apart against 1% (`jack_kinds2.py`).

Two passes on trial:

1. `thin_bars(s_audio=)` (`--light-alike`, tag `_li<P>a`, playtest `liPa`): the bars that had
   a thinned bar's onset rows and sound like it (cosine 0.5 or more, as the stretches that come
   back), earlier or later, are thinned alike, keeping the same rows.
2. `untangle_window(wide=True)` (`--untangle-wide`, tag `w` after `g`, playtest `ujmgw`): the jack
   pass within 1/4 beat in Insane too and 1/6 in Expert, where a tap with no lane free of a jack
   stays (`untangle_jacks(keep=True)`) instead of going: there the SR is short of the target.

Smoke checks (full-v4 on CPU, the 40 val songs of 10-12; the default's saved charts of
run_back; human level of these songs in brackets):

- The hold-up with the cap rule: rows over the cap as they were (0.0129; without the rule
  +0.0058), hold_share 0.146 -> 0.218 (0.207; without 0.219), Easy 0.132 -> 0.204 (0.191), SR
  +0.034 (Easy +0.015; without +0.035).
- Light bars at 0.5 alone / their look-alikes at 0.8 / at 0.5, Easy and Normal (15 songs): bars
  thinned 5.3 / 6.0 / 8.5 a song; of the pairs of bars that come back with a light bar in them
  the same rhythm Easy 0.088 / 0.088 / 0.143 (human 0.375), Normal 0.010 / 0.059 / 0.098 (0.793);
  every pair with a note, Easy 0.287 / 0.287 / 0.320 (0.518); notes -2.4% / -2.5% / -3.4%; F1
  -0.0016 / -0.0017 / -0.0026. At 0.8 few look-alikes are found: 0.5 it is.
- The jack windows wide, moving only (Insane and Expert, 15 songs): Insane 2.7 taps a song
  moved, fast jacks 0.345 -> 0.192 (0.063), SR +0.000, F1 -0.0004; Expert 1.3 moved, 1.29 ->
  1.24 (1.91), SR +0.005. Taking out the taps with no free lane too (5.7 a song in Insane): 0.047,
  but SR -0.022 there.
- No grammar violations.

`run_wide.sh` (patch 0050), on `$T`, 240 val songs:

0. The new default: run_space's default with long notes up to the human chart's share (and the
   cap rule) and light bars at 0.5, against run_space's default.
1. `--light 0.5 --light-alike` on run_space's default charts, against `--light 0.5` there
   (run_space).
2. The jack pass again on the new default's charts with `--untangle-wide`, against them.

Predictions (paired means; `section_light.py` for the pairs with light bars):

0. The new default against run_space's default, as the hold-up and light bars of run_space
   added (each ± a third of its change) but the cap: hold_share 0.21-0.22 (Easy 0.20-0.22),
   rows over the cap Easy 0.001-0.003 (human 0.005), Normal 0.027-0.031 (0.028); Easy bars with
   one or two onset rows 0.12-0.13; SR bias -0.033 to -0.042 (Easy +0.08 to +0.10, Hard -0.010
   to -0.020, Expert -0.22 to -0.24); F1 0.3824-0.3832; ln_f1 0.28-0.31; 180-220 more passes a
   song.
1. Look-alikes thinned: Easy 5-8 bars a song thinned (alone 4.4), Normal 2.5-5 (1.6); bars with
   one or two onset rows Easy 0.13-0.15, Normal 0.05-0.07 (human 0.129 / 0.038); of the pairs
   with a light bar the same rhythm Easy 0.18-0.25 (alone 0.143), Normal 0.10-0.25 (0.049); every
   pair with a note in Easy +0.01 to +0.03; notes Easy -0.3% to -1.5% more; F1 0 to -0.002; SR 0
   to -0.005.
2. The jack windows wide: Insane 2-5 taps a song moved, fast jacks 0.73 -> 0.40-0.55 (human
   0.39); Expert 1-4 moved, 3.3 -> 3.0-3.3 (1.8); SR within ±0.005; F1 within ±0.001; notes and
   chords as they were.

**Results (`run_wide.sh`, late morning of 10-12)** — paired means; bars, taps and the by-grade
means are per-song means; the pairs of returning bars with a light bar from `section_light.py`;
240 songs

| # | prediction | result | |
|---|---|---|---|
| 0 | the new default: hold_share 0.21-0.22 (Easy 0.20-0.22); rows over the cap Easy 0.001-0.003, Normal 0.027-0.031 | 0.216 (0.211); 0.0016, 0.0294 | hit ×4 |
| 0 | Easy bars with one or two onset rows 0.12-0.13; SR bias -0.033 to -0.042 (Easy +0.08 to +0.10, Hard -0.010 to -0.020, Expert -0.22 to -0.24) | 0.126; -0.040 (+0.091, -0.015, -0.228) | hit ×5 |
| 0 | F1 0.3824-0.3832; ln_f1 0.28-0.31; 180-220 more passes | 0.3828; 0.295; +196 | hit ×3 |
| 1 | repeats thinned alike: Easy 5-8 bars a song, Normal 2.5-5; one or two rows Easy 0.13-0.15, Normal 0.05-0.07 | 8.3, 3.1; 0.149, 0.052 | miss (just over), hit, hit, hit |
| 1 | pairs with a light bar the same rhythm Easy 0.18-0.25, Normal 0.10-0.25; every pair with a note in Easy +0.01 to +0.03 | 0.239 (human 0.308), 0.172 (0.396); +0.009 | hit, hit, miss (just under) |
| 1 | notes Easy -0.3% to -1.5% more; F1 0 to -0.002; SR 0 to -0.005 | -1.7%; Easy -0.0017; Easy -0.005 | miss (more), hit, hit |
| 2 | jack windows wide: Insane 2-5 taps moved, fast jacks 0.40-0.55; Expert 1-4 moved, 3.0-3.3 | 5.1, 0.50; 14.8, 2.79 | miss (just over), hit, miss (more), miss (lower) |
| 2 | SR within ±0.005; F1 within ±0.001; notes and chords as they were | -0.006 (Insane -0.013, Expert -0.037); +0.0001; as they were | miss, hit, hit |

All runs (per song; human: hold_share 0.218, Easy / Normal bars with one or two onset rows
0.129 / 0.038, chord_over Easy 0.005, fast_jacks Insane / Expert 0.39 / 1.82, section_copy 0.244,
alike_rhythm 0.622):

| run | hold_share | Easy / Normal 1-2-row bars | chord_over Easy / Normal | fast_jacks Insane / Expert | section_rhythm / copy | alike | SR bias (Easy / Normal / Hard / Insane / Expert) | density | F1 | passes |
|---|---|---|---|---|---|---|---|---|---|---|
| run_space's default N | 0.156 | 0.073 / 0.027 | 0.001 / 0.028 | 0.73 / 3.35 | 0.396 / 0.242 | 0.493 | -0.054 (+0.079 / +0.068 / -0.026 / -0.114 / -0.252) | 1.024 | 0.3825 | 386 |
| the new default M (hold-up with the cap rule, light 0.5) | 0.216 | 0.126 / 0.044 | 0.002 / 0.029 | 0.73 / 3.28 | 0.384 / 0.227 | 0.489 | -0.040 (+0.091 / +0.079 / -0.015 / -0.096 / -0.228) | 1.018 | 0.3828 | 583 |
| N with light 0.5 | 0.156 | 0.125 / 0.044 | 0.001 / 0.028 | 0.73 / 3.35 | 0.391 / 0.236 | 0.493 | -0.055 (+0.074 / +0.068 / ...) | 1.018 | 0.3825 | - |
| N with light 0.5, repeats alike | 0.155 | 0.149 / 0.052 | 0.001 / 0.028 | 0.73 / 3.35 | 0.387 / 0.234 | 0.489 | -0.056 (+0.070 / +0.068 / ...) | 1.014 | 0.3821 | - |
| M with the jack windows wide | 0.216 | 0.126 / 0.044 | 0.002 / 0.029 | 0.50 / 2.79 | 0.384 / 0.226 | 0.489 | -0.046 (+0.091 / +0.079 / -0.015 / -0.110 / -0.265) | 1.018 | 0.3829 | - |

What it says:

- **The new default came out as the two passes added**, the cap rule doing its part: rows over
  the cap in Easy 0.0016 (with the hold-up alone 0.013, human 0.005), long notes 0.216, Easy's
  light bars 0.126. The returning sections' copies fall a little (0.242 -> 0.227: Easy 0.43 ->
  0.38, towards the human 0.33; Expert 0.22 -> 0.20, human 0.18).
- **Repeats thinned alike make light bars come back as the human ones do**: of the pairs of bars
  of a returning stretch with a light bar in them, the same rhythm in Easy 0.143 -> 0.239 (human
  0.308), Normal 0.049 -> 0.172 (0.396), every pair with a note +0.009. They thin twice as many
  bars, so Easy's light bars pass the human a little (0.149 against 0.129) with notes -1.7% and
  SR -0.005 more (Easy still +0.07 over the target), F1 -0.0017. Adopted.
- **The jack windows wide**: in Insane, moving the chord taps (5 a song) brings the fast jacks
  0.73 -> 0.50 (human 0.39) for SR -0.013; in Expert 15 taps a song move and the SR falls 0.037,
  where it is short by 0.23 already and humans keep 1.8 per 100 notes: the jacks carry the SR
  there. Adopted for Insane only (`--untangle-insane`, 1/4 beat, moved only).

### 2026-10-12 · The default to play: a last check and the playtest of 10-12 (predictions)

From run_wide: the repeats of a thinned bar thinned alike (`--light-alike`) and the jack window
1/4 beat in Insane with the taps moved only (`--untangle-insane`, tag `i`, playtest `ujmgi`;
`sampler.UNTANGLE_INSANE`) are the default (DECISIONS 10-12). Nothing larger is waiting, and the
numbers have started to need the players (the section columns hid the light bars; what is left,
the SR over the target in Easy and Normal and under it from Insane up, the flow of streams, the
sameness of Insane and up, the long notes' releases, is a question of which of them players
notice): the default is held here and played.

`run_check.sh` (patch 0051), on `$T`, 240 val songs: the default (run_wide's M with the repeats
thinned alike and the Insane jack window), against M. Predictions (paired, as the two passes of
run_wide added): Easy 7-9 bars a song thinned, Normal 2.5-4; bars with one or two onset rows
Easy 0.14-0.16, Normal 0.05-0.06 (human 0.129 / 0.038); Insane fast jacks 0.48-0.56 (human
0.39), 4-6 taps a song moved and none taken out there, Expert as it was (3.28 ± 0.02); SR bias
-0.042 to -0.048 (Easy +0.08 to +0.09, Insane -0.105 to -0.112); density 1.012-1.016; F1
0.3822-0.3830; hold_share 0.214-0.218.

The playtest of 10-12 (`run_playtest.sh`, after the check): a pairs pack as on 10-04 and 10-07
(the human chart and the default's, blind, forced choice "which is human" and "which was more
fun"), 16 val songs at SR 1.5-4.5 and at most 150 s, none from an earlier pack (82 left out),
3 of them long-note charts (the human chart's share 0.37-0.45; `pick_playtest.py`), generated
from the audio with the human chart's stats (`st`, the long-note share asked), tag 1012.
Predictions (one player): the AI chart taken for human 1-4 of 16 (0 of 16 on 10-04, 0 of 10 on
10-06); the AI chart more fun in 3-7 of 16; fewer remarks on "random" or "nothing comes back"
than on 10-07 (no bite) and on 10-04 (no patterns), more on the low levels' density and on
jacks from Insane up; the long-note charts not given away by their long notes (10-06: PT13's
human chart gave itself away by its committed long-note style).

## Phase 3 Ablation A: Diffusion Design

_TBD — target Oct 14, 2026._

## Phase 3 Ablation B: Conditioning

### 2026-09-28 · Audio ablation (subset600-v1)

Questions written before the run: audio_ablation: onset CE가 other와 flat에서 얼마나 오를까? shift2는 other보다 더 나쁠까? shift2가 더 나쁘다면 모델이 소리의 종류보다 위치를 보고 있다는 뜻.

`audio_ablation.py`, 800 val chunks, the same masks in every condition; mean
over mask ratios 0.1-0.9 (onset CE: masked cells whose answer is a tap or a
hold start).

| mel | masked CE | onset CE | onset CE at 0.9 |
|---|---|---|---|
| real | 0.110 | 0.693 | 1.213 |
| other song | 0.149 (+0.040) | 1.015 (+0.322, +46%) | 1.611 |
| flat (zeros) | 0.151 (+0.041) | 1.238 (+0.545, +79%) | 2.048 |
| shifted 2 cells | 0.131 (+0.021) | 0.947 (+0.254, +37%) | 1.644 |

Answers: onset CE rises 46% with another song's audio and 79% with none.
shift2 is not worse than other (+37% vs +46%), but a 2-cell (1/6 beat) shift
already costs 80% of what swapping the song costs: the model reads where the
sound is, not only what it is. Another song beats no audio because the input is
beat-aligned, so any song puts its energy on strong beats. Empty and hold-body
cells come mostly from the grammar and the unmasked neighbours (all-cell CE
+0.04 only).

### 2026-09-29 · Audio ablation (full-v1)

Same script, chunks and masks as the subset entry.

| mel | masked CE | onset CE | onset CE at 0.9 |
|---|---|---|---|
| real | 0.090 | 0.571 | 1.097 |
| other song | 0.133 (+0.043) | 0.906 (+0.335, +59%) | 1.578 |
| flat (zeros) | 0.125 (+0.035) | 0.975 (+0.404, +71%) | 1.767 |
| shifted 2 cells | 0.112 (+0.022) | 0.828 (+0.257, +45%) | 1.642 |

Against the subset model: onset CE with real audio fell 18% (0.693 → 0.571)
and with no audio 21% (1.238 → 0.975), so the prior from the grammar and the
neighbours improved as much as the use of audio. Another song's audio now
costs more than no audio on all masked cells (0.133 vs 0.125; the subset had
0.149 vs 0.151): the full model trusts the audio more, and misleading audio
misleads it. The 2-cell shift still costs about 77% of a song swap (subset
79%), and at mask 0.9 it is worse than a song swap in both models (1.642 vs
1.578; subset 1.644 vs 1.611): with little of the chart left, sound in the
wrong place is worse than the wrong sound.

## Phase 3 Ablation C + Multi-key

_TBD — target Nov 15, 2026._
