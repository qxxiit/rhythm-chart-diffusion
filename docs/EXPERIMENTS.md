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

**Results (`run_check.sh`, afternoon of 10-12)** — the default (`F`) against run_wide's M, paired
means; bars and taps a song and the by-grade means are per-song means; 240 songs

| # | prediction | result | |
|---|---|---|---|
| 1 | Easy 7-9 bars a song thinned, Normal 2.5-4 | 8.2, 3.1 | hit, hit |
| 1 | bars with one or two onset rows Easy 0.14-0.16, Normal 0.05-0.06 (human 0.129 / 0.038) | 0.152, 0.053 | hit, hit |
| 2 | Insane fast jacks 0.48-0.56; 4-6 taps a song moved, none taken out; Expert 3.28 ± 0.02 | 0.518; 5.8 moved, none out; 3.281 | hit ×4 |
| 3 | SR bias -0.042 to -0.048 (Easy +0.08 to +0.09, Insane -0.105 to -0.112) | -0.0428 (+0.086, -0.104) | hit (at the edge), hit, miss (just short) |
| 3 | density 1.012-1.016; F1 0.3822-0.3830; hold_share 0.214-0.218 | 1.0136; 0.3825; 0.2161 | hit ×3 |

| | M (run_wide) | F (the default) | human |
|---|---|---|---|
| bars thinned a song, Easy / Normal | 4.3 / 1.7 | 8.2 / 3.1 | |
| bars with one or two onset rows, Easy / Normal | 0.126 / 0.044 | 0.152 / 0.053 | 0.129 / 0.038 |
| light_bars, Easy / Normal | 0.050 / 0.057 | 0.055 / 0.062 | 0.056 / 0.045 |
| jack taps a song, Insane: moved / taken out | 1.0 / 0.6 | 5.8 / 0 | |
| fast_jacks, Hard / Insane / Expert | 0.036 / 0.728 / 3.281 | 0.036 / 0.518 / 3.281 | 0.039 / 0.390 / 1.819 |
| SR bias (Easy / Normal / Hard / Insane / Expert) | -0.040 (+0.091 / +0.079 / -0.015 / -0.096 / -0.228) | -0.043 (+0.086 / +0.079 / -0.015 / -0.104 / -0.228) | 0 |
| density (Easy) | 1.018 (1.086) | 1.014 (1.068) | 1 |
| bar_lane_repeat / section_copy / alike_rhythm | 0.346 / 0.227 / 0.489 | 0.339 / 0.224 / 0.486 | 0.334 / 0.244 / 0.622 |
| F1@50 (Easy) | 0.3828 (0.2593) | 0.3825 (0.2576) | |

What it says:

- **The default came out as the two passes of run_wide added** (13 of 14). The one miss is the
  Insane SR, which fell less than predicted (-0.008 against -0.009 to -0.016): with the window at
  1/4 beat and keep, a tap with no free lane stays, where in M the 1/6-beat window took 0.6 a song
  out; the trial of run_wide ran the wide pass after M's, so it had both. Its fast jacks fall a
  little less for it (0.518 against 0.501 in the trial).
- Against M: F1 -0.0004 (CI -0.0007 to -0.0001; Easy -0.0017), bar_lane_repeat towards the human
  (0.346 -> 0.339; 0.334), light bars over all at the human level (0.0625 against 0.0626).
- **Past the human:** Easy's bars with one or two onset rows (0.152 against 0.129) and Normal's
  (0.053 against 0.038; light bars 0.062 against 0.045). Easy is still denser than the human
  charts (1.068) and over the target SR (+0.086), so the thinning is where the notes are; Normal's
  is a third over. The playtest has 4 Easy and 4 Normal songs (and 6 Hard, 2 Insane): whether the
  thinner bars play as empty is one of its questions.
- The default is held here: `run_playtest.sh` next. The 16 songs (`outputs/playtest_1012_keys.txt`):
  Easy 4 (SR 1.52-1.94), Normal 4 (2.14-2.68), Hard 6 (2.90-3.73), Insane 2 (4.24, 4.50); the
  long-note charts at SR 1.52 (0.37), 3.23 (0.45) and 3.33 (0.39); 45-146 s.

**Results (the playtest of 10-12, evening)** — one player, 7 of the 16 songs (exams; the other 9
after them): Normal 2, Hard 4, Insane 1 (PT08-PT13, PT16); left Easy 4, Normal 2, Hard 2, Insane 1.
`outputs/full-v4/playtest_1012/playtest/ratings.csv`, scored by `outputs/feel/playtest_1012_results.py`
(it prints which chart was the AI's: not before the other 9 are played)

| # | prediction | result | |
|---|---|---|---|
| 1 | the AI chart taken for human 1-4 of 16 | 0 of 7 (95% CI 0-35%) | miss |
| 2 | the AI chart more fun in 3-7 of 16 | 0 of 7 | miss |
| 3 | fewer remarks on "random" or "nothing comes back" than on 10-07 and 10-04 | none; one the other way ("the same things over and over", PT13) | hit |
| 4 | more on the low levels' density and on jacks from Insane up | no Easy song played; Normal's remarks on patterns, the Insane song's on 2-note chords and pacing | miss |
| 5 | the long-note charts not given away by their long notes | the one played (PT11, share 0.39) given away by its chords | hit |

What the remarks blamed, counted in the two charts of that song (AI / human):

| song | remark | AI / human |
|---|---|---|
| PT08 Hard | varied, clear patterns are good; more tap patterns like dfjkjfd, kjfdfjk, djfkdjfk, dkdkdkdfjfjfjf | stairs 108 / 183, jumptrills 90 / 25 |
| PT09 Hard | "too many lasers" (one lane hit over and over) | longest run on one lane 19 / 3, stairs 0 / 40 |
| PT13 Normal | the AI chart repeats the same things and drags; the human's short fast stairs before the downbeat keep it moving; clear patterns in pairs, varied | jumptrills 105 / 12, 3+ note chords 0 / 50 |
| PT16 Insane | the pace (NPS) needs to rise and fall; too fond of "pills" (2-note chords on adjacent lanes), plain repeats of them dull; locks in less | adjacent 2-note chords 219 / 139, jumptrills 177 / 91 |
| PT11 Hard (long notes) | both so-so, the AI's patterns more chaotic | 3+ note chords 74 / 1, jacks 25 / 89 |
| PT10 Hard, PT12 Normal | good charts move their long notes with one instrument; the human chart locks in | no metric yet |

On the 240 val songs (`outputs/feel/playtest_patterns.py`), per 100 onset rows, human / the model
with holes and rests (run_carry's T_ref) / the default F. Rows within half a beat of each other:
a jumptrill is a 2-note chord then another on the other two lanes; stairs (and rolls) three
single notes moving one way; trills three single notes a-b-a; a laser 4+ single notes on one lane.

| | all | Easy | Normal | Hard | Insane | Expert |
|---|---|---|---|---|---|---|
| jumptrills | 4.54 / 6.28 / 6.68 | 0.89 / 1.12 / 1.16 | 3.46 / 5.65 / 5.84 | 5.63 / 8.39 / 8.99 | 6.04 / 7.69 / 8.15 | 6.60 / 7.18 / 7.74 |
| stairs | 6.05 / 5.73 / 5.18 | 5.71 / 6.59 / 6.41 | 6.16 / 6.00 / 5.23 | 5.50 / 4.95 / 4.30 | 6.12 / 5.44 / 4.92 | 7.09 / 6.59 / 5.86 |
| trills | 2.88 / 2.51 / 2.42 | 2.40 / 1.73 / 1.92 | 3.15 / 2.18 / 2.28 | 2.67 / 2.33 / 2.12 | 3.16 / 3.15 / 3.02 | 3.49 / 3.39 / 3.01 |
| 3+ note chords | 6.57 / 6.92 / 7.60 | 0.02 / 0.42 / 0.05 | 1.36 / 2.06 / 2.31 | 8.72 / 9.08 / 9.86 | 10.12 / 10.24 / 11.62 | 9.42 / 10.66 / 11.66 |
| 2-note chords | 30.66 / 29.68 / 31.45 | 19.16 / 17.48 / 19.71 | 34.46 / 33.46 / 35.08 | 35.70 / 34.85 / 36.70 | 31.87 / 31.22 / 32.60 | 31.84 / 28.87 / 30.84 |
| of them adjacent | 17.65 / 17.11 / 17.70 | 10.72 / 10.49 / 11.62 | 19.75 / 19.66 / 20.50 | 20.15 / 20.18 / 20.76 | 18.96 / 17.40 / 17.58 | 18.97 / 16.59 / 16.86 |
| lasers | 0.03 / 0.06 / 0.05 | 0.00 / 0.10 / 0.10 | 0.03 / 0.07 / 0.05 | 0.03 / 0.01 / 0.01 | 0.04 / 0.08 / 0.05 | 0.02 / 0.13 / 0.14 |

What it says:

- **The remarks point one way: the flow among the chords.** The AI charts hold as many 2-note
  chords as the human ones (31.5 against 30.7 per 100 rows; adjacent 17.7 against 17.7), but
  alternate them as jumptrills 1.5 times as often (Normal and Hard 1.6-1.7 times) and run single
  notes as stairs and trills 0.85 times as often. Neither the chord share nor the pattern shares
  nor motion_pred showed it. It is there before the passes of 10-10 on (6.28, the model with its
  lane passes); those, which copy rhythms with the chords on them, add a little (6.68; stairs
  5.73 -> 5.18).
- **Lasers are rare on average (0.05 per 100 rows) but one song's 19 gave it away:** a measure
  to take per song, by its worst, not by the mean.
- **Not yet measured:** long notes moving with one instrument, the "locked-in" accents, the
  pace rising and falling within a section. No remark on "nothing comes back" (10-07's); one the
  other way.
- Next: the flow patterns (fewer jumptrills, more single-note stairs, rolls and trills; the
  longest run on one lane per song), then the long notes against the instruments (held over a
  sustained sound, released where it ends; on 10-07 0.67-0.70 of the releases fell on an onset
  row, human 0.76). The other 9 songs after the exams.

### 2026-10-15 · Where the notes are; rows asked again (predictions)

Asked what it would take to make charts players cannot tell from human ones, three measures on
the stored charts of the 240 val songs (no model; `outputs/feel/phrase_commit.py`,
`rhythm_faith.py`, `rhythm_faith_where.py`), then a probe with the model:

- **Phrase-level commitment is near the human (a guess that failed).** "The model mixes every
  pattern type everywhere instead of picking one per phrase": the mutual information between a
  4-bar phrase and the type of its flow events (two onset rows within half a beat: jack, trill,
  stair, zigzag, jumptrill, chordjack, mixed), above the same with the types shuffled within the
  song, is human 0.274 bits [0.249, 0.298], the default F 0.253 [0.235, 0.273], T_ref 0.248,
  full-v1 with no lane pass 0.247 (210 songs).
- **The lane passes shift the vocabulary.** Shares of the flow events, human / F: jack 1.5 / 0.9%,
  trill 4.3 / 3.4, stair 8.5 / 6.9, zigzag 15.3 / 15.3, jumptrill 6.4 / 9.4, chordjack 10.0 / 9.9,
  mixed 54.0 / 54.2. full-v1 with no lane pass, lane passes at 1, at 0.5: jumptrill 5.0 / 6.1 /
  7.8%, jack 2.1 / 0.8 / 0.3, trill 6.3 / 3.9 / 2.7, stair 8.8 / 10.2 / 10.4.
- **The largest gap is where the notes are.** Of a chart's onset rows, the share with no onset of
  the set's other charts (same audio, same timing, each on its own grid) within 5 ms, per song,
  173 songs with other charts in the set: human 0.051 [0.039, 0.064], F 0.138 [0.117, 0.160],
  T_ref 0.149, full-v1 raw 0.178. By grade, human / F: Normal 0.047 / 0.102, Hard 0.057 / 0.152,
  Insane 0.097 / 0.214, Expert 0.090 / 0.215. The comparison favours the human chart (a set's
  difficulties are made to agree), but not by 2.7 times. Where (per 100 onset rows, pooled),
  human / F: while another chart holds a long note 2.7 / 8.0, in a bar no other chart plays 0.07 /
  0.49, elsewhere (sounds or beats no mapper took) 3.7 / 9.1; of F's such rows 34% on whole or
  half beats (human 22%). F1 against the one human chart could not show this: it has no
  reference for what another mapper would share. It fits "it locks in less" (10-12) and the
  beats off in quiet parts (10-06).
- **The model doubts those notes.** 28 of the songs (Easy 3, Normal 5, Hard 9, Insane 8, Expert 3),
  every row without a long-note start asked with its free cells MASK and the chart around it
  (rows 8 apart together, windows every 4 bars, each answering its middle): p = 1 - prod(1 -
  p(TAP) - p(HOLD_START)) over the masked cells. Of F's 21,152 tap rows 17.9% are on new
  positions:

| F's tap rows, own context | new positions | the rest |
|---|---|---|
| p, mean (median) | 0.593 (0.690) | 0.908 (0.980) |
| below 0.2 / 0.3 / 0.5 | 20.7 / 26.1 / 36.0% | 1.3 / 1.8 / 3.6% |

  AUC 0.833. Below 0.3: 6.2 rows per 100 tap rows, 75% of them new; per song, new positions among
  the tap rows 16.4% -> 12.8%, 93.4% of the tap rows kept (0.5: 9.4 rows, 69% new; 11.3%, 89.8%
  kept). The human charts in their own context: 8.3% new, AUC 0.722, below 0.3 2.6 rows per 100
  (43% new). The empty rows the set's other charts play: p 0.119 on average against 0.012 for
  the other empty rows (AUC 0.823); above 0.7 3.0 rows per 100 tap rows, 59% of them on the
  others' positions (above 0.5: 6.2, 53%).

New: `evaluation.metrics.new_positions` and evaluate.py's columns `new_onsets`,
`human_new_onsets` (NaN without other charts in the set), `set_charts`; the pass
`sampler.recheck_notes` (`--recheck X`: rows with taps below X lose them; `--recheck-add Y`: empty
rows above Y get the tap the model likes most; tags `_rk0.3`, `_rk0.3a0.7`; off by default in
generate.py), after thin_bars, before untangle_jacks. Random-order sampling decides many rows
with little of the chart in view, and later passes copy rhythms around; this asks every row
again with all of it.

`run_recheck.sh` (patch 0054), on F's stored charts, 240 val songs: F rescored (the new columns
for F), F + `--recheck 0.3`, + `--recheck 0.3 --recheck-add 0.7`, + `--recheck 0.5`. Predictions
(new_onsets over the 174 songs whose set has other charts, one more than the measure above
used; the rest per song, all 240):

1. F rescored: new_onsets 0.136-0.140, human 0.049-0.053 (the stored-chart measure, through
   evaluate.py).
2. `--recheck 0.3`: 3.5-6 rows emptied per 100 onset rows; new_onsets 0.105-0.120; notes -3 to
   -6% (density 0.955-0.985); SR bias from -0.043 to -0.09 to -0.20, Easy from +0.086 to
   +0.02-+0.07; F1@50 -0.002 to +0.004, precision@50 +0.005 to +0.015; human_new_onsets as it was.
   Costs: breaks_per_100 from 0.102 to 0.105-0.14, section_rhythm and alike_rhythm 0 to -0.02.
3. `--recheck 0.3 --recheck-add 0.7`: 1.5-4 rows filled per 100 onset rows; new_onsets
   0.110-0.125; density 0.975-1.00; SR bias -0.06 to -0.15.
4. `--recheck 0.5`: 6-10 rows emptied; new_onsets 0.095-0.110; density 0.925-0.96; SR bias
   -0.15 to -0.35.

What would make it the default to play: new_onsets 0.02 or more lower with F1@50 within -0.003
and no new gap measured from the 10-12 playtest's remarks (flow types, lasers); the SR cost from
Insane up (already short of the target) is the open question, and the playtest the judge.

### 2026-10-15 · A judge: telling the charts from the human ones, 8 bars at a time (predictions)

The playtests ask a player which chart of a pair is human; a player has 16 songs in an evening.
New (patch 0055): `scripts/judge.py` trains a judge (`src/models/judge.py`: the full-v4
denoiser on the clean window, a linear score per row averaged over the rows) to tell a run's
charts from the human charts of the same songs, on pairs of 8-bar windows at the same rows
(windows every 4 bars, lanes mirrored at random, 3 epochs), the 240 val songs in 5 folds by
audio, so every song is judged by a judge that never saw it. It then scores the held-out
windows every 8 bars of the human chart, the run's and other runs' (`--score`). The judge
hears the audio (or silence, `--no-audio`) and is given each chart's own SR (rosu), not the
SR asked for. Numbers: auc (a run window above a human window), paired (the run's window above
the human chart's at the same rows: the playtest's question, asked of 8 bars), songs (the same
with the windows of a song averaged), by grade; for a scored run, its mean logit minus F's per
song. `Denoiser.hidden` gives the trunk's last hidden states (forward = head(hidden)).

`run_judge.sh` (after run_recheck): the judge of F with the audio and without, each scoring the
recheck runs, T_ref (10-10) and full-v1 with no pass. ~40 min each on the MacBook. Then
`outputs/feel/judge_results.py`: the rows of F's held-out windows on new positions against
the rest of its onset rows.

Predictions (F against the human charts):

1. With the audio: auc 0.88-0.97, paired 0.85-0.95; songs paired 0.97 or more (the playtest
   of 10-12 found the AI in 7 of 7).
2. By grade, auc rises with the notes in a window: Easy lowest (0.75-0.90), Insane and above
   highest (0.92-0.99).
3. Without the audio: auc 0.02-0.08 lower than with it (what the audio adds: where the notes
   are against the sound, the largest gap measured on 10-15).
4. Scored by the same judges: full-v1 with no pass more AI than F (+0.3 or more per song,
   paired 0.90 or more); T_ref within ±0.3 of F; `--recheck 0.3` below F by 0.02-0.15 per
   song (interval below 0), paired 0.005-0.03 lower; `--recheck 0.5` below F by more than
   0.3 does; 0.3 with taps added in between.
5. Per row (with the audio): F's onset rows on new positions score above its other onset rows,
   auc 0.60-0.75.

How it could fail: the judge finds a tell no player sees (e.g. the exact SR, or the grid's
fine positions of long-note ends) and wins on that alone; then auc is high everywhere and the
scored runs do not move. The rows (5) and the grades say where it looks.

### 2026-11-05 · The checks before a number counts; F without each family; M3; the sampler's settings (predictions)

docs/EVALUATION.md (patch 0056) fixes the question, the endpoints and the systems before
these results. New (patches 0057-0060), none of it in F: `train.py --loss-weight count`,
`--complementary`, `--block-mask` and val_more.csv / log_more.csv for the next model;
`--schedule cosine` in the sampler; `scripts/corrupt_test.py` (the human charts with 4
corruptions x 3 doses, src/evaluation/corrupt.py); `judge.py --shuffle-labels`, and a
`--score` run over fewer songs scored on those it has (the judge still trains on all of
`--ai`); `scripts/judge_vs_diag.py`; the calibration of the expected starts in
`probe_bars.py`; `scripts/results_table.py` (the one table, the diagnostics as the share inside
the human band of the grade). `run_1105.sh`, one phase at a time (240 val songs unless said):

Predictions, meta (~3 h):

1. F's judge with shuffled labels: auc and paired 0.40-0.60 pooled over the five folds (on
   random labels each fold's judge ranks by an arbitrary direction, so a fold or a grade may
   stray further). 0.7+ pooled is a bug in the pipeline, and no judge number counts until it
   is found.
2. The corruption test: new_onsets sensitive to new and timing (rho +1, agree 0.9+), invariant
   to lanes and jumptrill; chord_share sensitive to jumptrill (+); move_stair / move_trill /
   move_jack to lanes; hold_share falls with new and jumptrill (taps added) and is invariant to
   timing and lanes; bar_lane_repeat and section_copy fall with lanes. 25-40% of the columns
   blind to all four (most of the guideline checks and the dynamics columns).
3. F's judge scoring the corrupted human charts (minus F's own mean logit, paired against the
   human chart): rising with the dose for jumptrill (paired 0.60-0.75 at 0.8) and new (0.60-0.80
   at 8%), less for lanes (0.55-0.70 at 0.8), timing near 0.5 (0.50-0.60 at 0.4): F's tells are
   the flow among chords and the new positions, not the one-cell timing.
4. judge_vs_diag on F: R^2 over 5 folds 0.1-0.3, the top gaps among new_onsets, chord_share
   and move_*: most of what the judge sees is in no diagnostic yet.
5. The probe's calibration with a bar masked: expected starts over human 0.95-1.05; in the bars
   with fewer than 1 expected, humans leave 30-60% empty; with nothing in view, 0.90-1.10 over
   all but flatter (the bins under 1 expected hold 20-40% rests).

Predictions, families (~30 min), the mean logit minus F's per song under F's judge
(score phase): without rests -0.05 to +0.15; without reuse and copies +0.05 to +0.30; without
holds -0.10 to +0.10; without chords -0.10 to +0.10; without jacks 0.00 to +0.20. At least one
family's interval includes 0: the first candidate to leave the default.

Predictions, m3 (~3 h): M3 above F by +0.10 to +0.40 per song under F's judge; hold_share inside
the human band 10-25 points less often than F's; M3's own judge auc at or above F's.

Predictions, sweep (~5 h, first 60 songs, F's passes on each): cosine at 128 steps -0.10 to
+0.05 against F, new_onsets inside the band 0-5 points more often; 256 steps -0.05 to -0.15;
lane temperature 0.85-1: the flow among chords closer to the human rate and -0.20 to 0.00
against F, jacks up before the passes untangle them.

Predictions, ladder (~5 h): under F's judge M0 above M1 above T above F, M0 by +0.3 to +1.0
per song; each system's own judge auc 0.90 or more.

How it could fail: the judge finds a tell that no corruption makes (then 3 stays near 0.5
while its auc on F stays high, and 4's R^2 is low: the next diagnostic to look for is what
the judge's rows point at); a dose too weak for a metric (effect under 0.2 sd: neither
sensitive nor invariant, the doses are to be raised, not the criteria lowered).

### 2026-11-07 · run_1105 meta and the judges of 10-15: results; the diagnostics as a judge

`run_1105.sh meta` (240 val songs; predictions in the 11-05 entry) and `run_judge.sh` of 10-15
(predictions in the 10-15 entry), read on 11-06/07. Per-song rows were not read (the playtest
songs stay blind); everything here is aggregate.

**1. F's judge with shuffled labels:** auc 0.503, paired 0.499 pooled, 0.47-0.53 by grade
(34 min). Hit: the pipeline holds, the judge numbers count.

**2. The corruption test:** 32 of 62 columns sensitive to at least one corruption. Hits:
new_onsets (timing +3.96, new +0.80; invariant to lanes and jumptrills), chord_share
(jumptrill +1.17), move_trill and move_jack (lanes +1.26, +2.56), bar_lane_repeat (lanes
-0.74), hold_share invariant to timing and lanes. Misses: move_stair (lanes -0.50 but agree
0.79), hold_share falls with new and jumptrill only by 0.05-0.08 of the spread, section_copy
with lanes at rho -0.77; blind 30 of 62 (48%, predicted 25-40%), 12 of them untestable here
(9 long-note columns, which no corruption touches, and 3 comparison columns with no human
value).

**3. F's judge scoring the corrupted human charts** (paired: the corrupted window above the
human chart's at the same rows; songs: the same with a song's windows averaged):

| corruption | doses | paired | songs | mean logit | predicted at the largest dose |
|---|---|---|---|---|---|
| (F itself) | | 0.912 | 0.971 | +6.27 (human -5.57) | |
| timing (taps one cell over) | 10 / 20 / 40% | 0.932 / 0.947 / 0.939 | 0.988 / 0.988 / 0.988 | +1.87 / +3.58 / +4.27 | 0.50-0.60: miss |
| new (taps on empty rows) | 2 / 4 / 8% | 0.697 / 0.735 / 0.786 | 0.796 / 0.842 / 0.879 | -2.97 / -1.97 / -0.56 | 0.60-0.80, rising: hit |
| jumptrill | 0.2 / 0.4 / 0.8 | 0.571 / 0.562 / 0.431 | 0.617 / 0.633 / 0.467 | -4.41 / -4.99 / -6.19 | 0.60-0.75, rising: miss |
| lanes | 0.2 / 0.4 / 0.8 | 0.373 / 0.336 / 0.338 | 0.354 / 0.308 / 0.338 | -6.86 / -7.38 / -7.62 | 0.55-0.70: miss (the other side) |

One of four. Where the notes start is what the judge reacts to most: a tenth of the taps one
cell over is taken for AI in 98.8% of the songs, more than F itself. Shuffled lanes go the
other way, more human than the human charts; they add jacks (+2.56 of the spread), which F
has fewer of than the human charts (below). Jumptrills by grade: Easy and Normal 0.52-0.70 at
every dose, Expert 0.10-0.16 at 0.4 and 0.8. The prediction's reading ("F's tells are the flow
among chords and the new positions, not the one-cell timing") is half wrong.

**4. judge_vs_diag on F:** R^2 0.05 in sample, -0.03 over 5 folds, every |rho| 0.15 or less;
F's mean logit above the human chart's by +11.67 per song, in 97% of the songs. Miss
(0.1-0.3). Two readings were open: the judge sees what no diagnostic does (a), or it is
saturated and the size of a song's gap says little (b). The diagnostics judge below decides
for (b): the diagnostics together tell F's charts from the human ones nearly as well.

**5. The probe's calibration** (bar masked, oracle stats): expected starts over the human
chart's 0.993 (hit, 0.95-1.05); in the bars with under 1 expected the human charts leave 36%
empty (hit, 30-60%), though below 0.5 expected the model is too sure (0.25-0.5 expected: the
human charts have 1.03 there). With nothing in view: 1.022 (hit) and flatter (bars under 1
expected 1.6% of bars → 0.41%), those holding 43% rests (miss, 20-40%). What the context
carries: rests auc 0.986 → 0.917; bars with long notes 0.954 → 0.820 (within a song 0.916 →
0.699): where long notes go comes mostly from the chart around, little from the sound alone,
which random-order sampling has least of when it decides most cells.

**The judges of 10-15** (F against the human charts; the scored runs as mean logit minus F's
per song, 95% interval):

| | with the audio | without |
|---|---|---|
| F: auc / paired / songs | 0.900 / 0.912 / 0.971 | 0.857 / 0.882 / 0.963 |
| by grade, auc: Easy / Normal / Hard / Insane / Expert / Expert+ | 0.870 / 0.846 / 0.909 / 0.910 / 0.910 / 0.951 | 0.827 / 0.841 / 0.863 / 0.861 / 0.830 / 0.972 |
| `--recheck 0.3` | -0.53 [-1.01, -0.06] | -0.64 [-1.09, -0.19] |
| `--recheck 0.3 --recheck-add 0.7` | -0.81 [-1.22, -0.38] | -0.75 [-1.17, -0.32] |
| `--recheck 0.5` | -0.52 [-1.00, -0.03] | -0.69 [-1.17, -0.21] |
| T_ref (T + fill holes + rests) | -0.27 [-0.75, +0.20] | +0.23 [-0.25, +0.68] |
| full-v1, no pass | -4.23 [-4.89, -3.61] | -4.12 [-4.76, -3.51] |

Predictions (10-15): 1 hit (0.900, 0.912, 0.971); 2 half (Easy 0.870 inside 0.75-0.90 but
Normal lowest; Insane and Expert 0.91, under 0.92-0.99); 3 hit (the audio adds 0.043); 4 one of
five (T_ref within ±0.3; full-v1 far below F, not above; recheck 0.3 lower than predicted,
-0.53 against -0.02 to -0.15; 0.5 no lower than 0.3; 0.3 with taps added the lowest, not in
between); 5 (rows) not read yet. F_corrupt's own judge of F (same seed and data): auc 0.8997,
paired 0.9119: the judge is reproducible on the MacBook.

Read: the chart gives itself away, not its fit to the sound (0.04 of auc). Recheck lowers F by
0.5-0.8 logit, 4-7% of the 11.8 between F and the human charts: real and small. T_ref and F
are the same to the judge: the passes after the rests (reuse, long notes, chords, jacks) do not
move what it sees (the families will say which, if any). full-v1 with no pass, which has more
notes on new positions than F (17.8% against 13.8%, 10-15), scores far below F: the judge has
learned F's own tells, so under F's judge a run unlike F scores lower whatever its quality.
The score phase's numbers for M3 and the sweep need their own judges (M3 has one) or the
diagnostics judge beside them.

**The diagnostics as a judge** (new, `scripts/diag_judge.py`; F's per_song.csv, 55 columns,
no new_onsets in it): logistic 0.872 (paired 0.891), trees 0.935 (paired 0.937), labels
swapped in half the songs 0.50-0.53; by grade (trees) Easy 0.93, Normal 0.87, Hard 0.90,
Insane 0.98, Expert 0.99. Each family of columns alone (trees, a scratch run): long notes
0.84, lanes 0.84, rhythm 0.82. No single column above auc 0.66; band shares 0.73-1.00, median
0.87 (the human charts 0.80); F's spread under the human charts' for 42 of 55 columns (median
0.90x): too typical, inside every band.

| | F | human | F lower in |
|---|---|---|---|
| section_rhythm | 0.380 | 0.547 | 70% |
| alike_rhythm | 0.485 | 0.622 | 64% |
| bar_rhythm_repeat | 0.346 | 0.435 | 73% |
| hold_beats (spread 0.38x) | 0.760 | 1.004 | 67% |
| release_on_onset | 0.694 | 0.759 | 67% |
| move_stair (spread 0.50x) | 0.166 | 0.211 | 62% |
| move_jack | 0.031 | 0.044 | 67% |
| short_holds / quick_regrab | 0 in every song | 0.027 / 0.019 | (the human charts have some in 23% / 22% of songs) |

The trees lean most on move_stair, release_on_onset, move_jack, quick_regrab, short_holds,
hold_beats. F's gap over the 55 columns points the way of the corruptions the judge reacts to
(cosine with each corruption's effects at its largest dose: timing 0.70, new 0.58,
jumptrill 0.16, lanes 0.04). The rhythm repeats less than the human charts' at every scale
in spite of the reuse passes; the long notes have the asked share but not the human shape
(short, alike from song to song; clean_holds' guideline rules leave no short hold and no
quick regrab, which a fifth of the human charts have); fewer stairs and jacks.

Predictions for the diagnostics judge on the families and M3 (written before the families'
results were read), `python scripts/diag_judge.py ${F}_st-oracle $(cat
outputs/run_1105/no_*.path) ${F}_rk0.3a0.7_st-oracle --out outputs/run_1105/diag_judge.md`
(F rescored on 10-15 has new_onsets; all runs on the same columns):

1. F rescored with new_onsets: trees 0.94-0.97, new_onsets among the three it leans on most.
2. Each family out, trees auc minus F rescored's: rests -0.01 to +0.01; reuse +0.01 to +0.03
   (section_rhythm and alike_rhythm lower still); long notes +0.01 to +0.04 (hold_share short
   of the human charts' without the hold-up, the shape as it is); chords -0.01 to +0.01; jacks
   -0.02 to +0.02 (more jacks, toward the human rate, and the untangle's other effects gone).
3. Recheck 0.3 + 0.7: -0.01 to -0.03 (fewer new positions).
4. The five families in the same order by the diagnostics judge and by F's judge in the
   score phase (minus F per song): Spearman 0.5 or more.
5. M3: trees 0.01-0.04 above F (density and long-note share no longer the human chart's).

Also: the sweep phase takes about 8 h, not 5 (two of its six settings are 256 steps, twice
the time of 128).

### 2026-11-07 (2) · The families, M3 and the sweep under the judges; a judge is read for its own run; run_1107 (predictions)

`run_1105.sh families, m3, sweep, score, table` (predictions in the 11-05 entry and the first
11-07 entry) and the diagnostics judge on the families, recheck and M3, read on 11-07.
Aggregate only.

**The families** (F's judge F_1105: mean logit minus F's per song, 95% interval; the
diagnostics judge: trees auc against the human charts, F rescored 0.948):

| F without | F's judge | predicted (11-05) | diagnostics judge | single diagnostics moved away from the human charts (F → without) |
|---|---|---|---|---|
| rests | -0.27 [-0.52, -0.03] | -0.05 to +0.15: miss | 0.951 | steady_bars 0.355 → 0.329 (human 0.414) |
| reuse | +0.19 [-0.20, +0.60] | +0.05 to +0.30: hit | 0.986 | bar_rhythm_repeat 0.346 → 0.110 (0.435), alike_rhythm 0.485 → 0.162 (0.622), section_rhythm 0.380 → 0.117 (0.547) |
| holds | -0.14 [-0.49, +0.19] | -0.10 to +0.10: miss | 0.957 | ln_sections 0.322 (0.483), ln_scattered 0.311 (0.191), ln_bars 0.026 (0.064) |
| chords | -0.29 [-0.49, -0.11] | -0.10 to +0.10: miss | 0.948 | none clearly; chord_over 0.010 → 0.013 (0.012), the cap a guideline |
| jacks | -0.16 [-0.29, -0.05] | 0.00 to +0.20: miss | 0.954 | fast_jacks 0.442 → 0.546 (0.314); move_jack 0.031 either way |

One of five under F's judge (the revised prediction of 11-07 03:00 for reuse, within ±0.3,
also holds). Under F's judge every family is within 0.3 logit of F (2.5% of the 11.8 between F
and the human charts); without chords, rests or jacks the charts look a little less like F to
it. The diagnostics judge sees only reuse (0.948 → 0.986, paired 0.958 → 0.992; the model
alone repeats a bar's rhythm a quarter as often as the human charts). Each family moves at
least one diagnostic toward the human charts or keeps a guideline, so all five stay (the
charter's rule). The jack shortfall of F is not the untangle's (move_jack 0.031 with or
without it), which was the guess of the first 11-07 entry. Diagnostics judge predictions (first
11-07 entry): 1 hit (F rescored 0.948, new_onsets first); 2 three of five; 3 miss: recheck
0.954 (+0.006): new_onsets 0.138 → 0.124 but the repeats it breaks, bar_rhythm_repeat 0.346 →
0.295, alike_rhythm 0.485 → 0.402, section_rhythm 0.380 → 0.325. post_song runs recheck (and
the rests family's fill holes, rests, light bars) after the copies, deciding bar by bar: per-bar
edits belong before the copies or the same for a bar and its copies (an experiment for after
this week's no-new-pass rule).

**M3** (F without oracle inputs): its own judge auc 0.904, paired 0.913, songs 0.983 (F's own
0.900 / 0.912 / 0.975); the diagnostics judge 0.948 (F 0.948); under F's judge -0.83 [-1.50,
-0.18]; hold_share inside the band 0.713 (F 0.821). Predictions (11-05): own judge at or above
F's hit (0.904); hold share 10-25 points less often hit (-10.8); above F by +0.10 to +0.40
under F's judge miss (the other side). The first 11-07 entry's prediction 5 (trees +0.01 to
+0.04) miss: 0.000. Without the human chart's density and long-note share the charts are as
human to both judges that learned them.

**The sweep** (first 60 songs, F's passes on each; F's judge, minus F on the same songs):

| setting | F's judge | predicted (11-05) |
|---|---|---|
| cosine, 128 steps | -0.64 [-1.66, +0.45] | -0.10 to +0.05 |
| linear, 256 steps | -1.21 [-2.30, -0.05] | -0.05 to -0.15 |
| cosine, 256 steps | -1.46 [-2.56, -0.42] | |
| lane temperature 0.7 / 0.85 / 1 | -1.04 / -1.16 / -1.05 (intervals about ±0.8) | 0.85-1: -0.20 to 0.00 |

All lower than predicted and alike: no dose response in the lane temperature. The sweep runs are
new draws with today's code, F and T draws of 10-10; there was no draw of F's own settings
beside them (a gap in run_1105.sh), so the settings cannot be told from the redraw. The
results table set them (60 songs) beside F (240): their band shares, SR error and judge auc do
not compare (`results_table.py --songs-of` now restricts every row to one run's songs, the
judge columns again from each judge's windows; `diag_judge.py` judges the runs of one call on
the songs they share).

Read: a judge is read for the run it learned. F's judge put M3 0.83 below F though M3's own
judge and the diagnostics judge found M3 and F alike; it put full-v1 raw 4.2 below F (10-15)
with more notes on new positions; it does not see the repeats that reuse adds (a +0.19 that
includes 0, against 0.95 → 0.99 for the diagnostics), the tell F itself barely has (0.35
against 0.44) and so never learned. It shows how much of F's tells a run has, no more (DECISIONS
2026-11-07, EVALUATION.md). Runs are compared by their own judges' auc or by the diagnostics
judge, on the same songs.

Predictions for `run_1107.sh` (ctrl, sweepj, diag, town, ladder, table; written before any
of it ran), with the judge of no_reuse already running:
1. no_reuse's own judge, F scored by it: if an 8-bar judge sees the repeats, auc 0.93-0.98 and
   F below no_reuse by 1 logit or more per song; if not, auc near 0.90 and F within ±0.3.
2. The control (F's settings and seed drawn again today, 60 songs) under F_sweep: within ±0.4
   of F. At -0.6 or lower, the sweep's -1 was the redraw, not the settings.
3. Under F_sweep, each sweep run minus the control: cosine -0.3 to +0.3; 256 steps -0.5 to
   +0.1; cosine and 256 -0.6 to 0.0; lane temperature 0.7-1 -0.3 to -1.0 (the judge takes lane
   variety for human, as with the shuffled lanes). F_1105 and F_sweep agree within 0.3 on each.
4. The diagnostics judge on the 60 songs: every sweep run within ±0.03 of the control (60 songs
   cannot tell less); lane temperature 1 inside the human band of move_trill and move_jack 5
   points or more often than the control.
5. T's own judge: auc 0.91-0.96, above F's 0.90 (the passes take out some of what a judge
   learns).
6. The ladder: own judges M0 0.95-0.99, M1 0.93-0.98, both above T's and F's; under F's judge
   M0 and M1 below F by 1 to 5 logit (as full-v1 raw), against the 11-05 prediction (M0 above
   F by +0.3 to +1.0), which stands as written.

### 2026-11-08 · run_1107: the control, the sweep, the ladder of own judges; cosine on all songs (predictions)

`run_1107.sh` (11-07 18:20 to 11-08 04:52; predictions in 11-07 (2)), and no_reuse's own judge
before it. Aggregate only.

**no_reuse's own judge:** auc 0.925, paired 0.939, songs 0.992 (F's own 0.900 / 0.912 / 0.975);
F scored by it -0.95 [-1.26, -0.63] per song. Prediction 1 lands between its branches (0.93 and 1
logit not reached, ±0.3 far off): an 8-bar judge sees the repeats once it learns from charts
without them. F's judge missed them because F barely lacks them.

**The control** (F's settings and seed drawn again, 60 songs, F's passes): under F_sweep +0.000
[+0.000, +0.000], paired 0.500, and the diagnostics judge's numbers of F to the digit: the same
charts. Drawing is deterministic and the code of that path has not changed since 10-10; runs with
the same seed are exact pairs. Prediction 2 hit; the sweep's offsets are the settings'.

**The sweep** against it (F_sweep, minus F per song; F_1105 within 0.05 of each, the same seed and
data, so a check of reproduction, not a second judge; F's judge on the same 60 songs: auc 0.919):

| setting | F's judge | auc (60 songs) | per song against F, diagnostics (95% interval) |
|---|---|---|---|
| cosine, 128 steps | -0.60 [-1.62, +0.49] | 0.900 | new_onsets -0.015 [-0.029, -0.004], hold_beats +0.050 [+0.006, +0.097], move_jack -0.014 [-0.032, -0.003] |
| linear, 256 steps | -1.20 [-2.28, -0.03] | 0.897 | new_onsets -0.010 [-0.026, +0.003], release_on_onset +0.028 [+0.001, +0.057], move_jack -0.015 [-0.034, -0.002] |
| cosine, 256 steps | -1.42 [-2.53, -0.37] | 0.888 | new_onsets -0.016 [-0.026, -0.006] |
| lane temperature 0.7 / 0.85 / 1 | -1.03 / -1.14 / -1.04 | 0.886-0.890 | new_onsets ±0.001; motion_pred -0.017 / -0.025 / -0.034 (F 0.123, human 0.129) |

Prediction 3 missed on every range (all a little lower); 4: the diagnostics judge on 60 songs
(trees 0.855-0.920 against the control's 0.889) is as loose as its label-swapped check (0.52-0.65),
so the per-song pairs above are what counts. The schedule moves both judges and the diagnostics the
same way: about a quarter of F's excess of notes on new positions (0.116 against the human 0.052 on
these songs) gone, long notes a little nearer, jacks fewer. The lane temperature leaves the rhythm
alone and makes the lanes less predictable than the human charts': F's judge likes that as it liked
shuffled human lanes, the diagnostics do not. Not taken.

**The ladder** (240 songs; own judges; F's judge F_ladder, minus F; the diagnostics judge, trees,
56 columns, T on its 42):

| | own judge auc / paired / songs | F's judge | diagnostics judge |
|---|---|---|---|
| M0 (the model, random order, no pass) | 0.930 / 0.925 / 0.967 | -2.81 [-3.50, -2.12] | 0.993 |
| M1 (+ lanes forward, refine 2) | 0.938 / 0.944 / 0.988 | -0.10 [-0.67, +0.46] | 0.990 |
| T (+ loud bias, onset gate and taper) | 0.932 / 0.941 / 0.992 | -0.11 [-0.59, +0.36] | 0.986 (42 col.) |
| F without reuse | 0.925 / 0.939 / 0.992 | +0.19 [-0.20, +0.60] | 0.986 |
| F | 0.900 / 0.912 / 0.975 | | 0.948 |
| M3 (F without oracle inputs) | 0.904 / 0.913 / 0.983 | -0.83 [-1.50, -0.18] | 0.948 |

M0 → M1 → F (human): new_onsets 0.168 → 0.168 → 0.138 (0.051); motion_pred 0.046 → 0.118 → 0.121
(0.132); move_trill 0.219 → 0.151 → 0.175 (0.153); move_jack 0.049 → 0.027 → 0.031 (0.044);
bar_rhythm_repeat 0.102 → 0.102 → 0.346 (0.435); hold_share 0.170 → 0.170 → 0.216 (0.218).
Predictions: 5 hit (T 0.932); 6 half (M1 0.938 hit; M0 0.930, not 0.95-0.99 and not above T; under
F's judge M0 -2.81 hit, M1 -0.10 miss); 11-05's ladder (M0 above F by +0.3 to +1.0 under F's
judge) miss.

Read: the passes and sampler options carry the diagnostics a long way (0.99 → 0.95) and the judges
a short one (0.93 → 0.90, of which reuse 0.025). What a judge still finds in F is mostly what they
do not touch: where the notes start (0.168 → 0.138 against 0.051), the strongest direction of the
corruption probe. The lane refinement brings lane predictability and trills to the human level and
takes jacks below it (0.049 → 0.027): F's jack shortfall is the refinement's at temperature 0.5,
not the untangle's. F's judge reads that refinement's signature: M0 -2.81 under it, though its own
judge and the diagnostics find M0 the easiest to tell (11-07, a judge is read for its own run).

Predictions for `run_1108.sh` (cosine at 128 steps on all 240 songs, F's passes, its own judge
and the diagnostics judge with F rescored; written before it ran):
1. Against F per song: new_onsets -0.010 to -0.020, move_jack -0.005 to -0.015, hold_beats +0.02
   to +0.06.
2. Its own judge auc 0.880-0.900 (F 0.900): at most a small step, new positions falling by a fifth
   to a quarter of the excess.
3. The diagnostics judge (trees) within ±0.01 of F's 0.948.
How it could fail: an own judge at or above F's would say the remaining tell is not how many notes
sit on new positions but how they sit (the one-cell timing of the probe), which a schedule leaves.

run_1108 was restarted at song 19 with three processes sharing the GPU (`evaluate.py --shard K/N`
and `scripts/merge_shards.py`): one process drew a song in 65 s with the CPU at 10% and the GPU
mostly waiting on it. The seeds go by the song's index in the full run, so the shards' charts are
the one-process run's (tests/test_pipeline.py checks charts, rows and means).

### 2026-11-08 (2) · run_1108: cosine on all songs; what the judge reads; views (predictions)

`run_1108.sh` (11-08 06:06 to 09:44 on three processes: the draw merged 08:51, F's passes 09:10,
its own judge 09:43 in 1,990 s). Aggregate only. Against F per song (240 val songs, the same seeds;
95% intervals over songs):

| | F | cosine | human | prediction | |
|---|---|---|---|---|---|
| new_onsets | 0.138 | 0.118 (-0.020 [-0.034, -0.010]) | 0.051 | -0.010 to -0.020 | hit: a quarter (23%) of the excess gone |
| move_jack | 0.031 | 0.033 (+0.002 [-0.006, +0.011]) | 0.044 | -0.005 to -0.015 | miss |
| hold_beats | 0.755 | 0.771 (+0.017 [-0.012, +0.046]) | 1.004 | +0.02 to +0.06 | just under (before the passes +0.040 [+0.012, +0.070]) |
| own judge auc | 0.900 | 0.893 (-0.007 [-0.026, +0.011]) | | 0.880-0.900 | in range, within the noise |
| diagnostics judge, trees / logistic | 0.948 / 0.887 | 0.953 / 0.871 | | trees 0.948 ±0.01 | hit |

The failure branch (own judge at F's or above) is not taken, but the step is inside its interval:
"a small step" is not shown either. Of the sweep's 60 songs (the first 60 of these, and the same
charts to the byte, 60 of 60: the shards check on a real run), the jacks fell there (-0.014) and
rose on the other 180 (+0.007); long notes +0.050 there, +0.014 on the others; new positions -0.015
there, -0.022 on the others. What the sweep found with an interval just clear of 0 did not hold,
the new positions did. Of the 56 diagnostics, 13 moved outside their interval: nearer the human
charts alike_rhythm 0.480 → 0.532 (0.619), copy_far 0.046 → 0.060 (0.068), motion_pred 0.121 →
0.127 (0.132), fast_jacks 0.442 → 0.370 (0.314); further coverage 0.050 → 0.059 (0.042),
coverage_p2 (trill and jumptrill runs) 0.019 → 0.024 (0.010), loud_slope 0.206 → 0.216 (0.194),
and rest_bars 0.0072 → 0.0100 (0.0075) and light_bars 0.0625 → 0.0700 (0.0626) past the human
level. The mean gap over the 56, in human sd, 0.183 → 0.168 (0.363 → 0.348 before the passes).
Taken as the base (DECISIONS 2026-11-08): the same 128 steps, no cost.

**Why the judge barely moves when the new positions fall** (aggregate checks on run_1108's
charts and the judges' rows):
- `judge_vs_diag.py`: per song, no diagnostic explains how sure F's judge or cosine's is (ridge on
  the 8 nearest gaps, R^2 over 5 folds -0.03 and -0.12; every |rho| 0.25 or less; new_onsets
  +0.12 / +0.09). The judges are close to sure everywhere (+11 logit a song, AI above in 98-99% of
  the songs), which leaves this check little to work with.
- Snap (where in the beat's 12 cells the onsets sit, the 174 songs with one timing point): 1/1
  0.46, 1/2 0.33, 1/4 0.146 (human 0.156), 1/3 0.029 (0.034), 1/6 0.013 (0.016), 1/12 0.013
  (0.016); beats that mix 1/4 with a triplet 0.027 (0.038). The AI divides the beat as the
  mappers do; per window the judge's difference follows none of these by more than rho 0.14.
- The rows (rows.npz; a window's logit is the mean of its rows' scores): in the AI windows every
  kind of row scores about the same, +5 to +7 (F's judge: an AI onset on a row where the human
  chart has one +6.5, on a new position +6.3, an empty row +6.5, a long note's row +6.6), in the
  human windows -2 to -7. Onsets on new positions carry 0.14 of the 11.9 logits between the
  charts (1%). The judge does not point at notes; the rows only repeat the window's verdict.
- F's judge without audio (F-noaudio): 0.857 against 0.900 with it: most of the telling is in
  the chart.

So what is left is a property of the 8 bars as a whole, which taking a quarter of the excess
notes off new positions hardly changes. Which part of the chart carries it is the next question.

**Predictions for `run_1109.sh`** (written before it ran). The cosine run judged again with part
of every chart, the human one's too, taken away alike (`judge.py --view`), every chart shown with
the human chart's SR (the SR of the whole chart would tell what a view hides), a new judge each:
nested, each removes one more thing.
1. full (everything, the human SR): auc 0.875-0.900. Under 0.86 would make the SR a tell.
2. taps (long notes as taps): 0.80-0.88, a little carried by the holds (release_on_onset is
   among the three the diagnostics judge leans on; hold_beats 0.77 against 1.00).
3. rhythm (taps on lanes drawn at random: which rows have notes and how many): 0.72-0.85.
4. deaf (rhythm without audio): 0.62-0.78.
5. Epochs (--eval-epochs): the full view's auc after its first epoch within 0.01 of its third.
Readings: full minus taps is what the holds carry, taps minus rhythm the lanes, rhythm minus
deaf the rhythm against the music, deaf minus 0.5 the rhythm's own pattern. How it could fail:
rhythm at 0.86 or more puts nearly all of the tell in which rows get notes (the onset model, not
the passes, is the lever); taps under 0.65 puts it in the holds, rhythm under 0.65 with taps
above 0.86 in the lanes.

**Time from now on.** Every run says with its predictions when it may stop early, and is cut
there:
- A judge stops after a fold, from the second on, once its auc and 95% interval over songs lie
  between the same two bounds where the reading changes (`--stop-bounds`; run_1109: 0.65 and
  0.86). Two folds (~96 songs) give about ±0.03, enough for bounds 0.2 apart; ~14 min a judge
  instead of ~35. The songs of the folds run are reported (`judge_table.py` compares judges on
  the songs they all have). Stopping on an interval is optional stopping, so these are readings
  between bounds, not tests.
- A chain skips what an earlier answer makes moot: the rhythm view is not run when taps leaves
  little (under 0.65; it can only leave less), the deaf view not when rhythm does.
- Epochs: measured before they are cut (`--eval-epochs` scores the held-out songs after each
  epoch; the training is the same, tests/test_pipeline.py). If one epoch does, judges take a
  third of the time.
- Shards are shared by length (`evaluate.py shard_of`, longest first to the least loaded):
  run_1108's by index ended 15 min apart (shard 0 had 94 min of one-process work, the others 80
  and 84). Each process drew a song at x1.9-2.0 its one-process time, steady through the run
  (not heat), so three processes give x1.5: one process already used about two thirds of the
  GPU, not "mostly waiting" as written above.
- Draws next: an interim reading at a third and two thirds of the songs against the prediction
  (the rows as they are written), when the next draw is planned.

### 2026-11-08 (3) · run_1109: the views, and the SR they were shown with; again with the SR as shown (predictions)

`run_1109.sh` (11-08, 16:27 to 17:23). Every judge stopped after 2 of its 5 folds (14 min each,
four in ~58 min against ~2.3 h with all folds). On the 96 songs of those folds, each row's step
against the row above with its 95% interval over songs (`judge_table.py`):

| judge | shown | auc | step |
|---|---|---|---|
| cosine (11-08 (2), own SR, all folds) | everything | 0.901 | |
| view_full | everything, the human chart's SR | 0.944 | +0.043 [+0.025, +0.063] |
| view_taps | long notes as taps, human SR | 0.911 | -0.033 [-0.046, -0.020] |
| view_rhythm | taps on lanes drawn at random, human SR | 0.838 | -0.073 [-0.088, -0.057] |
| view_deaf | rhythm without audio, human SR | 0.785 | -0.053 [-0.075, -0.033] |

Prediction 1 missed upward (0.875-0.900), and it shows a flaw in the design (11-08 (2)): the
human chart's SR is the SR the run was asked for, so every AI chart was shown with a target it
misses by 0.25 on average (sr_error), and the judge read the mismatch, as src/models/judge.py
warns ("a judge given the request could tell 'too hard for what was asked'"). Every view had
that tell, so their levels are not the chart's: predictions 2-4 (taps 0.80-0.88 at 0.911,
rhythm 0.72-0.85 at 0.838, deaf 0.62-0.78 at 0.785) cannot be read. The steps are measured
under the same tell and all clear 0: the holds 0.033, the lanes 0.073, the audio 0.053.

The mismatch is a tell in its own right: a judge that knows the request finds the AI's
difficulty off, +0.043 on the same songs. A playtest that names the grade may let a player find
it as well; sr_error counts where the request is in view.

Epochs (prediction 5, missed): the full view's first epoch 0.918 against its third's 0.944. The
second is within 0.002 of the third in three judges (0.945 / 0.944, 0.913 / 0.911, 0.836 /
0.838) and 0.008 under in the fourth (0.777 / 0.785), with the learning rate still at a quarter
of its peak there (cosine decay): judges train 2 epochs from run_1110 (DECISIONS 2026-11-08).

**Predictions for `run_1110.sh`** (the same views, each chart with rosu's SR of the chart as
shown, `--sr view`; 2 epochs; written before it ran). On the same 96 songs:
1. full: 0.885-0.910 (the cosine run's own judge 0.901 there).
2. taps 0.86-0.90; 3. rhythm 0.78-0.86; 4. deaf 0.72-0.80.
5. The steps as under the human SR, which every view had: the holds -0.01 to -0.04, the lanes
   -0.04 to -0.09, the audio -0.03 to -0.07.
Readings as in 11-08 (2). How it could fail: full outside 0.88-0.92 would make the source of the
SR matter (own: the human SR from the .osu file, the run's from its tokens; view: both from
tokens).

### 2026-11-08 (4) · run_1110: what the judge reads, with the SR as shown; the rhythm against the music; alike bars (predictions)

`run_1110.sh` (11-08, ~18:36 to 19:29). On the 96 songs of the first two folds, each row's step
against the row above (95% over songs, `judge_table.py`):

| judge | shown | auc | step | prediction |
|---|---|---|---|---|
| view_full (run_1109) | everything, the human chart's SR | 0.944 | | |
| cosine (11-08 (2)) | everything, own SR | 0.901 | -0.043 [-0.063, -0.025] | |
| srv_full | everything, the SR as shown | 0.909 | +0.008 [-0.002, +0.019] | 0.885-0.910: hit |
| srv_taps | long notes as taps | 0.840 | -0.069 [-0.109, -0.034] | 0.86-0.90, step -0.01 to -0.04: miss |
| srv_rhythm | taps on lanes drawn at random | 0.742 | -0.098 [-0.147, -0.051] | 0.78-0.86, step -0.04 to -0.09: miss |
| srv_deaf | rhythm without audio | 0.606 (240 songs, 5 folds: 0.641 [0.596, 0.688]) | -0.136 [-0.168, -0.104] | 0.72-0.80, step -0.03 to -0.07: miss |

The source of the SR does not matter (+0.008, the interval holds 0): the own-SR judges read
as they ran. Of the full view's 0.409 above chance, removed in this order: the holds 0.069
(17%), the lanes 0.098 (24%), the rhythm against the music 0.136 (33%), the rhythm's own
pattern 0.106 (26%). Every step is larger than predicted, about twice its size under the
human SR (11-08 (3)), whose tell held up every view. And they overlap: without audio the full
view loses 0.043 (F-noaudio 0.857 against 0.900), the rhythm view 0.136, so the same tell
rides on the lanes, the holds and the rhythm against the music at once, and taking one away
leaves the others to find it. That is why one pass, or a quarter of the new positions, hardly
moves the judge. The largest single piece is where the notes sit against the music; the
rhythm's own pattern is near the human one (0.61-0.64 without audio). Time: 9 min a judge at 2
epochs and 2 folds; srv_deaf sat on its bound 0.65 and ran all 5 folds (23 min) though
nothing after it depended on it (DECISIONS 2026-11-08: such a judge runs `--max-folds 2`).
Epoch 1 within 0.007-0.013 of epoch 2.

**The rhythm against the music** (no run: the cosine run's charts and the human ones, the
3-band rise of the log-Mel (bands 0-19, 20-49, 50-79) at each row, its largest within ±2
frames, z within the song; beat positions on the 174 songs with one timing point). Per song,
AI minus human with 95% over songs:

| | human | AI | AI - human |
|---|---|---|---|
| onset rows | 0.196 | 0.186 | -0.010 [-0.014, -0.006] |
| the sound under a note (rise, z) | +0.205 | +0.224 | +0.019 [+0.003, +0.039] |
| rows of the loudest 10% of rises with a note | 0.271 | 0.269 | -0.002 [-0.010, +0.007] |
| the louder rows chosen within a beat position (auc): beat / 1/2 / 1/4 | 0.686 / 0.632 / 0.589 | 0.695 / 0.629 / 0.582 | +0.009 / -0.003 / -0.007, each holding 0 |
| the same within triplet positions | 0.612 | 0.569 | -0.042 [-0.072, -0.014] |
| bars alike in sound (top quarter, inside 8 bars) with the same rhythm | 0.419 | 0.354 | -0.064 [-0.094, -0.036] |
| of those, one or two rows apart | 0.277 | 0.321 | +0.044 [+0.019, +0.071] |

By position and loudness (pooled): notes on beats and half beats at 0.98-1.01 of the human
rate in every third of loudness; 1/4 at 0.86-0.92, triplets 0.70-0.75, 1/12 0.61-0.69, loud
or quiet alike; the half-beat chords a little larger (1.36-1.47 notes against 1.29-1.37). Notes
on new positions sit on quiet sounds in both (z -0.12 AI, -0.15 human; on the set's positions
+0.28, +0.24). So the AI follows the sound as closely as the mappers do; what differs is small:
the same rhythm when the music repeats (two thirds of that gap near-copies, one or two rows
apart, which align_alike at 2 nats leaves), fewer of the fine divisions, triplets that follow
the sound less. None large: the tell is spread, as the views said.

**Predictions for `run_1111.sh`** (the cosine draw with F's passes again, align_alike at 6
nats instead of 2; its own judge as srv_full, the same 2 folds; written before it ran):
1. Bars alike in sound with the same rhythm 0.354 → 0.38-0.42 (human 0.419); one or two rows
   apart 0.321 → 0.28-0.30. alike_rhythm 0.532 → 0.56-0.60 (0.619).
2. Its own judge against srv_full on the same 96 songs: a step of -0.015 to +0.005. A step of
   -0.015 or more (its interval clear of 0) makes repetition a piece the judge sees, worth a
   better pass; near 0 confirms the overlap, and the next lever acts on the whole window.
3. Onset rows, new_onsets and SR error within ±0.005 of the cosine run.

### 2026-11-08 (5) · Windows chosen by a judge (predictions)

The tell is spread over the 8-bar window and rides on several parts at once (11-08 (4)), so a
pass that fixes one part leaves the others to give the chart away. What sees the whole window
is a judge. `evaluate.py --select-judge J --select-k 4` draws every window of the song 4 times
(`sampler.sample_windows`: the 4 go through the denoiser together, one forward pass of 4 a
step) and keeps the draw the song's judge scores lowest (`src/models/select.py`). The judges
are a `judge.py --save-models` run's fold judges, each scoring only the songs its fold held
out, so a song is chosen for by a judge that never saw it. What a window's drawing decides is
its rhythm (the lane passes after it choose the lanes again), so the choosing judge is trained
on the rhythm view of the cosine draw before F's passes, with the human chart's SR (the SR a
draw is asked for). The judge that chooses never scores the result (docs/EVALUATION.md): fresh
judges with another seed and fold split, trained on the chosen run and on the cosine run, the
same 120 songs, all 5 folds each. `run_1112.sh`: the choosing judge, the first 120 val songs
drawn so on three processes, F's passes, the fresh judges, the steps.

Stopped early (`scripts/select_watch.py`): after 30 songs, if the mean gain of the choice (a
window's chosen draw under the mean of its 4, in the judge's logits) is under a tenth of the
judge's gap between the run's windows and the human ones, the draws of a window are too alike
for a choice to carry the chart far, and the drawing stops there (STOPPED next to the first
shard's log); nothing after it runs.

Predictions (written before it ran):
1. The choosing judge (rhythm view, human SR, the cosine draw before the passes, 5 folds): auc
   0.78-0.88.
2. The gain over the first 30 songs: 0.3-1.0 logits a window, 10-35% of that judge's gap: the
   drawing goes on.
3. The fresh judges, the chosen run minus the cosine run on the same 120 songs: full view
   -0.01 to -0.04, rhythm view -0.02 to -0.06.
4. Notes within ±3% of the cosine run; sr_error -0.00 to -0.05 (the choosing judge reads the
   difficulty against the SR asked for); new_onsets 0 to -0.015.
5. A song at 1.5-2.5 times the cosine draw's time a process.
How it could fail: the gain under a tenth stops it, and the tell is the model's rather than
the draw's (retraining is the lever); a large gain and no step under the fresh judges, and the
choice learned the choosing judge's own quirks.

### 2026-11-08 (6) · run_1111 and run_1112: alike bars at 6 nats; windows chosen by a judge; 8 draws on all songs (predictions)

`run_1111.sh` (about 20:05 to 20:41). align_alike at 6 nats instead of 2, F's passes on the cosine
draw; its own judge as srv_full (the same 96 songs):

| | cosine (2 nats) | 6 nats | difference (95%) | human | prediction |
|---|---|---|---|---|---|
| own judge auc | 0.909 | 0.922 | +0.013 [+0.005, +0.022] | | -0.015 to +0.005: miss, the other way |
| alike_rhythm (bars alike in sound anywhere) | 0.532 | 0.582 | +0.050 [+0.038, +0.062] | 0.622 | 0.56-0.60: hit |
| bars alike in sound inside 8 bars, the same rhythm | 0.354 | 0.352 | -0.002 [-0.005, +0.001] | 0.419 | 0.38-0.42: miss |
| of those, one or two rows apart | 0.321 | 0.324 | +0.003 | 0.277 | 0.28-0.30: miss |
| bars aligned, a song | 5.0 | 6.5 | +1.5 | | |
| new_onsets, density, SR error | | | +0.0002, +0.24%, +0.002 | | within ±0.005: hit |

The pass aligns bars alike in sound wherever they are (audio cosine 0.8+), so the near-copies
inside 8 bars stay; and the 1.5 bars a song more it aligns (a rhythm taken over, its lanes
chosen again) make the charts easier to tell, not harder. Not taken; whether align_alike at
2 nats costs the judge too is a pass-family question for later (`--alike` off, ~35 min).

`run_1112.sh` (20:41 to ~23:40). The choosing judge (the rhythm view of the cosine draw before
F's passes, the human SR, 5 folds kept): auc 0.877 [0.864, 0.889], its gap between the run's
windows and the human ones 5.58 logits (prediction 1, 0.78-0.88: hit). The 120 songs drawn
with 4 draws a window (21:06 to 22:33): the chosen draw lay 2.45 logits under the mean of its
window's 4 on average, 45% of the gap (the watch read 2.53 after 30 songs and let it go on;
prediction 2, 0.3-1.0 and 10-35%: miss, the draws of a window differ far more than
expected), and a song took x1.06 the plain draw's time a process (prediction 5, 1.5-2.5:
miss: the 4 draws go through the denoiser in one pass, and the GPU at one window a pass had
the room). Fresh judges (seed 1, all 5 folds of the 120 songs, trained on 96 songs a fold):

| on the same 120 songs | cosine | chosen | difference (95%) | prediction |
|---|---|---|---|---|
| full view auc | 0.858 | 0.818 | -0.040 [-0.079, -0.001] | -0.01 to -0.04: hit, at the edge |
| songs (each song's windows averaged) | 0.967 | 0.900 | | |
| density | | | -2.2% [-3.6, -0.7] | ±3%: hit |
| new_onsets (human 0.051) | 0.115 | 0.100 | -0.015 [-0.040, +0.004] | 0 to -0.015: at the edge |
| sr_error | 0.185 | 0.204 | +0.019 [-0.016, +0.056] | 0 to -0.05: miss, inside 0 |
| move_jack (human 0.051) | 0.034 | 0.043 | +0.009 [-0.000, +0.022] | |
| motion_pred (human 0.128) | 0.130 | 0.139 | +0.009 [+0.002, +0.016] | |
| rhythm_rep1 (human 0.269) | 0.266 | 0.288 | +0.022 [+0.001, +0.043] | |

The first step any setting has made under a judge that did not choose: a window's draws differ
in how they read, and a judge that never saw the song can tell which reads human. By grade
(small numbers): Easy 0.795 → 0.720, Hard 0.878 → 0.798, Expert 0.883 → 0.771; Normal and
Insane level. The jacks come up toward the human rate, the lane motion and the one-bar repeats
go a little past it. Judges trained on 96 songs a fold are weaker than the standard ones (the
cosine run: 0.858 here, 0.901 under the standard judge on 96 songs), so the playtest gate (a
system's own standard judge at songs 0.90 or under, or auc 0.85 or under) is not read off
these. The rhythm view's fresh judges come with the next entry.

**Predictions for `run_1113.sh`** (written before it ran): 8 draws a window, all 240 songs,
the same choosing judge; fresh standard judges (seed 1, --sr view, 2 epochs, 2 folds) on the
chosen run and the cosine run, the same 96 songs.
1. The gain: 3.0-3.6 logits a window (the lowest of 8 against the lowest of 4).
2. A song at x1.1-1.4 the plain draw's time a process.
3. The chosen run minus the cosine run under the standard judge: -0.03 to -0.07; the chosen
   run's songs 0.90-0.96.
4. Density -1% to -4%; new_onsets -0.01 to -0.03.
The playtest gate is read off 3: at songs 0.90 or under, or auc 0.85 or under, a quick pairs
check becomes worth the blind songs. How it could fail: no larger step than 4 draws made puts
the limit in what the choosing judge sees (the rhythm alone), not in the number of draws.

### 2026-11-09 (1) · run_1113: 8 draws on all songs; the lanes chosen too (predictions)

run_1112's rhythm-view fresh judges (the entry before left them for this one): the cosine run
0.664, the chosen run 0.535, -0.129 [-0.227, -0.034] (prediction 3, -0.02 to -0.06: miss,
far larger). In what the choosing judge sees, the chosen charts are nearly human (0.5 is
chance). The diagnostics judge on the 120 songs: logistic 0.852 → 0.831, trees 0.941 → 0.939.

`run_1113.sh` (23:35 to 04:42; the drawing 4 h 26 min). 8 draws a window, all 240 val songs,
the same choosing judge (sel_rhythm); fresh standard judges (seed 1, --sr view, 2 epochs, 2
folds) on the chosen run and the cosine run, the same 96 songs:

| | cosine | 8 draws | difference (95%) | human | prediction |
|---|---|---|---|---|---|
| standard judge auc | 0.911 | 0.878 | -0.033 [-0.068, +0.002] | | -0.03 to -0.07: hit, at the edge |
| songs (each song's windows averaged) | 0.969 | 0.958 | | | 0.90-0.96: hit |
| gain (the choosing judge's logits a window) | | 3.36 | the same 120 songs: 4 draws 2.44, 8 draws 3.42 | | 3.0-3.6: hit |
| a song's time (the plain draw's, a process) | | x1.74 | 4 draws x1.05 | | x1.1-1.4: miss |
| density | | | -2.4% | | -1% to -4%: hit |
| new_onsets | 0.118 | 0.108 | -0.010 [-0.016, -0.004] | 0.051 | -0.01 to -0.03: hit, at the edge |
| sr_error | 0.246 | 0.279 | +0.033 [+0.002, +0.065] | | |
| rhythm_rep1 | 0.255 | 0.329 | | 0.266 | |
| motion_pred | 0.127 | 0.151 | | 0.132 | |
| release_on_onset | 0.692 | 0.729 | | 0.759 | |
| hold_share | 0.215 | 0.203 | | 0.218 | |
| move_jack | 0.033 | 0.040 | | 0.044 | |

The diagnostics judge: logistic 0.871 → 0.860, trees 0.953 → 0.938. The playtest gate (songs
0.90 or under, or auc 0.85 or under) is not passed. The choosing judge's scores went down 40%
further than with 4 draws, the fresh judge's step not (-0.033 here, -0.040 with 4 draws under
the weaker 120-song judges): the failure reading written before it ran, the limit is what the
choosing judge sees (the rhythm alone), not the number of draws. 8 draws also push the one-bar
repeats and the lane motion's predictability past the human charts and spread the SR, for
x1.74 the time: 4 draws stay the setting (DECISIONS 2026-11-09).

**The lanes chosen too** (`run_1114.sh`). run_1112's rhythm-view fresh judge fell to 0.535
(0.664 for the cosine run), the full view only to 0.818: what is left to tell is in the lanes
and the long notes, which the choosing judge never saw. The lane passes now draw 4 versions of
the lanes at once (`sampler.forward_lanes_k`, `refine_lanes_k`: one forward pass of 4 a row;
the versions share the rhythm, chord sizes and holds) and the song keeps, span by span, the
version a full-view judge scores lowest (`sampler.choose_lanes`, `evaluate.py
--lane-select-judge J --lane-select-k 4`). The spans start 4 bars into the judges' windows
(`lane_spans`: row 0, then L/2, L/2 + L, ...), so a seam between two versions lies in the
middle of a window a judge scores; where the versions on its two sides differ, the 2 beats on
each side are chosen again with both sides in view (`refine_rows`, the refine pass on those
rows alone), so no lane pattern breaks there. The choosing judge: the full view of the cosine
draw before F's passes, the human chart's SR, 5 folds kept (sel_lanes). forward_lanes chooses
every lane from the rhythm, chord sizes and holds alone (the lanes a chart had never enter; a
test checks it), so the lanes are chosen on run_1112's chosen draw as saved (`--from-charts`,
`sampler.relane_song`): the same 120 songs and rhythm, lanes as its passes would have drawn
them, then F's passes; the drawing is not done again. Fresh judge as s1_sel (seed 1, --sr view,
2 epochs, all 5 folds of the 120 songs), so the step is paired with run_1112's on the same
songs. Stopped early as run_1112: after 30 songs, a lane gain (the chosen version under the
mean of its 4, in that judge's logits) under a tenth of its gap stops it (`select_watch.py
--field lane`).

Predictions (written before it ran):
1. The choosing judge (full view, human SR, the cosine draw before the passes, 5 folds): auc
   0.92-0.96 (the rhythm view's 0.877 and the lanes and long notes on top); its gap 7-10 logits.
2. The lane gain: 0.5-1.5 logits a span, 7-20% of the gap (the versions of a window's lanes
   read less apart than its rhythm draws, 2.45); the run goes on.
3. The fresh full-view judge, the lanes chosen minus run_1112's chosen run on the same 120
   songs: -0.01 to -0.04; songs under 0.90.
4. Lanes toward the human charts: move_jack 0.043 → 0.044-0.051, motion_pred 0.139 →
   0.125-0.137 (human 0.128); the rhythm as it was (density within ±0.5%, new_onsets within
   ±0.005: only F's passes can move it); sr_error 0 to -0.03.
5. A song at 60-100 s a process (the lane passes are half the cosine draw's 117 s, at x1.1
   for 4 versions; the seams and F's passes on top).
How it could fail: a gain under a tenth stops it, and the lanes' tell is the lane model's
habit rather than a draw's luck (a lane model trained on its own, or the taps view as the
judge, are the levers); a gain and no step, and F's passes undo the lanes (carry_rhythm,
copy_sections and align_alike choose the lanes of the bars they rewrite again) or the choice
learned the choosing judge's quirks; a step with the lane motion past the human, and the
choice overshoots as 8 draws did.

### 2026-11-09 (2) · run_1114: the lanes chosen by a judge; what is left of the chosen run's tell

`run_1114.sh` (06:08 to 07:47; the choosing judge 26 min, the lanes and F's passes on 120 songs
59 min, the fresh judge 12 min). Against run_1112's chosen run (the same 120 songs and rhythm,
the same fresh judge settings):

| | windows chosen (run_1112) | lanes chosen too | difference (95%) | human | prediction |
|---|---|---|---|---|---|
| the choosing judge sel_lanes (240 songs) | | auc 0.956, songs 0.962 | | | 0.92-0.96: hit |
| its gap | | 10.85 logits | | | 7-10: miss, a little over |
| the lane gain (30 songs / 120) | | 1.53 (14.1%) / 1.44 (13.3%) | | | 0.5-1.5, 7-20%: hit |
| fresh full-view judge auc | 0.818 | 0.835 | +0.017 [-0.017, +0.052] | | -0.01 to -0.04: miss |
| songs | 0.855 | 0.872 | | | under 0.90: hit |
| motion_pred | 0.139 | 0.129 | -0.010 [-0.014, -0.006] | 0.128 | 0.125-0.137: hit |
| move_jack | 0.043 | 0.046 | +0.003 [-0.012, +0.016] | 0.051 | 0.044-0.051: hit |
| lane_rep1 | 0.082 | 0.072 | -0.010 [-0.027, +0.009] | 0.063 | |
| copy_near | 0.039 | 0.035 | -0.004 [-0.011, +0.002] | 0.033 | |
| density, new_onsets, sr_error | | | +0.14%, +0.0002, -0.003 [-0.013, +0.007] | | ±0.5%, ±0.005: hit; 0 to -0.03: at 0 |
| a song's time, a process | | 87 s (median 66) | | | 60-100 s: hit |

The diagnostics judge: logistic 0.831 → 0.811, trees 0.939 → 0.917. The fresh judge's row gap
between the run and the human charts went 5.72 → 5.89 (+0.17 [-0.45, +0.83]), where the choosing
judge's 13% carried over whole would have taken it down about 0.77. Whether the seams took back
what the choice gained, the rows cannot say (corrected 11-09 evening): their scores in the bands
drawn again (192 ± 24 rows into a window), against the window's other rows and net of the
judge's human rows, moved +0.08 [-0.05, +0.21] logits from run_1112's, but a judge's rows repeat
its window's verdict (11-07), so a seam that gave a window away would raise every row of it
alike; first read here as "the seams are not read", which the rows cannot show. The failure reading written
before it ran, a gain and no step: F's passes rewrite about 33 bars a song (carry 17, sections
12, aligned 4, of about 130), too few to undo it all; the lane statistics moved toward the human
charts, so the lanes changed, but what a judge that did not choose reads in the lanes did not.
The rhythm's choice carried over (the rhythm view 0.664 → 0.535) because the draws of a window
differ in what every judge reads; the versions of the lanes may differ in detail that one judge
reads one way and another another, or the seams may cost what they gain (one with the seams at
the windows' edges and a judge on windows half a window off would tell). Not taken (DECISIONS
2026-11-09).

What is left of run_1112's tell, view by view (fresh judges, seed 1, the same 120 songs, each
view the one above with one part more; `s1_sel_taps` 08:58 to 09:11):

| view | auc (95%) | songs | minus the row above (95%) | adds |
|---|---|---|---|---|
| rhythm (lanes at random, long notes as taps) | 0.535 [0.458, 0.605] | 0.520 | | the rhythm, against the music |
| taps (long notes as taps) | 0.701 [0.646, 0.753] | 0.742 | +0.167 [+0.083, +0.252] | the lanes |
| full | 0.818 [0.775, 0.857] | 0.855 | +0.117 [+0.055, +0.176] | the long notes |

Both steps are larger than in the cosine run under the srv judges (96 songs: rhythm 0.742, taps
0.840, full 0.909; the lanes +0.098, the long notes +0.069): with the rhythm's tell gone, what it
covered shows (the views overlap). The lanes carry more of what is left, the long notes much of
the rest; choosing does not reach the lanes, and the long notes are not yet touched.

### 2026-11-09 (3) · The lanes' and the long notes' settings on the chosen draw (predictions)

What is left of the chosen run's tell is in the lanes and the long notes (11-09 (2)), and a
choice among versions of the lanes does not carry over, so the lever is how the passes draw
them. The settings are tried on run_1112's chosen draw as saved: `evaluate.py --relane` draws
its lanes again once at the settings given (`sampler.relane_song` without a judge; as for
run_1114, the lanes come out as a new drawing's would), or only F's passes run again; each run
is read by its own fresh judge (seed 1, --sr view, 2 epochs, all 5 folds of the 120 songs),
paired with run_1112's s1_sel. `run_1115.sh`:
- deaf: a taps-view fresh judge on run_1112's chosen run that hears silence: the taps view's
  0.701 minus it is what the lanes against the music (and the rhythm against it, 0.035 in the
  rhythm view) tell.
- rt1: F's passes with the releases of the long notes they add at temperature 1 instead of 0.5
  (`--release-temp 1`, new; refine_holds' release_temperature was fixed at 0.5): the chosen
  run's long notes last 0.76 beats, the human ones 0.94, and the releases are drawn near the
  likeliest.
- lg0: the lanes drawn again without the guidance towards the chart stats (`--lane-guidance 0`
  instead of 2; F's passes' lanes too).
- lt1: the lanes drawn again at temperature 1 instead of 0.5 (F's passes' lanes too).
- nohu (not in the chain): F's passes without the long notes topped up to the human share
  (the chosen draw has 0.126 before them, the human charts 0.225).

Predictions (written before it ran):
1. The deaf taps judge 0.62-0.69: the music adds +0.01 to +0.08 to the taps view (part of the
   lanes' tell is the lanes against the melody, most the lane patterns themselves).
2. rt1: hold_beats 0.76 → 0.80-0.88 (human 0.94), release_on_onset 0.728 → 0.73-0.76 (0.784),
   the long-note share within ±0.005; the fresh judge minus s1_sel -0.03 to +0.01.
3. lg0: the jack and trill rates off the chart stats' (move_jack, move_trill further from the
   human rates), motion_pred 0.12-0.15; the fresh judge -0.02 to +0.03.
4. lt1: motion_pred 0.139 → 0.10-0.12 (human 0.128; the 11-07 sweep: -0.034 at temperature 1);
   the fresh judge -0.03 to +0.03.
5. Times: deaf 13 min, rt1 about 25, lg0 and lt1 about 75 each (the lanes drawn again: 60).
A setting is taken when its step's 95% interval lies under 0; the ones taken go together next.
How it could fail: none taken, and the lanes' and the long notes' tell is not in how the passes
draw them but in the model (training towards them is the lever); the deaf judge as high as the
taps view, and the music is not where the lanes give the chart away.

### 2026-11-09 (4) · run_1115: the lanes' and the long notes' settings, none taken

`run_1115.sh` (09:35 to 12:01). Fresh judges on the same 120 songs, against run_1112's chosen run
(full view 0.818, taps 0.701):

| | auc (95%) | songs | minus run_1112 (95%) | prediction | taken |
|---|---|---|---|---|---|
| taps view, deaf | 0.665 [0.603, 0.722] | 0.713 | the music adds +0.036 [+0.017, +0.055] | 0.62-0.69, +0.01 to +0.08: hit | |
| rt1: the releases at temperature 1 | 0.826 [0.790, 0.863] | 0.865 | +0.008 [-0.010, +0.026] | -0.03 to +0.01: hit | no |
| lg0: the lanes without guidance | 0.837 [0.803, 0.870] | 0.862 | +0.019 [-0.011, +0.055] | -0.02 to +0.03: hit | no |
| lt1: the lanes at temperature 1 | 0.807 [0.770, 0.842] | 0.824 | -0.011 [-0.052, +0.026] | -0.03 to +0.03: hit | no |

| per song (run_1112 → the setting; 95%) | run_1112 | human | rt1 | lg0 | lt1 |
|---|---|---|---|---|---|
| hold_beats | 0.758 | 0.936 | 0.772 (+0.014 [+0.007, +0.023]) | 0.761 | 0.759 |
| release_on_onset | 0.728 | 0.784 | 0.720 (-0.008 [-0.012, -0.004]) | 0.724 | 0.725 |
| motion_pred | 0.139 | 0.128 | 0.139 | 0.143 | 0.097 (-0.042 [-0.049, -0.037]) |
| move_jack | 0.043 | 0.051 | 0.043 | 0.017 (-0.026 [-0.047, -0.011]) | 0.047 |
| move_trill | 0.173 | 0.152 | 0.172 | 0.156 | 0.178 |
| lane_rep1 | 0.082 | 0.063 | 0.076 | 0.082 | 0.032 (-0.050 [-0.072, -0.031]) |
| copy_near | 0.039 | 0.033 | 0.038 | 0.044 | 0.018 (-0.021 [-0.028, -0.014]) |
| sr_error | 0.204 | | 0.202 | 0.195 | 0.191 |
| a song's time, a process | 15 s (F's passes) | | 14 s | 44 s | 74 s |

The diagnostics judge (logistic / trees; run_1112 0.831 / 0.939): rt1 0.827 / 0.932, lg0
0.890 / 0.943 (leaning on move_jack), lt1 0.850 / 0.932. Predictions 2-4 on the metrics: rt1's
hold_beats 0.80-0.88 and release_on_onset 0.73-0.76 missed (0.772, 0.720: further off); lg0's
jacks off the chart stats hit (0.017 against the human 0.051), its motion_pred hit; lt1's
motion_pred 0.10-0.12 at the edge (0.097). Times: lg0 42 min for 75 (the lanes drawn again
once take 44 s a song, run_1114's 4 versions and seams 87).

Reading. The lanes' tell is mostly in the patterns themselves (0.665 without the music); the
lanes against the melody add +0.036. The guidance towards the chart stats is what keeps the
jacks at the human rate: without it they nearly go, and the judges see it. Temperature 1 is
the one setting the judges lean towards (the songs 0.855 → 0.824), not past the rule, and it
overshoots: motion, one-bar lane repeats and near copies all fall under the human charts; the
human values sit between 0.5 and 1 (motion_pred at about 0.63, lane_rep1 0.69, copy_near
0.64). The releases' temperature hardly moves the long notes' length: the long notes added
by F's top-up (181 a song) are shorter than the model's own (0.52 against 0.63 beats over all
long notes), yet as many of either end 3 cells before the next note in the lane (21%; within
6 cells 58% and 53%), so what keeps them short is where the model puts the ends there, not
the lanes in the way. None taken: the lane guidance stays 2, the lane temperature 0.5, the
releases' 0.5 (DECISIONS 2026-11-09).

### 2026-11-09 (5) · The lanes at temperature 1 in the taps view; the long notes drawn by the model (predictions)

A taps-view fresh judge on run_1115's lt1 run (seed 1, 5 folds, the same 120 songs; after the
entry above, so not in its rule):

| view (fresh judges) | run_1112 | lanes at temperature 1 | difference (95%) |
|---|---|---|---|
| taps | 0.701 (songs 0.742) | 0.607 (songs 0.632) | -0.095 [-0.165, -0.031] |
| full | 0.818 (songs 0.855) | 0.807 (songs 0.824) | -0.011 [-0.052, +0.026] |
| the lanes' step (taps minus rhythm) | +0.167 [+0.083, +0.252] | +0.072 [-0.019, +0.158] | |
| the long notes' step (full minus taps) | +0.117 [+0.055, +0.176] | +0.200 [+0.125, +0.277] | |

At temperature 1 the lanes give the charts away far less, in spite of their statistics now
past the human charts' (motion_pred 0.097 against 0.128); the full view hardly moves because
the long notes alone are enough to tell (the views overlap: the judge needs one tell). The
lanes at 1 go into the next run (DECISIONS 2026-11-09), to be kept or dropped by its full view.

The long notes. F's top-up adds 181 a song, shorter than the model's own (0.52 against 0.63
beats over all long notes; the songs' mean 0.848 before the passes, 0.758 after, the human
0.936), and the releases' temperature does not lengthen them (11-09 (4)). So the model draws
more of them itself: `--hold-bias 0.5` (about 1.6 times as many start; the chosen draw had
0.126 of the onsets as long notes, the human charts 0.225) and the top-up fills only what is
left. `run_1116.sh`: run_1112's draw again (the windows chosen among 4 by sel_rhythm, the same
seeds) with the hold bias and the lanes at temperature 1, F's passes (their lanes at 1 too),
fresh full-view and taps-view judges as s1_sel's, so the steps run_1112 → the lanes at 1 →
the long notes drawn are read on the same 120 songs in both views.

Predictions (written before it ran):
1. The long-note share before the passes 0.126 → 0.17-0.22, after them 0.215-0.225; the top-up
   181 → 40-110 a song.
2. hold_beats 0.759 → 0.80-0.90 (human 0.936); release_on_onset 0.725 → 0.73-0.77 (0.784).
3. The full view minus the lanes-at-1 run -0.02 to -0.06 (the long notes' step +0.200
   shrinks); the taps view within ±0.03 of it (the same lane setting; the rhythm drawn again).
4. Density within ±3%, new_onsets within ±0.01, sr_error within ±0.03 of run_1112's.
5. The drawing as run_1112's (1 h 27 min), the run about 2 h 10 min.
The run becomes the system if its full view is under run_1112's at 95%. How it could fail:
the long notes' step does not shrink, and their tell is not how many or how long but where
the model puts them (a view that shows them to the choosing judge, or training, is the
lever); the taps view rises, and the bias moved the rhythm (onsets turned into long notes).

### 2026-11-09 (6) · run_1116: the hold bias at 0.5 overshoots; at the human share (predictions)

`run_1116.sh` (17:30 to 19:41; the drawing about 1 h 30 min). Fresh judges on the same 120 songs:

| view (fresh judges) | run_1112 | lanes at 1 | + hold bias 0.5 | minus the lanes-at-1 run (95%) | prediction |
|---|---|---|---|---|---|
| full | 0.818 | 0.807 | 0.805 (songs 0.847) | -0.002 [-0.050, +0.043] | -0.02 to -0.06: miss |
| taps | 0.701 | 0.607 | 0.766 (songs 0.794) | +0.159 [+0.101, +0.223] | within ±0.03: miss |
| the long notes' step (full minus taps) | +0.117 | +0.200 | +0.039 | | |

| per song | human | run_1112 | lanes at 1 | + bias 0.5 | minus lanes at 1 (95%) | prediction |
|---|---|---|---|---|---|---|
| long-note share before F's passes | | 0.126 | 0.126 | 0.374 | | 0.17-0.22: miss |
| long-note share | 0.225 | 0.215 | 0.217 | 0.369 | +0.153 [+0.128, +0.176] | 0.215-0.225: miss |
| top-up (long notes a song) | | 181 | 181 | 17 | -164 | 40-110: miss |
| hold_beats | 0.936 | 0.758 | 0.749 | 0.811 | +0.062 [+0.029, +0.094] | 0.80-0.90: hit |
| release_on_onset | 0.784 | 0.728 | 0.728 | 0.706 | -0.023 [-0.053, +0.007] | 0.73-0.77: miss |
| ln_short | 0.157 | 0.118 | 0.115 | 0.192 | +0.078 [+0.054, +0.106] | |
| density (against the human) | | 0.987 | 0.989 | 1.009 | +2.0% [+0.3, +3.7] | ±3%: hit |
| new_onsets | 0.051 | 0.100 | 0.100 | 0.126 | +0.025 [+0.011, +0.050] | ±0.01: miss |
| motion_pred | 0.128 | 0.139 | 0.097 | 0.075 | -0.022 [-0.028, -0.015] | |
| rhythm_rep1 | 0.269 | 0.288 | 0.289 | 0.251 | -0.038 [-0.060, -0.019] | |
| sr_error | | 0.204 | 0.191 | 0.284 | +0.094 [+0.046, +0.147] | ±0.03: miss |

The diagnostics judge (logistic / trees): run_1112 0.831 / 0.939, lanes at 1 0.850 / 0.932, the
bias 0.869 / 0.927 (leaning on motion_pred, new_onsets, move_stair). Not the system (its full
view is not under run_1112's).

Reading. The bias works on every step of the drawing, not once: 0.5 tripled the long notes
(0.126 → 0.374, past the human 0.225), so prediction 1 was off by far. What it showed: the long
notes the model draws itself, even too many, give the chart away much less than the topped-up
ones (their own step +0.200 → +0.039, with the lanes alike), but so many long notes move the
rest, the rhythm and the lanes around them (the taps view +0.159, more new onsets, the SR
error up), so the full view stays. The failure reading written before it ran (the taps view
rises, the bias moved the rhythm), with the overshoot as the cause.

**At the human share** (`run_1118.sh`). The share's log-odds rose 2.84 a unit of bias (-1.93 to
-0.52), so 0.2 puts the model's own long notes at about 0.21 of the onsets and the top-up fills
the little left; otherwise run_1116 again (the windows chosen among 4 by sel_rhythm, the same
seeds, the lanes at temperature 1, F's passes, fresh full- and taps-view judges).

Predictions (written before it ran):
1. The long-note share before F's passes 0.18-0.25, after them 0.22-0.26; the top-up 20-90 a
   song.
2. hold_beats 0.78-0.85, ln_short 0.12-0.17.
3. The taps view, minus the lanes-at-1 run (0.607): -0.02 to +0.06.
4. The full view, minus the lanes-at-1 run (0.807): -0.02 to -0.08; songs under 0.80.
5. new_onsets within +0.015 of it, density within ±2%, sr_error within ±0.04.
The run becomes the system if its full view is under run_1112's (0.818) at 95%. How it could
fail: the taps view rises again, and any bias moves the rhythm and lanes with the long notes
(they are drawn together); the full view stays with the long notes' step small, and the tell
moved rather than shrank; then the choosing judge sees the long notes (`run_1117.sh`, ready:
the holds view, `judge.view_tokens` "holds", the rhythm view's random lanes with every long
note kept whole in its lane).

### 2026-11-09 (7) · The windows chosen by a judge that sees the long notes (predictions)

`run_1117.sh`, queued after run_1118 (its results not yet in). run_1112's choosing judge saw the
rhythm view, and its choice carried over to that view (0.664 → 0.535) and no further. Here the
choosing judge sees the holds view (`judge.view_tokens` "holds": the rhythm view's random lanes,
every long note kept whole in its lane): sel_holds, trained as sel_rhythm (the cosine draw
before F's passes, the human chart's SR, 2 epochs, all 5 folds kept), and a window's 4 draws are
chosen by their rhythm and long notes together; the rest as run_1118 (the hold bias 0.2, the
lanes at temperature 1, F's passes, fresh full- and taps-view judges), so the step against
run_1118 on the same 120 songs is the choice's.

Predictions (written before it ran):
1. sel_holds: auc 0.88-0.93 (sel_rhythm's 0.877 with the long notes on top); its gap 6-9 logits.
2. The select gain 2.5-3.5 logits (40-50% of its gap): the run goes on.
3. The fresh full view minus run_1118's: -0.02 to -0.06; the taps view minus run_1118's: -0.03
   to +0.02 (the rhythm is still chosen; the long notes as taps add little to it).
4. hold_beats and release_on_onset toward the human charts' (+0.01 to +0.05 each); the long-note
   share within ±0.02 of run_1118's.
5. About 2 h 30 min (the choosing judge 25 min, the drawing as run_1118's).
How it could fail: the step is no larger than noise, and what is left in the long notes is not
in the draws' differences but in the model's habits (as for the lanes, 11-09 (2)); the taps view
rises, and choosing by the long notes chose against the rhythm.

### 2026-11-10 (1) · run_1118 and run_1117: the long notes at the human share, chosen with them; the system on 240 songs (predictions)

`run_1118.sh` (21:26 to 01:37, with about an hour where the computer slept) and `run_1117.sh`
(01:37 to 04:52) on the same 120 songs, fresh judges as s1_sel:

| view (fresh judges) | run_1112 | lanes at 1 | + hold bias 0.2 (run_1118) | + windows chosen by the holds view (run_1117) | prediction |
|---|---|---|---|---|---|
| full | 0.818 | 0.807 | 0.773 (songs 0.796) | 0.743 (songs 0.768) | 1118: -0.02 to -0.08 hit, songs under 0.80 hit; 1117: -0.02 to -0.06 hit |
| minus the column before (95%) | | -0.011 [-0.052, +0.026] | -0.034 [-0.079, +0.010] | -0.030 [-0.093, +0.026] | |
| minus run_1112 (95%) | | | -0.045 [-0.101, +0.011] | -0.075 [-0.137, -0.019] | |
| taps | 0.701 | 0.607 | 0.614 | 0.699 | 1118: -0.02 to +0.06 hit; 1117: -0.03 to +0.02 miss (+0.085 [+0.039, +0.132]) |
| the long notes' step (full minus taps) | +0.117 | +0.200 | +0.159 | +0.044 | |

| per song | human | run_1112 | lanes at 1 | + bias 0.2 | + holds view | prediction |
|---|---|---|---|---|---|---|
| long-note share before F's passes | | 0.126 | 0.126 | 0.205 | 0.259 | 1118: 0.18-0.25 hit |
| long-note share | 0.225 | 0.215 | 0.217 | 0.239 | 0.271 | 1118: 0.22-0.26 hit; 1117: ±0.02 of 1118 miss (+0.032) |
| top-up (a song) | | 181 | 181 | 78 | 34 | 1118: 20-90 hit |
| hold_beats | 0.936 | 0.758 | 0.759 | 0.786 | 0.805 | 1118: 0.78-0.85 hit; 1117: +0.01 to +0.05 hit (+0.020, n.s.) |
| ln_short | 0.157 | 0.118 | 0.113 | 0.144 | 0.161 | 1118: 0.12-0.17 hit |
| release_on_onset | 0.784 | 0.728 | 0.725 | 0.718 | 0.713 | 1117: +0.01 to +0.05 miss |
| new_onsets | 0.051 | 0.100 | 0.100 | 0.114 | 0.120 | 1118: within +0.015 hit |
| density (against the human) | | 0.987 | 0.989 | 0.990 | 1.008 | 1118: ±2% hit |
| rhythm_rep1 | 0.269 | 0.288 | 0.289 | 0.296 | 0.275 | |
| sr_error | | 0.204 | 0.191 | 0.202 | 0.226 | 1118: ±0.04 hit |

sel_holds (the holds view of the cosine draw before F's passes, human SR, 240 songs): auc 0.929
(songs 0.938), gap 8.83 (predictions 0.88-0.93, 6-9: hit); its select gain 3.11 logits, 35% of
the gap (2.5-3.5 hit; 40-50% miss). The diagnostics judge (logistic / trees): run_1112 0.831 /
0.939, lanes at 1 0.850 / 0.932, bias 0.2 0.863 / 0.942, holds view 0.854 / 0.939.

Reading. The bias at 0.2 puts the model's own long notes near the human share and leaves the
rhythm and lanes as they were (the taps view as the lanes-at-1 run's); the long notes' step
falls a little (+0.200 → +0.159). Choosing the windows by the holds view takes it down to
+0.044, the long notes now read nearly human, but the taps view rises back to run_1112's
(+0.085 against run_1118): the choice of a window weighs its long notes with its rhythm and
gives some of the rhythm away (the failure reading written for it), and it leans to more long
notes than the human charts (0.271). On the full view the three steps add up: 0.818 → 0.743,
-0.075 [-0.137, -0.019] against run_1112, under 0 at 95% (the rule of 11-09 (5)): the system
from now on (DECISIONS 2026-11-10), to be read on the standard judges.

**The system on 240 songs** (`run_1119.sh`). The playtest gate is read on the standard judges
(240 songs, seed 1, --sr view, 2 epochs, 2 folds: the cosine run 0.911, run_1113's 8 draws
0.878 on the same 96 songs), so the other 120 songs are drawn and passed as run_1117's
(`evaluate.py --reuse R`, new: the songs R has keep R's charts, the others drawn with the
seeds a 240-song run gives them).

Predictions (written before it ran):
1. The standard judge: auc 0.78-0.84 (the 120-song fresh judges read 0.04-0.06 under the
   standard ones: the cosine run 0.858 against 0.911), songs 0.84-0.92.
2. Minus the cosine run on the songs they share: -0.07 to -0.13; minus run_1113's 8 draws:
   -0.04 to -0.10.
3. The gate (the standard judge's songs 0.90 or under, or its auc 0.85 or under): passed, by
   the auc.
4. The other 120 songs as run_1117's: the long-note share 0.25-0.29, hold_beats 0.78-0.83.
5. About 2 h 30 min (the 120 songs drawn 2 h, F's passes 12 min, the judge 20 min).
How it could fail: the standard judge reads the system as run_1113's (0.86 or over), and the
120-song judges' step was in their own noise.

### 2026-11-11 (1) · Where the system's tell is left; its lanes at 0.7 and its hold bias at 0.1 (predictions)

`run_1120.sh`, queued after run_1119 (whose standard-judge reading is not in yet). On the
system's first 120 songs (run_1117), fresh judges as s1_sh's (seed 1, --sr view, 2 epochs,
5 folds):
- views: the system in the rhythm view and the holds view, run_1112's in the holds view: how
  much of what the holds-view choice gave up is the rhythm (the taps view rose 0.614 → 0.699
  against run_1118), and how human the rhythm with the long notes now reads.
- lt07: the system's draw with its lanes drawn again at temperature 0.7 (`--relane`; at 1 the
  lane motion, the one-bar lane repeats and the near copies fell under the human charts', whose
  values lie between 0.5 and 1, at about 0.63-0.69), then F's passes (their lanes at 0.7).
- hb1: the system drawn with the hold bias 0.1 (the holds-view choice took the long notes to
  0.271 of the onsets, over the human 0.225), then F's passes.

Predictions (written before it ran):
1. The rhythm view of the system 0.56-0.66 (run_1112's 0.535); the holds view of the system
   0.60-0.70, of run_1112 0.68-0.78.
2. lt07: motion_pred 0.085 → 0.10-0.12 (human 0.128), lane_rep1 0.044 → 0.05-0.07 (0.063); the
   taps view minus the system's -0.04 to +0.04, the full view -0.03 to +0.03.
3. hb1: the long-note share before F's passes 0.19-0.23 (0.259 at 0.2), after 0.23-0.26 (0.225);
   hold_beats 0.78-0.82; the full view minus the system's -0.02 to +0.03, the taps view -0.03 to
   +0.03.
4. Times: views 40 min, lt07 about 65, hb1 about 2 h 45 min.
A setting replaces the system's when its full view is under the system's at 95%. How it could
fail: the rhythm view of the system near run_1112's, and the taps view's rise was its lanes,
not its rhythm; neither setting moves the full view, and the system's settings stay.

### 2026-11-11 (2) · run_1119: the system on 240 songs, the playtest gate passed; run_1120 results

`run_1119.sh` (02:35 to 11:21 on 11-11, its merge stopped once: merge_shards.py would not write
over run_1117's directory without `--force`, fixed in the script and run after run_1120). The
standard judges (seed 1, --sr view, 2 epochs, 2 folds), the same 96 songs:

| | auc (95%) | songs | minus the row above (95%) | prediction |
|---|---|---|---|---|
| the cosine run | 0.911 [0.887, 0.934] | 0.923 | | |
| run_1113's 8 draws | 0.878 [0.845, 0.909] | 0.881 | -0.033 [-0.068, +0.002] | |
| the system | 0.816 [0.779, 0.850] | 0.837 | -0.062 [-0.105, -0.019] | 0.78-0.84 hit; songs 0.84-0.92: under it |
| the system minus the cosine run | | | -0.095 [-0.131, -0.061] | -0.07 to -0.13: hit |

The playtest gate (the standard judge's songs 0.90 or under, or its auc 0.85 or under) is
passed on both counts; the auc's interval reaches 0.850. The 120 songs drawn now: the long-note
share 0.250 (their human charts 0.212; prediction 0.25-0.29 hit), hold_beats 0.868 (0.78-0.83:
over), sr_error 0.276 (the first 120 songs 0.226).

`run_1120.sh` (05:15 to 10:46), fresh judges on the system's first 120 songs:

| | the system | the setting | minus the system (95%) | prediction | taken |
|---|---|---|---|---|---|
| the rhythm view (run_1112's 0.535) | 0.603 | | +0.068 [-0.016, +0.147] against run_1112 | 0.56-0.66: hit | |
| the holds view (run_1112's 0.712) | 0.733 | | +0.021 [-0.043, +0.082] against run_1112 | 0.60-0.70: miss | |
| the lanes at temperature 0.7, full | 0.743 | 0.771 | +0.028 [+0.003, +0.057] | -0.03 to +0.03: hit, at the edge | no |
| the lanes at 0.7, taps | 0.699 | 0.693 | -0.006 [-0.043, +0.029] | -0.04 to +0.04: hit | |
| the hold bias 0.1, full | 0.743 | 0.793 | +0.050 [+0.008, +0.092] | -0.02 to +0.03: miss | no |
| the hold bias 0.1, taps | 0.699 | 0.681 | -0.019 [-0.063, +0.026] | -0.03 to +0.03: hit | |

At 0.7 the lanes' statistics come nearer the human charts' (motion_pred 0.085 → 0.101, human
0.128; lane_rep1 0.043 → 0.050, 0.063: both predictions hit) and the judges read them as less
human: they prefer lanes less predictable than the statistics. At the hold bias 0.1 the long
notes come nearer the human share (0.271 → 0.243; before F's passes 0.259 → 0.214, both
predictions hit) but the top-up adds 61 a song instead of 34, and the full view worsens: the
topped-up long notes are the tell (11-09 (6)), so fewer of them is worth more than the share.
In the rhythm and long notes alone (the holds view) the system is no more human than run_1112;
its gain is in the lanes (with the long notes there, the lanes add +0.010 over the holds view
against run_1112's +0.106). The system's settings stay (DECISIONS 2026-11-10).

### 2026-11-14 (1) · The windows chosen by the rhythm judge and the holds judge together (predictions)

The system's choice by the holds view gave some of the rhythm away (above). `run_1121.sh`: each
window's 4 draws are scored by sel_rhythm and sel_holds, each over its own gap, summed
(`evaluate.py --select-judge J1 J2`, `select.combined_scorer`); the rest as the system. On all
240 val songs, against the system's 240 by fresh 5-fold judges (paired on all 240), the full
and the rhythm view.

Predictions (written before it ran):
1. The select gain (the two judges' gains over their gaps, summed): 0.5-0.8 (alone: sel_rhythm
   0.42, sel_holds 0.35 of their gaps).
2. The system in 5-fold judges on 240 songs: full 0.78-0.82, rhythm 0.58-0.64.
3. The two judges' run minus the system: full -0.01 to -0.04, rhythm -0.03 to -0.08.
4. The long-note share 0.24-0.27, hold_beats within ±0.03 of the system's.
5. The drawing about 4 h, F's passes 25 min, the four judges 2 h.
It becomes the system if its full view's step lies under 0 at 95%. How it could fail: the
rhythm view does not fall, and the rhythm given away was the hold bias's, not the choice's (the
holds view already carries the rhythm, so the two judges add little to one).

(Not run on 11/14: the pilot's findings below came first; run_1121 waits.)

### 2026-11-14 (2) · The pilot's tells: empty stretches, mirrored answers; the rhythm drawn beat by beat (predictions)

The pilot (the user, pack v2, the system's charts): no sheet filled, two tells reported. (a)
Stretches with nothing at all, not short, mid-song: "a buggy AI". (b) A human chart answers a
pattern with a left-right mirrored one, the AI too seldom. Measured afterwards (aggregates;
`fatal.py` in the device VM, numpy only; the 240 val songs, the human chart's span from its
first to its last note):

| | human | system draw | after F's passes |
|---|---|---|---|
| longest empty stretch a song, 90th percentile (beats) | 7.9 | 8.8 | 14.0 |
| songs with an empty stretch of 8+ beats | 8% | 11% | 18% |
| songs with a fatal one (8+ beats, the human chart 1+ onset a beat there) | 0% | 3% | 5% |
| 1 beat: the next beat has the same rhythm (busy beat pairs) | 25.0% | 12.2% | 14.6% |
| of those (alternations aside), mirrored | 10.7% | 8.0% | 7.2% |
| of those, the same | 3.4% | 5.7% | 5.0% |
| 2 beats: the next 2 have the same rhythm | 16.9% | 7.9% | 9.2% |
| of those, mirrored | 7.0% | 5.0% | 3.3% |

- The fatal stretches (10 in the draw, 18 after the passes) all lie in the quieter part of
  their song (below its 40th loudness percentile), where the human chart goes on following
  something. 13 of the 18 were all but empty in the draw already; the rest pass joins and
  lengthens them (songs with one: 3.2 bars cleared by it, the others 0.7).
- Human charts (3,000 train charts) answer a beat of the same rhythm with its mirror 11.7% of
  the time against 6.1% for the same rhythm anywhere else in the song, an exact repeat 4.9%:
  the mirror is the answer they choose, at 2 beats and a bar too, in every grade.
- The model knows it (`mirror_probe.py`, 40 val charts, forward_lanes' past+rhythm view,
  temperature 1): at the first row where a human chart mirrors, its top lane set is the
  mirror 77% of the time (2 beats: 89%; where the human chart does something else 26-33%,
  chance 20%), and it goes on with a mirror once begun (85%). The AI mirrors at about 40% of
  the human rate mostly because its rhythm repeats beat to beat half as often.
- The rhythm is drawn in random order, ~9 cells a step anywhere in the window: neighbouring
  beats are drawn without each other (the parallel-decoding failure CoDiLA, ICML 2026, fixes
  with a local autoregressive decoder). `--order block` already draws beat by beat, left to
  right, each beat with the ones before it complete (block diffusion with a beat a block).

`run_1122.sh`: the first 120 val songs drawn twice with the system's settings but one draw a
window (sample_windows does not batch the 4 draws for block order yet), in random order and
in block order, the same seeds; F's passes; fresh 120-song judges on both, full and rhythm
view (paired); then `scripts/stems.py` for every audio file (the val and test audio first).

Predictions (written before it ran):
1. Busy beat pairs whose next beat has the same rhythm, random order: 11-14% in the draw,
   13-16% after the passes; block order: 16-22% in the draw, 17-24% after.
2. Of those, mirrored (after the passes): random 6-9%, block 7-11%.
3. Empty stretches after the passes, songs with 8+ beats: random 15-21%, block 15-28% (left
   to right can hand an empty beat on as context, as the confidence order did on 10-03).
4. Fresh judges, block minus random: full view -0.05 to +0.02, rhythm view -0.08 to 0.00.
5. Time a song within ±15% of the random order's.
If block order's rhythm view lies under 0 at 95% and its fatal stretches are not more, the
4 draws get batched for it and it goes into the system run (the windows chosen among 4); if
its full view lies under 0 at 95%, at once. How it could fail: the beat-to-beat repeats do
not rise (the model itself does not repeat a beat it has just seen), or they rise as exact
copies, not as the human charts' answers.

Results (11/14 23:58 - 11/15 03:52; the first 120 val songs, aggregates):

| | human | random | block | prediction |
|---|---|---|---|---|
| fresh judge, full view | | 0.834 | 0.984, +0.150 [+0.118, +0.187] | -0.05 to +0.02 ✗ |
| fresh judge, rhythm view | | 0.738 | 0.967, +0.229 [+0.185, +0.277] | -0.08 to 0.00 ✗ |
| sr_error | | 0.227 | 1.024 | |
| notes, AI over human (median, 10th-90th pct) | | 1.00 (0.90-1.16) | 1.24 (0.97-1.65) | |
| next beat of the same rhythm, draw / after the passes | 26.6% | 9.3% / 12.3% | 1.4% / 2.1% | 11-14 / 13-16 ✗; 16-22 / 17-24 ✗ |
| of those mirrored, after the passes | 8.4% | 6.2% | 9.4% (few pairs) | 6-9 ✓; 7-11 ✓ |
| songs with an 8+ beat empty stretch, after | 8% | 18% | 21% | 15-21 ✓; 15-28 ✓ |
| songs with a fatal one, after | 0% | 5% | 11% | |
| drawing time | | 80 min | 85 min (+6%) | ±15% ✓ |

The diagnostics judge: random 0.865 (trees 0.941), block 0.995 (0.994). Reading: block order
is not taken. A model trained on random masks has hardly seen a window whose every later
beat is MASK: not knowing what follows, it overfills (1.24 times the notes, sr_error four
times), drops the beat-to-beat repeats instead of making them, and leaves more empty
stretches. Block diffusion trains for its order (BD3-LM's block-causal objective); drawing
in order needs a model, or a head, trained for it (block-masked fine-tuning, or CoDiLA's
small autoregressive decoder), not only the order. The system's own draw (the windows chosen
among 4) repeats the next beat's rhythm more than one draw a window (12.2% on 240 songs
against 9.3% here): the choosing judge gives some of it back.

### 2026-11-15 (1) · full-v4 taught the stems (predictions)

The fatal empty stretches lie where the mix gets quiet but a human chart goes on following
something (EXPERIMENTS 2026-11-14 (2)), and the user's reading of the pilot: the model does
not hear what to follow there. `scripts/stems.py` (run_1122's last phase, 5,693 audio files,
none failed) gives every audio file htdemucs's drums, bass, other and vocals as 16 bands each
on the mixture's frames, standardized per song (`mel.MelStore(stems=True)`: the 80 bands of
the mix, then the 64). `train.py --init outputs/full-v4/best.pt --stems`: full-v4's config
and weights, its audio_conv widened from 80 to 144 input channels with zero weights
(`diffusion.widen_audio`: it starts as full-v4 exactly), 20k steps at lr 1e-4 (warmup 500),
full-v4's recipe otherwise (row-mask 0.5, chart stats, batch 16). A model reads the first
config.n_mels channels, so the judges and the choosing judge (full-v4 trunks) take the same
frames. Control: the same 20k steps without the stems (outputs/ft-v1), for val_ce only.

`run_1123.sh`: waits for the stems, trains, draws the first 120 val songs as the system does
(the windows chosen among 4 by sel_holds, hold bias 0.2, lanes at temperature 1, F's passes)
with the new model, and reads it against the system there with the 120-song fresh judges
(full view s1_sh 0.743, rhythm view s1_sh_rhythm 0.603), then the control.

Predictions (written before it ran):
1. val_ce (full-v4 0.0740): ft-v1 0.0735-0.0742, stems-v1 0.0715-0.0738, under ft-v1;
   val_ce@0.9 (the audio does most of the work; full-v4 0.1734) falls the most:
   stems-v1 0.160-0.172.
2. After the passes, songs with a fatal empty stretch: the system 5% (on 240) -> 0-4%;
   with an 8+ beat empty stretch 18% -> 10-17%.
3. The next beat of the same rhythm (after the passes): 14.6% -> 13-18%.
4. Fresh judges against the system: full view -0.06 to +0.01, rhythm view -0.06 to +0.02.
5. sr_error 0.20-0.25 (the system 0.226 on its first 120).
It goes into the system if its full view lies under 0 at 95%; if the full view is not worse
and the fatal stretches fall, the stems stay for the next system run with the pass fixes.
How it could fail: 20k steps are too few for the new channels to matter (val_ce as ft-v1's),
or the stems help the likelihood but the passes and the choice decide the empty stretches
(the rest pass joins them whatever the draw).

### 2026-11-15 (2) · The pilot's tells in the passes: no long empty stretch from the rest pass, a beat answered (predictions)

Two passes on the system's draw, from what 2026-11-14 (2) measured:
- `--rest-max-run 8` (sampler.rest_bars max_run): the rest pass leaves a bar when clearing
  it would leave 8 beats or more with nothing at all around it (longest_empty_run), and puts
  the bar back as it was. The fatal stretches began in the draw (13 of 18) and the pass
  joined and lengthened them (songs with one: 3.2 bars cleared, the others 0.7); a single
  emptied bar between notes stays allowed (human charts leave 0.85% of their bars empty).
- `--answer 0` (sampler.answer_beats, after align_alike): every beat is offered the rhythm of
  the beat before it (3+ onset rows, no long note across either's edges, the onset count
  within 15% or 1, not already the same rhythm); one forward pass with the beat MASK scores
  both rhythms row by row (onset_count_logp, as copy_bars), and the earlier one comes in
  when it scores at least as well (bias 0 nats). Its lanes are drawn again by the model
  (refine_rows at the lane temperature 1, the lane guidance): the model, which knows the
  mirror, decides between the mirror, the repeat and the rest.

`run_1124.sh` (after run_1123): F's passes with each, and with both, on the system's draw
of the 240 val songs (full-v4); the 120-song fresh judges (full and rhythm view) on the first
120 against the system's s1_sh / s1_sh_rhythm; fatal.py on all 240 (the device VM).

Predictions (written before it ran), against the system after F's passes on the 240:
1. Rest guard: songs with an 8+ beat empty stretch 18% -> 11-15%, with a fatal one 5% ->
   2-4%; rest bars cleared a song 0.8 -> 0.5-0.7; the judges within ±0.02 (n.s.).
2. Answers: the next beat of the same rhythm 14.6% -> 18-26%; of those mirrored 7.2% -> 7-12%,
   the same 5.0% -> 4-8%; notes within ±3%; 15-60 beats answered a song; full view -0.04 to
   +0.01, rhythm view -0.06 to +0.01.
3. Both: the two above together, neither undoing the other.
Each is taken if its full view's step lies under 0 at 95%; if not, it stays a candidate when
its rhythm view's does, or when the fatal stretches fall and the full view is not worse.
How it could fail: the model scores the rhythm before as worse almost everywhere (few
answers: the draw's beats are what the model likes), or answers too often with exact copies
(repeats over the human 3.4-4.9%), which a judge sees as a pattern copied.

Results of 2026-11-15 (1), run_1123 (11/15 20:00 - 11/16 00:15; the first 120 val songs):

| | human | system | stems-v1 | prediction |
|---|---|---|---|---|
| val_ce (full-v4 0.07396; ft-v1, 20k more steps without the stems, 0.07260) | | | 0.07209 | 0.0715-0.0738 ✓, under ft-v1 ✓ (by 0.0005) |
| val_ce@0.9 (full-v4 0.17335, ft-v1 0.17144) | | | 0.17124 | 0.160-0.172 ✓, falls the most ✗ |
| fresh judge, full view | | 0.743 | 0.734, -0.008 [-0.068, +0.049] | -0.06 to +0.01 ✓ |
| fresh judge, rhythm view | | 0.603 | 0.680, +0.077 [+0.008, +0.152] | -0.06 to +0.02 ✗ |
| songs with a fatal empty stretch, after the passes | 0% | 6% | 8% | 0-4% ✗ |
| songs with an 8+ beat empty stretch, after | 8% | 18% | 21% | 10-17% ✗ |
| the next beat of the same rhythm, after | 26.6% | 14.6% | 14.4% | 13-18% ✓ |

Reading: not taken. ft-v1 shows most of the val_ce gain was the 20k steps more (full-v4 was
not done learning); the stems added 0.0005, so 20k steps from zero weights did not teach the
model to use them. Whether the rhythm view's step is the stems' or the retraining's (the row
biases and the choosing judge were set on full-v4's draws) is not known: ft-v1 was not drawn.

Results of 2026-11-15 (2), run_1124 (11/16 00:20 - 02:37; F's passes on the system's draw of
the 240 val songs; judges on the first 120 against s1_sh / s1_sh_rhythm):

| | human | system | rest guard | answers | both | prediction |
|---|---|---|---|---|---|---|
| songs with an 8+ beat empty stretch | 8% | 18% | 12% | 18% | 12% | guard 11-15% ✓ |
| songs with a fatal one | 0% | 5% | 3% | 5% | 3% | guard 2-4% ✓ |
| longest empty stretch, 90th pct (beats) | 7.9 | 14.0 | 10.0 | 14.0 | 10.0 | |
| rest bars cleared a song (kept by the guard) | | 0.8 | 0.47 (0.28) | | 0.47 (0.28) | 0.5-0.7 ✗ (just under) |
| the next beat of the same rhythm | 25.0% | 14.6% | 14.6% | 26.6% | 26.6% | answers 18-26% ✗ (just over) |
| of those mirrored / the same | 10.7 / 3.4% | 7.2 / 5.0% | | 11.8 / 6.8% | 11.8 / 6.8% | 7-12 ✓ / 4-8 ✓ |
| beats answered a song; notes | | | | 16.2; +0.2% | 16.2 | 15-60 ✓; ±3% ✓ |
| fresh judge, full view | | 0.743 | +0.001 [-0.003, +0.004] | +0.024 [-0.003, +0.055] | +0.023 [-0.003, +0.053] | guard ±0.02 ✓; answers -0.04 to +0.01 ✗ |
| fresh judge, rhythm view | | 0.603 | +0.013 [-0.003, +0.028] | +0.044 [+0.001, +0.090] | +0.034 [-0.013, +0.080] | guard ±0.02 ✓; answers -0.06 to +0.01 ✗ |

The diagnostics judge (logistic / trees): 0.889 / 0.950 for all four. Reading: the rest guard
takes out what the pass added to the empty stretches at no cost to the judges (the 3% left
were empty in the draw): a candidate for the system (the rule: the fatal stretches fall, the
full view not worse). The answers bring the beat-to-beat rhythm and the mirror to the human
rates, but the rhythm judge likes them less: answering wherever the model does not object
(bias 0) copies beats the music does not ask for, and the exact repeats run at twice the
human rate. Not taken at bias 0.

### 2026-11-16 (1) · The stems learned from the start; the answers made stricter (predictions, results)

(1) stems-v2: `train.py --stems` from the start on full-v4's recipe exactly (60k steps, lr
3e-4, warmup 1000, batch 16, row-mask 0.5, the chart stats, seed 0), so full-v4 is its
control; the system's draw with it on the first 120 val songs, F's passes, the fresh judges
against s1_sh / s1_sh_rhythm. (2) `--answer -1` and `-2` with `--rest-max-run 8` on the
system's draw of the 240 (full-v4), against the rest guard alone (s1_rgf 0.743, s1_rgf_rhythm
0.616). `run_1125.sh`.

Predictions (written before it ran):
1. stems-v2's val_ce 0.0720-0.0738 (full-v4 0.07396), val_ce@0.9 0.166-0.173 (0.17335);
   training 3.2-3.8 h.
2. Its songs with a fatal empty stretch after F's passes: 2-6% (the system 6% on these 120).
3. Its fresh judges against the system: full view -0.06 to +0.02, rhythm view -0.04 to +0.06.
4. Answers at -1 / -2: 6-12 / 3-8 beats a song; the next beat of the same rhythm 18-24% /
   16-21%; exact repeats among them 5-7%; against the guard alone, full view -0.02 to +0.02,
   rhythm view -0.02 to +0.03.
Either is taken if its full view lies under 0 at 95%; stems-v2 also stays if its val_ce is
under full-v4's by 1%+ and its judges are not worse (then a GPU sweep of its training).

Results, run_1125 (11/16 03:13 - 10:01):

| | human | system (full-v4) | stems-v2 | prediction |
|---|---|---|---|---|
| val_ce | | 0.07396 | 0.07269 (-1.7%) | 0.0720-0.0738 ✓ |
| val_ce@0.9 | | 0.17335 | 0.17198 | 0.166-0.173 ✓ |
| training | | | 3.6 h | 3.2-3.8 h ✓ |
| fresh judge, full view (first 120) | | 0.743 | 0.826, +0.083 [+0.037, +0.133] | -0.06 to +0.02 ✗ |
| fresh judge, rhythm view | | 0.603 | 0.731, +0.128 [+0.076, +0.190] | -0.04 to +0.06 ✗ |
| diagnostics judge, logistic / trees | | 0.854 / 0.939 | 0.849 / 0.945 | |
| songs with a fatal empty stretch, after the passes | 0% | 6% | 8% | 2-6% ✗ |
| the next beat of the same rhythm, after | 25.0% | 14.6% | 10.9% | |
| hold share (song mean) | 0.225 | 0.271 | 0.315 | |
| density ratio; sr_error; f1@50 | | 1.008; 0.226; 0.382 | 1.036; 0.280; 0.386 | |

| answers (with the rest guard, against it alone: s1_rgf 0.743 / 0.616) | bias -1 | bias -2 | prediction (-1 / -2) |
|---|---|---|---|
| beats answered a song | 6.7 | 3.4 | 6-12 ✓ / 3-8 ✓ |
| the next beat of the same rhythm (240; human 25.0%, guard 14.6%) | 19.7% | 17.2% | 18-24% ✓ / 16-21% ✓ |
| of those mirrored / the same (human 10.7 / 3.4%) | 9.7 / 5.9% | 9.0 / 5.7% | the same 5-7% ✓ |
| songs with a fatal empty stretch | 3% | 3% | |
| fresh judge, full view | +0.021 [-0.002, +0.048] | +0.011 [-0.000, +0.023] | -0.02 to +0.02 ✗ (just over) / ✓ |
| fresh judge, rhythm view | +0.057 [+0.015, +0.110] | +0.025 [-0.014, +0.065] | -0.02 to +0.03 ✗ / ✓ |

Reading: stems-v2 not taken as it stands. Its val_ce is the stems' first like-for-like gain
(-1.7% at the same 60k steps), but its charts through the system are clearly worse: the
system's settings were set on full-v4's draws (hold bias 0.2, the loud and onset biases, the
choosing judge sel_holds trained on full-v4's draws, F's passes), and on stems-v2 they give
too many long notes (0.315, the human 0.225, the system 0.271) and a looser SR. Whether the
stems help the charts is open until the settings are set again for it (the hold bias first;
a raw draw of each model with nothing on top tells the models apart) - each a 2-3 h run here.
The answers: fewer of them cost the judges less, but none of the three biases helps either
view; the beat answered from the one before is itself the tell. Not taken; the pass is left
off.

### 2026-11-16 (2) · stems-v2 or the system around it: the judges' trunk, and a raw draw of each model (predictions, results)

run_1125 left two things set on full-v4 between stems-v2 and its charts' verdict: (1) the
fresh judges start from full-v4's trunk (judge.py --ckpt; trained 2 epochs, lr 1e-4), which
may see full-v4's own kind of mistake less; (2) the system's settings (hold bias 0.2, the loud
and onset biases, the choosing judge sel_holds trained on full-v4's draws, F's passes), with
which stems-v2's charts come out with too many long notes (0.315; human 0.225, the system
0.271). judge.py now reads the stems when --ckpt is a stems model, so a judge can start from
stems-v2's trunk.

`run_1126.sh`: (1) fresh full- and rhythm-view judges from stems-v2's trunk (t2_) on the
system's run (s1_sh's) and on stm2f (s1_stm2's); (2) a raw draw of each model on the first 120
val songs with the system's decoding only (cosine, forward lanes at temperature 1, refine 2,
lane guidance 2, the oracle stats, sample seed 0, one draw a window: no choice, no bias, no
pass), judged from both trunks (s1_raw4 / s1_raw2, t2_raw4 / t2_raw2); fatal.py on both.

Predictions (written before it ran):
1. From stems-v2's trunk, stm2f against the system: full view +0.03 to +0.10, rhythm view
   +0.06 to +0.15 (most of run_1125's gap stays: the long notes and the SR are in the charts).
2. full-v4's raw draw, full view 0.78-0.88 from full-v4's trunk (the system's 0.743 is what the
   biases, the choice and the passes add).
3. stems-v2's raw draw minus full-v4's: full view -0.05 to +0.03, rhythm view -0.05 to +0.04,
   from either trunk; its hold share above full-v4's by 0.02-0.06; songs with a fatal empty
   stretch 2-6% for both.
stems-v2 goes on to be tuned (the hold bias, then a choosing judge on its own draws) only if
its raw draw is ahead in the full view from both trunks (under 0 at 95% from one, not over 0
from the other); otherwise full-v4 stays the system through the freeze (11/29) and the stems
wait for a GPU. If the trunk changes the verdict on run_1125's pair, every judge comparing two
models is read from both trunks from here on.

Results, run_1126 (11/16 19:28 - 11/17 03:21; the first 120 val songs):

| | full-v4's trunk (s1_) | stems-v2's trunk (t2_) | prediction |
|---|---|---|---|
| the system, full view | 0.743 | 0.833 | |
| stm2f minus the system, full view | +0.083 [+0.037, +0.133] | +0.023 [-0.009, +0.058] | t2 +0.03 to +0.10 ✗ (under) |
| the system, rhythm view | 0.603 | 0.757 | |
| stm2f minus the system, rhythm view | +0.128 [+0.076, +0.190] | +0.032 [-0.008, +0.075] | t2 +0.06 to +0.15 ✗ (under) |
| full-v4's raw draw, full view | 0.885 | 0.918 | s1 0.78-0.88 ✗ (just over) |
| stems-v2's raw minus full-v4's, full view | +0.019 [-0.012, +0.050] | -0.013 [-0.039, +0.010] | -0.05 to +0.03 ✓ ✓ |
| full-v4's raw draw, rhythm view | 0.767 | 0.863 | |
| stems-v2's raw minus full-v4's, rhythm view | +0.067 [+0.029, +0.107] | +0.000 [-0.028, +0.029] | -0.05 to +0.04 ✗ ✓ |

The diagnostics judge (no trunk; logistic / trees) on the raw draws: full-v4 0.967 / 0.975,
stems-v2 0.965 / 0.981. The raw draws (human / full-v4 / stems-v2): hold share 0.225 / 0.159 /
0.198 (prediction +0.02 to +0.06 ✓, +0.039); density ratio 1.056 / 1.067; sr_error 0.185 /
0.214; f1@50 0.376 / 0.377; songs with a fatal empty stretch 1% / 2% (2-6% ✗ / ✓).

Where the empty stretches come from (fatal.py on the same 120 songs, full-v4):

| | human | raw draw | + the biases (run_1122's rnd) | + the choice among 4 (sys draw) | + F's passes (the system) | + the rest guard |
|---|---|---|---|---|---|---|
| songs with an 8+ beat empty stretch | 8% | 6% | 12% | 11% | 18% | 12% |
| songs with a fatal one | 0% | 1% | 2% | 3% | 6% | 3% |
| longest empty stretch, 90th pct (beats) | 7.9 | 7.0 | 9.6 | 8.1 | 13.0 | 8.1 |
| the next beat of the same rhythm | 26.6% | 9.2% | 9.3% | 12.4% | 14.6% | 14.6% |

Reading: each judge favours its own trunk's model, by about 0.03 in the full view (both ways)
and up to 0.07-0.10 in the rhythm view from full-v4's; run_1125's verdict was mostly that. On
raw draws the two models are level (the full view +0.019 / -0.013 from the two trunks, the
diagnostics judge level), so the stems' -1.7% val_ce does not show in the charts: stems-v2 is
not tuned (the rule: ahead from both trunks), full-v4 stays the system through the freeze,
and the stems wait for a GPU (longer training, its own settings). Model comparisons are read
from both trunks from here on. stems-v2's trunk is the sharper judge (the system 0.833 against
0.743): a candidate trunk for the choosing judge. The empty stretches are not the model's:
the raw draw has fewer than the human charts; the loud and onset biases double them (they push
quiet bars to empty), the passes add the rest, and the rest guard takes the passes' share
back out.

### 2026-11-17 (1) · The patterns followed: a unit's first row drawn freely, the rest lower (predictions, results)

The pilot (11/14), again after run_1126: the AI's patterns are hard to make out, and so their
repeats, mirrored or varied, do not show either (not stairs alone). On the first 120 val songs
(patterns.py; human / the system of 11/09, lanes at 0.5 / the system, lanes at 1 since 11/10):

| | human | lanes at 0.5 | lanes at 1 (the system) |
|---|---|---|---|
| motion_pred (the lane motion's predictability over chance: stairs, trills, unnamed motifs) | 0.128 | 0.139 | 0.085 |
| run length (events a run) | 7.1 | 8.8 | 5.7 |
| coverage (of the events in runs; chance-corrected repeats) | 0.042 | 0.083 | 0.029 |
| moves: stair / trill / jack / leap | 0.199 / 0.152 / 0.051 / 0.127 | 0.163 / 0.173 / 0.043 / 0.131 | 0.148 / 0.197 / 0.040 / 0.139 |
| the next beat of the same rhythm; of those mirrored | 26.6%; 8.4% | 18.3%; 9.4% | 14.6%; 8.2% |

The mirrored share of repeated rhythms is human already; what is missing is the figure. One
temperature for every row moves the charts along a line from blurred (1) to held too long
(0.5); the human charts are on neither end: a figure followed while it lasts, a new one, or the
last one mirrored or varied, where it ends. sampler.unit_temperature: the lane passes draw the
first onset row of each unit (a beat, `--lane-unit 12`) at the lane temperature and its later
onset rows at `--lane-inner-temp`, so the figure is picked freely and then followed (forward
lanes sees the lanes chosen so far; refine sees both sides).

`run_1127.sh`: the system's draw (run_1119's sys) on the first 120 val songs, its lanes drawn
again (`--relane`: forward lanes, refine 2, lane guidance 2) at 1 (ul0, the control), in units
of a beat at 0.5 (ul1), in units of two beats at 0.5 (ul2), each with the rest guard and F's
passes; fresh full- and taps-view judges, each against ul0.

Predictions (written before it ran), against ul0 (about the system: motion_pred 0.085, run
length 5.7, coverage 0.029):
1. ul1: motion_pred 0.10-0.13, run length 6.5-8.0, coverage 0.035-0.060; trills down to
   0.16-0.19, stairs up to 0.15-0.19; the mirrored share of repeated rhythms 7-12%; judges
   full view -0.03 to +0.02, taps view -0.04 to +0.04.
2. ul2: further: motion_pred 0.11-0.14, coverage 0.04-0.07, run length 7-9; taps view -0.02 to
   +0.06 (towards 11/09's held-too-long patterns).
A setting is taken if its pattern numbers land nearer the human charts than ul0's (motion_pred
within 0.02, coverage under 1.5 times the human) and its full view's step is not over 0 at 95%;
then the user plays it against ul0 on a few songs. How it could fail: a beat is too short for a
figure (most runs cross beats), so ul1 changes little; or the lower rows turn every figure into
the one the model likes best (trills), which the move mix shows.

Results, run_1127 (11/17 04:4x - 10:04; the first 120 val songs):

| | human | ul0 (all at 1) | ul1 (a beat) | ul2 (two beats) | prediction (ul1 / ul2) |
|---|---|---|---|---|---|
| motion_pred | 0.128 | 0.084 | 0.099 | 0.107 | 0.10-0.13 ✗ (just under) / 0.11-0.14 ✗ (just under) |
| run length | 7.1 | 6.0 | 6.7 | 7.0 | 6.5-8.0 ✓ / 7-9 ✓ |
| coverage | 0.042 | 0.032 | 0.039 | 0.047 | 0.035-0.060 ✓ / 0.04-0.07 ✓ |
| moves: stair / trill | 0.199 / 0.152 | 0.154 / 0.203 | 0.164 / 0.184 | 0.162 / 0.188 | stair 0.15-0.19 ✓, trill 0.16-0.19 ✓ |
| jack / leap | 0.051 / 0.127 | 0.042 / 0.134 | 0.037 / 0.133 | 0.039 / 0.137 | |
| the next beat of the same rhythm | 26.6% | 14.3% | 14.0% | 14.7% | |
| of those mirrored / the same | 8.4 / 3.6% | 6.5 / 3.7% | 14.7 / 7.0% | 17.1 / 9.0% | mirrored 7-12% ✗ |
| songs with a fatal empty stretch | 0% | 3% | 3% | 3% | |
| fresh judge, full view | | 0.773 | +0.023 [-0.016, +0.057] | +0.025 [-0.008, +0.053] | -0.03 to +0.02 ✗ (just over) |
| fresh judge, taps view | | 0.695 | +0.006 [-0.023, +0.033] | +0.003 [-0.023, +0.026] | ✓ / ✓ |
| diagnostics judge, logistic / trees | | 0.858 / 0.941 | 0.811 / 0.923 | 0.819 / 0.921 | |

Reading: the units move every pattern number towards the human charts (ul2: the run length
human, motion_pred half way, the trills fewer), the diagnostics judge (no trunk) likes both
(-0.04 to -0.05 logistic), and the taps view, which put 11/09's lanes at 0.5 everywhere 0.095
behind lanes at 1, is level here: the figure is followed without the whole chart held. Two
costs: the full view leans worse (+0.02, not at 95%), and a beat that repeats the last one's
rhythm now mirrors or copies its lanes twice as often as a human chart (the model's own choice
once the rows are drawn lower). By the rule neither is taken (motion_pred misses "within 0.02"
by 0.009 / 0.001); the user plays them blind (outputs/lanes_try_1117: five songs not in a
playtest pack, ul0 / ul1 / ul2 as X / Y / Z in a different order each song, the human chart
named; the key in key_open_after_playing.txt) before it is decided.

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
