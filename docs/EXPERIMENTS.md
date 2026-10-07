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
