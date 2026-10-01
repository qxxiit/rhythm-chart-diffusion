# Experiment Log

> One entry per significant training run. Newest at top.
> Add results to `paper/` when ready for the paper.

| Date | Run ID | Phase | Model | Key Config | Val F1 | Test F1 | wandb | Notes |
|------|--------|-------|-------|------------|--------|---------|-------|-------|
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
  3 or 6 cells long (21, 49 and 39 of them), B has 10. The model does not know
  a tap chart from a long-note chart and spreads holds over every song; short
  holds come from the sampler closing holds cell by cell (DECISIONS 2026-10-01,
  long notes).

Next: `pattern_probe.py` (does the model know the lanes when the rest of the
chart is visible?) and `evaluate.py --refine 2` at lane temperatures 0.5 and 1
against T = 128 alone, with the replicate (`--sample-seed 1`) for the noise;
`hold_stats.py` for the human long-note levels behind the clean-up defaults.

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
