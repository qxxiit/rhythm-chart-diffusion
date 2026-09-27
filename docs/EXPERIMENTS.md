# Experiment Log

> One entry per significant training run. Newest at top.
> Add results to `paper/` when ready for the paper.

| Date | Run ID | Phase | Model | Key Config | Val F1 | Test F1 | wandb | Notes |
|------|--------|-------|-------|------------|--------|---------|-------|-------|
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

## Phase 3 Ablation C + Multi-key

_TBD — target Nov 15, 2026._
