# Experiment Log

> One entry per significant training run. Newest at top.
> Add results to `paper/` when ready for the paper.

| Date | Run ID | Phase | Model | Key Config | Val F1 | Test F1 | wandb | Notes |
|------|--------|-------|-------|------------|--------|---------|-------|-------|
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

## Phase 3 Ablation A: Diffusion Design

_TBD — target Oct 14, 2026._

## Phase 3 Ablation B: Conditioning

_TBD — target Oct 28, 2026._

## Phase 3 Ablation C + Multi-key

_TBD — target Nov 15, 2026._
