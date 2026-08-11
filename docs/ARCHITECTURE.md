# Code Architecture

> Map of the `src/` modules and how they fit together.
> Status: ✅ implemented · 🚧 in design · ⚪ not started. Last updated 2026-08-11.

## Overview

```
osu! API ──> [scripts/fetch_metadata.py] ──> beatmapsets.jsonl        ✅
                       │
                       ▼
             [scripts/make_target_list.py] ──> targets.txt            ✅
                       │
                       ▼
             [scripts/download_data.py] ──> *.osz                     ✅
                       │
                       ▼
             [scripts/extract_osz.py] ──> data/raw/{sid}/             ✅
                       │
       ┌───────────────┴───────────────┐
       ▼                               ▼
audio (.mp3)                      chart (.osu)
       │                               │
       │                    [data/chart_parser.py] ✅
       │                               │
       ▼                               ▼
[data/preprocess.py] 🚧        [data/tokenizer.py] 🚧
 mel-spectrogram                  chart tokens
       └───────────────┬───────────────┘
                       ▼
              [data/dataset.py] ⚪
              PyTorch Dataset
                       │
                       ▼
           [training/trainer.py] ⚪
                       │
          ┌────────────┴────────────┐
          ▼                         ▼
[models/transformer.py] ⚪   [models/diffusion.py] ⚪
(Phase 1 baseline)           (Phase 2 main)
          └────────────┬────────────┘
                       ▼
             [evaluation/*.py] ⚪
             F1, mAP, diversity
                       │
                       ▼
             [unity_export/] ⚪
             Unity JSON chart
```

## Module Details

### `src/data/`
- ✅ **`osu_api.py`**: osu! API v2 client (client-credentials auth, beatmapset search with cursor pagination, 429 handling).
- ✅ **`chart_parser.py`**: Parses `.osu` text into `Chart` / `Note` objects — hit objects, timing points, key count, lane derivation, hold notes. See [`DATA_FORMAT.md`](DATA_FORMAT.md).
- 🚧 **`tokenizer.py`**: Converts chart events to integer tokens and back. Vocabulary definition lives here. *Scheme under design — the decision determines sequence length and the diffusion transition matrix, so it gates everything downstream.*
- 🚧 **`preprocess.py`**: log-Mel extraction with beat-aligned hop (1/48 beat), BPM-derived frame timing.
- ⚪ **`dataset.py`**: PyTorch `Dataset` returning (spectrogram, chart tokens) pairs. Song-level train/val/test split.

> Data acquisition currently lives in `scripts/` rather than `src/data/`, since each stage is a standalone batch job with its own resume state. Reusable pieces (API client, parser) are in `src/data/`.

### `src/models/` ⚪
- **`encoders.py`**: Audio encoders. Default: small 1D CNN. Pluggable: MERT / AST (Phase 3 ablation).
- **`transformer.py`**: Autoregressive Transformer encoder–decoder (Yi et al. 2023 baseline).
- **`diffusion.py`**: Discrete diffusion — forward (absorbing/mask) process, audio-conditioned denoiser, reverse sampling.

### `src/training/` ⚪
- **`trainer.py`**: Training loop. Optimizer, scheduler, checkpointing, logging. Low-budget options (AMP, gradient accumulation) belong here.
- **`losses.py`**: Cross-entropy for AR; masked CE for diffusion.

### `src/evaluation/` ⚪
- **`f1_metric.py`**: Donahue 2017 F1 protocol with ±30 ms tolerance.
- **`tiou_map.py`**: Temporal IoU mean Average Precision (TAL style).
- **`diversity.py`**: n-gram diversity, lane balance entropy.

> Evaluation is shared by both models and should be written once, before either is trained, so baseline and diffusion numbers are directly comparable.

### `src/unity_export/` ⚪
- **`chart_to_json.py`**: Converts model output tokens to the team's Unity JSON chart format.

### `src/utils/` ⚪
- **`logging.py`**: wandb wrapper, structured logging.

## Config (Hydra)

Configs live in `configs/`. Composition:

```yaml
# configs/config.yaml (default)
defaults:
  - model: transformer_baseline
  - data: osu_mania_4k
  - training: default
  - _self_
```

```bash
# Override at the command line:
python scripts/train.py model=diffusion training.batch_size=8
```

> Hydra is wired into `configs/` but not yet used by any running script; the data pipeline scripts take plain `argparse` flags. They will be migrated when training begins.

## Entry Points (`scripts/`)

```bash
# Data pipeline — run in this order
python scripts/fetch_metadata.py       # ✅ metadata for all ranked mania sets
python scripts/make_target_list.py     # ✅ filter to sets with 4K difficulties
python scripts/download_data.py        # ✅ fetch .osz (mirror fallback, resumable)
python scripts/extract_osz.py          # ✅ keep 4K .osu + audio, discard the archive
python scripts/build_catalog.py        # ✅ catalog.csv reference table
python scripts/check_osz.py            # ✅ integrity check on downloaded archives

# Model pipeline — stubs
python scripts/preprocess_data.py      # 🚧 raw → tensors
python scripts/train.py                # ⚪
python scripts/evaluate.py             # ⚪
python scripts/inference.py            # ⚪
```

## Known issues

- `download_data.py` decides "already done" from the presence of `data/osz/{sid}.osz`, but `extract_osz.py` deletes archives after extraction — so a rerun re-downloads everything. Should check `data/raw/{sid}` instead. See [`DECISIONS.md`](DECISIONS.md).
- Progress logs are buffered (no `flush()`), so long runs look stalled; check file mtimes rather than the log.
