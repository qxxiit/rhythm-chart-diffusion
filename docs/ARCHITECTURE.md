# Code Architecture

> Map of the `src/` modules and how they fit together.

## Overview

```
audio (.mp3) ──┐
               ├──> [data/preprocess.py] ──> mel-spectrogram
osu (.osu) ────┘                              chart tokens
                                                   │
                                                   ▼
                                          [data/dataset.py]
                                          PyTorch Dataset
                                                   │
                                                   ▼
                                       [training/trainer.py]
                                                   │
                                      ┌────────────┴────────────┐
                                      ▼                         ▼
                            [models/transformer.py]   [models/diffusion.py]
                            (Phase 1 baseline)        (Phase 2 main)
                                      │                         │
                                      └────────────┬────────────┘
                                                   ▼
                                       [evaluation/*.py]
                                       F1, mAP, diversity
                                                   │
                                                   ▼
                                       [unity_export/]
                                       Unity JSON chart
```

## Module Details

### `src/data/`

- **`download.py`**: Fetches osu!mania beatmaps via the osu! API. Resumable, rate-limit aware.
- **`osu_parser.py`**: Parses `.osu` text files (hit objects, timing points, metadata).
- **`preprocess.py`**: Extracts mel-spectrogram, applies beat alignment, computes BPM-based hop length.
- **`tokenizer.py`**: Converts chart events to integer tokens and back. Vocab definition lives here.
- **`dataset.py`**: PyTorch `Dataset` returning (spectrogram, chart tokens) pairs. Handles train/val/test split (song-level).

### `src/models/`

- **`encoders.py`**: Audio encoders. Default: small 1D CNN. Pluggable: MERT / AST (Phase 3 ablation).
- **`transformer.py`**: Autoregressive Transformer encoder-decoder (Yi 2023 baseline).
- **`diffusion.py`**: Discrete diffusion denoiser. Includes forward process, reverse sampling.

### `src/training/`

- **`trainer.py`**: Training loop. Optimizer, scheduler, checkpoint, logging.
- **`losses.py`**: Cross-entropy for AR, masked CE for diffusion.

### `src/evaluation/`

- **`f1_metric.py`**: Donahue 2017 F1 protocol with ±30ms tolerance.
- **`tiou_map.py`**: Temporal IoU mean Average Precision (TAL style).
- **`diversity.py`**: n-gram diversity, lane balance entropy.

### `src/unity_export/`

- **`chart_to_json.py`**: Converts model output tokens to our Unity game's JSON chart format.

### `src/utils/`

- **`logging.py`**: Wandb wrapper, structured logging.

## Config (Hydra)

Configs live in `configs/`. Composition:

```yaml
# configs/config.yaml (default)
defaults:
  - model: transformer_baseline
  - data: osu_mania_4k
  - training: default
  - _self_

# Override at command line:
# python scripts/train.py model=diffusion training.batch_size=8
```

## Entry Points (scripts/)

All scripts use Hydra and load configs from `configs/`.

```bash
python scripts/download_data.py     # Phase 0–1: data acquisition
python scripts/preprocess_data.py   # Phase 1: turn raw → tensors
python scripts/train.py             # Phase 1–3: training
python scripts/evaluate.py          # Phase 1–3: evaluation
python scripts/inference.py         # Phase 4: single-song demo
```
