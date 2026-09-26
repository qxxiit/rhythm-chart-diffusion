# Code Architecture

> Map of the `src/` modules and how they fit together.
> Status: ✅ implemented · 🚧 in design · ⚪ not started. Last updated 2026-09-26.

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
[data/preprocess.py] 🚧        [data/tokenizer.py] ✅
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
- ✅ **`tokenizer.py`**: `encode` / `decode` between charts and the `[n_chunks, 384, 4]` token grid (1/12-beat cells re-originated at each red line, vocabulary of 7), `grammar_violations`, and the onset-first collision rule. Full-dataset check: `scripts/validate_tokenizer.py`. See design doc §2.
- ✅ **`chart_writer.py`**: `Chart` → `.osu` that osu! opens as a local mania difficulty.
- ✅ **`cache.py`**: the `data/cache` file layout that the token stage, the mel stage and the Dataset share (no torch).
- ✅ **`dataset.py`**: `ChunkDataset` of (x0, mel, s, b) chunks read from `data/cache`.
- 🚧 **mel pipeline**: fixed-hop log-Mel resampled at `BeatGrid.frame_times` (4 frames per token cell, 1/48 beat inside a timing section), written to `data/cache/mel/{key}.npy` in the layout above.

> Data acquisition currently lives in `scripts/` rather than `src/data/`, since each stage is a standalone batch job with its own resume state. Reusable pieces (API client, parser) are in `src/data/`.

### `src/models/`
- ✅ **`diffusion.py`**: D-32 denoiser (Conv1D audio encoder, 6 blocks of self-attention + cross-attention + FFN, SR and tempo embeddings, ~6.87M parameters), the absorbing forward process and the continuous-time loss. Design doc §4.7-4.10; deliberate differences are listed in the module docstring.
- ✅ **`sampler.py`**: reverse process with grammar-constrained unmasking (random or confidence order) and song generation by continuation or independent chunks (§4.8).
- ⚪ **`transformer.py`**: AR-4 / AR-32 baselines (Yi et al. setting), W5.

### Training (`scripts/train.py`) ✅
AdamW, warmup + cosine, gradient clipping and accumulation, bf16 autocast on CUDA, validation at fixed mask ratios, `last.pt` / `best.pt`, `--resume`, optional wandb, and `--overfit N` with a rebuild check. Plain `argparse`; the Hydra configs are not used.

### `src/evaluation/`
- ✅ **`metrics.py`**: onset F1 at ±20/±50 ms with lanes on or off (greedy one-to-one matching, closest pairs first) and the grammar violation rate.
- ✅ **`sr.py`**: local star rating with rosu-pp (pinned in `requirements.txt`); `scripts/check_sr.py` measures how closely it tracks the API's SR.
- ⚪ rho (audio/chart self-similarity correlation) and pattern clarity: waiting on the exact definitions in design doc §4.11.
- ⚪ long-note mAP@tIoU: deferred.

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

# Model pipeline
python scripts/validate_tokenizer.py   # ✅ tokenizer invariants on every chart
python scripts/build_manifest.py       # ✅ data/manifest.csv: key, split (by audio_key), SR
python scripts/preprocess_data.py      # ✅ token cache (mel: 🚧 mel pipeline)
python scripts/train.py --overfit 10   # ✅ milestone check; drop --overfit for a full run
python scripts/sample.py --ckpt ... --key ...   # ✅ generate one song -> playable .osu
python scripts/evaluate.py --ckpt ...  # ✅ F1, violation rate, SR error, density (rho: ⚪)
python scripts/check_sr.py             # ✅ local SR vs API SR, and what tokenization moves
python scripts/tempo_density.py        # ✅ tempo vs notes per beat inside each SR grade
```

## Known issues

- `download_data.py` decides "already done" from the presence of `data/osz/{sid}.osz`, but `extract_osz.py` deletes archives after extraction — so a rerun re-downloads everything. Should check `data/raw/{sid}` instead. See [`DECISIONS.md`](DECISIONS.md).
- Progress logs are buffered (no `flush()`), so long runs look stalled; check file mtimes rather than the log.
