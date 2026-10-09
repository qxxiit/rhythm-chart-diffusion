# Audio-Conditioned Discrete Diffusion for Rhythm Game Chart Generation

> Discrete diffusion model for generating rhythm-game charts from audio.
> POSTECH UGRP 2026. **Open source** from day one — contributions and feedback welcome.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)
[![Status: Pre-alpha](https://img.shields.io/badge/status-pre--alpha-orange.svg)](MASTERPLAN.md)

## Overview

Rhythm game charts (note placement data) are normally authored by hand, taking hours to tens of hours per song. This project develops an automatic chart generation system: given an audio file (mp3/wav), the model produces a playable chart.

Existing learning-based approaches generate charts token by token (autoregressive), which makes it hard to maintain structural consistency across a whole song and offers little control over difficulty or pattern style. We investigate **discrete diffusion** as an alternative — generating the note sequence as a whole and refining it iteratively, conditioned on audio.

The project compares two approaches on the same dataset:

1. **Baseline** — Beat-Aligned autoregressive Transformer (Yi et al., ISMIR 2023), reproduced as a reference point
2. **Proposed** — Audio-conditioned discrete diffusion (D3PM-style masking with a cross-attention denoiser)

See [`MASTERPLAN.md`](MASTERPLAN.md) for the full project plan.

## Status

🚧 **In development.** Started July 2026. Target completion: January 2027.

| Phase | Status |
|---|---|
| Phase 0 — Setup & data pipeline | ✅ **Complete** (Aug 2026) |
| Phase 1 — Problem formulation & baseline | 🟡 Formulation, tokenizer, split, audio pipeline and evaluation done; AR baseline on hold (generator first, Sep 30) |
| Phase 2 — Discrete diffusion | 🟡 D-32 trained on the full training set (val CE 0.072) and evaluated on 240 val songs; sampler fixed; charts for any audio file (Sep 2026) |
| Phase 3 — Ablation & multi-key | ⚪ Not started (audio ablations of the subset and full models are in `docs/EXPERIMENTS.md`) |
| Phase 4 — Unity integration & user study | ⚪ Not started |
| Phase 5 — Paper | ⚪ Not started |

### Dataset (built Aug 2026)

| | |
|---|---|
| Ranked osu!mania beatmapsets surveyed | 7,363 |
| Sets containing 4K difficulties | 5,882 |
| Sets downloaded | 5,882 (0 failures) |
| **4K charts (`.osu`) parsed** | **18,452** (0 parse failures) |

Collection, extraction, and parsing are fully scripted and reproducible. Raw beatmap data is **not redistributed** — the pipeline downloads from the official API and community mirrors using the beatmapset ID list, so anyone can rebuild the same dataset from scratch.

## Quick Start

### Requirements

- Python ≥ 3.11
- PyTorch ≥ 2.0 (CUDA or MPS)
- ~30 GB free disk space (`.osz` archives are discarded after extracting 4K charts and audio)
- osu! API v2 credentials (`OSU_CLIENT_ID`, `OSU_CLIENT_SECRET`) — register an OAuth application at <https://osu.ppy.sh/home/account/edit>

### Setup

```bash
git clone https://github.com/qxxiit/rhythm-chart-diffusion.git
cd rhythm-chart-diffusion

# Environment (choose one)
conda env create -f environment.yml && conda activate rhythm-chart   # Option A
python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt   # Option B

cp .env.example .env    # then fill in OSU_CLIENT_ID / OSU_CLIENT_SECRET

python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

### Building the Dataset

The pipeline runs in four stages; each step is resumable and skips work already done.

```bash
# 1. Fetch metadata for all ranked mania beatmapsets (cursor pagination, resumable)
python scripts/fetch_metadata.py

# 2. Filter to sets containing 4K difficulties -> data/metadata/targets.txt
python scripts/make_target_list.py

# 3. Download .osz archives (mirror fallback, rate limited, logged)
python scripts/download_data.py --output data/osz

# 4. Extract 4K .osu charts + audio only, discard the rest
python scripts/extract_osz.py
python scripts/build_catalog.py       # -> data/metadata/catalog.csv
```

Downloading the full set takes several hours. Run steps 3 and 4 concurrently to keep disk usage flat — `extract_osz.py` removes each archive once its charts are extracted.

### Parsing Charts

```python
from src.data.chart_parser import parse_osu

chart = parse_osu("data/raw/1154776/xxx.osu")
chart.key_count   # 4
chart.bpm         # 202.0
chart.notes       # [Note(time_ms, lane, end_ms), ...]  end_ms set for hold notes
```

### Training and Inference

```bash
python scripts/build_manifest.py        # data/manifest.csv: cache key, split by song, SR
python scripts/preprocess_data.py       # token cache
python scripts/preprocess_data.py --mel # log-Mel per audio file (see docs/ARCHITECTURE.md)
python scripts/train.py --overfit 10    # milestone check; drop --overfit for a full run
python scripts/build_style.py           # data/style.csv: genre and mapper per chart
python scripts/train.py --steps 60000 --val-every 2000 --style data/style.csv --run full-v2
python scripts/build_chart_stats.py     # data/chart_stats.csv: long-note share, jack and trill rate per chart
python scripts/train.py --steps 60000 --val-every 2000 --row-mask 0.5 \
    --chart-stats data/chart_stats.csv --run full-v4   # those as inputs, like the SR
python scripts/sample.py --ckpt outputs/<run>/best.pt --key <key>   # a dataset song -> .osu + .osz
python scripts/evaluate.py --ckpt outputs/<run>/best.pt --per-song --n 0   # F1, SR error, rho, patterns
python scripts/compare_runs.py <eval dir> <eval dir> ...   # paired over songs, 95% intervals
python scripts/rescore.py <eval dir> ...   # new pattern columns for an earlier run (charts.npz)
python scripts/evaluate.py --ckpt ... --per-song --n 0 --copy-bias 0 --from-charts <eval dir>
                                        # bar copies on saved charts, no new sampling
python scripts/train.py --steps 60000 --val-every 2000 --row-mask 0.5 \
    --chart-stats data/chart_stats.csv --snapshot-every 1000 --run fit-v4   # the path saved
python scripts/fit_viz.py outputs/fit-v4   # the path on the loss surface (its PCA plane),
                                        # the straight line, Adam's step per coordinate
```

The cache layout is in `src/data/cache.py`. Not implemented: the AR baselines (on hold).

### Make a chart for your own song

```bash
python scripts/generate.py --audio "song.mp3" --sr 2 3.5 5          # tempo and offset estimated
python scripts/generate.py --audio "song.mp3" --bpm 174 --offset 1234 --sr 3.5
python scripts/generate.py --audio "song.mp3" --timing timed.osu --sr 3.5   # red lines of an .osu
python scripts/generate.py --audio "song.mp3" --sr 3.5 --lanes sampled --refine 0   # raw sampler output
python scripts/generate.py --audio "song.mp3" --sr 3.5 --hold-share 0      # no long notes (0.3: a long-note chart)
python scripts/generate.py --ckpt outputs/full-v2/best.pt --list-styles     # genres and mappers it knows
python scripts/generate.py --ckpt outputs/full-v2/best.pt --audio "song.mp3" --sr 3.5 \
    --genre electronic --mapper <name>  # a model trained with --style
python scripts/generate.py --ckpt outputs/full-v4/best.pt --audio "song.mp3" --sr 3.5 \
    --stats ln=0.4,jack=0.02            # a model trained with --chart-stats: a long-note chart, few jacks
python scripts/generate.py --ckpt outputs/full-v4/best.pt --audio "song.mp3" --sr 3.5 --stats sample \
    --lane-guidance 1                   # the style of a random human chart of about that SR; the
                                        # lane passes pushed towards it (jacks, trills)
```

Writes one 4K difficulty per star rating and an `.osz` under `outputs/generated/`; open it with osu! (lazer: double-click). By default the lanes are chosen again after sampling (`--lanes forward --refine 2`): a left-to-right pass, then two sweeps with the whole chart in view, which brought pattern clarity on the val songs to the human level (EXPERIMENTS 2026-10-02/03) at about 1.6x the time. Then bars are copied where the audio repeats (`--copy-bias 0`, `--no-copy` to turn it off), which brought whole-bar repetition to the human rate (EXPERIMENTS 2026-10-03, result of 10-04). Quiet bars are thinned (`--loud-bias 0.1`, 0 to turn off), which brought the loudness dynamics to the human level with the SR unchanged (EXPERIMENTS 2026-10-06), and notes are less likely to start where nothing in the music starts (`--onset-bias 1`, 0 to turn off), which cut the notes off the human rhythm by a fifth with long notes kept (EXPERIMENTS 2026-10-07; about 0.1 SR under the target, the loss in the hard grades). From SR 2.7 up the gate weakens, to 0.35 of it from SR 5 (`--onset-taper 2.7,5,0.35`, `off` for the full gate), which brought the hard grades' SR back near the target (EXPERIMENTS 2026-10-08), and a stream that misses exactly one note gets it where the model would start one with probability 0.5 or more (`--holes 0.5`, 0 to turn off). Last, the bars where the model, asked about the whole bar with the chart around it, expects under one note start are left empty (`--rest 1`, 0 to turn off), which brought the bars without a note to the human count (EXPERIMENTS 2026-10-07 night). The model places notes on the beat grid it is given: the estimate assumes one constant tempo, so for songs whose tempo changes, time the song in the osu! editor and pass `--timing`.

## Repository Structure

```
src/
  data/               # osu! API client, chart parser, tokenizer, dataset classes
  models/             # Model architectures (transformer baseline, diffusion)
  training/           # Training loops, losses
  evaluation/         # F1, mAP@tIoU, diversity metrics
  unity_export/       # Convert model output to Unity JSON
  utils/

configs/              # Hydra configs
scripts/              # Entry points (see pipeline above)
notebooks/            # Exploration and chart visualization
tests/
docs/                 # Architecture, data format, decisions, experiment notes
paper/                # Drafts and reading notes
```

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for module details, [`docs/DATA_FORMAT.md`](docs/DATA_FORMAT.md) for the chart representation, and [`docs/DECISIONS.md`](docs/DECISIONS.md) for design decisions and known technical debt.

## Citation

```bibtex
@misc{rhythm-chart-diffusion-2026,
  title  = {Rhythm Game Chart Generation via Audio-Conditioned Discrete Diffusion},
  author = {Moon, Jeonghyun and Ji, Hyunwoo and Pyo, Youngbok},
  year   = {2026},
  note   = {POSTECH UGRP},
  url    = {https://github.com/qxxiit/rhythm-chart-diffusion}
}
```

## References

- [1] Donahue, Lipton, McAuley. *Dance Dance Convolution.* ICML 2017.
- [2] Takada et al. *GenéLive!: Generating Rhythm Actions in Love Live!* AAAI 2023.
- [3] Yi, Lee, Lee. *Beat-Aligned Spectrogram-to-Sequence Generation of Rhythm-Game Charts.* arXiv:2311.13687, 2023.
- [4] Austin et al. *Structured Denoising Diffusion Models in Discrete State-Spaces (D3PM).* NeurIPS 2021.
- [5] Ho, Jain, Abbeel. *Denoising Diffusion Probabilistic Models.* NeurIPS 2020.

## License

MIT License — see [`LICENSE`](LICENSE). The license covers this code only; beatmap content remains under its original authors' terms and is not distributed here.

## Contributing

This is primarily a student research project, but we welcome:
- Issue reports (bugs, unclear documentation, dataset/preprocessing edge cases)
- Suggestions on model design, evaluation metrics, or related literature
- PRs for fixes or improvements (please open an issue first to discuss)

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for workflow conventions.

## Acknowledgments

- Mentor: Prof. Wonhwa Kim (POSTECH CSE)
- POSTECH Center for Innovative Education (UGRP)
