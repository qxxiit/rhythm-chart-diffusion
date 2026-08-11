# Audio-Conditioned Discrete Diffusion for Rhythm Game Chart Generation

> Discrete diffusion model for generating rhythm-game charts from audio.
> POSTECH UGRP 2026. **Open source** from day one — contributions and feedback welcome.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
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
| Phase 1 — Problem formulation & baseline | 🟡 In progress — tokenization and model design |
| Phase 2 — Discrete diffusion | ⚪ Not started |
| Phase 3 — Ablation & multi-key | ⚪ Not started |
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

- Python ≥ 3.10
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

Not yet available — the model is still being designed. Entry points (`scripts/train.py`, `scripts/evaluate.py`, `scripts/inference.py`) exist as stubs and will be filled in as Phases 1–2 progress.

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
