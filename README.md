# Rhythm Game Chart Generation via Audio-Conditioned Discrete Diffusion

> ActFusion-inspired discrete diffusion model for generating rhythm-game charts from audio.
> POSTECH UGRP 2026. **Open source** from day one — contributions and feedback welcome.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)
[![Status: Pre-alpha](https://img.shields.io/badge/status-pre--alpha-orange.svg)](MASTERPLAN.md)

## Overview

This project develops an automatic chart generation system for rhythm games. Given an audio file (mp3/wav), the model produces a playable chart for 4–8 key rhythm games. We compare:

1. **Baseline**: Beat-Aligned autoregressive Transformer (Yi et al., ISMIR 2023)
2. **Main contribution**: ActFusion-inspired audio-conditioned discrete diffusion (Gong et al., NeurIPS 2024 → audio domain)

See [`MASTERPLAN.md`](MASTERPLAN.md) for the full 9-month project plan.

## Status

🚧 **In development.** Project started May 2026. Target completion: January 2027.

| Phase | Period | Status |
|---|---|---|
| Phase 0 — Setup | 2026.05 ~ 06 | 🟡 In progress |
| Phase 1 — Baseline | 2026.07 ~ 08 | ⚪ Not started |
| Phase 2 — Diffusion | 2026.08 ~ 09 | ⚪ Not started |
| Phase 3 — Ablation | 2026.10 ~ 11 | ⚪ Not started |
| Phase 4 — Demo + Eval | 2026.11 | ⚪ Not started |
| Phase 5 — Paper | 2026.12 | ⚪ Not started |
| Phase 6 — Submission | 2027.01 | ⚪ Not started |

## Quick Start

### Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.0 (with CUDA or MPS support)
- 4TB+ free disk space (for osu!mania dataset)

### Setup

```bash
# Clone the repo (replace qxxiit with your actual GitHub handle)
git clone https://github.com/qxxiit/rhythm-chart-diffusion.git
cd rhythm-chart-diffusion

# Create environment (choose one)
# Option A: conda
conda env create -f environment.yml
conda activate rhythm-chart

# Option B: pip
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

### Data Preparation

```bash
# Download osu!mania 4K ranked beatmaps (requires osu! API key in .env)
python scripts/download_data.py --mode mania-4k --status ranked --output data/raw

# Preprocess: extract mel-spectrograms and tokenize charts
python scripts/preprocess_data.py --input data/raw --output data/processed
```

### Training

```bash
# Train baseline (autoregressive Transformer)
python scripts/train.py model=transformer_baseline data=osu_mania_4k

# Train diffusion model (Phase 2)
python scripts/train.py model=diffusion data=osu_mania_4k
```

### Evaluation

```bash
python scripts/evaluate.py --checkpoint outputs/<run_id>/best.ckpt
```

### Inference (single song)

```bash
python scripts/inference.py \
  --checkpoint outputs/<run_id>/best.ckpt \
  --audio path/to/song.mp3 \
  --output path/to/chart.json
```

## Repository Structure

```
src/                  # All Python source code
  data/               # Data download, preprocessing, dataset classes
  models/             # Model architectures (transformer, diffusion)
  training/           # Training loops, losses
  evaluation/         # F1, mAP@tIoU, diversity metrics
  unity_export/       # Convert model output to Unity JSON
  utils/              # Common utilities

configs/              # Hydra configs (model, data, training)
scripts/              # Entry points (download, preprocess, train, evaluate)
notebooks/            # Jupyter notebooks (exploration, visualization)
tests/                # Unit tests
docs/                 # Architecture, data format, experiment notes
paper/                # Paper drafts and reading notes
```

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for module details.

## Citation

If you use this code in academic work, please cite:

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
- [2] Takada et al. *GenéLive!* AAAI 2023.
- [3] Yi, Lee, Lee. *Beat-Aligned Spectrogram-to-Sequence Generation of Rhythm-Game Charts.* arXiv:2311.13687, 2023.
- [4] Gong, Kwak, Cho. *ActFusion: a Unified Diffusion Model for Action Segmentation and Anticipation.* NeurIPS 2024.
- [5] Austin et al. *Structured Denoising Diffusion Models in Discrete State-Spaces (D3PM).* NeurIPS 2021.

## License

MIT License — see [`LICENSE`](LICENSE).

## Contributing

This is primarily a student research project, but we welcome:
- Issue reports (bugs, unclear documentation, dataset/preprocessing edge cases)
- Suggestions on model design, evaluation metrics, or related literature
- PRs for fixes or improvements (please open an issue first to discuss)

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for our internal workflow conventions.

## Acknowledgments

- Mentor: Prof. Wonhwa Kim (POSTECH CSE)
- POSTECH Center for Innovative Education (UGRP)
