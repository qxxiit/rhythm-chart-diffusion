# Audio-Conditioned Discrete Diffusion for Rhythm Game Chart Generation

> Discrete diffusion model for generating rhythm-game charts from audio.
> POSTECH UGRP 2026. **Open source** from day one — contributions and feedback welcome.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)
[![Status: research prototype](https://img.shields.io/badge/status-research%20prototype-orange.svg)](MASTERPLAN.md)

## Overview

Rhythm game charts (note placement data) are normally authored by hand, taking hours to tens of hours per song. This project develops an automatic chart generation system: given an audio file (mp3/wav), the model produces a playable osu!mania 4K chart.

Existing learning-based approaches generate charts token by token (autoregressive), which makes it hard to keep a whole song consistent and offers little control over difficulty or style. We use **discrete diffusion** instead: the note grid of a window is generated as a whole by masked denoising, conditioned on the audio, the target star rating (SR) and the chart's style, and then refined by passes that act on the whole song.

The question the project is judged by: **can a player tell our chart from the human chart of the same song and difficulty?** (`docs/EVALUATION.md`). Since the team meeting of Sep 30 the generator comes first; the autoregressive baseline (Yi et al., ISMIR 2023) is on hold. See [`MASTERPLAN.md`](MASTERPLAN.md) for the plan.

## Status (Nov 2026)

| Phase | Status |
|---|---|
| Phase 0 — Setup & data pipeline | ✅ Complete (Aug 2026) |
| Phase 1 — Problem formulation & baseline | ✅ Formulation, tokenizer, split, audio pipeline, evaluation; the AR baseline on hold (Sep 30) |
| Phase 2 — Discrete diffusion | ✅ D-32 denoiser (6.9M parameters); full-v4 (the chart's style as inputs) is the system's model |
| Phase 3 — Ablation | 🟡 Sampler, passes, judges and the stems (htdemucs) tried run by run: [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) |
| Phase 4 — User study | 🟡 Pilot playtest Nov 14; the main blind playtest (about 20 players x 20 test songs) Nov 30 – Dec 6, after the freeze on Nov 29 |
| Phase 5 — Paper | ⚪ Not started |

### How close it is

Judges (classifier two-sample tests, `scripts/judge.py`) learn to tell the system's charts from the human charts of the same songs; an AUC of 0.5 means they cannot. On the first 120 val songs, 8 bars at a time, judges from full-v4's trunk:

| | full view | rhythm view (lanes hidden) |
|---|---|---|
| full-v4 alone (one draw a window; no choice, biases or passes) | 0.885 | 0.767 |
| the system | 0.743 [0.692, 0.787] | 0.603 [0.541, 0.660] |

A sharper judge, from the trunk of the model that also hears the stems, reads the system at 0.833: the charts are not yet indistinguishable. What playing them showed (empty stretches, blurred patterns, short long notes over busy parts) and what was done about each is in the log.

### The system

`sh scripts/system.sh test 0` makes the system's charts for a split's songs (DECISIONS through 2026-11-18):

1. **Draw.** full-v4, each window drawn four times; a judge that sees the long notes (`sel_holds`) keeps the draw it finds most human. Cosine schedule (128 steps), the lanes decided row by row and refined twice, quiet bars thinned, fewer note starts where nothing in the music starts, a slight push towards long notes.
2. **Passes on the whole song.** Rhythms carried over and sections that come back copied, bars that sound alike aligned, chords on the strong sounds, long notes up to the human chart's share, rests and holes, light bars in Easy and Normal, jacks untangled, the chord cap in Easy, no long empty stretch, in songs whose human chart is mostly taps a short long note repeated in its lane in a busy part made a tap, and long notes let go as the next note comes.

The choosing judge is trained first on a plain cosine draw of the val songs (`python scripts/judge.py --ckpt outputs/full-v4/best.pt --ai <the draw> --view holds --sr human --epochs 2 --save-models --name sel_holds`; `outputs/` is not in git). The run scripts of every experiment are in [`experiments/runs/`](experiments/runs/) with a one-line outcome each.

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
python scripts/build_chart_stats.py     # data/chart_stats.csv: long-note share, jack and trill rate per chart
python scripts/train.py --steps 60000 --val-every 2000 --row-mask 0.5 \
    --chart-stats data/chart_stats.csv --run full-v4     # ~3.5 h on an M-series MacBook (MPS)
python scripts/stems.py                 # optional: htdemucs stems per audio file (train.py --stems)

python scripts/evaluate.py --ckpt outputs/full-v4/best.pt --per-song --n 0   # F1, SR error, patterns
python scripts/judge.py --ckpt outputs/full-v4/best.pt --ai <eval dir>      # the C2ST judge
python scripts/diag_judge.py <eval dir> ...                                 # the diagnostics judge
sh scripts/system.sh val 120            # the system on 120 val songs (see above)
python scripts/playtest_pack.py --single --from-run <eval dir> ...          # a blind playtest pack
```

Every option of `evaluate.py` (the passes on saved charts with `--from-charts`, the views and judges) is listed in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md). The cache layout is in `src/data/cache.py`. Not implemented: the AR baselines (on hold).

### Make a chart for your own song

```bash
python scripts/generate.py --ckpt outputs/full-v4/best.pt --audio "song.mp3" --sr 3.5 \
    --stats ln=0 --lane-guidance 2 --schedule cosine --lane-temp 1 --hold-bias 0.2
                                        # a tap chart (ln=0.4: a long-note chart; the audio
                                        # does not say which a song gets, so choose)
python scripts/generate.py --audio "song.mp3" --bpm 174 --offset 1234 --sr 2 3.5 5
python scripts/generate.py --audio "song.mp3" --timing timed.osu --sr 3.5   # red lines of an .osu
```

Writes one 4K difficulty per star rating and an `.osz` under `outputs/generated/`; open it with osu! (lazer: double-click). `generate.py` runs the draw and the passes of the system except the judge's choice among draws and the passes added after Nov 14 (those run on saved charts, `evaluate.py --from-charts`); without `--ckpt` it uses full-v1 and the October settings. The tempo estimate assumes one constant tempo: for songs whose tempo changes, time the song in the osu! editor and pass `--timing`.

## Repository Structure

```
src/
  data/               # osu! API client, chart parser, tokenizer, mel and stems, datasets
  models/             # the denoiser (diffusion.py), the sampler and its passes, judges
  training/           # training loop, losses
  evaluation/         # F1, SR, patterns, long notes, playability, structure
  unity_export/       # (planned) model output to Unity JSON
scripts/              # entry points; scripts/system.sh is the system
experiments/runs/     # the run script of every experiment, with an index
docs/                 # architecture, data format, evaluation, decisions
docs/experiments/     # the experiment log, one file a month (index: docs/EXPERIMENTS.md)
notebooks/            # exploration and chart visualization
paper/                # drafts and reading notes
tests/
data/, outputs/       # not in git: the dataset (rebuilt by the pipeline) and every run
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
