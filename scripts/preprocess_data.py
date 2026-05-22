"""Preprocess raw osu! data into training-ready tensors.

Reads raw .osu chart files and audio from data/raw,
extracts mel-spectrograms with beat alignment,
tokenizes charts, saves processed tensors.

Usage:
    python scripts/preprocess_data.py --input data/raw --output data/processed

TODO (Phase 1, week of 7/3):
- [ ] Parse .osu files (src/data/osu_parser.py)
- [ ] Extract mel-spectrogram (80 mel bins, FFT 512)
- [ ] Compute hop length = 1/48 beat from BPM (TimingPoints)
- [ ] Apply beat alignment per Yi et al. 2023
- [ ] Tokenize charts (src/data/tokenizer.py)
- [ ] Save as .npy or .pt (per song or sharded)
- [ ] Build train/val/test split file (song-level, no leakage)
"""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/raw", help="Raw data directory")
    parser.add_argument("--output", default="data/processed", help="Processed output directory")
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    raise NotImplementedError(
        f"TODO: implement preprocessing pipeline.\n"
        f"  input={args.input}, output={args.output}"
    )


if __name__ == "__main__":
    main()
