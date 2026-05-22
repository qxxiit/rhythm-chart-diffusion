"""Single-song inference: audio file → playable chart.

Usage:
    python scripts/inference.py \\
        --checkpoint outputs/<run_id>/best.ckpt \\
        --audio path/to/song.mp3 \\
        --output path/to/chart.json \\
        --difficulty hard \\
        --keys 4

The output is in the team's Unity rhythm game JSON format.

TODO (Phase 4, week of 11/16):
- [ ] Load checkpoint
- [ ] Load and preprocess audio (mel-spectrogram, beat alignment)
- [ ] Run model inference (autoregressive or diffusion sampling)
- [ ] Detokenize chart
- [ ] Convert to Unity JSON (src/unity_export/chart_to_json.py)
- [ ] Save output
"""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--audio", required=True, help="Input .mp3 or .wav")
    parser.add_argument("--output", required=True, help="Output .json chart")
    parser.add_argument("--difficulty", default="normal", choices=["easy", "normal", "hard", "insane"])
    parser.add_argument("--keys", type=int, default=4, choices=[4, 5, 6, 7, 8])
    args = parser.parse_args()

    raise NotImplementedError(
        f"TODO: implement single-song inference.\n"
        f"  audio={args.audio} → chart={args.output} ({args.keys}K, {args.difficulty})"
    )


if __name__ == "__main__":
    main()
