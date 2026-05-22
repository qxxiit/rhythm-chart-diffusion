"""Evaluate a trained checkpoint on test set.

Usage:
    python scripts/evaluate.py --checkpoint outputs/<run_id>/best.ckpt

Outputs F1, mAP@tIoU, n-gram diversity to stdout and saves report JSON.

TODO (Phase 1, week of 7/29):
- [ ] Load checkpoint
- [ ] Build test dataloader
- [ ] Run inference on test set
- [ ] Compute metrics (src/evaluation/)
- [ ] Save report (JSON + markdown table)
"""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint .ckpt")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--output", default="reports/", help="Where to save metrics report")
    args = parser.parse_args()

    raise NotImplementedError(f"TODO: implement evaluation. ckpt={args.checkpoint}")


if __name__ == "__main__":
    main()
