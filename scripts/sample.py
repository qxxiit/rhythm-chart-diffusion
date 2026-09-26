"""Generate a chart for one cached song and write a playable .osu.

    python scripts/sample.py --ckpt outputs/<run>/best.pt --key <manifest key>
    python scripts/sample.py --ckpt ... --key ... --sr 4.5 --steps 64 --order confidence

Uses the song's cached mel and its real timing (BPM and offset are given,
design doc §4.4) and writes <ckpt dir>/samples/<key>_....osu that refers to the
original audio file. Copy it into that beatmap's folder under osu!/Songs and
press F5 in song select. Also prints a quick comparison with the real chart
(onsets in the same cell and lane); proper F1 at ±20/50 ms is for the
evaluation code (§4.11).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from src.data.cache import read_manifest
from src.data.chart_parser import _kv, _split_sections, parse_osu
from src.data.chart_writer import write_osu
from src.data.tokenizer import HOLD_START, PAD, TAP, decode, grammar_violations, make_metas
from src.models.diffusion import load_denoiser, pick_device
from src.models.sampler import generate_song


def onset_match(pred: np.ndarray, real: np.ndarray) -> tuple[float, float]:
    keep = real != PAD
    p = np.isin(pred, (TAP, HOLD_START)) & keep
    r = np.isin(real, (TAP, HOLD_START)) & keep
    hit = (p & r).sum()
    return hit / max(p.sum(), 1), hit / max(r.sum(), 1)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--key", required=True, help="manifest key of the song")
    ap.add_argument("--sr", type=float, default=None, help="target SR (default: the real chart's)")
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--order", choices=["random", "confidence"], default="random")
    ap.add_argument("--mode", choices=["continue", "independent"], default="continue")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"))
    ap.add_argument("--root", type=Path, default=Path("data/raw"))
    ap.add_argument("--cache", type=Path, default=Path("data/cache"))
    ap.add_argument("--out", type=Path, default=None, help="default: <ckpt dir>/samples")
    ap.add_argument("--device", default="auto")
    a = ap.parse_args(argv)

    row = next((r for r in read_manifest(a.manifest) if r["key"] == a.key), None)
    if row is None:
        print(f"{a.key} is not in {a.manifest}", file=sys.stderr)
        return 2
    z = np.load(a.cache / "tokens" / f"{a.key}.npz")
    mel = np.load(a.cache / "mel" / f"{a.key}.npy").astype(np.float32)
    tps = [(float(t), float(bl)) for t, bl in z["timing_points"]]
    cell_offset, n_cells = int(z["cell_offset"]), int(z["n_cells"])
    sr = a.sr if a.sr is not None else float(z["sr"])

    model = load_denoiser(a.ckpt, pick_device(a.device))
    tokens = generate_song(model, mel, sr, tps, cell_offset, n_cells, steps=a.steps,
                           order=a.order, mode=a.mode, seed=a.seed)
    bad = len(grammar_violations(tokens))
    chart = decode(tokens, make_metas(tps, cell_offset, len(tokens), sr))

    src = a.root / row["path"]
    meta = _kv(_split_sections(src.read_text(encoding="utf-8-sig", errors="replace"))
               .get("Metadata", []))
    chart.audio_filename = parse_osu(src).audio_filename
    out_dir = a.out or a.ckpt.parent / "samples"
    out = out_dir / f"{a.key}_{a.mode}_{a.order}_T{a.steps}_s{sr:.2f}.osu"
    write_osu(out, chart, title=meta.get("Title", ""), artist=meta.get("Artist", ""),
              version=f"diffusion s={sr:.2f} {a.mode}/{a.order}/T{a.steps}")

    real = z["tokens"]
    same = real.shape == tokens.shape
    precision, recall = onset_match(tokens, real) if same else (float("nan"), float("nan"))
    print(f"{row['path']}\n  -> {out}")
    print(f"  {len(chart.notes)} notes (real chart {int(np.isin(real, (TAP, HOLD_START)).sum())}), "
          f"grammar violations {bad}, same-cell onset precision {precision:.3f} "
          f"recall {recall:.3f}")
    print("  copy the .osu into this beatmap's folder under osu!/Songs, then F5 in song select")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
