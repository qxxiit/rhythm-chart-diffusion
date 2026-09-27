"""How much does the denoiser use the audio? (val CE with the audio taken away)

    python scripts/audio_ablation.py --ckpt outputs/<run>/best.pt
    python scripts/audio_ablation.py --ckpt ... --split val --batches 100

The same val chunks and the same masks (fixed seeds, as train.py's validation)
are scored under four versions of the mel:

    real      the chunk's own log-Mel
    other     the log-Mel of another chunk in the (shuffled) batch: another song,
              almost always, at another tempo
    flat      zeros: the song's average spectrum (the input is standardized per
              song and band), no timing at all
    shift2    the chunk's own log-Mel moved 2 cells (8 frames) later, wrapped:
              right sound, wrong place

masked CE is over all masked cells; onset CE only over masked cells whose answer
is a tap or a hold start, where the audio should matter most (hold bodies and
empty cells are mostly implied by their neighbours). If "other" and "flat" are
close to "real", the model is filling charts from the grammar and the unmasked
cells, not from the music. "shift2" separates "uses the audio" from "uses where
in the audio".
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data.dataset import ChunkDataset
from src.data.tokenizer import HOLD_START, PAD, TAP
from src.models.diffusion import N_CLASSES, load_denoiser, mask_tokens, pick_device

GAMMAS = (0.1, 0.3, 0.5, 0.7, 0.9)
CONDITIONS = ("real", "other", "flat", "shift2")


def variant(mel: torch.Tensor, name: str) -> torch.Tensor:
    if name == "real":
        return mel
    if name == "other":
        return torch.roll(mel, shifts=1, dims=0)                  # the next chunk's audio
    if name == "flat":
        return torch.zeros_like(mel)
    if name == "shift2":
        return torch.roll(mel, shifts=8, dims=1)
    raise ValueError(name)


@torch.no_grad()
def score(model, loader, device, batches: int) -> dict:
    sums = {(c, g, w): 0.0 for c in CONDITIONS for g in GAMMAS for w in ("all", "onset")}
    counts = dict.fromkeys(sums, 0.0)
    for i, batch in enumerate(loader):
        if i >= batches:
            break
        x0, s, b = batch["x0"].to(device), batch["s"].to(device), batch["b"].to(device)
        mel = batch["mel"].to(device)
        onset = (x0 == TAP) | (x0 == HOLD_START)
        target = torch.where(x0 == PAD, torch.zeros_like(x0), x0)
        for j, g in enumerate(GAMMAS):
            gen = torch.Generator().manual_seed(10_000 * i + j)        # same masks everywhere
            x_t, masked = mask_tokens(x0, torch.full((x0.shape[0],), g, device=device), gen)
            for c in CONDITIONS:
                logits = model(x_t, variant(mel, c), s, b).float()
                ce = F.cross_entropy(logits.reshape(-1, N_CLASSES), target.reshape(-1),
                                     reduction="none").view_as(x0)
                for w, m in (("all", masked), ("onset", masked & onset)):
                    sums[(c, g, w)] += float((ce * m).sum())
                    counts[(c, g, w)] += float(m.sum())
    return {k: sums[k] / max(counts[k], 1.0) for k in sums}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"))
    ap.add_argument("--cache", type=Path, default=Path("data/cache"))
    ap.add_argument("--batches", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args(argv)

    ds = ChunkDataset(a.manifest, a.cache, (a.split,))
    if len(ds) == 0:
        print(f"no {a.split} chunks with log-Mel", file=sys.stderr)
        return 2
    device = pick_device(a.device)
    model = load_denoiser(a.ckpt, device)
    loader = DataLoader(ds, batch_size=a.batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(a.seed))
    res = score(model, loader, device, a.batches)

    print(f"{a.ckpt}: {min(a.batches, len(loader)) * a.batch_size:,} {a.split} chunks, "
          "same masks in every condition")
    for w, title in (("all", "masked CE (all masked cells)"),
                     ("onset", "onset CE (masked taps and hold starts)")):
        print(f"\n{title}")
        print(f"  {'mask':>5s} " + " ".join(f"{c:>8s}" for c in CONDITIONS)
              + "   " + " ".join(f"{c + '-real':>12s}" for c in CONDITIONS[1:]))
        for g in (*GAMMAS, "mean"):
            vals = [np.mean([res[(c, x, w)] for x in GAMMAS]) if g == "mean" else res[(c, g, w)]
                    for c in CONDITIONS]
            print(f"  {g!s:>5s} " + " ".join(f"{v:8.4f}" for v in vals) + "   "
                  + " ".join(f"{v - vals[0]:+12.4f}" for v in vals[1:]))
    out = a.ckpt.parent / f"audio_ablation_{a.split}.json"
    out.write_text(json.dumps({f"{c}@{g}/{w}": round(v, 5) for (c, g, w), v in res.items()},
                              indent=1))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
