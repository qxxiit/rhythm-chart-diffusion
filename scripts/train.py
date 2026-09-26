"""Train the D-32 denoiser (design doc §4.9-4.10, §6).

    python scripts/train.py --overfit 10                      # milestone: 10 charts, same for val
    python scripts/train.py --steps 50000 --batch-size 32     # full run on the train split
    python scripts/train.py --resume outputs/<run>/last.pt    # continue after an interruption

Reads data/manifest.csv and data/cache (tokens + mel, layout in src/data/cache.py).
Writes outputs/<run>/: config.json, log.csv (every --log-every steps), val.csv,
last.pt and best.pt. Before a run, write the prediction (final loss, when it
converges, how it could fail) in docs/EXPERIMENTS.md (design doc §6, 예측 기록).

--overfit N trains and validates on the same N charts, then samples chunk 0
of the first one and counts matching cells. The loss should approach 0 and the
sample should rebuild the chart; if not, suspect the code before the
hyperparameters (§6, 사전 점검).
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dataset import ChunkDataset
from src.data.tokenizer import HOLD_START, MASK, PAD, TAP, grammar_violations
from src.models.diffusion import (
    Denoiser,
    DenoiserConfig,
    diffusion_loss,
    n_params,
    pick_device,
)
from src.models.sampler import sample_window

VAL_GAMMAS = (0.1, 0.3, 0.5, 0.7, 0.9)


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"))
    ap.add_argument("--cache", type=Path, default=Path("data/cache"))
    ap.add_argument("--out", type=Path, default=Path("outputs"))
    ap.add_argument("--run", default=None, help="run name (default: time stamp)")
    ap.add_argument("--overfit", type=int, default=0, help="train and validate on N charts")
    ap.add_argument("--max-charts", type=int, default=None, help="subset of the train split")
    ap.add_argument("--window", choices=["chunk", "bar"], default="chunk",
                    help="training windows: the fixed chunks, or a random bar line inside each")
    ap.add_argument("--steps", type=int, default=None, help="optimizer steps (3000 with --overfit)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.98))
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--no-audio-add", action="store_true",
                    help="cross-attention only, as in the design doc (실험 큐 5)")
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--d-ff", type=int, default=1024)
    ap.add_argument("--device", default="auto", help="auto / cuda / mps / cpu")
    ap.add_argument("--no-amp", action="store_true", help="disable bf16 autocast on CUDA")
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--val-every", type=int, default=1000)
    ap.add_argument("--val-batches", type=int, default=50)
    ap.add_argument("--sample-steps", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", type=Path, default=None)
    ap.add_argument("--wandb", default=None, help="wandb project name (optional)")
    a = ap.parse_args(argv)
    if a.steps is None:
        a.steps = 3000 if a.overfit else 20000
    a.warmup = min(a.warmup, max(1, a.steps // 10))
    return a


def param_groups(model: torch.nn.Module, weight_decay: float) -> list[dict]:
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        skip = p.ndim < 2 or name in ("pos", "audio_pos") or name.startswith("tok_emb")
        (no_decay if skip else decay).append(p)
    return [{"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0}]


def lr_lambda(warmup: int, total: int):
    def f(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = min(1.0, (step - warmup) / max(1, total - warmup))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return f


def to_device(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


@torch.no_grad()
def validate(model, loader, device, amp, max_batches: int) -> dict:
    """Masked CE at fixed mask ratios with fixed masks, so runs are comparable."""
    model.eval()
    sums = {g: 0.0 for g in VAL_GAMMAS}
    n = 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        batch = to_device(batch, device)
        for j, g in enumerate(VAL_GAMMAS):
            gen = torch.Generator().manual_seed(10_000 * i + j)    # CPU: same masks on any device
            gamma = torch.full((batch["x0"].shape[0],), g, device=device)
            with amp():
                _, info = diffusion_loss(model, batch["x0"], batch["mel"], batch["s"], batch["b"],
                                         gamma=gamma, generator=gen)
            sums[g] += info["masked_ce"]
        n += 1
    model.train()
    out = {f"val_ce@{g}": sums[g] / max(n, 1) for g in VAL_GAMMAS}
    out["val_ce"] = sum(out.values()) / len(VAL_GAMMAS)
    return out


def overfit_check(model, ds: ChunkDataset, device, steps: int) -> dict:
    """Sample chunk 0 of the first chart from all-MASK and compare with the chart."""
    item = ds[0]
    x0 = item["x0"].numpy()
    x = np.where(x0 == PAD, PAD, MASK)
    more = ds.charts[0]["n_cells"] > len(x0)                  # the song goes on past chunk 0
    out = sample_window(model, x, item["mel"].to(device), float(item["s"]), float(item["b"]),
                        steps=steps, order="confidence", rng=np.random.default_rng(0),
                        left_closed=True, right_closed=not more)
    real = x0 != PAD
    onset_true = np.isin(x0, (TAP, HOLD_START)) & real
    onset_pred = np.isin(out, (TAP, HOLD_START)) & real
    hit = (onset_true & onset_pred & (out == x0)).sum()
    return {
        "cells_match": float((out == x0)[real].mean()),
        "onset_precision": float(hit / max(onset_pred.sum(), 1)),
        "onset_recall": float(hit / max(onset_true.sum(), 1)),
        "grammar_violations": len(grammar_violations(out, closed=False)),
    }


def append_csv(path: Path, header: list[str], row: list) -> None:
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(header)
        w.writerow(row)


def save(path: Path, **state) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.replace(path)                                # never leave a half-written checkpoint


def main(argv=None) -> int:
    a = parse_args(argv)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    device = pick_device(a.device)
    use_amp = device.type == "cuda" and not a.no_amp

    def amp():
        if use_amp:
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    # --- data ---
    if a.overfit:
        first = ChunkDataset(a.manifest, a.cache, ("train",), max_charts=a.overfit)
        keys = [c["key"] for c in first.charts]
        val_ds = first
        train_ds = ChunkDataset(a.manifest, a.cache, ("train",), keys=keys, window=a.window)
    else:
        train_ds = ChunkDataset(a.manifest, a.cache, ("train",), max_charts=a.max_charts,
                                window=a.window)
        val_ds = ChunkDataset(a.manifest, a.cache, ("val",))           # always the fixed chunks
        keys = None
    if len(train_ds) == 0:
        print("no training chunks: run scripts/build_manifest.py and scripts/preprocess_data.py, "
              "and check that data/cache/mel exists", file=sys.stderr)
        return 2
    loader = DataLoader(train_ds, batch_size=a.batch_size, shuffle=True,
                        drop_last=len(train_ds) >= a.batch_size, num_workers=a.num_workers,
                        pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_ds, batch_size=a.batch_size, shuffle=False,
                            num_workers=a.num_workers)

    # --- model ---
    config = DenoiserConfig(d_model=a.d_model, lane_dim=a.d_model // 4, n_layers=a.layers,
                            n_heads=a.heads, d_ff=a.d_ff, dropout=a.dropout,
                            audio_add=not a.no_audio_add)
    model = Denoiser(config).to(device)
    opt = torch.optim.AdamW(param_groups(model, a.weight_decay), lr=a.lr, betas=tuple(a.betas))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(a.warmup, a.steps))
    step, best = 0, math.inf
    if a.resume:
        ckpt = torch.load(a.resume, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        sched.load_state_dict(ckpt["scheduler"])
        step, best = ckpt["step"], ckpt["best"]
        run_dir = a.resume.parent
    else:
        run_dir = a.out / (a.run or time.strftime("%Y%m%d-%H%M%S"))
        run_dir.mkdir(parents=True, exist_ok=True)
    args_json = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()}
    (run_dir / "config.json").write_text(json.dumps(
        {"args": args_json, "model": asdict(config), "overfit_keys": keys,
         "device": str(device), "params": n_params(model)}, indent=2))
    print(f"{run_dir}: {n_params(model):,} params on {device}, {len(train_ds):,} train chunks "
          f"from {len(train_ds.charts):,} charts, {len(val_ds):,} val chunks")

    wandb = None
    if a.wandb:
        import wandb as _wandb
        wandb = _wandb
        wandb.init(project=a.wandb, name=run_dir.name, config=args_json)


    def checkpoint(name: str) -> None:
        save(run_dir / name, model=model.state_dict(), optimizer=opt.state_dict(),
             scheduler=sched.state_dict(), step=step, best=best, config=asdict(config),
             args=args_json)

    # --- loop ---
    model.train()
    micro, seen, t0 = 0, 0, time.time()
    running = {"loss": 0.0, "masked_ce": 0.0, "mask_rate": 0.0}
    n_running = 0
    while step < a.steps:
        for batch in loader:
            batch = to_device(batch, device)
            with amp():
                loss, info = diffusion_loss(model, batch["x0"], batch["mel"], batch["s"], batch["b"])
            (loss / a.grad_accum).backward()
            micro += 1
            seen += batch["x0"].shape[0]
            running["loss"] += loss.item()
            running["masked_ce"] += info["masked_ce"]
            running["mask_rate"] += info["mask_rate"]
            n_running += 1
            if micro % a.grad_accum:
                continue
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip))
            opt.step()
            opt.zero_grad(set_to_none=True)
            sched.step()
            step += 1

            if step % a.log_every == 0 or step == a.steps:
                rate = seen / (time.time() - t0)
                row = {k: v / n_running for k, v in running.items()}
                lr = sched.get_last_lr()[0]
                append_csv(run_dir / "log.csv",
                           ["step", "loss", "masked_ce", "mask_rate", "lr", "grad_norm",
                            "chunks_per_s"],
                           [step, f"{row['loss']:.5f}", f"{row['masked_ce']:.5f}",
                            f"{row['mask_rate']:.3f}", f"{lr:.2e}", f"{grad_norm:.3f}",
                            f"{rate:.1f}"])
                print(f"step {step:6d}  loss {row['loss']:.4f}  masked_ce {row['masked_ce']:.4f}"
                      f"  lr {lr:.2e}  |g| {grad_norm:.2f}  {rate:.0f} chunks/s", flush=True)
                if wandb:
                    wandb.log({**row, "lr": lr, "grad_norm": grad_norm}, step=step)
                running = dict.fromkeys(running, 0.0)
                n_running, seen, t0 = 0, 0, time.time()

            if step % a.val_every == 0 or step == a.steps:
                val = validate(model, val_loader, device, amp, a.val_batches)
                append_csv(run_dir / "val.csv", ["step", *val.keys()],
                           [step, *(f"{v:.5f}" for v in val.values())])
                print(f"  val_ce {val['val_ce']:.4f}  " + "  ".join(
                    f"@{g} {val[f'val_ce@{g}']:.3f}" for g in VAL_GAMMAS), flush=True)
                if wandb:
                    wandb.log(val, step=step)
                if val["val_ce"] < best:
                    best = val["val_ce"]
                    checkpoint("best.pt")
                checkpoint("last.pt")
                seen, t0 = 0, time.time()                # keep validation out of chunks/s
            if step >= a.steps:
                break

    if a.overfit:
        check = overfit_check(model, val_ds, device, a.sample_steps)
        (run_dir / "overfit_check.json").write_text(json.dumps(check, indent=2))
        print("overfit check (chunk 0 of the first chart, sampled from all-MASK): " + ", ".join(
            f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in check.items()))
    if wandb:
        wandb.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
