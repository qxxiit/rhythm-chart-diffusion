"""Train the D-32 denoiser (design doc §4.9-4.10, §6).

    python scripts/train.py --overfit 10                      # milestone: 10 charts, same for val
    python scripts/train.py --steps 50000 --batch-size 32     # full run on the train split
    python scripts/train.py --resume outputs/<run>/last.pt    # continue after an interruption
    python scripts/train.py --steps 60000 --val-every 2000 --style data/style.csv --run full-v2
    python scripts/train.py --steps 60000 --val-every 2000 --row-mask 0.5 \
        --chart-stats data/chart_stats.csv --run full-v4
    python scripts/train.py --steps 60000 --val-every 2000 --row-mask 0.5 \
        --chart-stats data/chart_stats.csv --snapshot-every 1000 --run fit-v4

Reads data/manifest.csv and data/cache (tokens + log-Mel, layout in src/data/cache.py).
--fake-mel reads the plumbing-only mel that encodes the answer instead (never report it).
Writes outputs/<run>/: config.json, log.csv (every --log-every steps), val.csv,
last.pt and best.pt (and traj/ with --snapshot-every). Before a run, write the
prediction (final loss, when it converges, how it could fail) in docs/EXPERIMENTS.md
(design doc §6, 예측 기록).

--style data/style.csv (scripts/build_style.py) adds the genre and mapper inputs
(src/data/style.py): the vocab is built from the train charts and saved in the
checkpoint. In training, with probability --style-drop both labels are dropped
(set to "no label"), and otherwise each one on its own with the same probability,
so the model also learns the style-free distribution (classifier-free guidance,
sampler). Validation reports val_ce with the charts' own labels and val_ce_null
without any; best.pt follows val_ce.

--chart-stats data/chart_stats.csv (scripts/build_chart_stats.py) adds the chart's own
long-note share, jack rate and trill rate as inputs (src/data/chartstats.py), as
quantile buckets of the train charts (the edges are saved in the checkpoint). With
probability --stats-drop all three are dropped, and otherwise each one on its own with
the same probability. val_ce_null is then without style and without stats.

--overfit N trains and validates on the same N charts, then samples chunk 0
of the first one and counts matching cells. The loss should approach 0 and the
sample should rebuild the chart; if not, suspect the code before the
hyperparameters (§6, 사전 점검).

--snapshot-every N saves the path of the run for scripts/fit_viz.py, in outputs/<run>/traj
(Trajectory below): the parameters at step 0, N/10, N/5, N/2, every multiple of N and the
last step. It draws no random numbers, so a run with snapshots trains as the same run
without them. --optim sgd (plain gradient descent, or with --momentum) and --betas 0 0
(AdamW with no averaging: the normalized gradient, each coordinate moved by lr in the sign
of its gradient) are there to compare optimizers from the same start (notebook 6.5).
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

from src.data import chartstats
from src.data.dataset import ChunkDataset
from src.data.style import MIN_MAPPER_CHARTS, build_vocab, encode, read_style, sizes
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
    ap.add_argument("--fake-mel", action="store_true",
                    help="data/cache/fake_mel (encodes the answer): plumbing tests only")
    ap.add_argument("--window", choices=["chunk", "bar"], default="chunk",
                    help="training windows: the fixed chunks, or a random bar line inside each")
    ap.add_argument("--steps", type=int, default=None, help="optimizer steps (3000 with --overfit)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.98))
    ap.add_argument("--optim", choices=["adamw", "sgd"], default="adamw",
                    help="sgd: gradient descent with --momentum (default 0: plain)")
    ap.add_argument("--momentum", type=float, default=0.0, help="--optim sgd")
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--row-mask", type=float, default=0.0,
                    help="share of chunks masked by whole rows (all 4 lanes) instead of cell "
                         "by cell; validation always masks cell by cell")
    ap.add_argument("--no-audio-add", action="store_true",
                    help="cross-attention only, as in the design doc (실험 큐 5)")
    ap.add_argument("--style", type=Path, default=None,
                    help="data/style.csv: condition on genre and mapper (default: no style)")
    ap.add_argument("--min-mapper-charts", type=int, default=MIN_MAPPER_CHARTS,
                    help="--style: mappers with fewer train charts share one 'other' label")
    ap.add_argument("--style-drop", type=float, default=0.1,
                    help="--style: probability of dropping both labels, and then each one")
    ap.add_argument("--chart-stats", type=Path, default=None,
                    help="data/chart_stats.csv: condition on the chart's long-note share, jack "
                         "and trill rate (default: none)")
    ap.add_argument("--stats-drop", type=float, default=0.15,
                    help="--chart-stats: probability of dropping all stats, and then each one")
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
    ap.add_argument("--snapshot-every", type=int, default=0,
                    help="save the parameters to <run>/traj for scripts/fit_viz.py; 0 = off")
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


def drop_style(genre: torch.Tensor, mapper: torch.Tensor, null: tuple[int, int],
               p: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Condition dropout: both labels -> null with probability p, then each with p."""
    if p <= 0:
        return genre, mapper
    u = torch.rand(3, genre.shape[0], device=genre.device)
    both = u[0] < p
    return (torch.where(both | (u[1] < p), torch.full_like(genre, null[0]), genre),
            torch.where(both | (u[2] < p), torch.full_like(mapper, null[1]), mapper))


def drop_stats(stats: torch.Tensor, null: int, p: float) -> torch.Tensor:
    """Condition dropout for chart stats [B, n]: all -> null with probability p, then each."""
    if p <= 0:
        return stats
    u_all = torch.rand(stats.shape[0], 1, device=stats.device)
    u_each = torch.rand(stats.shape, device=stats.device)
    return torch.where((u_all < p) | (u_each < p), torch.full_like(stats, null), stats)


@torch.no_grad()
def validate(model, loader, device, amp, max_batches: int,
             null: tuple[int, int] | None = None, stats_null: int | None = None) -> dict:
    """Masked CE at fixed mask ratios with fixed masks, so runs are comparable. With style
    (null = the no-label indices) or chart stats (stats_null = the no-value bucket): val_ce
    with each chart's own, val_ce_null without any."""
    model.eval()
    sums = {g: 0.0 for g in VAL_GAMMAS}
    sum_null = 0.0
    n = 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        batch = to_device(batch, device)
        for j, g in enumerate(VAL_GAMMAS):
            gamma = torch.full((batch["x0"].shape[0],), g, device=device)
            conds = [(batch.get("genre"), batch.get("mapper"), batch.get("stats"))]
            if null is not None or stats_null is not None:
                conds.append((
                    None if null is None else torch.full_like(batch["genre"], null[0]),
                    None if null is None else torch.full_like(batch["mapper"], null[1]),
                    None if stats_null is None else torch.full_like(batch["stats"], stats_null)))
            for k, (genre, mapper, stats) in enumerate(conds):
                gen = torch.Generator().manual_seed(10_000 * i + j)   # CPU: same masks anywhere
                with amp():
                    _, info = diffusion_loss(model, batch["x0"], batch["mel"], batch["s"],
                                             batch["b"], gamma=gamma, generator=gen,
                                             genre=genre, mapper=mapper, stats=stats)
                if k == 0:
                    sums[g] += info["masked_ce"]
                else:
                    sum_null += info["masked_ce"] / len(VAL_GAMMAS)
        n += 1
    model.train()
    out = {f"val_ce@{g}": sums[g] / max(n, 1) for g in VAL_GAMMAS}
    out["val_ce"] = sum(out.values()) / len(VAL_GAMMAS)
    if null is not None or stats_null is not None:
        out["val_ce_null"] = sum_null / max(n, 1)
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


def snapshot_steps(every: int, total: int) -> list[int]:
    """--snapshot-every: 0, every / 10, / 5, / 2 (the first steps move the most), every
    multiple of every, and the last step."""
    if every <= 0:
        return []
    early = {s for s in (0, every // 10, every // 5, every // 2) if s < every}
    return sorted(early | set(range(every, total + 1, every)) | {total})


class Trajectory:
    """outputs/<run>/traj, what scripts/fit_viz.py reads:

    params_<step>.npy   float32 [P], the parameters in model.parameters() order
                        (torch.nn.utils.parameters_to_vector), after that many steps
    adam_<step>.npz     AdamW only, from step 1: exp_avg (m), exp_avg_sq (v) and the step
                        count t at a fixed sample of coordinates (up to SAMPLE per tensor,
                        sample_idx.npy: positions in the flat vector)
    meta.json           names and shapes in that order, the optimizer, betas and eps
    steps.csv           step, lr (of the next step) of each snapshot as it is written
    """

    SAMPLE = 2048

    def __init__(self, run_dir: Path, model: torch.nn.Module, opt, a: argparse.Namespace):
        self.dir = run_dir / "traj"
        self.dir.mkdir(parents=True, exist_ok=True)
        named = list(model.named_parameters())
        self.params = [p for _, p in named]
        self.opt = opt if isinstance(opt, torch.optim.AdamW) else None
        rng = np.random.default_rng(0)                 # its own generator: training's untouched
        self.local, flat, offset = [], [], 0
        for p in self.params:
            idx = np.sort(rng.choice(p.numel(), min(p.numel(), self.SAMPLE), replace=False))
            self.local.append(torch.as_tensor(idx, device=p.device))
            flat.append(offset + idx)
            offset += p.numel()
        np.save(self.dir / "sample_idx.npy", np.concatenate(flat))
        group = opt.param_groups[0]
        (self.dir / "meta.json").write_text(json.dumps({
            "names": [n for n, _ in named], "shapes": [list(p.shape) for p in self.params],
            "optim": a.optim, "betas": list(group["betas"]) if "betas" in group else None,
            "eps": group.get("eps"), "momentum": group.get("momentum"), "lr": a.lr,
            "steps": a.steps}, indent=1))

    def save(self, step: int, lr: float) -> None:
        vec = torch.nn.utils.parameters_to_vector(self.params).detach().float().cpu().numpy()
        np.save(self.dir / f"params_{step:06d}.npy", vec)
        if self.opt is not None and step > 0:
            m, v, t = [], [], 0.0
            for p, idx in zip(self.params, self.local, strict=True):
                state = self.opt.state.get(p, {})
                if "exp_avg" not in state:              # no gradient yet: no step either
                    m.append(np.zeros(len(idx), np.float32))
                    v.append(np.zeros(len(idx), np.float32))
                    continue
                m.append(state["exp_avg"].reshape(-1)[idx].float().cpu().numpy())
                v.append(state["exp_avg_sq"].reshape(-1)[idx].float().cpu().numpy())
                t = max(t, float(state["step"]))
            np.savez(self.dir / f"adam_{step:06d}.npz", m=np.concatenate(m),
                     v=np.concatenate(v), t=np.float64(t))
        append_csv(self.dir / "steps.csv", ["step", "lr"], [step, f"{lr:.4e}"])


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
        first = ChunkDataset(a.manifest, a.cache, ("train",), max_charts=a.overfit,
                             fake_mel=a.fake_mel)
        keys = [c["key"] for c in first.charts]
        train_ds = ChunkDataset(a.manifest, a.cache, ("train",), keys=keys, window=a.window,
                                fake_mel=a.fake_mel)
        val_ds = first
    else:
        train_ds = ChunkDataset(a.manifest, a.cache, ("train",), max_charts=a.max_charts,
                                window=a.window, fake_mel=a.fake_mel)
        val_ds = ChunkDataset(a.manifest, a.cache, ("val",),           # always the fixed chunks
                              fake_mel=a.fake_mel)
        keys = None
    vocab, null, spec, stats_null = None, None, None, None
    if a.resume:                                           # a style run keeps its vocab
        saved = torch.load(a.resume, map_location="cpu", weights_only=True)
        vocab, spec = saved.get("style"), saved.get("chart_stats")
        if (vocab is None) != (a.style is None):
            print("--resume: give --style exactly when the run was started with it",
                  file=sys.stderr)
            return 2
        if (spec is None) != (a.chart_stats is None):
            print("--resume: give --chart-stats exactly when the run was started with it",
                  file=sys.stderr)
            return 2
    if a.style is not None:
        labels = read_style(a.style)
        if vocab is None:
            vocab = build_vocab(labels, [c["key"] for c in train_ds.charts], a.min_mapper_charts)
        null = sizes(vocab)
        coded = {k: encode(vocab, v["genre_id"], v["mapper_id"]) for k, v in labels.items()}
        for ds in {id(train_ds): train_ds, id(val_ds): val_ds}.values():
            ds.style, ds.null_style = coded, null
    if a.chart_stats is not None:
        table = chartstats.read_stats(a.chart_stats)
        if spec is None:
            spec = chartstats.build_spec(table, [c["key"] for c in train_ds.charts])
        stats_null = chartstats.null_index(spec)
        coded_stats = {k: chartstats.encode(spec, v) for k, v in table.items()}
        for ds in {id(train_ds): train_ds, id(val_ds): val_ds}.values():
            ds.stats, ds.null_stats = coded_stats, (stats_null,) * len(spec["names"])
    if len(train_ds) == 0:
        where = "data/cache/fake_mel" if a.fake_mel else "data/cache/logmel (preprocess_data.py --mel)"
        print("no training chunks: run scripts/build_manifest.py and scripts/preprocess_data.py, "
              f"and check {where}", file=sys.stderr)
        return 2
    loader = DataLoader(train_ds, batch_size=a.batch_size, shuffle=True,
                        drop_last=len(train_ds) >= a.batch_size, num_workers=a.num_workers,
                        pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_ds, batch_size=a.batch_size, shuffle=False,
                            num_workers=a.num_workers)

    # --- model ---
    config = DenoiserConfig(d_model=a.d_model, lane_dim=a.d_model // 4, n_layers=a.layers,
                            n_heads=a.heads, d_ff=a.d_ff, dropout=a.dropout,
                            audio_add=not a.no_audio_add,
                            n_genres=null[0] if null else 0, n_mappers=null[1] if null else 0,
                            n_stats=len(spec["names"]) if spec else 0,
                            stat_bins=spec["bins"] if spec else 0)
    model = Denoiser(config).to(device)
    if a.optim == "sgd":
        opt = torch.optim.SGD(param_groups(model, a.weight_decay), lr=a.lr, momentum=a.momentum)
    else:
        opt = torch.optim.AdamW(param_groups(model, a.weight_decay), lr=a.lr,
                                betas=tuple(a.betas))
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
         "device": str(device), "params": n_params(model), "style": vocab,
         "chart_stats": spec}, indent=2))
    print(f"{run_dir}: {n_params(model):,} params on {device}, {len(train_ds):,} train chunks "
          f"from {len(train_ds.charts):,} charts, {len(val_ds):,} val chunks")
    if vocab is not None:
        known = sum(int(m) < len(vocab["mappers"]) for _, m in
                    (train_ds.style.get(c["key"], null) for c in train_ds.charts))
        print(f"style: {len(vocab['genres'])} genres, {len(vocab['mappers'])} mappers with >= "
              f"{vocab['min_charts']} train charts ({known:,} of {len(train_ds.charts):,} charts), "
              f"drop {a.style_drop}")
    if spec is not None:
        have = sum(c["key"] in train_ds.stats for c in train_ds.charts)
        print(f"chart stats: {', '.join(spec['names'])}, "
              f"{'/'.join(str(len(e) + 1) for e in spec['edges'])} buckets, {have:,} of "
              f"{len(train_ds.charts):,} train charts, drop {a.stats_drop}")

    wandb = None
    if a.wandb:
        import wandb as _wandb
        wandb = _wandb
        wandb.init(project=a.wandb, name=run_dir.name, config=args_json)


    def checkpoint(name: str) -> None:
        save(run_dir / name, model=model.state_dict(), optimizer=opt.state_dict(),
             scheduler=sched.state_dict(), step=step, best=best, config=asdict(config),
             args=args_json, **({"style": vocab} if vocab is not None else {}),
             **({"chart_stats": spec} if spec is not None else {}))

    snaps = set(snapshot_steps(a.snapshot_every, a.steps))
    traj = Trajectory(run_dir, model, opt, a) if snaps else None
    if traj is not None and step == 0:
        traj.save(0, sched.get_last_lr()[0])

    # --- loop ---
    model.train()
    micro, seen, t0 = 0, 0, time.time()
    running = {"loss": 0.0, "masked_ce": 0.0, "mask_rate": 0.0}
    n_running = 0
    while step < a.steps:
        for batch in loader:
            batch = to_device(batch, device)
            genre, mapper = batch.get("genre"), batch.get("mapper")
            if null is not None:
                genre, mapper = drop_style(genre, mapper, null, a.style_drop)
            stats = batch.get("stats")
            if stats_null is not None:
                stats = drop_stats(stats, stats_null, a.stats_drop)
            with amp():
                loss, info = diffusion_loss(model, batch["x0"], batch["mel"], batch["s"],
                                            batch["b"], row_mask=a.row_mask, genre=genre,
                                            mapper=mapper, stats=stats)
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
            if traj is not None and step in snaps:
                traj.save(step, sched.get_last_lr()[0])

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
                val = validate(model, val_loader, device, amp, a.val_batches, null, stats_null)
                append_csv(run_dir / "val.csv", ["step", *val.keys()],
                           [step, *(f"{v:.5f}" for v in val.values())])
                print(f"  val_ce {val['val_ce']:.4f}  " + "  ".join(
                    f"@{g} {val[f'val_ce@{g}']:.3f}" for g in VAL_GAMMAS)
                    + (f"  null {val['val_ce_null']:.4f}" if "val_ce_null" in val else ""),
                    flush=True)
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
