"""The path of training on the loss surface: UDL notebook 6.5 (Adam) at 6.9M parameters.

    python scripts/fit_viz.py outputs/fit-v4 --ref outputs/full-v4/best.pt
    python scripts/fit_viz.py outputs/opt-adam outputs/opt-sgd0.05 outputs/opt-sgd0.5 \
        outputs/opt-norm --out outputs/opt-compare --grid 21 --chunks 32
    python scripts/fit_viz.py outputs/fit-v4 --replot          # the figures again, no model

The notebook draws a loss of two parameters as contours and the optimizer's steps on top.
A run here has 6.9M parameters, so its path is drawn on the plane that holds most of it
(Li et al. 2018, Visualizing the Loss Landscape of Neural Nets, §7): the top two principal
directions d1, d2 of the snapshots' offsets from the last one, theta_i - theta_T
(train.py --snapshot-every). The path is the points (a_i, b_i) = ((theta_i - theta_T).d1,
(theta_i - theta_T).d2), the surface the loss at theta_T + a d1 + b d2 on a grid around
them. The loss is val_ce's: masked CE at the mask ratios 0.1 .. 0.9 with fixed masks,
averaged over the ratios, here on --chunks fixed chunks of --split (default train: the
loss the optimizer goes down), with each chart's own chart stats and style.

What the plane leaves out is reported next to it: the share of the path's variance along
d1 and d2, and at every snapshot the loss at theta_i next to the loss at its projection on
the plane. Also: the loss on the straight line from theta_0 through theta_T (Goodfellow et
al. 2015; a bump would be a barrier between the start and the end), the distance from
theta_0, and for AdamW runs the step Adam takes per coordinate, |m^ / (sqrt(v^) + eps)| in
units of the learning rate, against the coordinate's gradient scale sqrt(v^) (notebook 6.5,
eqs. 6.13-6.17): gradient descent moves each coordinate in proportion to its gradient,
Adam by about the same distance whatever the gradient. With several runs (same model and
data, e.g. optimizers from the same start) each one gets its own plane.

Writes to --out (default <first run>/fit_viz): fit_viz.json (the numbers and the grids),
fit_viz.npz (Adam's samples), fit_path.png, fit_curves.png and, for AdamW runs, fit_adam.png.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import torch
import torch.nn.functional as F
from train import VAL_GAMMAS

from src.data import chartstats
from src.data.dataset import ChunkDataset
from src.data.style import encode, read_style, sizes
from src.data.tokenizer import PAD
from src.models.diffusion import N_CLASSES, Denoiser, DenoiserConfig, mask_tokens, pick_device

BLOCK = 1 << 18                       # columns per pass over the snapshots (float64)
NOTEBOOK_CMAP = ("2a0902", "4b1e19", "6f352b", "924f40", "b36c58", "d08b72", "eaae91",
                 "fbd5b6", "ffffe0")  # notebook 6.5's colour ramp: dark = low loss
PATH_COLOR = "#a0d9d3"                # and its path colour
RUN_COLORS = ("#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b")


# ---------------------------------------------------------------------------
# the snapshots and their plane
# ---------------------------------------------------------------------------

def read_traj(run: Path) -> dict:
    """The snapshots of a run (memory-mapped), up to the last one with finite parameters."""
    d = run / "traj"
    if not (d / "meta.json").exists():
        raise FileNotFoundError(f"{d}: no snapshots (train.py --snapshot-every)")
    files = sorted(d.glob("params_*.npy"), key=lambda f: int(f.stem.split("_")[1]))
    X = [np.load(f, mmap_mode="r") for f in files]
    n_ok = 0
    while n_ok < len(X) and np.isfinite(X[n_ok]).all():
        n_ok += 1
    steps = [int(f.stem.split("_")[1]) for f in files]
    return {"meta": json.loads((d / "meta.json").read_text()), "steps": steps[:n_ok],
            "X": X[:n_ok], "diverged_after": steps[n_ok - 1] if 0 < n_ok < len(X) else None}


def plane(X: list[np.ndarray]) -> dict:
    """Li et al.'s plane of snapshots X[0 .. n-1], n >= 4.

    The offsets M_i = X_i - X_{n-1} (i < n - 1), mean-centred as sklearn's PCA does, give
    the top two principal directions d [2, P] (unit vectors) and their shares of the
    offsets' variance. coords [n, 2]: every snapshot's offset projected on d1, d2 (the
    last one at 0, 0). gram [n, n]: M M^T with M_{n-1} = 0, all the distances between
    snapshots. Two passes over the snapshots, BLOCK columns at a time."""
    n, P = len(X), len(X[0])
    if n < 4:
        raise ValueError(f"{n} snapshots: the plane needs at least 4")
    gram = np.zeros((n, n))
    for lo in range(0, P, BLOCK):
        M = np.stack([np.asarray(x[lo:lo + BLOCK], dtype=np.float64) for x in X])
        M -= M[-1]
        gram += M @ M.T
    m = n - 1
    H = np.eye(m) - 1.0 / m                              # centring: H 1 = 0
    lam, U = np.linalg.eigh(H @ gram[:m, :m] @ H)
    order = np.argsort(lam)[::-1]
    lam, U = np.clip(lam[order], 0.0, None), U[:, order]
    if lam[1] <= 1e-12 * max(lam[0], 1e-300):
        raise ValueError("the snapshots lie on a line: no second direction")
    U2, root = U[:, :2], np.sqrt(lam[:2])
    d = np.zeros((2, P), dtype=np.float64)              # d_k = M^T u_k / sqrt(lam_k): u_k is
    for lo in range(0, P, BLOCK):                       # orthogonal to 1, so H drops out
        M = np.stack([np.asarray(x[lo:lo + BLOCK], dtype=np.float64) for x in X[:m]])
        M -= np.asarray(X[-1][lo:lo + BLOCK], dtype=np.float64)
        d[:, lo:lo + BLOCK] = (U2.T @ M) / root[:, None]
    coords = np.zeros((n, 2))
    coords[:m] = gram[:m, :m] @ U2 / root                 # M_i . d_k = (G u_k)_i / sqrt(lam_k)
    return {"d": d, "coords": coords, "explained": (lam[:2] / lam.sum()).tolist(),
            "gram": gram}


def distances(gram: np.ndarray) -> tuple[np.ndarray, float]:
    """||theta_i - theta_0|| for every snapshot, and the path length sum ||theta_i+1 - theta_i||."""
    diag = np.diag(gram)
    from_init = np.sqrt(np.clip(diag + diag[0] - 2 * gram[0], 0, None))
    steps = np.sqrt(np.clip(diag[1:] + diag[:-1] - 2 * np.diag(gram, 1), 0, None))
    return from_init, float(steps.sum())


# ---------------------------------------------------------------------------
# the loss at a parameter vector
# ---------------------------------------------------------------------------

class Surface:
    """The loss of a run's model at any parameter vector: val_ce's recipe on fixed chunks
    and masks (the same chunks for every run with the same data and --seed)."""

    def __init__(self, run: Path, a: argparse.Namespace, device: torch.device):
        cfg = json.loads((run / "config.json").read_text())
        args = cfg["args"]
        self.device = device
        self.model = Denoiser(DenoiserConfig(**cfg["model"])).to(device).eval()
        self.params = list(self.model.parameters())
        manifest = a.manifest or Path(args["manifest"])
        cache = a.cache or Path(args["cache"])
        ds = ChunkDataset(manifest, cache, (a.split,), fake_mel=bool(args.get("fake_mel")))
        if cfg.get("style") is not None:                       # as train.py sets them up
            vocab = cfg["style"]
            ds.style = {k: encode(vocab, v["genre_id"], v["mapper_id"])
                        for k, v in read_style(Path(args["style"])).items()}
            ds.null_style = sizes(vocab)
        if cfg.get("chart_stats") is not None:
            spec = cfg["chart_stats"]
            ds.stats = {k: chartstats.encode(spec, v)
                        for k, v in chartstats.read_stats(Path(args["chart_stats"])).items()}
            ds.null_stats = (chartstats.null_index(spec),) * len(spec["names"])
        if len(ds) == 0:
            raise ValueError(f"no {a.split} chunks in {manifest}")
        rng = np.random.default_rng(a.seed)
        pick = np.sort(rng.choice(len(ds), min(a.chunks, len(ds)), replace=False))
        self.n_chunks = len(pick)
        pairs = []
        for c, i in enumerate(pick):
            item = ds[int(i)]
            for j, g in enumerate(VAL_GAMMAS):
                gen = torch.Generator().manual_seed(100_003 * j + c + 7919 * a.seed)
                x_t, masked = mask_tokens(item["x0"][None], torch.tensor([g]), gen)
                pairs.append((item, x_t[0], masked[0], j))
        cond = [k for k in ("genre", "mapper", "stats") if k in pairs[0][0]]
        self.batches, self.count = [], np.zeros(len(VAL_GAMMAS))
        for lo in range(0, len(pairs), a.batch_size):
            part = pairs[lo:lo + a.batch_size]
            x0 = torch.stack([p[0]["x0"] for p in part])
            g = torch.tensor([p[3] for p in part])
            batch = {"x_t": torch.stack([p[1] for p in part]),
                     "mel": torch.stack([p[0]["mel"] for p in part]),
                     "s": torch.stack([torch.as_tensor(p[0]["s"]) for p in part]).float(),
                     "b": torch.stack([torch.as_tensor(p[0]["b"]) for p in part]).float(),
                     "target": torch.where(x0 == PAD, torch.zeros_like(x0), x0),
                     "masked": torch.stack([p[2] for p in part]).float(),
                     "onehot": F.one_hot(g, len(VAL_GAMMAS)).float()}
            for k in cond:
                batch[k] = torch.stack([torch.as_tensor(p[0][k]) for p in part]).long()
            np.add.at(self.count, g.numpy(), batch["masked"].sum(dim=(1, 2)).numpy())
            self.batches.append({k: v.to(device) for k, v in batch.items()})
        self.cond = cond

    def vector(self) -> np.ndarray:
        return torch.nn.utils.parameters_to_vector(self.params).detach().float().cpu().numpy()

    @torch.no_grad()
    def __call__(self, vec: torch.Tensor) -> float:
        offset = 0
        for p in self.params:                       # copied in: the model never aliases vec
            p.copy_(vec[offset:offset + p.numel()].view_as(p))
            offset += p.numel()
        sums = torch.zeros(len(VAL_GAMMAS), device=self.device)
        for bt in self.batches:
            logits = self.model(bt["x_t"], bt["mel"], bt["s"], bt["b"],
                                **{k: bt[k] for k in self.cond}).float()
            ce = F.cross_entropy(logits.reshape(-1, N_CLASSES), bt["target"].reshape(-1),
                                 reduction="none").view_as(bt["masked"])
            sums += (ce * bt["masked"]).sum(dim=(1, 2)) @ bt["onehot"]
        per = sums.cpu().double().numpy() / np.maximum(self.count, 1)   # MPS: no float64
        return float(per.mean())


def as_device(x: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.array(x, dtype=np.float32)).to(device)    # a copy: x is a memmap


# ---------------------------------------------------------------------------
# Adam's step per coordinate
# ---------------------------------------------------------------------------

def group_of(name: str) -> str:
    if name.startswith(("tok_emb", "pos", "audio_pos", "stat_emb", "genre_emb", "mapper_emb")):
        return "embeddings"
    if name.startswith("audio_"):
        return "audio input"
    if name.startswith(("s_mlp", "b_mlp")):
        return "SR / beat MLPs"
    for key, label in (("self_attn", "self-attention"), ("cross_attn", "cross-attention"),
                       (".ff.", "feed-forward")):
        if key in name:
            return label
    if "ln_" in name or "norm" in name:
        return "LayerNorms"
    return "output head" if name.startswith("head") else "other"


def adam_view(run: Path, meta: dict, at: float) -> tuple[dict | None, dict]:
    """Per snapshot: Adam's step |m^ / (sqrt(v^) + eps)| (in units of lr) and the gradient
    scale sqrt(v^) at the sampled coordinates; the scatter at the snapshot nearest `at` of
    the run, and medians by parameter group there."""
    files = sorted((run / "traj").glob("adam_*.npz"), key=lambda f: int(f.stem.split("_")[1]))
    if meta.get("optim") != "adamw" or not files:
        return None, {}
    b1, b2 = meta["betas"]
    eps = meta["eps"]
    sizes_ = [math.prod(s) for s in meta["shapes"]]
    tensor = np.searchsorted(np.cumsum(sizes_), np.load(run / "traj" / "sample_idx.npy"),
                             side="right")
    groups = np.array([group_of(meta["names"][t]) for t in tensor])
    out = {"steps": [], "u_p10": [], "u_median": [], "u_p90": [], "g_p5": [], "g_median": [],
           "g_p95": [], "eps": eps}
    target = at * meta["steps"]
    pick = min(range(len(files)), key=lambda i: abs(int(files[i].stem.split("_")[1]) - target))
    scatter = {}
    for i, f in enumerate(files):
        z = np.load(f)
        t = float(z["t"])
        if t < 1:
            continue
        m_hat = z["m"].astype(np.float64) / (1 - b1 ** t)
        v_hat = z["v"].astype(np.float64) / (1 - b2 ** t)
        ok = np.isfinite(m_hat) & np.isfinite(v_hat) & (v_hat > 0)
        g = np.sqrt(v_hat[ok])
        u = np.abs(m_hat[ok]) / (g + eps)
        u_pos = u[u > 0]
        if not len(u_pos):
            continue
        out["steps"].append(int(f.stem.split("_")[1]))
        for k, q in (("u_p10", 10), ("u_median", 50), ("u_p90", 90)):
            out[k].append(float(np.percentile(u, q)))
        for k, q in (("g_p5", 5), ("g_median", 50), ("g_p95", 95)):
            out[k].append(float(np.percentile(g, q)))
        if i == pick:
            keep = u > 0
            scatter = {"x": np.log10(g[keep]).astype(np.float32),
                       "y": np.log10(u[keep]).astype(np.float32)}
            out["scatter_step"] = out["steps"][-1]
            out["decades_g"] = float(np.diff(np.percentile(scatter["x"], [5, 95]))[0])
            out["decades_u"] = float(np.diff(np.percentile(scatter["y"], [5, 95]))[0])
            out["groups"] = {
                name: {"coords": int((groups[ok] == name).sum()),
                       "g_median": float(np.median(g[groups[ok] == name])),
                       "u_median": float(np.median(u[groups[ok] == name]))}
                for name in sorted(set(groups[ok]))}
    return (out if out["steps"] else None), scatter


# ---------------------------------------------------------------------------
# one run
# ---------------------------------------------------------------------------

def analyse(run: Path, a: argparse.Namespace, device: torch.device, ref: Path | None) -> tuple:
    tr = read_traj(run)
    X, steps = tr["X"], tr["steps"]
    pl = plane(X)
    from_init, length = distances(pl["gram"])
    surface = Surface(run, a, device)
    T = as_device(X[-1], device)
    d1, d2 = as_device(pl["d"][0], device), as_device(pl["d"][1], device)
    c = pl["coords"]
    lo, hi = c.min(axis=0), c.max(axis=0)
    span = np.maximum(hi - lo, 1e-6)
    ga = np.linspace(lo[0] - a.margin * span[0], hi[0] + a.margin * span[0], a.grid)
    gb = np.linspace(lo[1] - a.margin * span[1], hi[1] + a.margin * span[1], a.grid)
    print(f"{run}: {len(steps)} snapshots ({steps[0]} .. {steps[-1]}), PC1 {pl['explained'][0]:.1%}"
          f" PC2 {pl['explained'][1]:.1%} of the path, |theta_T - theta_0| {from_init[-1]:.2f}, "
          f"path {length:.2f}; {surface.n_chunks} chunks x {len(VAL_GAMMAS)} mask ratios",
          flush=True)

    true = [surface(as_device(x, device)) for x in X]
    on_plane = [surface(T + float(ai) * d1 + float(bi) * d2) for ai, bi in c]
    print(f"  loss at the snapshots {true[0]:.4f} -> {true[-1]:.4f}", flush=True)
    Z = np.full((len(gb), len(ga)), np.nan)
    t0 = time.time()
    for i, bv in enumerate(gb):
        for j, av in enumerate(ga):
            Z[i, j] = surface(T + float(av) * d1 + float(bv) * d2)
        if i == 0 or i == len(gb) - 1:
            left = (time.time() - t0) / (i + 1) * (len(gb) - i - 1)
            print(f"  grid row {i + 1}/{len(gb)}, {left / 60:.1f} min left", flush=True)
    line = {}
    if a.line:
        alpha = np.linspace(0.0, 1.2, a.line)
        X0 = as_device(X[0], device)
        line = {"alpha": alpha.tolist(), "loss": [surface(X0 + float(al) * (T - X0))
                                                  for al in alpha]}
        inside = np.array(line["loss"])[alpha <= 1.0 + 1e-9]
        line["monotone"] = bool(np.all(np.diff(inside) <= 1e-6))
        line["argmin_alpha"] = float(alpha[int(np.nanargmin(line["loss"]))])
    mark = None
    if ref is not None:
        sd = torch.load(ref, map_location="cpu", weights_only=True)["model"]
        surface.model.load_state_dict(sd)
        vec = surface.vector().astype(np.float64)
        off = vec - np.asarray(X[-1], dtype=np.float64)
        ab = pl["d"] @ off
        dist = float(np.linalg.norm(off))
        mark = {"path": str(ref), "a": float(ab[0]), "b": float(ab[1]), "dist_to_final": dist,
                "off_plane": float(math.sqrt(max(dist ** 2 - float(ab @ ab), 0.0))),
                "loss": surface(as_device(vec, device))}
    adam, scatter = adam_view(run, tr["meta"], a.adam_at)
    k = np.unravel_index(np.nanargmin(Z), Z.shape) if np.isfinite(Z).any() else (0, 0)
    meta = tr["meta"]
    res = {"run": str(run), "name": run.name, "optim": meta.get("optim"), "lr": meta.get("lr"),
           "betas": meta.get("betas"), "momentum": meta.get("momentum"),
           "total_steps": meta.get("steps"), "steps": steps,
           "diverged_after": tr["diverged_after"], "explained": pl["explained"],
           "dist_init_final": float(from_init[-1]), "path_length": length,
           "dist_from_init": from_init.tolist(), "coords": c.tolist(),
           "loss_true": true, "loss_plane": on_plane,
           "grid": {"a": ga.tolist(), "b": gb.tolist(),
                    "loss": [[None if not np.isfinite(v) else float(v) for v in row] for row in Z]},
           "grid_min": {"a": float(ga[k[1]]), "b": float(gb[k[0]]), "loss": float(Z[k])},
           "line": line, "ref": mark, "adam": adam}
    return res, scatter


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def title_of(r: dict) -> str:
    if r["optim"] == "sgd":
        how = f"SGD lr {r['lr']:g}" + (f", momentum {r['momentum']:g}" if r["momentum"] else "")
    elif r["betas"] and max(r["betas"]) == 0:
        how = f"normalized gradient (AdamW betas 0 0), lr {r['lr']:g}"
    else:
        how = f"AdamW lr {r['lr']:g}" + (f", betas {r['betas'][0]:g} {r['betas'][1]:g}"
                                          if r["betas"] else "")
    end = f", diverged after step {r['diverged_after']:,}" if r["diverged_after"] else ""
    return f"{r['name']}: {how}{end}"


def grid_of(r: dict) -> np.ndarray:
    return np.array([[np.nan if v is None else v for v in row] for row in r["grid"]["loss"]])


def color_range(r: dict) -> tuple[float, float] | None:
    """The plane's loss range drawn in colour: its minimum up to 1.2 x the path's highest."""
    Z = grid_of(r)
    if not np.isfinite(Z).any():
        return None
    lo = float(np.nanmin(Z))
    hi = min(float(np.nanmax(Z)), 1.2 * float(np.nanmax(np.array(r["loss_true"], dtype=float))))
    return lo, max(hi, lo * 1.05)


def draw_plane(fig, ax, r: dict, cmap, scale: tuple[float, float] | None) -> None:
    """One run's plane as notebook 6.5 draws it. scale: the colour range (shared by runs)."""
    from matplotlib.colors import LogNorm
    Z = grid_of(r)
    ax.set_title(title_of(r), fontsize=10)
    if scale is None or not np.isfinite(Z).any():
        ax.text(0.5, 0.5, "no finite loss on the plane", ha="center", transform=ax.transAxes)
        return
    A, B = np.meshgrid(r["grid"]["a"], r["grid"]["b"])
    lo, hi = scale
    levels = np.geomspace(lo, hi, 41)
    Zm = np.ma.masked_invalid(np.clip(Z, lo, None))
    filled = ax.contourf(A, B, Zm, levels=levels, cmap=cmap, norm=LogNorm(lo, hi), extend="max")
    ax.contour(A, B, Zm, levels=levels[::4], colors=["#80808080"], linewidths=0.7)
    c = np.array(r["coords"])
    ax.plot(c[:, 0], c[:, 1], "-", color=PATH_COLOR, lw=1.3)
    ax.plot(c[:, 0], c[:, 1], ".", color=PATH_COLOR, ms=9)
    ax.plot(*c[0], "o", mfc="none", mec="white", ms=11, mew=1.5, label="start (step 0)")
    ax.plot(*c[-1], "*", color="white", ms=14, label=f"end (step {r['steps'][-1]:,})")
    for i in sorted(set(np.linspace(0, len(c) - 1, 6).round().astype(int))):
        ax.annotate(f"{r['steps'][i]:,}", c[i], textcoords="offset points", xytext=(6, 4),
                    color="white", fontsize=8)
    if r.get("ref"):
        m = r["ref"]
        ax.plot(m["a"], m["b"], "x", color="#ff8c69", ms=11, mew=2,
                label=f"{Path(m['path']).parent.name} ({m['dist_to_final']:.3g} away)")
    ax.legend(loc="best", fontsize=8, framealpha=0.6)
    ax.text(0.98, 0.02, f"start to end {r['dist_init_final']:.3g}, path {r['path_length']:.3g}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
            bbox={"boxstyle": "round", "fc": "white", "alpha": 0.6, "lw": 0})
    e1, e2 = r["explained"]
    ax.set_xlabel(f"PC1 of the path ({e1:.0%} of its variance)")
    ax.set_ylabel(f"PC2 ({e2:.0%})")
    bar = fig.colorbar(filled, ax=ax)
    ticks = [t for t in (0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1, 1.5, 2, 3, 5)
             if lo <= t <= hi]
    bar.set_ticks(ticks)
    bar.set_ticklabels([f"{t:g}" for t in ticks])
    bar.minorticks_off()
    bar.set_label("masked CE (val_ce's recipe)")


def plot_all(res: dict, scatters: dict, out: Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("udl", ["#" + h for h in NOTEBOOK_CMAP])
    runs, written = res["runs"], []

    cols = 1 if len(runs) == 1 else (2 if len(runs) <= 4 else 3)
    rows = math.ceil(len(runs) / cols)
    size = (7.5, 6.5) if len(runs) == 1 else (6.4, 5.4)
    fig, axes = plt.subplots(rows, cols, figsize=(size[0] * cols, size[1] * rows), squeeze=False)
    ranges = [x for x in map(color_range, runs) if x is not None]
    shared = (min(x[0] for x in ranges), max(x[1] for x in ranges)) if ranges else None
    for ax, r in zip(axes.flat, runs, strict=False):                  # the same colours for the same loss
        draw_plane(fig, ax, r, cmap, shared if color_range(r) else None)
    for ax in list(axes.flat)[len(runs):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out / "fit_path.png", dpi=130)
    plt.close(fig)
    written.append(out / "fit_path.png")

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(17, 4.8))
    for k, r in enumerate(runs):
        col = RUN_COLORS[k % len(RUN_COLORS)]
        st = np.array(r["steps"])
        name = r["name"]
        if len(runs) == 1:
            log = read_rows(Path(r["run"]) / "log.csv")
            if log:
                s = np.array([float(x["step"]) for x in log])
                y = np.array([float(x["masked_ce"]) for x in log])
                w = max(1, len(y) // 100)                   # a centred mean, no padding
                y = np.convolve(y, np.ones(w) / w, mode="valid")
                s = s[(w - 1) // 2:(w - 1) // 2 + len(y)]
                ax1.plot(s, y, color="#bbbbbb", lw=1, label="train batches (log.csv, smoothed)")
            val = read_rows(Path(r["run"]) / "val.csv")
            if val:
                ax1.plot([float(x["step"]) for x in val], [float(x["val_ce"]) for x in val],
                         "s", color="#555555", ms=4, label="val_ce (val.csv)")
        ax1.plot(st, r["loss_true"], ".-", color=col, label=f"{name}: at the snapshot")
        ax1.plot(st, r["loss_plane"], "--", color=col, alpha=0.8,
                 label=f"{name}: at its projection on the plane")
        if r["line"]:
            ax2.plot(r["line"]["alpha"], r["line"]["loss"], ".-", color=col, label=name)
        ax3.plot(st, r["dist_from_init"], ".-", color=col, label=name)
    lin = max(1, min(r["steps"][1] for r in runs if len(r["steps"]) > 1))
    for ax in (ax1, ax3):
        ax.set_xscale("symlog", linthresh=lin)
        ax.set_xlabel("step")
    ax1.set_yscale("log")
    ax1.set_ylabel("masked CE")
    ax1.set_title("loss along the path (the gap: what the plane leaves out)", fontsize=10)
    ax1.legend(fontsize=7)
    ax2.axvline(1.0, color="#999999", lw=0.8, ls=":")
    ax2.set_yscale("log")
    ax2.set_xlabel(r"$\alpha$ in $\theta_0 + \alpha\,(\theta_T - \theta_0)$")
    ax2.set_ylabel("masked CE")
    ax2.set_title("straight line from the start through the end", fontsize=10)
    ax2.legend(fontsize=8)
    ax3.set_ylabel(r"$\|\theta - \theta_0\|$")
    ax3.set_title("distance from the start", fontsize=10)
    ax3.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fit_curves.png", dpi=130)
    plt.close(fig)
    written.append(out / "fit_curves.png")

    adams = [(k, r) for k, r in enumerate(runs) if r.get("adam") and str(k) in scatters]
    if adams:
        fig, axes = plt.subplots(1, len(adams) + 1, figsize=(5.6 * (len(adams) + 1), 4.8),
                                 squeeze=False)
        for ax, (k, r) in zip(axes[0], adams, strict=False):
            x, y = scatters[str(k)]["x"], scatters[str(k)]["y"]
            ax.hexbin(x, y, gridsize=60, bins="log", cmap="viridis", mincnt=1)
            xm, ym = float(np.median(x)), float(np.median(y))
            xs = np.array([np.percentile(x, 1), np.percentile(x, 99)])
            ax.plot(xs, ym + (xs - xm), color="#ff8c69", lw=1.5,
                    label="gradient descent: step ∝ gradient")
            ad = r["adam"]
            if ad.get("eps") and math.log10(ad["eps"]) > float(x.min()) - 0.5:
                ax.axvline(math.log10(ad["eps"]), color="#999999", ls=":", lw=1,
                           label="ε (left of it: step ∝ gradient / ε)")
            ax.set_xlabel(r"gradient scale $\log_{10}\sqrt{\hat v}$")
            ax.set_ylabel(r"Adam's step / lr: $\log_{10}\,|\hat m| / (\sqrt{\hat v} + \epsilon)$")
            ax.set_title(f"{r['name']}, step {ad['scatter_step']:,}: 5-95% spans "
                         f"{ad['decades_g']:.1f} vs {ad['decades_u']:.1f} decades", fontsize=9)
            ax.legend(fontsize=8, loc="lower right")
        ax = axes[0][-1]
        for k, r in adams:
            col = RUN_COLORS[k % len(RUN_COLORS)]
            ad = r["adam"]
            ax.plot(ad["steps"], ad["u_median"], ".-", color=col, label=f"{r['name']}: median")
            ax.fill_between(ad["steps"], ad["u_p10"], ad["u_p90"], color=col, alpha=0.15)
        ax.set_xscale("symlog", linthresh=max(1, min(r["adam"]["steps"][0] for _, r in adams)))
        ax.set_yscale("log")
        ax.set_xlabel("step")
        ax.set_ylabel("Adam's step / lr (10-90% band)")
        ax.set_title("how far Adam moves a coordinate, in units of lr", fontsize=10)
        first = Path(adams[0][1]["run"]) / "traj" / "steps.csv"
        lrs = read_rows(first)
        if lrs:
            tw = ax.twinx()
            tw.plot([float(x["step"]) for x in lrs], [float(x["lr"]) for x in lrs], ":",
                    color="#777777", label="lr")
            tw.set_ylabel("lr (dotted)")
        ax.legend(fontsize=8, loc="lower left")
        fig.tight_layout()
        fig.savefig(out / "fit_adam.png", dpi=130)
        plt.close(fig)
        written.append(out / "fit_adam.png")
    return written


# ---------------------------------------------------------------------------

def summary_lines(res: dict) -> list[str]:
    lines = []
    for r in res["runs"]:
        e1, e2 = r["explained"]
        lines.append(f"{title_of(r)}")
        lines.append(f"  plane: PC1 {e1:.1%} + PC2 {e2:.1%} = {e1 + e2:.1%} of the path; "
                     f"|theta_T - theta_0| {r['dist_init_final']:.3g}, path {r['path_length']:.3g}")
        gap = np.abs(np.array(r["loss_plane"]) - np.array(r["loss_true"]))
        lines.append(f"  loss {r['loss_true'][0]:.4f} -> {r['loss_true'][-1]:.4f}; plane vs "
                     f"snapshot: max gap {np.nanmax(gap):.4f}, median {np.nanmedian(gap):.4f}; "
                     f"grid minimum {r['grid_min']['loss']:.4f} at "
                     f"({r['grid_min']['a']:.3g}, {r['grid_min']['b']:.3g})")
        if r["line"]:
            lines.append(f"  straight line: {'monotone' if r['line']['monotone'] else 'NOT monotone'}"
                         f" on [0, 1], minimum at alpha {r['line']['argmin_alpha']:.2f}")
        if r["ref"]:
            m = r["ref"]
            lines.append(f"  {m['path']}: {m['dist_to_final']:.4g} from the end "
                         f"({m['off_plane']:.4g} off the plane), loss {m['loss']:.4f}")
        if r["adam"] and "decades_g" in r["adam"]:
            ad = r["adam"]
            i = ad["steps"].index(ad["scatter_step"])
            lines.append(f"  Adam at step {ad['scatter_step']:,}: gradient scale spans "
                         f"{ad['decades_g']:.1f} decades (5-95%), the step {ad['decades_u']:.1f}; "
                         f"median step {ad['u_median'][i]:.3f} lr")
    return lines


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="+", type=Path,
                    help="run directories with traj/ (train.py --snapshot-every)")
    ap.add_argument("--out", type=Path, default=None, help="default <first run>/fit_viz")
    ap.add_argument("--grid", type=int, default=25, help="points per axis of the plane")
    ap.add_argument("--margin", type=float, default=0.25,
                    help="the grid beyond the path, as a share of its span")
    ap.add_argument("--line", type=int, default=25,
                    help="points on the straight line, alpha 0 .. 1.2; 0 = off")
    ap.add_argument("--chunks", type=int, default=64, help="chunks the loss is measured on")
    ap.add_argument("--split", default="train")
    ap.add_argument("--batch-size", type=int, default=32,
                    help="(chunk, mask ratio) pairs per forward pass")
    ap.add_argument("--seed", type=int, default=0, help="the chunks and their masks")
    ap.add_argument("--adam-at", type=float, default=0.25,
                    help="Adam's scatter at the snapshot nearest this share of the run")
    ap.add_argument("--ref", type=Path, default=None,
                    help="a checkpoint to mark on the first run's plane")
    ap.add_argument("--replot", action="store_true",
                    help="redraw the figures from <out>/fit_viz.json and .npz")
    ap.add_argument("--manifest", type=Path, default=None, help="default: the run's")
    ap.add_argument("--cache", type=Path, default=None, help="default: the run's")
    ap.add_argument("--device", default="auto")
    a = ap.parse_args(argv)
    out = a.out or a.runs[0] / "fit_viz"
    out.mkdir(parents=True, exist_ok=True)

    if a.replot:
        if not (out / "fit_viz.json").exists():
            print(f"--replot: no {out / 'fit_viz.json'}", file=sys.stderr)
            return 2
        res = json.loads((out / "fit_viz.json").read_text())
        scatters = {}
        if (out / "fit_viz.npz").exists():
            z = np.load(out / "fit_viz.npz")
            for k in {f.split("_")[0] for f in z.files}:
                scatters[k] = {"x": z[f"{k}_x"], "y": z[f"{k}_y"]}
    else:
        for run in a.runs:
            if not (run / "traj" / "meta.json").exists():
                print(f"{run}: no traj/ (train.py --snapshot-every)", file=sys.stderr)
                return 2
        device = pick_device(a.device)
        res = {"eval": {"split": a.split, "chunks": a.chunks, "gammas": list(VAL_GAMMAS),
                        "seed": a.seed, "grid": a.grid, "margin": a.margin}, "runs": []}
        scatters = {}
        for k, run in enumerate(a.runs):
            try:
                r, sc = analyse(run, a, device, a.ref if k == 0 else None)
            except ValueError as e:                       # e.g. a run that blew up at once
                print(f"{run}: skipped ({e})", file=sys.stderr)
                continue
            if sc:
                scatters[str(len(res["runs"]))] = sc
            res["runs"].append(r)
        if not res["runs"]:
            return 2
        (out / "fit_viz.json").write_text(json.dumps(res, indent=1))
        np.savez_compressed(out / "fit_viz.npz", **{f"{k}_{x}": v for k, sc in scatters.items()
                                                   for x, v in sc.items()})
    for line in summary_lines(res):
        print(line)
    for path in plot_all(res, scatters, out):
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
