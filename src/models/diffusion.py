"""Masked (absorbing-state) discrete diffusion for 4K charts. Design doc §4.7-4.10.

    Denoiser        f(x_t, mel, s, b[, genre, mapper, stats]) -> logits [B, 384, 4, 5] over x0
    Styled          a Denoiser with a fixed style and chart stats, called as f(x_t, mel, s, b)
                    by the sampler; with classifier-free guidance if asked
    sample_gamma    mask ratios for a batch, stratified over (0, 1]
    mask_tokens     forward process: each non-PAD position becomes MASK with probability gamma
                    (or, for chunks picked by row_mask, each whole row of 4 lanes)
    diffusion_loss  continuous-time loss  E_gamma[ (1/gamma) * sum_masked CE ] / N

Deliberate differences from the design doc:
  - The head outputs the 5 clean classes directly. The doc outputs 7 logits and
    sets MASK/PAD to -inf; the resulting distributions are identical.
  - The audio memory gets its own learned positional embedding. Without one,
    cross-attention has no way to tell which frames belong to which cell.
  - Memory row i is also added to cell i before the blocks (audio_add). Cell i
    and frames 4i..4i+3 are aligned by construction, and cross-attention alone
    has to learn that diagonal first: on a toy set whose "audio" spells out the
    answer, a 2-layer model with cross-attention only was still at masked CE
    0.72 after 600 steps, and 0.09 after 200 with the addition. Cross-attention
    stays for context; audio_add=False gives the design-doc model (실험 큐 5).
  - Pre-LN blocks (LayerNorm inside each residual branch) plus a final LayerNorm,
    and a LayerNorm on the audio memory.
  - PAD cells are excluded as attention keys.
  - The loss is divided by N = 384 * 4, so it reads as an NLL bound per position.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from src.data.tokenizer import MASK, PAD, K, L

N_CLASSES = 5                          # EMPTY, TAP, HOLD_START, HOLD_BODY, HOLD_END
VOCAB = 7                              # + MASK, PAD as inputs


@dataclass
class DenoiserConfig:
    d_model: int = 256
    lane_dim: int = 64                 # d_model = K * lane_dim: the 4 lanes are concatenated
    n_layers: int = 6
    n_heads: int = 4
    d_ff: int = 1024
    n_mels: int = 80
    frames_per_cell: int = 4           # r = 48 / 12
    dropout: float = 0.0
    s_scale: float = 100.0             # SR 0..10 -> 0..1000 before the sinusoid
    b_scale: float = 100.0             # log(beat ms) 5..7.6 -> 500..760
    audio_add: bool = True             # also add memory row i to cell i (see module docstring)
    n_genres: int = 0                  # style inputs (src/data/style.py); 0 = none, as in
    n_mappers: int = 0                 # full-v1. Index n_genres / n_mappers is "no label"
    n_stats: int = 0                   # chart-stat inputs (src/data/chartstats.py); 0 = none
    stat_bins: int = 0                 # buckets per stat; index stat_bins is "no value"


def sinusoidal(x: torch.Tensor, dim: int) -> torch.Tensor:
    """[B] scalars -> [B, dim] sin/cos features (the usual diffusion time embedding)."""
    half = dim // 2
    freqs = torch.exp(-math.log(10_000.0) * torch.arange(half, device=x.device) / half)
    args = x.float()[:, None] * freqs[None, :]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class Block(nn.Module):
    """self-attention (cell <-> cell), cross-attention (cell -> audio), FFN; pre-LN residuals."""

    def __init__(self, c: DenoiserConfig):
        super().__init__()
        d = c.d_model
        self.ln_self, self.ln_cross, self.ln_ff = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.self_attn = nn.MultiheadAttention(d, c.n_heads, dropout=c.dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d, c.n_heads, dropout=c.dropout, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, c.d_ff), nn.GELU(), nn.Linear(c.d_ff, d))
        self.drop = nn.Dropout(c.dropout)

    def forward(self, h: torch.Tensor, mem: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
        q = self.ln_self(h)
        h = h + self.drop(self.self_attn(q, q, q, key_padding_mask=pad, need_weights=False)[0])
        q = self.ln_cross(h)
        h = h + self.drop(self.cross_attn(q, mem, mem, key_padding_mask=pad, need_weights=False)[0])
        return h + self.drop(self.ff(self.ln_ff(h)))


class Denoiser(nn.Module):
    """f(x_t, mel, s, b[, genre, mapper, stats]) -> logits over x0.

    x_t  [B, L, K] long     tokens with MASK where the forward process erased them
    mel  [B, L * r, F]      log-Mel frames, r = 4 per cell
    s    [B]                chart SR
    b    [B]                mean beat length of the chunk, ms
    genre, mapper  [B] long style indices (style.encode), only with config.n_genres /
                            n_mappers > 0; None or the index n_genres / n_mappers = no
                            label. Their embeddings join s and b in the per-chunk condition.
    stats [B, n_stats] long buckets of the chart's stats (chartstats.encode), only with
                            config.n_stats > 0; None or stat_bins = no value. One embedding
                            table per stat, summed into the condition as well.
    ->   [B, L, K, 5]
    There is no time input: the fraction of MASK tokens already says how far
    along the reverse process is (design doc §4.10).
    """

    def __init__(self, config: DenoiserConfig | None = None):
        super().__init__()
        c = self.config = config or DenoiserConfig()
        if c.d_model != K * c.lane_dim:
            raise ValueError("d_model must equal K * lane_dim")
        d = c.d_model
        self.tok_emb = nn.Embedding(VOCAB, c.lane_dim)          # shared by the 4 lanes
        self.pos = nn.Parameter(torch.randn(L, d) * 0.02)
        self.audio_conv = nn.Conv1d(c.n_mels, d, kernel_size=c.frames_per_cell,
                                    stride=c.frames_per_cell)
        self.audio_pos = nn.Parameter(torch.randn(L, d) * 0.02)
        self.audio_norm = nn.LayerNorm(d)
        self.s_mlp = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))
        self.b_mlp = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))
        if c.n_genres:
            self.genre_emb = nn.Embedding(c.n_genres + 1, d)          # + "no label"
            nn.init.normal_(self.genre_emb.weight, std=0.02)
        if c.n_mappers:
            self.mapper_emb = nn.Embedding(c.n_mappers + 1, d)
            nn.init.normal_(self.mapper_emb.weight, std=0.02)
        if c.n_stats:
            if c.stat_bins < 1:
                raise ValueError("n_stats needs stat_bins >= 1")
            self.stat_emb = nn.Embedding(c.n_stats * (c.stat_bins + 1), d)   # + "no value"
            nn.init.normal_(self.stat_emb.weight, std=0.02)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.n_layers))
        self.out_norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, K * N_CLASSES)

    def forward(self, x_t: torch.Tensor, mel: torch.Tensor, s: torch.Tensor,
                b: torch.Tensor, genre: torch.Tensor | None = None,
                mapper: torch.Tensor | None = None,
                stats: torch.Tensor | None = None) -> torch.Tensor:
        batch = x_t.shape[0]
        pad = (x_t == PAD).all(dim=-1)                           # [B, L] whole PAD rows
        pad[:, 0] &= ~pad.all(dim=1)                             # never mask every key

        h = self.tok_emb(x_t).reshape(batch, L, -1) + self.pos
        cond = self.s_mlp(sinusoidal(s * self.config.s_scale, h.shape[-1])) \
            + self.b_mlp(sinusoidal(torch.log(b) * self.config.b_scale, h.shape[-1]))
        for name, n, idx in (("genre", self.config.n_genres, genre),
                             ("mapper", self.config.n_mappers, mapper)):
            if n:
                if idx is None:
                    idx = torch.full((batch,), n, dtype=torch.long, device=x_t.device)
                cond = cond + getattr(self, f"{name}_emb")(idx)
            elif idx is not None:
                raise ValueError(f"this model has no {name} input (trained without --style)")
        c = self.config
        if c.n_stats:
            if stats is None:
                stats = torch.full((batch, c.n_stats), c.stat_bins, dtype=torch.long,
                                   device=x_t.device)
            offset = torch.arange(c.n_stats, device=x_t.device) * (c.stat_bins + 1)
            cond = cond + self.stat_emb(stats.long() + offset).sum(dim=1)
        elif stats is not None:
            raise ValueError("this model has no chart-stat input (trained without --chart-stats)")
        h = h + cond[:, None, :]

        mem = self.audio_conv(mel.transpose(1, 2)).transpose(1, 2)   # [B, L, d]
        mem = self.audio_norm(mem + self.audio_pos)
        if self.config.audio_add:
            h = h + mem

        for block in self.blocks:
            h = block(h, mem, pad)
        return self.head(self.out_norm(h)).reshape(batch, L, K, N_CLASSES)


class Styled(nn.Module):
    """A Denoiser with a fixed style for sampling: forward(x_t, mel, s, b), as sampler calls it.

    genre, mapper   vocab indices (style.encode), None = no label
    stats           chart-stat buckets (chartstats.encode), one per stat; None = none given
    guidance        classifier-free guidance w: logits = c + w * (c - u), c with the style
                    and stats and u without any: two forward passes per call. w = 0: c alone.
    """

    def __init__(self, model: Denoiser, genre: int | None = None, mapper: int | None = None,
                 guidance: float = 0.0, stats=None):
        super().__init__()
        self.model, self.config = model, model.config
        self.genre, self.mapper, self.guidance = genre, mapper, guidance
        if stats is not None and not self.config.n_stats:
            raise ValueError("this model has no chart-stat input (trained without --chart-stats)")
        if stats is not None and len(stats) != self.config.n_stats:
            raise ValueError(f"{len(stats)} chart stats for a model with {self.config.n_stats}")
        self.stats = None if stats is None else tuple(int(v) for v in stats)
        if self.stats is not None and not all(0 <= v <= self.config.stat_bins for v in self.stats):
            raise ValueError(f"chart-stat buckets {self.stats} outside 0..{self.config.stat_bins}")

    def _stats(self, stats, like: torch.Tensor) -> torch.Tensor | None:
        if not self.config.n_stats:
            return None
        row = stats if stats is not None else (self.config.stat_bins,) * self.config.n_stats
        return torch.tensor(row, dtype=torch.long, device=like.device).expand(like.shape[0], -1)

    def _idx(self, v: int | None, n: int, like: torch.Tensor) -> torch.Tensor | None:
        if not n:
            if v is not None:
                raise ValueError("this model has no style input (trained without --style)")
            return None
        return torch.full((like.shape[0],), n if v is None else v, dtype=torch.long,
                          device=like.device)

    def forward(self, x_t, mel, s, b):
        c = self.config
        out = self.model(x_t, mel, s, b, genre=self._idx(self.genre, c.n_genres, x_t),
                         mapper=self._idx(self.mapper, c.n_mappers, x_t),
                         stats=self._stats(self.stats, x_t))
        if self.guidance:
            free = self.model(x_t, mel, s, b, genre=self._idx(None, c.n_genres, x_t),
                              mapper=self._idx(None, c.n_mappers, x_t),
                              stats=self._stats(None, x_t))
            out = out + self.guidance * (out - free)
        return out


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# ---------------------------------------------------------------------------
# forward process and loss
# ---------------------------------------------------------------------------

def sample_gamma(batch: int, device=None, generator: torch.Generator | None = None,
                 eps: float = 1e-3) -> torch.Tensor:
    """Mask ratios gamma_i = (u + i / B) mod 1, clipped below at eps.

    Stratified: one uniform draw shifted by i/B covers (0, 1] evenly within a
    batch, which cuts the variance of the 1/gamma weight (design doc §4.9).
    A generator draws on its own device, so a CPU generator gives the same
    ratios on CPU, CUDA and MPS.
    """
    u = torch.rand((), generator=generator,
                   device=generator.device if generator is not None else device)
    gamma = (u.to(device) + torch.arange(batch, device=device) / batch) % 1.0
    return gamma.clamp_min(eps)


def mask_tokens(x0: torch.Tensor, gamma: torch.Tensor,
                generator: torch.Generator | None = None,
                rows: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """q(x_t | x0): every non-PAD position independently becomes MASK with probability gamma.
    rows: optional bool [B]; in those chunks whole rows (all lanes of a cell) are masked
    together with probability gamma, so the model also learns to fill a row from the
    rows around it (which lanes, not only whether). Returns (x_t, masked)."""
    gen_device = generator.device if generator is not None else x0.device
    u = torch.rand(x0.shape, generator=generator, device=gen_device)
    if rows is not None:
        u_row = torch.rand((*x0.shape[:2], 1), generator=generator, device=gen_device)
        u = torch.where(rows.to(gen_device).view(-1, 1, 1), u_row.expand_as(u), u)
    masked = (u.to(x0.device) < gamma.view(-1, 1, 1)) & (x0 != PAD)
    return torch.where(masked, torch.full_like(x0, MASK), x0), masked


def diffusion_loss(model: nn.Module, x0: torch.Tensor, mel: torch.Tensor, s: torch.Tensor,
                   b: torch.Tensor, gamma: torch.Tensor | None = None,
                   generator: torch.Generator | None = None,
                   row_mask: float = 0.0, genre: torch.Tensor | None = None,
                   mapper: torch.Tensor | None = None,
                   stats: torch.Tensor | None = None) -> tuple[torch.Tensor, dict]:
    """L = E_gamma[ (1/gamma) * sum over masked positions of -log p(x0) ] / N   (§4.9).

    The 1/gamma weight makes every mask ratio count equally: a chunk masked at
    ratio gamma has about gamma * N masked positions. Divided by N = L * K it is
    an upper bound on the per-position NLL.
    row_mask: share of chunks masked row by row (mask_tokens rows=); the 1/gamma
    weight stays right, since every position is still masked with probability gamma.
    genre, mapper: style indices for a model with style inputs (None: not passed);
    stats: chart-stat buckets for a model with chart-stat inputs (None: not passed).
    Returns (loss, info) where info holds unweighted diagnostics.
    """
    if gamma is None:
        gamma = sample_gamma(x0.shape[0], x0.device, generator)
    rows = None
    if row_mask > 0:
        rows = torch.rand(x0.shape[0], generator=generator,
                          device=generator.device if generator is not None else x0.device)
        rows = rows < row_mask
    x_t, masked = mask_tokens(x0, gamma, generator, rows)
    style = {k: v for k, v in (("genre", genre), ("mapper", mapper), ("stats", stats))
             if v is not None}
    logits = model(x_t, mel, s, b, **style).float()
    target = torch.where(x0 == PAD, torch.zeros_like(x0), x0)       # PAD is never scored
    ce = F.cross_entropy(logits.reshape(-1, N_CLASSES), target.reshape(-1),
                         reduction="none").view_as(x0)
    per_chunk = (ce * masked).sum(dim=(1, 2)) / gamma
    loss = per_chunk.mean() / (L * K)
    with torch.no_grad():
        n_masked = masked.sum()
        info = {
            "masked_ce": float((ce * masked).sum() / n_masked.clamp_min(1)),
            "mask_rate": float(n_masked / (x0 != PAD).sum().clamp_min(1)),
        }
    return loss, info


def load_denoiser(path, device="cpu") -> Denoiser:
    """Model from a checkpoint written by scripts/train.py, in eval mode. model.style is
    the style vocab it was trained with (style.py), None for a model without style inputs;
    model.chart_stats the chart-stat spec (chartstats.py), None without chart-stat inputs."""
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = Denoiser(DenoiserConfig(**ckpt["config"])).to(device)
    model.load_state_dict(ckpt["model"])
    model.style = ckpt.get("style")
    model.chart_stats = ckpt.get("chart_stats")
    return model.eval()


def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
