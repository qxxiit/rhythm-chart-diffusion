"""Masked (absorbing-state) discrete diffusion for 4K charts. Design doc §4.7-4.10.

    Denoiser        f(x_t, mel, s, b) -> logits [B, 384, 4, 5] over the clean state x0
    sample_gamma    mask ratios for a batch, stratified over (0, 1]
    mask_tokens     forward process: each non-PAD position becomes MASK with probability gamma
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
    """f(x_t, mel, s, b) -> logits over x0.

    x_t  [B, L, K] long     tokens with MASK where the forward process erased them
    mel  [B, L * r, F]      log-Mel frames, r = 4 per cell
    s    [B]                chart SR
    b    [B]                mean beat length of the chunk, ms
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
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.n_layers))
        self.out_norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, K * N_CLASSES)

    def forward(self, x_t: torch.Tensor, mel: torch.Tensor, s: torch.Tensor,
                b: torch.Tensor) -> torch.Tensor:
        batch = x_t.shape[0]
        pad = (x_t == PAD).all(dim=-1)                           # [B, L] whole PAD rows
        pad[:, 0] &= ~pad.all(dim=1)                             # never mask every key

        h = self.tok_emb(x_t).reshape(batch, L, -1) + self.pos
        cond = self.s_mlp(sinusoidal(s * self.config.s_scale, h.shape[-1])) \
            + self.b_mlp(sinusoidal(torch.log(b) * self.config.b_scale, h.shape[-1]))
        h = h + cond[:, None, :]

        mem = self.audio_conv(mel.transpose(1, 2)).transpose(1, 2)   # [B, L, d]
        mem = self.audio_norm(mem + self.audio_pos)
        if self.config.audio_add:
            h = h + mem

        for block in self.blocks:
            h = block(h, mem, pad)
        return self.head(self.out_norm(h)).reshape(batch, L, K, N_CLASSES)


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
                generator: torch.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """q(x_t | x0): every non-PAD position independently becomes MASK with probability gamma.
    Returns (x_t, masked)."""
    u = torch.rand(x0.shape, generator=generator,
                   device=generator.device if generator is not None else x0.device)
    masked = (u.to(x0.device) < gamma.view(-1, 1, 1)) & (x0 != PAD)
    return torch.where(masked, torch.full_like(x0, MASK), x0), masked


def diffusion_loss(model: nn.Module, x0: torch.Tensor, mel: torch.Tensor, s: torch.Tensor,
                   b: torch.Tensor, gamma: torch.Tensor | None = None,
                   generator: torch.Generator | None = None) -> tuple[torch.Tensor, dict]:
    """L = E_gamma[ (1/gamma) * sum over masked positions of -log p(x0) ] / N   (§4.9).

    The 1/gamma weight makes every mask ratio count equally: a chunk masked at
    ratio gamma has about gamma * N masked positions. Divided by N = L * K it is
    an upper bound on the per-position NLL.
    Returns (loss, info) where info holds unweighted diagnostics.
    """
    if gamma is None:
        gamma = sample_gamma(x0.shape[0], x0.device, generator)
    x_t, masked = mask_tokens(x0, gamma, generator)
    logits = model(x_t, mel, s, b).float()
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
    """Model from a checkpoint written by scripts/train.py, in eval mode."""
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = Denoiser(DenoiserConfig(**ckpt["config"])).to(device)
    model.load_state_dict(ckpt["model"])
    return model.eval()


def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
