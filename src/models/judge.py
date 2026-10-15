"""Human or AI: a judge over 8-bar windows of a chart and its audio (실험 2026-10-15).

    Judge(trunk, audio=True)    f(x, mel, s, b) -> logit [B] (positive: the window looks
                                generated); rows(...) -> [B, L] the score of each row

The trunk is the denoiser (src/models/diffusion.py) the charts were made with, started from
its checkpoint: embeddings, audio memory and blocks, on the clean window (no MASK). It already
knows what human charts look like against the audio, which a judge trained on a few thousand
windows could not learn from scratch. A linear score per row is averaged over the rows that
are not PAD, so the window's logit is the mean of its rows' and the rows say where the tells
are. audio=False gives the judge silence (zeros, the song's mean after normalisation) for the
audio: what it then finds is in the chart alone.

s is the chart's own SR (rosu), not the SR it was asked for: a player sees the chart, not the
request, and a judge given the request could tell "too hard for what was asked" instead.
"""

from __future__ import annotations

import torch
from torch import nn

from src.data.tokenizer import PAD
from src.models.diffusion import Denoiser


class Judge(nn.Module):
    def __init__(self, trunk: Denoiser, audio: bool = True):
        super().__init__()
        self.trunk, self.audio = trunk, audio
        self.score = nn.Linear(trunk.config.d_model, 1)
        nn.init.zeros_(self.score.weight)              # starts at 0: no opinion
        nn.init.zeros_(self.score.bias)

    def rows(self, x: torch.Tensor, mel: torch.Tensor, s: torch.Tensor,
             b: torch.Tensor) -> torch.Tensor:
        """[B, L] the score of each row (its share of the window's logit, times the rows)."""
        if not self.audio:
            mel = torch.zeros_like(mel)
        return self.score(self.trunk.hidden(x, mel, s, b)).squeeze(-1)

    def forward(self, x: torch.Tensor, mel: torch.Tensor, s: torch.Tensor,
                b: torch.Tensor) -> torch.Tensor:
        r = self.rows(x, mel, s, b).float()
        keep = ~(x == PAD).all(dim=-1)
        return (r * keep).sum(dim=1) / keep.sum(dim=1).clamp_min(1)
