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

import zlib

import numpy as np
import torch
from torch import nn

from src.data.tokenizer import EMPTY, HOLD_BODY, HOLD_END, HOLD_START, PAD, TAP, K
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


def view_tokens(tokens: np.ndarray, view: str, seed=(0,)) -> np.ndarray:
    """A chart [rows, K] as --view shows it; the same for every chart of a song, so a judge can
    only use what is kept.

        full    the chart
        taps    long notes as taps at their heads (bodies and tails empty): nothing of which
                notes are held, nor for how long
        rhythm  taps, and each onset row's notes on lanes drawn at random (as many as it had,
                seed: the song and the chart): which rows have notes and how many, nothing of
                the lanes
        holds   the rhythm view with the long notes kept: each onset row's taps and long-note
                heads on lanes drawn at random among those no long note holds, every long note
                in its lane to its tail (holds_view): which rows have notes, how many, which
                are held and for how long, nothing of the lanes
    """
    if view == "full":
        return tokens
    if view == "holds":
        return holds_view(tokens, seed)
    x = np.array(tokens, copy=True)
    x[x == HOLD_START] = TAP
    x[(x == HOLD_BODY) | (x == HOLD_END)] = EMPTY
    if view == "rhythm":
        n = (x == TAP).sum(axis=1)
        rows = np.flatnonzero(n)
        perm = np.argsort(np.random.default_rng(list(seed)).random((len(rows), K)), axis=1)
        x[rows] = EMPTY
        x[np.repeat(rows, n[rows]), perm[np.arange(K)[None, :] < n[rows][:, None]]] = TAP
    return x


def holds_view(tokens: np.ndarray, seed=(0,)) -> np.ndarray:
    """view_tokens' "holds": a chart (any shape [..., K], no MASK) with its notes moved to lanes
    drawn at random row by row, a long note's body and tail following its head's new lane. A
    body or tail with no head above it (a window that starts inside a long note) gets a lane
    where it first shows. PAD rows stay; the rows' counts of taps and heads and every long
    note's length stay, so the grammar holds."""
    x = np.asarray(tokens)
    rows = x.reshape(-1, K)
    out = np.full_like(rows, EMPTY)
    rng = np.random.default_rng(list(seed))
    held = {}                                        # the chart's lane -> the view's lane
    for r, row in enumerate(rows):
        if (row == PAD).all():
            out[r] = PAD
            continue
        ends = []
        for lane in np.flatnonzero((row == HOLD_BODY) | (row == HOLD_END)):
            if lane not in held:                     # a long note from above the window
                free = [o for o in range(K) if out[r, o] == EMPTY and o not in held.values()]
                held[lane] = free[int(rng.integers(len(free)))]
            out[r, held[lane]] = row[lane]
            if row[lane] == HOLD_END:
                ends.append(lane)
        free = [o for o in range(K) if out[r, o] == EMPTY and o not in held.values()]
        free = [free[i] for i in rng.permutation(len(free))]
        heads = np.flatnonzero(row == HOLD_START)
        taps = int((row == TAP).sum())
        for lane, o in zip(heads, free, strict=False):
            out[r, o] = HOLD_START
            held[lane] = o
        out[r, free[len(heads):len(heads) + taps]] = TAP
        for lane in ends:
            del held[lane]
    return out.reshape(x.shape)


def stable_hash(s: str) -> int:
    return zlib.crc32(s.encode("utf-8"))
