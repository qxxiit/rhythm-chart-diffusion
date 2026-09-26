"""Reverse process (design doc §4.8): fill MASK cells with the denoiser under the hold grammar.

    sample_window   one 384-cell window, T steps; step t opens ~1/t of the MASK cells left
    generate_song   a whole song, window by window
        mode="continue"     every window after the first keeps the last 2 bars of the
                            previous one fixed and fills 6 new bars (§4.8 청크 이어 생성)
        mode="independent"  every chunk on its own, holds closed at chunk edges (실험 큐 3c)

Grammar per lane: x[i+1] in {BODY, END}  <=>  x[i] in {START, BODY}.
A cell being opened only looks at neighbours that are already open, so every
adjacent pair is checked by whichever of the two opens later; cells opened in
the same step go left to right. PAD, and the outside of a closed window edge,
count as EMPTY. A run of MASK cells between two open cells can always be
filled, so sampling never gets stuck.
"""

from __future__ import annotations

import numpy as np
import torch

from src.data.beat_grid import from_timing_points
from src.data.tokenizer import (
    BAR,
    BEATS_PER_CHUNK,
    EMPTY,
    HOLD_BODY,
    HOLD_END,
    HOLD_START,
    MASK,
    PAD,
    D,
    K,
    L,
)
from src.models.diffusion import N_CLASSES

_OPENS = np.array([False, False, True, True, False])     # START, BODY leave a hold open
_NEEDS = np.array([False, False, False, True, True])     # BODY, END continue one


def allowed(left: int | None, right: int | None) -> np.ndarray:
    """Classes 0..4 a cell may take between these neighbours (None = not open yet)."""
    ok = np.ones(N_CLASSES, dtype=bool)
    if left is not None:                 # a hold open on the left must go on; none may not
        ok &= _NEEDS if left in (HOLD_START, HOLD_BODY) else ~_NEEDS
    if right is not None:                # a cell the right continues must leave a hold open
        ok &= _OPENS if right in (HOLD_BODY, HOLD_END) else ~_OPENS
    return ok


def _neighbour(v: int) -> int | None:
    return None if v == MASK else (EMPTY if v == PAD else int(v))


@torch.no_grad()
def denoiser_probs(model, x: np.ndarray, mel: torch.Tensor, s: float, b: float) -> np.ndarray:
    """p(x0 | x) for one window: [L, K, 5] float64."""
    device = mel.device
    was_training = model.training
    model.eval()
    logits = model(torch.as_tensor(x, dtype=torch.long, device=device)[None], mel[None],
                   torch.tensor([s], dtype=torch.float32, device=device),
                   torch.tensor([b], dtype=torch.float32, device=device))
    model.train(was_training)
    return torch.softmax(logits.float(), dim=-1)[0].cpu().numpy().astype(np.float64)


def sample_window(model, x: np.ndarray, mel: torch.Tensor, s: float, b: float, *,
                  steps: int = 32, order: str = "random", rng: np.random.Generator | None = None,
                  left_closed: bool = True, right_closed: bool = True) -> np.ndarray:
    """Fill the MASK cells of one window. Cells that are not MASK are kept as they are.

    x      [L, K] tokens: MASK where to generate, PAD past the end of the song
    mel    [L * 4, n_mels] tensor on the model's device
    order  "random": each MASK cell opens with probability 1/t at step t (§4.8)
           "confidence": the round(n / t) most confident MASK cells open (MaskGIT, 큐 9)
    left_closed / right_closed: whether the window edge is a song edge, where a
           hold cannot come in or stay open.
    """
    if order not in ("random", "confidence"):
        raise ValueError(f"unknown order {order!r}")
    x = np.array(x, dtype=np.int64)
    rng = rng or np.random.default_rng()
    for t in range(steps, 0, -1):
        todo = np.argwhere(x == MASK)
        if len(todo) == 0:
            break
        probs = denoiser_probs(model, x, mel, s, b)
        if t == 1:
            pick = todo
        elif order == "random":
            pick = todo[rng.random(len(todo)) < 1.0 / t]
        else:
            confidence = probs[todo[:, 0], todo[:, 1]].max(axis=-1)
            pick = todo[np.argsort(-confidence, kind="stable")[:max(1, round(len(todo) / t))]]

        for c, k in pick[np.lexsort((pick[:, 0], pick[:, 1]))]:     # lane by lane, left to right
            left = _neighbour(x[c - 1, k]) if c > 0 else (EMPTY if left_closed else None)
            right = _neighbour(x[c + 1, k]) if c + 1 < L else (EMPTY if right_closed else None)
            ok = allowed(left, right)
            p = probs[c, k] * ok
            p = p / p.sum() if p.sum() > 0 else ok / ok.sum()
            x[c, k] = rng.choice(N_CLASSES, p=p)
    return x


def generate_song(model, mel: np.ndarray, s: float, timing_points, cell_offset: int,
                  n_cells: int, *, steps: int = 32, order: str = "random",
                  mode: str = "continue", prefix_cells: int = 2 * BAR,
                  seed: int = 0) -> np.ndarray:
    """Chart tokens for a whole song: [n_chunks, L, K] int8, rows >= n_cells are PAD.

    mel           [n_frames, n_mels] frames of the whole song on the token grid
                  (frame 4r + j belongs to token row r), as in data/cache/mel
    s             target SR, the same for every window
    timing_points, cell_offset
                  the song's timing and token origin; b is computed per window from them
    n_cells       non-PAD rows (how far the song goes)
    Decode the result with tokenizer.make_metas(timing_points, cell_offset, n_chunks, s).
    """
    if mode not in ("continue", "independent"):
        raise ValueError(f"unknown mode {mode!r}")
    device = next(model.parameters()).device
    r = model.config.frames_per_cell
    n_chunks = -(-n_cells // L)
    song = np.full(((n_chunks + 1) * L, K), PAD, dtype=np.int64)   # +1 chunk for overhang
    song[:n_cells] = MASK

    frames = np.asarray(mel, dtype=np.float32)
    need = song.shape[0] * r
    if len(frames) < need:                                          # past the audio: silence
        fill = frames.min() if frames.size else 0.0
        frames = np.concatenate([frames, np.full((need - len(frames), frames.shape[1]), fill,
                                                 dtype=np.float32)])
    frames = torch.as_tensor(frames[:need], device=device)

    grid = from_timing_points(timing_points)
    rng = np.random.default_rng(seed)

    def beat_len(row0: int) -> float:
        start = row0 - cell_offset
        return (grid.time_from_cell(start + L, D) - grid.time_from_cell(start, D)) / BEATS_PER_CHUNK

    def fill(row0: int, left_closed: bool, right_closed: bool) -> None:
        song[row0:row0 + L] = sample_window(
            model, song[row0:row0 + L], frames[row0 * r:(row0 + L) * r], s, beat_len(row0),
            steps=steps, order=order, rng=rng, left_closed=left_closed, right_closed=right_closed)

    if mode == "independent":
        for c in range(n_chunks):
            fill(c * L, left_closed=True, right_closed=True)
    else:
        stride = L - prefix_cells
        row0 = 0
        while True:
            last = row0 + L >= n_cells
            fill(row0, left_closed=True, right_closed=last)   # left edge: song start or fixed prefix
            if last:
                break
            row0 += stride
    return song[:n_chunks * L].reshape(n_chunks, L, K).astype(np.int8)
