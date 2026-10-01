"""Reverse process (design doc §4.8): fill MASK cells with the denoiser under the hold grammar.

    sample_window   one 384-cell window, T steps; step t opens ~1/t of the MASK cells left
    generate_song   a whole song, window by window
        mode="continue"     every window after the first keeps the last 2 bars of the
                            previous one fixed and fills 6 new bars (§4.8 청크 이어 생성)
        mode="independent"  every chunk on its own, holds closed at chunk edges (실험 큐 3c)
    refine_lanes    afterwards: re-choose the lanes of tap rows with the whole rest of
                    the chart as context, rhythm and chord sizes kept (generate_song's
                    refine= sweeps). Sampling commits lanes early, when little is open
                    yet; this lets every lane choice see what came after it as well.
    clean_holds     afterwards: a release followed too soon by an onset in its lane
                    moves back, and holds shorter than min_hold cells become taps.
                    Cells are sampled one by one, so a hold can be closed by the grammar
                    rather than by the model: an onset sampled as HOLD_START must be
                    followed by BODY or END even where the model expects nothing, which
                    makes holds of 1-3 cells, and an early END splits a long hold into
                    END-then-START pairs. hold_bias (log scale) on the hold classes sets
                    how readily holds start at all (-inf: none).

Grammar per lane: x[i+1] in {BODY, END}  <=>  x[i] in {START, BODY}.
A cell being opened only looks at neighbours that are already open, so every
adjacent pair is checked by whichever of the two opens later; cells opened in
the same step go left to right. PAD, and the outside of a closed window edge,
count as EMPTY. A run of MASK cells between two open cells can always be
filled, so sampling never gets stuck.
"""

from __future__ import annotations

from itertools import combinations

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
    TAP,
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


ORDERS = ("random", "confidence", "noisy")


_HOLD = np.array([False, False, True, True, True])     # START, BODY, END


def sample_window(model, x: np.ndarray, mel: torch.Tensor, s: float, b: float, *,
                  steps: int = 32, order: str = "random", rng: np.random.Generator | None = None,
                  left_closed: bool = True, right_closed: bool = True,
                  temperature: float = 1.0, hold_bias: float = 0.0) -> np.ndarray:
    """Fill the MASK cells of one window. Cells that are not MASK are kept as they are.

    x      [L, K] tokens: MASK where to generate, PAD past the end of the song
    mel    [L * 4, n_mels] tensor on the model's device
    order  "random": each MASK cell opens with probability 1/t at step t (§4.8)
           "confidence": the round(n / t) most confident MASK cells open (MaskGIT, 큐 9)
           "noisy": the same count, ranked by log confidence + temperature * (t / steps) *
                    Gumbel noise (MaskGIT's choice temperature): close to random early,
                    close to confidence late. temperature 0 = "confidence"
    left_closed / right_closed: whether the window edge is a song edge, where a
           hold cannot come in or stay open.
    hold_bias: added to the log probability of START, BODY and END wherever a cell has
           a choice (where the grammar forces a hold to go on, all of its options are
           hold classes and the bias cancels). -inf: no holds.
    """
    hold_scale = float(np.exp(min(hold_bias, 20.0)))
    if order not in ORDERS:
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
            score = np.log(probs[todo[:, 0], todo[:, 1]].max(axis=-1) + 1e-12)
            if order == "noisy" and temperature > 0:
                score = score + temperature * (t / steps) * rng.gumbel(size=len(todo))
            pick = todo[np.argsort(-score, kind="stable")[:max(1, round(len(todo) / t))]]

        for c, k in pick[np.lexsort((pick[:, 0], pick[:, 1]))]:     # lane by lane, left to right
            left = _neighbour(x[c - 1, k]) if c > 0 else (EMPTY if left_closed else None)
            right = _neighbour(x[c + 1, k]) if c + 1 < L else (EMPTY if right_closed else None)
            ok = allowed(left, right)
            p = probs[c, k] * ok
            if hold_scale != 1.0:
                p = np.where(_HOLD, p * hold_scale, p)
            p = p / p.sum() if p.sum() > 0 else ok / ok.sum()
            x[c, k] = rng.choice(N_CLASSES, p=p)
    return x


def _lane_sets(free: list[int], n: int, p_tap: np.ndarray, p_empty: np.ndarray,
               temperature: float, rng: np.random.Generator) -> tuple[int, ...]:
    """Pick n of the free lanes for the taps of one row: the model's cells are
    independent given the context, so a lane set scores sum log p(TAP) over its lanes
    + sum log p(EMPTY) over the other free lanes; sampled at the temperature
    (0 = the best set)."""
    sets = list(combinations(free, n))
    score = np.array([sum(np.log(p_tap[k] + 1e-12) for k in c)
                      + sum(np.log(p_empty[k] + 1e-12) for k in free if k not in c)
                      for c in sets])
    if temperature <= 0:
        return sets[int(np.argmax(score))]
    w = np.exp((score - score.max()) / temperature)
    return sets[int(rng.choice(len(sets), p=w / w.sum()))]


def clean_holds(song: np.ndarray, *, min_hold: int = 3, release_gap: int = 2) -> np.ndarray:
    """Long notes as players expect them; [n_rows, K] tokens, changed in place.

    1. A release followed by an onset in its lane with fewer than release_gap empty
       cells between moves back until there are release_gap (the onset wins, as in
       the tokenizer's collision rule).
    2. A hold shorter than min_hold cells (1/12 beat each) becomes a tap at its start.
    Onsets never move, so the rhythm, F1 and the lane patterns stay as they were;
    only releases change. The grammar holds throughout.
    """
    onset_classes = (TAP, HOLD_START)
    for k in range(song.shape[1]):
        col = song[:, k]
        for start in np.flatnonzero(col == HOLD_START):
            end = start + 1
            while end < len(col) and col[end] == HOLD_BODY:
                end += 1
            if end >= len(col) or col[end] != HOLD_END:
                continue                                  # open at the edge: leave it
            if release_gap > 0:
                ahead = np.flatnonzero(np.isin(col[end + 1:end + 1 + release_gap], onset_classes))
                if len(ahead):
                    new_end = end + 1 + int(ahead[0]) - release_gap - 1
                    col[max(new_end, start) + 1:end + 1] = EMPTY
                    if new_end > start:
                        col[new_end] = HOLD_END
                    end = max(new_end, start)
            if end - start < max(min_hold, 1):
                col[start] = TAP
                col[start + 1:end + 1] = EMPTY
    return song


def _after_release(song: np.ndarray, row: int, lane: int, release_gap: int) -> bool:
    """True if an onset at (row, lane) would come too soon after a release in its lane."""
    if release_gap <= 0:
        return False
    return bool(np.any(song[max(0, row - release_gap):row, lane] == HOLD_END))


def refine_lanes(model, song: np.ndarray, frames: torch.Tensor, s: float, beat_len,
                 n_cells: int, *, sweeps: int = 2, temperature: float = 0.5, stride: int = 8,
                 rng: np.random.Generator | None = None, release_gap: int = 0) -> np.ndarray:
    """Re-choose which lanes the taps of each row use; rhythm, chord sizes and holds stay.

    song      [n_rows, K] tokens of the whole song (no MASK), changed in place and returned
    frames    [n_rows * r, n_mels] tensor on the model's device; beat_len(row0) -> b
    Windows of L rows at a stride of L / 2; each window re-chooses the rows of its
    middle half (the first and last window also their outer quarter), so every row
    is decided with context on both sides. Within a window, one pass per residue
    class of the row index mod stride: those rows' free cells (TAP or EMPTY) become
    MASK and the rest of the chart stays visible; rows that far apart barely
    constrain each other. Changing a TAP into EMPTY or back never breaks the hold
    grammar, so holds are left exactly as they are; with release_gap, a tap is not
    moved into the cells just after a release in its lane (clean_holds' rule).
    """
    rng = rng or np.random.default_rng()
    fpc = model.config.frames_per_cell
    half, quarter = L // 2, L // 4
    for _ in range(sweeps):
        w0 = 0
        while True:
            last = w0 + L >= n_cells
            lo = 0 if w0 == 0 else w0 + quarter
            hi = min(w0 + L, n_cells) if last else w0 + 3 * quarter
            for j in rng.permutation(stride):
                todo = []
                for row in range(lo + (j - lo) % stride, hi, stride):
                    cells = song[row]
                    free = [k for k in range(K) if cells[k] in (EMPTY, TAP)
                            and not _after_release(song, row, k, release_gap)]
                    n = int(sum(cells[k] == TAP for k in free))
                    if 0 < n < len(free):
                        todo.append((row, free, n))
                if not todo:
                    continue
                x = song[w0:w0 + L].astype(np.int64).copy()
                for row, free, _ in todo:
                    x[row - w0, free] = MASK
                probs = denoiser_probs(model, x, frames[w0 * fpc:(w0 + L) * fpc], s, beat_len(w0))
                for row, free, n in todo:
                    p = probs[row - w0]
                    chosen = _lane_sets(free, n, p[:, TAP], p[:, EMPTY], temperature, rng)
                    song[row, free] = EMPTY
                    song[row, list(chosen)] = TAP
            if last:
                break
            w0 += half
    return song


def generate_song(model, mel: np.ndarray, s: float, timing_points, cell_offset: int,
                  n_cells: int, *, steps: int = 32, order: str = "random",
                  mode: str = "continue", prefix_cells: int = 2 * BAR,
                  seed: int = 0, temperature: float = 1.0, refine: int = 0,
                  lane_temperature: float = 0.5, hold_bias: float = 0.0, min_hold: int = 3,
                  release_gap: int = 2) -> np.ndarray:
    """Chart tokens for a whole song: [n_chunks, L, K] int8, rows >= n_cells are PAD.

    mel           [n_frames, n_mels] frames of the whole song on the token grid
                  (frame 4r + j belongs to token row r): mel.MelStore.chart(...)
    s             target SR, the same for every window
    timing_points, cell_offset
                  the song's timing and token origin; b is computed per window from them
    n_cells       non-PAD rows (how far the song goes)
    refine        sweeps of refine_lanes after sampling (0 = none), at lane_temperature
    hold_bias     log-scale bias on starting holds (sample_window); -inf = no holds
    min_hold, release_gap
                  clean_holds after sampling (0, 0 = keep the holds as sampled)
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
            steps=steps, order=order, rng=rng, left_closed=left_closed, right_closed=right_closed,
            temperature=temperature, hold_bias=hold_bias)

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
    if min_hold > 0 or release_gap > 0:
        clean_holds(song, min_hold=min_hold, release_gap=release_gap)
    if refine:
        refine_lanes(model, song, frames, s, beat_len, n_cells, sweeps=refine,
                     temperature=lane_temperature, rng=rng, release_gap=release_gap)
    return song[:n_chunks * L].reshape(n_chunks, L, K).astype(np.int8)
