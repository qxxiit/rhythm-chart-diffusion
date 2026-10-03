"""Reverse process (design doc §4.8): fill MASK cells with the denoiser under the hold grammar.

    sample_window   one 384-cell window, T steps; step t opens ~1/t of the MASK cells left
                    (order "block": beat by beat, left to right; steps 0: one cell per
                    forward pass, the ceiling of parallel sampling)
    generate_song   a whole song, window by window
        mode="continue"     every window after the first keeps the last 2 bars of the
                            previous one fixed and fills 6 new bars (§4.8 청크 이어 생성)
        mode="independent"  every chunk on its own, holds closed at chunk edges (실험 큐 3c)
    forward_lanes   afterwards (lanes="forward"): choose the lanes of every tap row once
                    more, left to right, with the lanes before it as just chosen and the
                    rhythm after it known, but not its lanes. One forward pass per row.
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


STATS = {"passes": 0}           # forward passes so far (evaluate.py reports them per song)


@torch.no_grad()
def denoiser_probs(model, x: np.ndarray, mel: torch.Tensor, s: float, b: float) -> np.ndarray:
    """p(x0 | x) for one window: [L, K, 5] float64."""
    STATS["passes"] += 1
    device = mel.device
    was_training = model.training
    model.eval()
    logits = model(torch.as_tensor(x, dtype=torch.long, device=device)[None], mel[None],
                   torch.tensor([s], dtype=torch.float32, device=device),
                   torch.tensor([b], dtype=torch.float32, device=device))
    model.train(was_training)
    return torch.softmax(logits.float(), dim=-1)[0].cpu().numpy().astype(np.float64)


ORDERS = ("random", "confidence", "noisy", "block")


def steps_name(steps: int) -> str:
    """How file and directory names show a step count: 128, or seq for steps 0."""
    return "seq" if steps <= 0 else str(steps)


def _class_weight(hold_bias: float, empty_bias: float) -> np.ndarray | None:
    """Multipliers on p(class) for hold_bias / empty_bias (None when both are 0)."""
    if hold_bias == 0 and empty_bias == 0:
        return None
    hold = float(np.exp(min(hold_bias, 20.0)))
    return np.array([float(np.exp(min(empty_bias, 20.0))), 1.0, hold, hold, hold])


def _spread(cells: np.ndarray, count: int, group: np.ndarray,
            rng: np.random.Generator) -> np.ndarray:
    """count of the cells, at most one per group as long as groups last: random
    round robin (first a random cell of each group, groups in random order, then
    second cells, ...)."""
    if count <= 0:
        return cells[:0]
    perm = rng.permutation(len(cells))
    g = np.asarray(group)[perm]
    by_group = np.argsort(g, kind="stable")             # groups together, each in perm order
    sorted_g = g[by_group]
    starts = np.concatenate([[0], np.flatnonzero(np.diff(sorted_g)) + 1])
    sizes = np.diff(np.concatenate([starts, [len(g)]]))
    rank = np.empty(len(g), dtype=np.int64)             # rank of each (permuted) cell in its group
    rank[by_group] = np.arange(len(g)) - np.repeat(starts, sizes)
    by_rank = perm[np.argsort(rank, kind="stable")]
    return cells[np.sort(by_rank[:count])]


def _open_cells(x: np.ndarray, pick: np.ndarray, probs: np.ndarray, rng: np.random.Generator,
                left_closed: bool, right_closed: bool, weight: np.ndarray | None) -> None:
    """Draw the picked cells from probs under the grammar; cells of one step go lane by
    lane, left to right, so each sees the neighbours opened before it."""
    for c, k in pick[np.lexsort((pick[:, 0], pick[:, 1]))]:
        left = _neighbour(x[c - 1, k]) if c > 0 else (EMPTY if left_closed else None)
        right = _neighbour(x[c + 1, k]) if c + 1 < L else (EMPTY if right_closed else None)
        ok = allowed(left, right)
        p = probs[c, k] * ok
        if weight is not None:
            p = p * weight
        p = p / p.sum() if p.sum() > 0 else ok / ok.sum()
        x[c, k] = rng.choice(N_CLASSES, p=p)


def sample_window(model, x: np.ndarray, mel: torch.Tensor, s: float, b: float, *,
                  steps: int = 32, order: str = "random", rng: np.random.Generator | None = None,
                  left_closed: bool = True, right_closed: bool = True,
                  temperature: float = 1.0, hold_bias: float = 0.0, empty_bias: float = 0.0,
                  spread: bool = False) -> np.ndarray:
    """Fill the MASK cells of one window. Cells that are not MASK are kept as they are.

    x      [L, K] tokens: MASK where to generate, PAD past the end of the song
    mel    [L * 4, n_mels] tensor on the model's device
    steps  forward passes for the window. 0: one cell per pass ("sequential"), so no two
           cells are ever drawn from the same prediction: the ceiling of parallel
           sampling for the order, at one pass per MASK cell (1,152-1,536 per window)
    order  "random": each MASK cell opens with probability 1/t at step t (§4.8)
           "confidence": the round(n / t) most confident MASK cells open (MaskGIT, 큐 9)
           "noisy": the same count, ranked by log confidence + temperature * (t / steps) *
                    Gumbel noise (MaskGIT's choice temperature): close to random early,
                    close to confidence late. temperature 0 = "confidence"
           "block": beats (D = 12 rows) left to right; each beat's MASK cells open in
                    random order over its share of the steps (the steps go to the beats
                    in proportion to their MASK cells, at least one each), with every
                    beat before it complete and every beat after it still MASK
                    (semi-autoregressive, as block diffusion / LLaDA's semi-AR sampling)
    spread: cells that open in the same step come from different beats ("random") or
           different rows ("block") as long as there are enough of them; how many open
           stays as without it, only which go together changes (same forward passes)
    left_closed / right_closed: whether the window edge is a song edge, where a
           hold cannot come in or stay open.
    hold_bias: added to the log probability of START, BODY and END wherever a cell has
           a choice (where the grammar forces a hold to go on, all of its options are
           hold classes and the bias cancels). -inf: no holds.
    empty_bias: added to the log probability of EMPTY the same way: > 0 fewer notes,
           < 0 more (to compare samplers at the same density). Biases change what a cell
           becomes, never which cells open (the confidence score is unbiased).
    """
    if order not in ORDERS:
        raise ValueError(f"unknown order {order!r}")
    x = np.array(x, dtype=np.int64)
    rng = rng or np.random.default_rng()
    weight = _class_weight(hold_bias, empty_bias)
    if order == "block":
        _sample_blocks(model, x, mel, s, b, steps, rng, left_closed, right_closed, weight, spread)
        return x
    sequential = steps <= 0
    if sequential:
        steps = int((x == MASK).sum())
    for t in range(steps, 0, -1):
        todo = np.argwhere(x == MASK)
        if len(todo) == 0:
            break
        probs = denoiser_probs(model, x, mel, s, b)
        if t == 1:
            pick = todo
        elif order == "random":
            if sequential:
                pick = todo[[int(rng.integers(len(todo)))]]
            else:
                chosen = rng.random(len(todo)) < 1.0 / t
                pick = (_spread(todo, int(chosen.sum()), todo[:, 0] // D, rng) if spread
                        else todo[chosen])
        else:
            score = np.log(probs[todo[:, 0], todo[:, 1]].max(axis=-1) + 1e-12)
            if order == "noisy" and temperature > 0:
                score = score + temperature * (t / steps) * rng.gumbel(size=len(todo))
            pick = todo[np.argsort(-score, kind="stable")[:max(1, round(len(todo) / t))]]
        _open_cells(x, pick, probs, rng, left_closed, right_closed, weight)
    return x


def _block_steps(counts: np.ndarray, steps: int) -> np.ndarray:
    """steps over blocks in proportion to their MASK cells, at least one each;
    steps 0: one per MASK cell."""
    if steps <= 0:
        return counts.copy()
    cum = np.concatenate([[0], np.cumsum(counts)])
    edges = np.round(steps * cum / cum[-1]).astype(np.int64)
    return np.maximum(np.diff(edges), 1)


def _sample_blocks(model, x: np.ndarray, mel: torch.Tensor, s: float, b: float, steps: int,
                   rng: np.random.Generator, left_closed: bool, right_closed: bool,
                   weight: np.ndarray | None, spread: bool) -> None:
    """order="block" for sample_window: x is filled in place."""
    open_rows = np.flatnonzero((x == MASK).any(axis=1))
    if len(open_rows) == 0:
        return
    blocks = np.unique(open_rows // D)
    counts = np.array([int((x[blk * D:(blk + 1) * D] == MASK).sum()) for blk in blocks])
    for blk, n_steps in zip(blocks, _block_steps(counts, steps), strict=True):
        lo = blk * D
        for t in range(int(n_steps), 0, -1):
            todo = np.argwhere(x[lo:lo + D] == MASK)
            if len(todo) == 0:
                break
            todo[:, 0] += lo
            probs = denoiser_probs(model, x, mel, s, b)
            if t == 1:
                pick = todo
            elif steps <= 0:
                pick = todo[[int(rng.integers(len(todo)))]]
            else:
                chosen = rng.random(len(todo)) < 1.0 / t
                pick = (_spread(todo, int(chosen.sum()), todo[:, 0], rng) if spread
                        else todo[chosen])
            _open_cells(x, pick, probs, rng, left_closed, right_closed, weight)


def _lane_sets(free: list[int], n: int, p_tap: np.ndarray, p_empty: np.ndarray,
               temperature: float, rng: np.random.Generator,
               bonus: np.ndarray | None = None) -> tuple[int, ...]:
    """Pick n of the free lanes for the taps of one row: the model's cells are
    independent given the context, so a lane set scores sum log p(TAP) over its lanes
    + sum log p(EMPTY) over the other free lanes (+ bonus[k] for each chosen lane k);
    sampled at the temperature (0 = the best set)."""
    sets = list(combinations(free, n))
    score = np.array([sum(np.log(p_tap[k] + 1e-12) for k in c)
                      + sum(np.log(p_empty[k] + 1e-12) for k in free if k not in c)
                      + (sum(bonus[k] for k in c) if bonus is not None else 0.0)
                      for c in sets])
    if temperature <= 0:
        return sets[int(np.argmax(score))]
    w = np.exp((score - score.max()) / temperature)
    return sets[int(rng.choice(len(sets), p=w / w.sum()))]


# Long-note rules by target SR, from the human charts (docs/_stats/hold_stats.csv, train):
# min_hold is about the 10th percentile of long-note length (Easy 6, Normal 5, Hard and
# up 3 cells), release_gap keeps the next press in a lane at least 6 cells (Easy) or 3
# cells after a release, as for 99% (Easy) and 97% (all grades) of human long notes.
HOLD_RULES = ((2.0, 6, 5), (2.7, 5, 2), (float("inf"), 3, 2))   # (SR below, min_hold, gap)


def hold_rules(s: float) -> tuple[int, int]:
    """(min_hold, release_gap) for a target SR."""
    return next((m, g) for upper, m, g in HOLD_RULES if s < upper)


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


def _free_lanes(song: np.ndarray, row: int, release_gap: int) -> tuple[list[int], int]:
    """(lanes whose cell may hold a tap of this row, taps among them). A row has a lane
    choice when 0 < taps < len(lanes): its taps can move, holds and the rest never do."""
    cells = song[row]
    free = [k for k in range(K) if cells[k] in (EMPTY, TAP)
            and not _after_release(song, row, k, release_gap)]
    return free, int(sum(cells[k] == TAP for k in free))


def lane_choice_rows(song: np.ndarray, n_rows: int, release_gap: int = 0) -> list[tuple]:
    """(row, free lanes, taps) of every row before n_rows that has a lane choice."""
    out = []
    for row in range(min(n_rows, len(song))):
        free, n = _free_lanes(song, row, release_gap)
        if 0 < n < len(free):
            out.append((row, free, n))
    return out


def jack_bonus(song: np.ndarray, row: int, jack_bias: float) -> np.ndarray | None:
    """Log-score bonus per lane for repeating the lanes of the previous onset row, if
    it is at most a beat (D cells) earlier; None when there is nothing to add."""
    if jack_bias == 0:
        return None
    onset = np.isin(song[max(0, row - D):row], (TAP, HOLD_START)).any(axis=1)
    hits = np.flatnonzero(onset)
    if len(hits) == 0:
        return None
    prev = song[max(0, row - D) + hits[-1]]
    return np.where(np.isin(prev, (TAP, HOLD_START)), jack_bias, 0.0)


def forward_lanes(model, song: np.ndarray, frames: torch.Tensor, s: float, beat_len,
                  n_cells: int, *, temperature: float = 0.5, rng: np.random.Generator | None = None,
                  release_gap: int = 0, past: int = L - L // 4,
                  jack_bias: float = 0.0) -> np.ndarray:
    """Choose the lanes of every row with a lane choice once more, left to right.

    song      [n_rows, K] tokens of the whole song (no MASK), changed in place and returned
    frames    [n_rows * r, n_mels] tensor on the model's device; beat_len(row0) -> b
    For each such row, in time order: a window that starts `past` rows earlier (as far
    as the song allows) in which the row's free cells are MASK, the rows before it are
    as this pass left them, and every later row with a lane choice has its free cells
    MASK too. So the model sees the lanes chosen so far and where the later notes are
    not (their empty rows, holds and fixed chords), but not the lanes sampling gave
    the later taps, which were drawn with little context. The row's n lanes are then
    chosen as in refine_lanes (_lane_sets at the temperature). Rhythm, chord sizes and
    holds stay; with release_gap, taps keep out of the cells just after a release.
    One forward pass per row; pattern_probe.py's "past+rhythm" context measures how
    well the model chooses lanes this way on human charts.
    """
    rng = rng or np.random.default_rng()
    fpc = model.config.frames_per_cell
    todo = lane_choice_rows(song, n_cells, release_gap)
    waiting = np.zeros(song.shape, dtype=bool)          # free cells of rows not chosen yet
    for row, free, _ in todo:
        waiting[row, free] = True
    for row, free, n in todo:
        w0 = int(np.clip(row - past, 0, len(song) - L))
        x = song[w0:w0 + L].astype(np.int64).copy()
        x[waiting[w0:w0 + L]] = MASK
        probs = denoiser_probs(model, x, frames[w0 * fpc:(w0 + L) * fpc], s, beat_len(w0))
        p = probs[row - w0]
        chosen = _lane_sets(free, n, p[:, TAP], p[:, EMPTY], temperature, rng,
                            jack_bonus(song, row, jack_bias))
        song[row, free] = EMPTY
        song[row, list(chosen)] = TAP
        waiting[row] = False
    return song


def refine_lanes(model, song: np.ndarray, frames: torch.Tensor, s: float, beat_len,
                 n_cells: int, *, sweeps: int = 2, temperature: float = 0.5, stride: int = 8,
                 rng: np.random.Generator | None = None, release_gap: int = 0,
                 jack_bias: float = 0.0) -> np.ndarray:
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
                    free, n = _free_lanes(song, row, release_gap)
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
                    chosen = _lane_sets(free, n, p[:, TAP], p[:, EMPTY], temperature, rng,
                                        jack_bonus(song, row, jack_bias))
                    song[row, free] = EMPTY
                    song[row, list(chosen)] = TAP
            if last:
                break
            w0 += half
    return song


LANE_PASSES = ("sampled", "forward")


def generate_song(model, mel: np.ndarray, s: float, timing_points, cell_offset: int,
                  n_cells: int, *, steps: int = 32, order: str = "random",
                  mode: str = "continue", prefix_cells: int = 2 * BAR,
                  seed: int = 0, temperature: float = 1.0, refine: int = 0,
                  lane_temperature: float = 0.5, hold_bias: float = 0.0,
                  min_hold: int | None = None, release_gap: int | None = None,
                  lanes: str = "sampled", spread: bool = False,
                  empty_bias: float = 0.0, forward_temperature: float | None = None,
                  jack_bias: float = 0.0) -> np.ndarray:
    """Chart tokens for a whole song: [n_chunks, L, K] int8, rows >= n_cells are PAD.

    mel           [n_frames, n_mels] frames of the whole song on the token grid
                  (frame 4r + j belongs to token row r): mel.MelStore.chart(...)
    s             target SR, the same for every window
    timing_points, cell_offset
                  the song's timing and token origin; b is computed per window from them
    n_cells       non-PAD rows (how far the song goes)
    steps, order, temperature, spread, hold_bias, empty_bias
                  per window, see sample_window (steps 0 = one cell per forward pass)
    lanes         "forward": after sampling, forward_lanes re-chooses every row's lanes
                  left to right with the rhythm known, at forward_temperature (None:
                  lane_temperature)
    refine        sweeps of refine_lanes after that (0 = none), at lane_temperature
    jack_bias     both lane passes: log-score bonus for repeating the lanes of the onset
                  row just before (at most a beat): the model picks jacks at a fifth of
                  the human rate (EXPERIMENTS 2026-10-03)
    min_hold, release_gap
                  clean_holds after sampling; None = by the target SR (hold_rules),
                  0, 0 = keep the holds as sampled
    Order of work: sample, clean_holds, forward_lanes, refine_lanes.
    Decode the result with tokenizer.make_metas(timing_points, cell_offset, n_chunks, s).
    """
    if mode not in ("continue", "independent"):
        raise ValueError(f"unknown mode {mode!r}")
    if lanes not in LANE_PASSES:
        raise ValueError(f"unknown lanes {lanes!r}")
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
            temperature=temperature, hold_bias=hold_bias, empty_bias=empty_bias, spread=spread)

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
    auto_hold, auto_gap = hold_rules(s)
    min_hold = auto_hold if min_hold is None else min_hold
    release_gap = auto_gap if release_gap is None else release_gap
    if min_hold > 0 or release_gap > 0:
        clean_holds(song, min_hold=min_hold, release_gap=release_gap)
    if lanes == "forward":
        forward_lanes(model, song, frames, s, beat_len, n_cells, rng=rng, release_gap=release_gap,
                      jack_bias=jack_bias,
                      temperature=lane_temperature if forward_temperature is None
                      else forward_temperature)
    if refine:
        refine_lanes(model, song, frames, s, beat_len, n_cells, sweeps=refine,
                     temperature=lane_temperature, rng=rng, release_gap=release_gap,
                     jack_bias=jack_bias)
    return song[:n_chunks * L].reshape(n_chunks, L, K).astype(np.int8)
