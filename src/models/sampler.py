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
    copy_bars       afterwards (copy_bias): offer every bar a copy of an earlier bar
                    whose audio is similar, as it is or mirrored, and take it when the
                    model scores it within copy_bias of the bar it replaces. Human charts
                    copy about a sixth of their bars, most from far back (a section that
                    comes back); sampled ones almost none.
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
from src.evaluation.structure import ssm_audio
from src.models.diffusion import N_CLASSES, Styled

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


STATS = {"passes": 0, "copied_bars": 0, "changed_holds": 0, "rest_bars": 0,
         "filled_holes": 0, "carried_bars": 0,
         "tidied_holds": 0}             # passes, copies, hold changes, bars left empty,
                                         # stream holes filled, rhythms carried over, long
                                         # notes turned into taps
                                         # (evaluate.py reports them per song)


@torch.no_grad()
def denoiser_probs(model, x: np.ndarray, mel: torch.Tensor, s: float, b: float) -> np.ndarray:
    """p(x0 | x) for one window: [L, K, 5] float64."""
    STATS["passes"] += 2 if getattr(model, "guidance", 0) else 1     # guidance: two passes
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
                left_closed: bool, right_closed: bool, weight: np.ndarray | None,
                row_bias: np.ndarray | None = None,
                row_onset: np.ndarray | None = None) -> None:
    """Draw the picked cells from probs under the grammar; cells of one step go lane by
    lane, left to right, so each sees the neighbours opened before it. row_bias [L]: an
    extra log bias on EMPTY per row (sample_window's row_empty_bias); row_onset [L]: a log
    penalty on the onset classes (TAP, HOLD_START) per row (row_onset_bias), which leaves
    the body and release of a long note as they were."""
    for c, k in pick[np.lexsort((pick[:, 0], pick[:, 1]))]:
        left = _neighbour(x[c - 1, k]) if c > 0 else (EMPTY if left_closed else None)
        right = _neighbour(x[c + 1, k]) if c + 1 < L else (EMPTY if right_closed else None)
        ok = allowed(left, right)
        p = probs[c, k] * ok
        if weight is not None:
            p = p * weight
        if row_bias is not None and row_bias[c] != 0:
            p = p.copy()
            p[EMPTY] *= float(np.exp(min(row_bias[c], 20.0)))
        if row_onset is not None and row_onset[c] != 0:
            p = p.copy()
            p[[TAP, HOLD_START]] *= float(np.exp(-min(row_onset[c], 20.0)))
        p = p / p.sum() if p.sum() > 0 else ok / ok.sum()
        x[c, k] = rng.choice(N_CLASSES, p=p)


def sample_window(model, x: np.ndarray, mel: torch.Tensor, s: float, b: float, *,
                  steps: int = 32, order: str = "random", rng: np.random.Generator | None = None,
                  left_closed: bool = True, right_closed: bool = True,
                  temperature: float = 1.0, hold_bias: float = 0.0, empty_bias: float = 0.0,
                  spread: bool = False, row_empty_bias: np.ndarray | None = None,
                  row_onset_bias: np.ndarray | None = None) -> np.ndarray:
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
    row_empty_bias: [L] more of the same per row, on top of empty_bias (generate_song's
           loud_bias: fewer notes where the music is quiet)
    row_onset_bias: [L] subtracted from the log probability of TAP and HOLD_START per row
           (generate_song's onset_bias): fewer notes start there, while long notes that
           pass through go on (an EMPTY bias would also cut their bodies)
    """
    if order not in ORDERS:
        raise ValueError(f"unknown order {order!r}")
    x = np.array(x, dtype=np.int64)
    rng = rng or np.random.default_rng()
    weight = _class_weight(hold_bias, empty_bias)
    if order == "block":
        _sample_blocks(model, x, mel, s, b, steps, rng, left_closed, right_closed, weight, spread,
                       row_empty_bias, row_onset_bias)
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
        _open_cells(x, pick, probs, rng, left_closed, right_closed, weight, row_empty_bias,
                    row_onset_bias)
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
                   weight: np.ndarray | None, spread: bool,
                   row_bias: np.ndarray | None = None,
                   row_onset: np.ndarray | None = None) -> None:
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
            _open_cells(x, pick, probs, rng, left_closed, right_closed, weight, row_bias,
                        row_onset)


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


HOLD_REACH = L // 4             # refine_holds looks at most 8 beats past an onset


def _hold_groups(song: np.ndarray, n_cells: int, min_hold: int, release_gap: int):
    """refine_holds' order: windows as in refine_lanes (each deciding the onsets in its
    middle half), one lane at a time, every other onset of it. Yields (w0, lane, todo),
    todo = [(onset row, limit, candidate release rows)], built from the song as it is
    when the group comes up."""
    half, quarter = L // 2, L // 4
    w0 = 0
    while True:
        last = w0 + L >= n_cells
        lo = 0 if w0 == 0 else w0 + quarter
        hi = min(w0 + L, n_cells) if last else w0 + 3 * quarter
        for k in range(K):
            col = song[:, k]
            onset_rows = np.flatnonzero(np.isin(col[:n_cells], (TAP, HOLD_START)))
            for parity in (0, 1):
                todo = []
                for i, r in enumerate(onset_rows):
                    if not lo <= r < hi or i % 2 != parity:
                        continue
                    nxt = onset_rows[i + 1] if i + 1 < len(onset_rows) else n_cells
                    limit = min(nxt, r + HOLD_REACH, n_cells, w0 + L)
                    if col[r] == HOLD_START and np.any(col[r + 1:limit] == HOLD_BODY) \
                            and not np.any(col[r + 1:limit] == HOLD_END):
                        continue                    # a long note reaching past the limit stays
                    top = limit - 1 - (release_gap if limit == nxt < n_cells else 0)
                    todo.append((int(r), int(limit), np.arange(r + max(min_hold, 1), top + 1)))
                if todo:
                    yield w0, k, todo
        if last:
            break
        w0 += half


def refine_holds(model, song: np.ndarray, frames: torch.Tensor, s: float, beat_len,
                 n_cells: int, *, temperature: float = 1.0, release_temperature: float = 0.5,
                 min_hold: int = 3, release_gap: int = 2, hold_bias: float = 0.0,
                 hold_share: float | None = None,
                 rng: np.random.Generator | None = None) -> int:
    """Decide again, for every onset, tap or long note and where it is released, with the
    rest of the chart in view; onsets never move. Returns how many onsets changed.

    song      [n_rows, K] tokens of the whole song (no MASK), changed in place
    For one lane at a time, every other onset of the lane has its cells MASK from the
    onset up to the next onset in the lane (at most HOLD_REACH cells); everything else,
    the other lanes included, stays visible (_hold_groups). Two questions, one forward
    pass each:
      1. tap or long note: p(START) against p(TAP) on the onset cell (+ hold_bias, at
         temperature);
      2. for the long notes, with START now visible and the rest still MASK: the release
         at e with probability p(END at e) (at release_temperature, 0: the likeliest),
         over the e with e - onset >= min_hold and release_gap empty cells before the
         next onset (clean_holds' rules); none left: a tap.
    hold_share: instead of 1, the long notes go to the round(hold_share * onsets)
    onsets with the highest p(START) / (p(START) + p(TAP)) in a first sweep: as many
    long notes as asked, where the model expects them most (human charts pick a long-note
    style per chart: 7% have none and 10% have more than 47%; the sampler gives every
    song about 15%, 2026-10-04).
    A long note has one release, so p(END at e) is the model's distribution of where it
    ends. Scoring whole paths instead (sum log p(BODY) ... log p(END) against sum log
    p(EMPTY)) charged every body cell for the model's doubt whether there is a long note
    at all, and cut long notes by 40% on 8 val songs (2026-10-04).
    """
    rng = rng or np.random.default_rng()
    fpc = model.config.frames_per_cell
    changed = 0

    def pick(logit: np.ndarray, temp: float) -> int:
        if temp <= 0:
            return int(np.argmax(logit))
        w = np.exp((logit - logit.max()) / temp)
        return int(rng.choice(len(logit), p=w / w.sum()))

    def masked(w0: int, k: int, todo: list) -> np.ndarray:
        x = song[w0:w0 + L].astype(np.int64).copy()
        for r, limit, _ in todo:
            x[r - w0:limit - w0, k] = MASK
        return x

    chosen = None
    if hold_share is not None:
        p_hold = {}
        for w0, k, todo in _hold_groups(song, n_cells, min_hold, release_gap):
            probs = denoiser_probs(model, masked(w0, k, todo),
                                   frames[w0 * fpc:(w0 + L) * fpc], s, beat_len(w0))
            for r, _, ends in todo:
                if len(ends):
                    p = probs[r - w0, k]
                    p_hold[(r, k)] = p[HOLD_START] / (p[HOLD_START] + p[TAP] + 1e-12)
        want = int(np.rint(hold_share * np.isin(song[:n_cells], (TAP, HOLD_START)).sum()))
        chosen = set(sorted(p_hold, key=p_hold.get, reverse=True)[:max(want, 0)])

    for w0, k, todo in _hold_groups(song, n_cells, min_hold, release_gap):
        mel = frames[w0 * fpc:(w0 + L) * fpc]
        x = masked(w0, k, todo)
        if chosen is not None:
            held = [len(ends) > 0 and (r, k) in chosen for r, _, ends in todo]
        else:
            probs = denoiser_probs(model, x, mel, s, beat_len(w0))
            held = []
            for r, _, ends in todo:
                p = probs[r - w0, k]
                logit = np.log(np.array([p[TAP], p[HOLD_START] * np.exp(min(hold_bias, 20.0))])
                               + 1e-12)
                held.append(len(ends) > 0 and pick(logit, temperature) == 1)
        if any(held):
            for (r, _, _), h in zip(todo, held, strict=True):
                if h:
                    x[r - w0, k] = HOLD_START
            probs = denoiser_probs(model, x, mel, s, beat_len(w0))
        for (r, limit, ends), h in zip(todo, held, strict=True):
            new = np.full(limit - r, EMPTY, dtype=np.int64)
            if h:
                e = int(ends[pick(np.log(probs[ends - w0, k, HOLD_END] + 1e-12),
                                  release_temperature)])
                new[0], new[1:e - r], new[e - r] = HOLD_START, HOLD_BODY, HOLD_END
            else:
                new[0] = TAP
            if not np.array_equal(song[r:limit, k], new):
                song[r:limit, k] = new
                changed += 1
    STATS["changed_holds"] += changed
    return changed


LOUD_CLIP = 3.0                 # loudness z-scores beyond this count as this


LOUD_SIDES = ("quiet", "both")


def loudness_bias(mel: np.ndarray, n_cells: int, n_rows: int, loud_bias: float,
                  side: str = "quiet") -> np.ndarray:
    """[n_rows] log bias on EMPTY per row: -loud_bias * the loudness z-score of the row's
    bar (mean of the song-standardized log-Mel over the bar, standardized over the song's
    bars, clipped at LOUD_CLIP), 0 past n_cells. Human charts thin out more in quiet bars
    than the sampler does (EXPERIMENTS 2026-10-04: in the quietest bars, z < -1.5, the
    sampled charts have 17% more notes than the human ones; structure.dynamics' slope
    0.20 against 0.22): loud_bias > 0 puts fewer notes in quiet bars. side "quiet": only
    there (z < 0; louder bars keep their bias 0); "both": also more notes in loud bars,
    which already have more than the human charts and raised the SR by 0.09 at 0.1
    (EXPERIMENTS 2026-10-05)."""
    if side not in LOUD_SIDES:
        raise ValueError(f"unknown loudness side {side!r}")
    out = np.zeros(n_rows)
    n_bars = n_cells // BAR
    if loud_bias == 0 or n_bars < 2:
        return out
    frames = np.asarray(mel, dtype=np.float64)[:n_bars * BAR * 4]
    loud = frames.reshape(n_bars, BAR * 4, -1).mean(axis=(1, 2))
    z = np.clip((loud - loud.mean()) / (loud.std() + 1e-9), -LOUD_CLIP, LOUD_CLIP)
    if side == "quiet":
        z = np.minimum(z, 0.0)
    out[:n_bars * BAR] = np.repeat(-loud_bias * z, BAR)
    if n_cells > n_bars * BAR:                   # the part bar at the end: as the last bar
        out[n_bars * BAR:n_cells] = -loud_bias * z[-1]
    return out


def onset_strength(mel: np.ndarray, n_rows: int) -> np.ndarray:
    """[n_rows] the audio's onset strength per token row: spectral flux (the positive
    differences of the song-standardized log-Mel between frames, summed over bands), the
    largest over the row's frames, then the larger of the row and the next one (audio
    onsets peak about 18 ms after the note times, DECISIONS 2026-09-27); 0 past the audio."""
    fpc = 4
    x = np.asarray(mel, dtype=np.float64)
    m = min(n_rows, len(x) // fpc)
    out = np.zeros(n_rows)
    if m < 2:
        return out
    x = x[:m * fpc]
    d = np.maximum(np.diff(x, axis=0, prepend=x[:1]), 0.0).sum(axis=1)
    f = d.reshape(m, fpc).max(axis=1)
    out[:m] = np.maximum(f, np.append(f[1:], f[-1]))
    return out


ONSET_FULL = 0.1                # onset_gate: rows at or below this flux quantile get all of it


def onset_gate(mel: np.ndarray, n_cells: int, n_rows: int, beta: float) -> np.ndarray:
    """[n_rows] log penalty on starting a note (TAP, HOLD_START) per row from the audio's
    onset strength (onset_strength), ranked within the song: 0 for rows at or above the
    median, rising to beta at the ONSET_FULL quantile and below (silence included); 0 past
    n_cells. On 40 val songs (EXPERIMENTS 2026-10-06, quiet sections) the sampled onsets
    that miss the human rhythm by more than a cell lie below the song's median onset
    strength 62% of the time (in its bottom quarter 31%), the human onsets 26% (9%): the
    penalty takes notes away where the music gives them nothing to land on, more off the
    rhythm than on it. It applies to starts only: as an EMPTY bias it also cut the bodies
    of long notes, which sit on weak onsets by nature (long-note share -38% at 1, 10-06)."""
    out = np.zeros(n_rows)
    if beta == 0 or n_cells < 2:
        return out
    f = onset_strength(mel, n_cells)
    q = np.searchsorted(np.sort(f), f, side="left") / len(f)   # share of rows strictly weaker
    out[:n_cells] = beta * np.clip((0.5 - q) / (0.5 - ONSET_FULL), 0.0, 1.0)
    return out


# The gate by SR. Humans start more of their notes on weak onsets the harder the chart:
# of their onset rows below the song's median onset strength, Easy 5.5%, Normal 8.8%, Hard
# 15.7%, Insane 24.4%, Expert 28.7% (240 val songs, 2026-10-08): streams run through the
# weak onsets. The gate at 1 matches them up to Normal (5.5%, 8.7%) and cuts too much above
# (Insane 18.9%, Expert 22.3%), where the charts lose notes and SR (-0.27, -0.62) and their
# streams break. taper (lo, hi, floor): the gate at full strength up to SR lo, falling
# linearly to floor at hi and above; (2.7, 5.0, 0.35) is the guess the per-grade shares give
# when the share moves linearly with the gate's strength between 0 and 1.
GATE_TAPER = (2.7, 5.0, 0.35)


def parse_taper(text: str | None):
    """"LO,HI,FLOOR" -> (lo, hi, floor) for taper_factor; None, "", "off" or "0" -> None."""
    if text is None or text.strip().lower() in ("", "off", "0", "none"):
        return None
    try:
        lo, hi, floor = (float(v) for v in text.split(","))
    except ValueError:
        raise ValueError(f"onset taper {text!r}: LO,HI,FLOOR") from None
    if not (lo < hi and 0 <= floor <= 1):
        raise ValueError(f"onset taper {text!r}: LO < HI and 0 <= FLOOR <= 1")
    return lo, hi, floor


def taper_factor(s: float, taper) -> float:
    """The onset gate's strength factor at target SR s for taper = (lo, hi, floor), or 1."""
    if taper is None:
        return 1.0
    lo, hi, floor = taper
    return float(np.clip((hi - s) / (hi - lo), floor, 1.0))


# Stream holes. In the sampled charts an evenly spaced stream misses a note more often than in
# human ones (bars of 8+ onset rows evenly spaced: 28% against 41% per song, 2026-10-08; a
# stream of 8+ single notes ends at a missing note 28% of the time against 12.5%, 2026-10-03).
HOLE_GAPS = (3, 4, 6, 12)       # stream spacings in cells: 1/4, 1/3, 1/2, 1 beat


def stream_holes(song: np.ndarray, n_cells: int) -> list[int]:
    """Rows h before n_cells without a note start where a stream misses exactly one: for a
    gap g of HOLE_GAPS (the first that fits), note starts at h - 2g, h - g, h + g and h + 2g
    and none on the other rows from h - 2g to h + 2g."""
    on = np.isin(song[:n_cells], (TAP, HOLD_START)).any(axis=1)
    holes = []
    for h in np.flatnonzero(~on):
        for g in HOLE_GAPS:
            if h - 2 * g < 0 or h + 2 * g >= n_cells:
                continue
            span = on[h - 2 * g:h + 2 * g + 1]
            want = np.zeros(4 * g + 1, dtype=bool)
            want[[0, g, 3 * g, 4 * g]] = True
            if np.array_equal(span, want):
                holes.append(int(h))
                break
    return holes


def fill_holes(model, song: np.ndarray, frames: torch.Tensor, s: float, beat_len,
               n_cells: int, *, threshold: float, release_gap: int = 0) -> int:
    """Fill the stream holes (stream_holes) the model wants filled; returns how many.

    For each hole, the row's free lanes (EMPTY, not too soon after a release) MASK and the
    rest of the chart around it in view (a window centred on the row): if the row starts a
    note with probability at least threshold (1 - the product over the masked lanes of
    1 - p(TAP) - p(HOLD_START)), a tap goes in the masked lane with the highest p(TAP).
    A tap in an EMPTY cell keeps the grammar.
    """
    fpc = model.config.frames_per_cell
    filled = 0
    for h in stream_holes(song, n_cells):
        free = [k for k in _free_lanes(song, h, release_gap)[0] if song[h, k] == EMPTY]
        if not free:
            continue
        w0 = int(np.clip(h - L // 2, 0, len(song) - L))
        x = song[w0:w0 + L].astype(np.int64).copy()
        x[h - w0, free] = MASK
        probs = denoiser_probs(model, x, frames[w0 * fpc:(w0 + L) * fpc], s, beat_len(w0))
        p = probs[h - w0, free]
        if 1.0 - float(np.prod(1.0 - p[:, TAP] - p[:, HOLD_START])) < threshold:
            continue
        song[h, free[int(np.argmax(p[:, TAP]))]] = TAP
        filled += 1
    STATS["filled_holes"] += filled
    return filled


# Bar copies. Human charts repeat whole bars where the music repeats; on the 240 val
# charts (EXPERIMENTS 2026-10-03) 17% of the bars with 4+ onsets copy an earlier bar,
# as it is or mirrored, and the source is the earlier bar whose audio is most similar
# 71% of the time, among the 3 most similar 85%; when that bar's similarity is 0.9 or
# more, 36% of bars copy it, below 0.5 3%. The model's own charts copy 1.4% of bars.
# Of the busy human bars (4+ onset rows), the nearest earlier copy is 1-2 bars back for
# 3.4%, 3-8 for 5.9%, 9 or more for 7.6%: a whole bar comes back from a section before,
# while the next bar changes its lanes (structure.phrase_repeats). copy_bars from the
# nearest bars made 8.5% / 3.8% / 2.2% (2026-10-11): min_lag keeps the near ones out.
COPY_TOP = 3                    # sources: the most similar earlier bars ...
COPY_MIN_SIM = 0.5              # ... with at least this audio cosine (ssm_audio)
COPY_MIN_ONSETS = 4             # a source bar has at least this many onset rows
COPY_DENSITY = 0.15             # a copy keeps the bar's onset count within 15% (or 1)


def _closed_bar(song: np.ndarray, r0: int) -> bool:
    """True if no long note crosses either edge of the bar that starts at row r0."""
    return not (np.isin(song[r0], (HOLD_BODY, HOLD_END)).any()
                or np.isin(song[r0 + BAR - 1], (HOLD_START, HOLD_BODY)).any())


def onset_count_logp(p_onset: np.ndarray) -> np.ndarray:
    """[rows, K] onset probability per cell -> [rows, K + 1] log P(the row has n onsets),
    the lanes taken as independent (as the model's cells are, given the context)."""
    dist = np.zeros((len(p_onset), p_onset.shape[1] + 1))
    dist[:, 0] = 1.0
    for k in range(p_onset.shape[1]):
        q = p_onset[:, k:k + 1]
        dist = dist * (1 - q) + np.pad(dist[:, :-1], ((0, 0), (1, 0))) * q
    return np.log(dist + 1e-12)


def copy_bars(model, song: np.ndarray, frames: torch.Tensor, s: float, beat_len,
              n_cells: int, s_audio: np.ndarray, *, copy_bias: float,
              top: int = COPY_TOP, min_sim: float = COPY_MIN_SIM, mirror: bool = True,
              min_lag: int = 1) -> int:
    """Offer every bar a copy of an earlier bar with similar audio; returns how many it took.

    song      [n_rows, K] tokens of the whole song (no MASK), changed in place
    s_audio   [n_bars, n_bars] audio similarity of the song's bars (structure.ssm_audio)
    Bars in time order. Sources: the top bars at least min_lag before it by audio
    similarity, at least min_sim, with COPY_MIN_ONSETS onset rows and an onset count within
    COPY_DENSITY of the bar's own; the whole bar is copied (rhythm, lanes and long notes),
    as it is or mirrored (lane k <-> 3 - k). One forward pass with the bar MASK and the rest
    of the chart around it (a window centred on it) gives p(cell) for the bar's cells. The
    rhythm decides: a version scores, row by row, log P(that many onsets in the row)
    (onset_count_logp), the bar as it is too, and the first source in order of audio
    similarity whose score + copy_bias reaches the bar's own replaces it (copy_bias in nats
    per bar: how much worse a copy may fit the music and still be taken). Between a source
    as it is and mirrored, sum log p over the cells picks (the lanes at the bar's edges).
    Taking the best-scoring source instead, or scoring cell by cell, preferred sparse copies
    (10 val songs, 2026-10-03: -5% to -12% notes): with the bar MASK the model spreads a
    sure note over the lanes, and the likeliest version is the one without the uncertain
    notes. Bars with a long note across an edge neither copy nor are copied, so the grammar
    holds; run clean_holds afterwards for the release gap at the edges. A copied bar can be
    copied again.
    """
    fpc = model.config.frames_per_cell
    n_bars = min(n_cells // BAR, len(s_audio))
    rows, lanes = np.arange(BAR), np.arange(K)
    copied = 0

    def counts(version: np.ndarray) -> np.ndarray:
        return np.isin(version, (TAP, HOLD_START)).sum(axis=1)

    for b in range(max(min_lag, 1), n_bars):
        r0 = b * BAR
        if not _closed_bar(song, r0):
            continue
        current = song[r0:r0 + BAR].copy()
        n_now = int(counts(current).sum())
        sources: list[np.ndarray] = []
        sims = s_audio[b, :b - max(min_lag, 1) + 1]
        for a in np.argsort(-sims, kind="stable")[:top]:
            a0 = int(a) * BAR
            src = song[a0:a0 + BAR].copy()
            if sims[a] >= min_sim and _closed_bar(song, a0) \
                    and np.isin(src, (TAP, HOLD_START)).any(axis=1).sum() >= COPY_MIN_ONSETS \
                    and abs(int(counts(src).sum()) - n_now) <= max(1, COPY_DENSITY * n_now) \
                    and not np.array_equal(src, current):
                sources.append(src)
        if not sources:
            continue
        w0 = int(np.clip(r0 - (L - BAR) // 2, 0, len(song) - L))
        x = song[w0:w0 + L].astype(np.int64).copy()
        x[r0 - w0:r0 - w0 + BAR] = MASK
        probs = denoiser_probs(model, x, frames[w0 * fpc:(w0 + L) * fpc], s, beat_len(w0))
        bar = probs[r0 - w0:r0 - w0 + BAR]
        logp = onset_count_logp(bar[..., TAP] + bar[..., HOLD_START])
        cell_logp = np.log(bar + 1e-12)
        own = float(logp[rows, counts(current)].sum())
        for src in sources:
            if float(logp[rows, counts(src)].sum()) + copy_bias < own:
                continue
            versions = [src, src[:, ::-1]] if mirror else [src]
            fit = [float(cell_logp[rows[:, None], lanes[None, :], v].sum()) for v in versions]
            song[r0:r0 + BAR] = versions[int(np.argmax(fit))]
            copied += 1
            break
    STATS["copied_bars"] += copied
    return copied


def bar_probs(model, song: np.ndarray, frames: torch.Tensor, s: float, beat_len,
              r0: int) -> np.ndarray:
    """p(x0) [BAR, K, 5] for the bar that starts at row r0, with the bar MASK and the rest
    of the chart around it (a window centred on it, as copy_bars); song as copy_bars."""
    fpc = model.config.frames_per_cell
    w0 = int(np.clip(r0 - (L - BAR) // 2, 0, len(song) - L))
    x = song[w0:w0 + L].astype(np.int64).copy()
    x[r0 - w0:r0 - w0 + BAR] = MASK
    probs = denoiser_probs(model, x, frames[w0 * fpc:(w0 + L) * fpc], s, beat_len(w0))
    return probs[r0 - w0:r0 - w0 + BAR]


# Bars left empty. Human charts leave 0.85% of the bars inside a song without a note (240
# val songs, 2026-10-07); the sampled charts fill 77% of those (the onset gate included).
# The audio does not find them: those bars are quiet in the middle (loudness z -1.3) but
# spread wide, and humans fill 94% of the bars at z < -2. A whole empty bar is a joint
# choice of 192 cells, which cell-by-cell sampling seldom makes even where each cell is
# unlikely to hold a note, the way it seldom kept a lane pattern (EXPERIMENTS 2026-10-02):
# so ask the model about the whole bar, with the rest of the chart in view.


def rest_bars(model, song: np.ndarray, frames: torch.Tensor, s: float, beat_len,
              n_cells: int, *, threshold: float) -> int:
    """Leave empty the bars where the model expects almost no notes; returns how many.

    For every whole bar with a note start, in time order, bar_probs with the bar MASK
    gives the expected number of note starts in it, the sum of p(TAP) + p(HOLD_START) over
    its cells; below threshold the bar's taps and the long notes that start in it (to
    their ends) are taken out. Long notes that started before the bar are kept. Taking
    notes out keeps the grammar and the release gaps, so no clean_holds is needed.
    """
    cleared = 0
    for b in range(n_cells // BAR):
        r0 = b * BAR
        starts = np.isin(song[r0:r0 + BAR], (TAP, HOLD_START))
        if not starts.any():
            continue
        p = bar_probs(model, song, frames, s, beat_len, r0)
        if float((p[..., TAP] + p[..., HOLD_START]).sum()) >= threshold:
            continue
        for row, k in zip(*np.nonzero(starts), strict=True):
            r = r0 + int(row)
            e = r + 1
            if song[r, k] == HOLD_START:
                while e < len(song) and song[e, k] == HOLD_BODY:
                    e += 1
                if e < len(song) and song[e, k] == HOLD_END:
                    e += 1
            song[r:e, k] = EMPTY
        cleared += 1
    STATS["rest_bars"] += cleared
    return cleared


# Rhythm carried over. Human charts give a bar the rhythm (onset rows) of the bar before 27%
# of the time, of the bar 2 earlier 31%, 4 earlier 29%, 8 earlier 23%; the sampled charts
# about half as often (14 / 15 / 12 / 9%; structure.phrase_repeats, 240 val songs,
# 2026-10-10): "the same pattern does not go on for long" in the playtest of 10-07. When
# they repeat the bar before, humans change its lanes 94% of the time. A bar's rhythm is a
# joint choice of 48 rows, like a rest (rest_bars), so ask the model about the whole bar.
CARRY_LAGS = (1, 2, 4, 8)       # carry_rhythm: the earlier bars a bar may take its rhythm from


def _tap_bar(cells: np.ndarray) -> bool:
    """True if the cells hold taps only: no long note starts, runs or ends in them."""
    return bool(np.isin(cells, (EMPTY, TAP)).all())


def _place_rhythm(song: np.ndarray, r0: int, counts: np.ndarray, src: np.ndarray,
                  release_gap: int) -> bool:
    """The bar at row r0 gets counts[i] taps on its row i, in the lanes the source bar src
    used where they are free (not just after a release in the lane), then in the other free
    lanes; EMPTY elsewhere. False, and the bar as it was, if a row has too few free lanes."""
    old = song[r0:r0 + BAR].copy()
    song[r0:r0 + BAR] = EMPTY
    for i in np.flatnonzero(counts):
        row = r0 + int(i)
        free = [k for k in range(K) if not _after_release(song, row, k, release_gap)]
        if len(free) < counts[i]:
            song[r0:r0 + BAR] = old
            return False
        lanes = [k for k in free if src[i, k] == TAP] + [k for k in free if src[i, k] != TAP]
        song[row, lanes[:counts[i]]] = TAP
    return True


def _relane_bar(model, song: np.ndarray, frames: torch.Tensor, s: float, beat_len, r0: int, *,
                temperature: float, rng: np.random.Generator, release_gap: int = 0,
                stride: int = 8) -> None:
    """refine_lanes for the bar at row r0 alone: its rows of each residue class mod stride
    get their free cells MASK, with the rest of the chart in view (a window centred on the
    bar), and their lanes chosen again (_lane_sets); one forward pass per class."""
    fpc = model.config.frames_per_cell
    w0 = int(np.clip(r0 - (L - BAR) // 2, 0, len(song) - L))
    for j in rng.permutation(stride):
        todo = []
        for row in range(r0 + int(j), r0 + BAR, stride):
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
            chosen = _lane_sets(free, n, p[:, TAP], p[:, EMPTY], temperature, rng)
            song[row, free] = EMPTY
            song[row, list(chosen)] = TAP


def carry_rhythm(model, song: np.ndarray, frames: torch.Tensor, s: float, beat_len,
                 n_cells: int, s_audio: np.ndarray | None, *, bias: float, lane_model=None,
                 lags=CARRY_LAGS, temperature: float = 0.5,
                 rng: np.random.Generator | None = None, release_gap: int = 0) -> int:
    """Let a bar take over the rhythm of an earlier bar when the model likes it as well;
    returns how many bars did.

    song      [n_rows, K] tokens of the whole song (no MASK), changed in place
    s_audio   [n_bars, n_bars] audio similarity of the song's bars (None: order by lag)
    Bars in time order; a bar and its sources hold taps only (_tap_bar) and COPY_MIN_ONSETS+
    onset rows. Sources: the bars lags earlier whose onset rows differ from the bar's and
    whose note count is within COPY_DENSITY of its own. With the bar MASK and the chart
    around it (bar_probs), a rhythm scores log P(that many onsets) row by row
    (onset_count_logp), as in copy_bars; the first source in order of audio similarity
    (then lag) whose score + bias reaches the bar's own, and whose chords fit the free
    lanes (_place_rhythm), gives the bar its rhythm (the rows and their chord sizes) and,
    to start from, its lanes; then the bar's lanes are chosen again with
    the chart in view (_relane_bar, lane_model at the temperature), so the rhythm goes on
    and the lanes move, as in human charts. A carried bar can pass its rhythm on.
    """
    rng = rng or np.random.default_rng()
    lane_model = lane_model or model
    rows = np.arange(BAR)
    carried = 0
    for b in range(1, n_cells // BAR):
        r0 = b * BAR
        if not _tap_bar(song[r0:r0 + BAR]):
            continue
        now = (song[r0:r0 + BAR] == TAP).sum(axis=1)
        if (now > 0).sum() < COPY_MIN_ONSETS:
            continue
        sources = []
        for lag in lags:
            a0 = (b - lag) * BAR
            if a0 < 0 or not _tap_bar(song[a0:a0 + BAR]):
                continue
            src = song[a0:a0 + BAR].copy()
            c = (src == TAP).sum(axis=1)
            if (c > 0).sum() < COPY_MIN_ONSETS or np.array_equal(c > 0, now > 0) \
                    or abs(int(c.sum()) - int(now.sum())) > max(1, COPY_DENSITY * now.sum()):
                continue
            sim = float(s_audio[b, b - lag]) if s_audio is not None and b < len(s_audio) else 0.0
            sources.append((-sim, lag, c, src))
        if not sources:
            continue
        p = bar_probs(model, song, frames, s, beat_len, r0)
        logp = onset_count_logp(p[..., TAP] + p[..., HOLD_START])
        own = float(logp[rows, now].sum())
        for _, _, c, src in sorted(sources, key=lambda t: (t[0], t[1])):
            if float(logp[rows, c].sum()) + bias < own \
                    or not _place_rhythm(song, r0, c, src, release_gap):
                continue
            _relane_bar(lane_model, song, frames, s, beat_len, r0, temperature=temperature,
                        rng=rng, release_gap=release_gap)
            carried += 1
            break
    STATS["carried_bars"] += carried
    return carried


# Long notes among taps. Of the long notes human charts start, 48% are in bars where at least
# half the starts are long notes and 19% in bars where under a quarter are; the sampled
# charts 34% and 29% (structure.ln_placement, 240 val songs, 2026-10-10), and their long
# notes are shorter (0.75 against 1.00 beats): a short long note dropped into a run of taps,
# "long notes mixed with taps unlike a human" and "too fond of holding while tapping" in the
# playtest of 10-07.
TIDY_SHARE = 0.25               # tidy_holds: bars where under this share ...
TIDY_MIN = 4                    # ... of at least this many note starts are long notes


def tidy_holds(song: np.ndarray, n_cells: int, *, max_beats: float) -> int:
    """Long notes among taps become taps; returns how many.

    In every whole bar with TIDY_MIN+ note starts of which under TIDY_SHARE are long notes,
    each long note that starts there and lasts less than max_beats beats becomes a tap at
    its start (its body and end EMPTY). Taking cells out keeps the grammar and the release
    gaps; onsets stay, so the rhythm and F1 do too."""
    changed = 0
    for b in range(n_cells // BAR):
        r0 = b * BAR
        cells = song[r0:r0 + BAR]
        n = int(np.isin(cells, (TAP, HOLD_START)).sum())
        long_rows, long_lanes = np.nonzero(cells == HOLD_START)
        if n < TIDY_MIN or not len(long_rows) or len(long_rows) >= TIDY_SHARE * n:
            continue
        for i, k in zip(long_rows.tolist(), long_lanes.tolist(), strict=True):
            r = r0 + i
            e = r + 1
            while e < len(song) and song[e, k] == HOLD_BODY:
                e += 1
            if e < len(song) and song[e, k] == HOLD_END and e - r < max_beats * D:
                song[r, k] = TAP
                song[r + 1:e + 1, k] = EMPTY
                changed += 1
    STATS["tidied_holds"] += changed
    return changed


LANE_PASSES = ("sampled", "forward")


def _song_frames(model, mel: np.ndarray, n_rows: int,
                 n_cells_for_ssm: int = 0) -> tuple[torch.Tensor, np.ndarray | None]:
    """(frames for n_rows token rows on the model's device, padded with silence past the
    audio; the bars' audio similarity over the first n_cells_for_ssm cells, or None)."""
    device = next(model.parameters()).device
    frames = np.asarray(mel, dtype=np.float32)
    need = n_rows * model.config.frames_per_cell
    if len(frames) < need:                                          # past the audio: silence
        fill = frames.min() if frames.size else 0.0
        frames = np.concatenate([frames, np.full((need - len(frames), frames.shape[1]), fill,
                                                 dtype=np.float32)])
    n_bars = n_cells_for_ssm // BAR
    s_audio = ssm_audio(frames, n_bars) if n_bars >= 2 else None
    return torch.as_tensor(frames[:need], device=device), s_audio


def _beat_len_fn(timing_points, cell_offset: int):
    """row0 -> mean beat length (ms) of the window starting at token row row0 (the b input)."""
    grid = from_timing_points(timing_points)

    def beat_len(row0: int) -> float:
        start = row0 - cell_offset
        return (grid.time_from_cell(start + L, D) - grid.time_from_cell(start, D)) / BEATS_PER_CHUNK
    return beat_len


def post_song(model, tokens: np.ndarray, mel: np.ndarray, s: float, timing_points,
              cell_offset: int, n_cells: int, *, holds: bool = False,
              hold_share: float | None = None,
              copy_bias: float | None = None, min_hold: int | None = None,
              release_gap: int | None = None, seed: int = 0, genre: int | None = None,
              mapper: int | None = None, stats=None, rest: float | None = None,
              holes: float | None = None, carry: float | None = None,
              ln_tidy: float | None = None, lane_guidance: float | None = None,
              lane_temperature: float = 0.5, copy_min_lag: int = 1) -> np.ndarray:
    """The passes after the lane passes on a finished chart ([n_chunks, L, K] tokens as
    generate_song returns them, e.g. from evaluate.py's charts.npz), in generate_song's
    order: refine_holds (holds=True or a hold_share), carry_rhythm (carry: the bias; its
    lanes with lane_guidance at lane_temperature), copy_bars (copy_bias, sources at least
    copy_min_lag bars back), clean_holds,
    tidy_holds (ln_tidy: the longest long note in beats it turns into a tap), fill_holes
    (holes: the threshold), rest_bars (rest: the threshold); a new array."""
    base = model
    if genre is not None or mapper is not None or stats is not None:
        model = Styled(base, genre, mapper, stats=stats)
    lane_model = model if lane_guidance is None else Styled(base, genre, mapper, lane_guidance,
                                                            stats=stats)
    song = np.concatenate([np.asarray(tokens, dtype=np.int64).reshape(-1, K),
                           np.full((L, K), PAD, dtype=np.int64)])
    frames, s_audio = _song_frames(model, mel, len(song),
                                   n_cells if copy_bias is not None or carry is not None else 0)
    beat_len = _beat_len_fn(timing_points, cell_offset)
    auto_hold, auto_gap = hold_rules(s)
    min_hold = auto_hold if min_hold is None else min_hold
    release_gap = auto_gap if release_gap is None else release_gap
    rng = np.random.default_rng(seed)
    if holds or hold_share is not None:
        refine_holds(model, song, frames, s, beat_len, n_cells, min_hold=min_hold,
                     release_gap=release_gap, hold_share=hold_share, rng=rng)
    if carry is not None:
        carry_rhythm(model, song, frames, s, beat_len, n_cells, s_audio, bias=carry,
                     lane_model=lane_model, temperature=lane_temperature, rng=rng,
                     release_gap=release_gap)
    if copy_bias is not None and s_audio is not None \
            and copy_bars(model, song, frames, s, beat_len, n_cells, s_audio,
                          copy_bias=copy_bias, min_lag=copy_min_lag) \
            and (min_hold > 0 or release_gap > 0):
        clean_holds(song, min_hold=min_hold, release_gap=release_gap)
    if ln_tidy is not None:
        tidy_holds(song, n_cells, max_beats=ln_tidy)
    if holes is not None:
        fill_holes(model, song, frames, s, beat_len, n_cells, threshold=holes,
                   release_gap=release_gap)
    if rest is not None:
        rest_bars(model, song, frames, s, beat_len, n_cells, threshold=rest)
    return song[:-L].reshape(np.shape(tokens)).astype(np.int8)


def generate_song(model, mel: np.ndarray, s: float, timing_points, cell_offset: int,
                  n_cells: int, *, steps: int = 32, order: str = "random",
                  mode: str = "continue", prefix_cells: int = 2 * BAR,
                  seed: int = 0, temperature: float = 1.0, refine: int = 0,
                  lane_temperature: float = 0.5, hold_bias: float = 0.0,
                  min_hold: int | None = None, release_gap: int | None = None,
                  lanes: str = "sampled", spread: bool = False,
                  empty_bias: float = 0.0, forward_temperature: float | None = None,
                  jack_bias: float = 0.0, copy_bias: float | None = None,
                  genre: int | None = None, mapper: int | None = None,
                  style_guidance: float = 0.0, holds: bool = False,
                  hold_share: float | None = None, loud_bias: float = 0.0,
                  loud_side: str = "quiet", stats=None,
                  lane_guidance: float | None = None, onset_bias: float = 0.0,
                  onset_taper=None, rest: float | None = None,
                  holes: float | None = None, carry: float | None = None,
                  ln_tidy: float | None = None, copy_min_lag: int = 1) -> np.ndarray:
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
    holds         refine_holds after the lane passes: tap or long note, and the release,
                  decided again with the whole chart in view; hold_share (implies holds):
                  that share of the onsets become long notes, where the model expects them
    copy_bias     copy_bars after that with this bias in nats per bar (None: no copies);
                  the audio similarity of bars comes from mel; sources at least
                  copy_min_lag bars back
    loud_bias     while sampling, log bias on EMPTY of -loud_bias * the bar's loudness
                  z-score (loudness_bias): fewer notes where the music is quiet; loud_side
                  "quiet" (only there) or "both" (also more where it is loud)
    onset_bias    while sampling, log penalty of up to onset_bias on starting a note on the
                  rows where the audio's onset strength is below the song's median
                  (onset_gate): fewer notes where nothing in the music starts
    genre, mapper style indices (style.encode) for a model trained with --style; None =
                  no label. style_guidance w > 0: classifier-free guidance towards the
                  style and stats (diffusion.Styled), two forward passes for every one
    stats         chart-stat buckets (chartstats.encode: long-note share, jack rate, trill
                  rate) for a model trained with --chart-stats; None = none given, which
                  is what that model samples when no one chooses
    lane_guidance the lane passes' own guidance weight towards the style and stats (None:
                  style_guidance). The lane passes ask with the rest of the chart in view,
                  where the context outweighs the stats; the long notes are already decided
                  by then, so guidance there moves the lanes (jacks, trills) without
                  overshooting the long-note share (EXPERIMENTS 2026-10-06)
    onset_taper   (lo, hi, floor): the gate's strength falls with the target SR (taper_factor,
                  GATE_TAPER); None: onset_bias at every SR
    carry         carry_rhythm with this bias after refine_holds (None: none): a bar takes
                  the rhythm of an earlier bar (1, 2, 4 or 8 before) the model likes as well,
                  its lanes chosen again by the lane passes' model
    ln_tidy       tidy_holds after the copies (None: none): in bars of taps, long notes
                  shorter than this many beats become taps
    holes         fill_holes with this threshold before rest_bars (None: none)
    rest          rest_bars last with this threshold (None: none): a bar whose expected
                  note starts, asked with the bar MASK and the chart around it, fall
                  below it is left empty
    min_hold, release_gap
                  clean_holds after sampling; None = by the target SR (hold_rules),
                  0, 0 = keep the holds as sampled
    Order of work: sample, clean_holds, forward_lanes, refine_lanes, refine_holds,
    carry_rhythm, copy_bars, clean_holds again if a bar was copied, tidy_holds, fill_holes,
    rest_bars.
    Decode the result with tokenizer.make_metas(timing_points, cell_offset, n_chunks, s).
    """
    if mode not in ("continue", "independent"):
        raise ValueError(f"unknown mode {mode!r}")
    if lanes not in LANE_PASSES:
        raise ValueError(f"unknown lanes {lanes!r}")
    lane_model = None
    if lane_guidance is not None and lane_guidance != style_guidance:
        lane_model = Styled(model, genre, mapper, lane_guidance, stats=stats)
    if genre is not None or mapper is not None or style_guidance or stats is not None:
        model = Styled(model, genre, mapper, style_guidance, stats=stats)
    lane_model = lane_model or model
    r = model.config.frames_per_cell
    n_chunks = -(-n_cells // L)
    song = np.full(((n_chunks + 1) * L, K), PAD, dtype=np.int64)   # +1 chunk for overhang
    song[:n_cells] = MASK
    frames, s_audio = _song_frames(model, mel, len(song),
                                   n_cells if copy_bias is not None or carry is not None else 0)
    beat_len = _beat_len_fn(timing_points, cell_offset)
    rng = np.random.default_rng(seed)
    row_bias = loudness_bias(mel, n_cells, len(song), loud_bias, loud_side) if loud_bias else None
    row_onset = onset_gate(mel, n_cells, len(song), onset_bias * taper_factor(s, onset_taper)) \
        if onset_bias else None

    def fill(row0: int, left_closed: bool, right_closed: bool) -> None:
        song[row0:row0 + L] = sample_window(
            model, song[row0:row0 + L], frames[row0 * r:(row0 + L) * r], s, beat_len(row0),
            steps=steps, order=order, rng=rng, left_closed=left_closed, right_closed=right_closed,
            temperature=temperature, hold_bias=hold_bias, empty_bias=empty_bias, spread=spread,
            row_empty_bias=None if row_bias is None else row_bias[row0:row0 + L],
            row_onset_bias=None if row_onset is None else row_onset[row0:row0 + L])

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
        forward_lanes(lane_model, song, frames, s, beat_len, n_cells, rng=rng,
                      release_gap=release_gap,
                      jack_bias=jack_bias,
                      temperature=lane_temperature if forward_temperature is None
                      else forward_temperature)
    if refine:
        refine_lanes(lane_model, song, frames, s, beat_len, n_cells, sweeps=refine,
                     temperature=lane_temperature, rng=rng, release_gap=release_gap,
                     jack_bias=jack_bias)
    if holds or hold_share is not None:
        refine_holds(model, song, frames, s, beat_len, n_cells, min_hold=min_hold,
                     release_gap=release_gap, hold_share=hold_share, rng=rng)
    if carry is not None:
        carry_rhythm(model, song, frames, s, beat_len, n_cells, s_audio, bias=carry,
                     lane_model=lane_model, temperature=lane_temperature, rng=rng,
                     release_gap=release_gap)
    copied = copy_bias is not None and s_audio is not None \
        and copy_bars(model, song, frames, s, beat_len, n_cells, s_audio, copy_bias=copy_bias,
                      min_lag=copy_min_lag)
    if copied and (min_hold > 0 or release_gap > 0):
        clean_holds(song, min_hold=min_hold, release_gap=release_gap)
    if ln_tidy is not None:
        tidy_holds(song, n_cells, max_beats=ln_tidy)
    if holes is not None:
        fill_holes(model, song, frames, s, beat_len, n_cells, threshold=holes,
                   release_gap=release_gap)
    if rest is not None:
        rest_bars(model, song, frames, s, beat_len, n_cells, threshold=rest)
    return song[:n_chunks * L].reshape(n_chunks, L, K).astype(np.int8)
