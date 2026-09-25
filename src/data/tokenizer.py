"""Chart <-> token grid (design doc §2).

    encode(chart)          -> tokens [n_chunks, L, K] int8, one ChunkMeta per chunk, EncodeStats
    decode(tokens, metas)  -> Chart
    grammar_violations(x)  -> (cell, lane) positions that break the hold grammar

Time axis: BeatGrid.cell_index / time_from_cell with D = 12 cells per beat,
re-originated at every red line (§2.3). Audio frames for a chunk come from
BeatGrid.frame_times(meta.start_cell, L) on the same cells, 4 per cell.

Collision policy (§2.7): onsets win, releases give way. Per lane, in time order:
  - an onset that lands on the previous hold's end cell shortens that hold by
    one cell (a hold shortened to zero length becomes a tap);
  - an onset that lands anywhere else inside the previous note (duplicate or
    overlapping objects) drops the later note.
F1 scores onsets only; the release-aware metric (mAP@tIoU) is deferred.
Without this rule the later onset was lost and hold_body followed hold_end.

Look at one chart next to its .osu file (run from the repo root):
    python -m src.data.tokenizer "data/raw/<set>/<difficulty>.osu"
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.data.beat_grid import from_chart, from_timing_points
from src.data.chart_parser import Chart, Note, parse_osu

D, L, K = 12, 384, 4                 # cells per beat, cells per chunk (32 beats), lanes
BEATS_PER_CHUNK = L // D             # 32
EMPTY, TAP, HOLD_START, HOLD_BODY, HOLD_END, MASK, PAD = range(7)
X0_VALUES = (EMPTY, TAP, HOLD_START, HOLD_BODY, HOLD_END)
MAX_CELLS = 100_000                  # ~8,300 beats. Real charts: p95 10,498 cells.
                                     # Extreme-BPM gimmick sections blow past this.


class ChartTooLong(ValueError):
    """The chart spans more than max_cells cells (usually an extreme-BPM section)."""


@dataclass
class ChunkMeta:
    chunk_idx: int
    start_cell: int            # BeatGrid cell of this chunk's row 0 (may be negative)
    cell_offset: int           # token index = grid cell + cell_offset, same for every chunk
    beat_len_ms: float         # b: mean beat length over the chunk's 32 beats
    sr: float                  # s: the chart's SR, same for every chunk
    timing_points: list[tuple[int, float]]


@dataclass
class EncodeStats:
    n_notes: int               # notes in the input chart
    n_onsets: int              # TAP + HOLD_START, counted in the output tokens
    n_dropped: int             # notes removed (duplicate / overlapping onset)
    n_shortened: int           # holds that gave their end cell to the next onset
    n_shortened_to_tap: int    #   ... of which ended up as taps
    n_demoted: int             # holds whose start and end rounded to one cell -> tap
    n_cells: int               # non-PAD cells
    lane_notes: np.ndarray     # [K] input notes per lane
    lane_onsets: np.ndarray    # [K] onsets per lane, counted in the tokens
    lane_dropped: np.ndarray   # [K] dropped notes per lane

    @property
    def balanced(self) -> bool:
        """Every input note is in the tokens or counted as dropped, lane by lane."""
        return (self.n_onsets + self.n_dropped == self.n_notes
                and np.array_equal(self.lane_onsets + self.lane_dropped, self.lane_notes))


# ---------------------------------------------------------------------------
# grammar
# ---------------------------------------------------------------------------

def grammar_violations(tokens: np.ndarray, *, closed: bool = True) -> np.ndarray:
    """Positions (cell, lane) that break the token grammar. Empty array = valid.

    tokens: [n_cells, K], or [n_chunks, L, K] read back to back as one song.

    values   only 0..4 and PAD
    holds    x[i+1] in {BODY, END}  <=>  x[i] in {START, BODY}   (per lane; pairs
             touching PAD are skipped). A violation is reported at cell i+1.
    PAD      whole rows only, and only as a suffix
    closed   True  (a whole song): the first cell cannot continue a hold and the
                   last non-PAD cell cannot leave one open.
             False (one window, e.g. a chunk): holds may cross both edges.

    Used by decode (precondition), by the evaluation's grammar-violation rate
    and as the rule behind constrained sampling (design doc §4.8, §4.11).
    """
    x = np.asarray(tokens)
    x = x.reshape(-1, x.shape[-1])
    if x.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    pad = x == PAD
    bad = ~(np.isin(x, X0_VALUES) | pad)                  # MASK or garbage

    pad_row = pad.all(axis=1)
    bad |= pad & ~pad_row[:, None]                        # PAD mixed into a note row
    if pad_row.any():
        first = int(np.argmax(pad_row))
        bad[first:][~pad_row[first:]] = True              # note rows after PAD began

    opens = np.isin(x, (HOLD_START, HOLD_BODY))           # a hold is open after this cell
    needs = np.isin(x, (HOLD_BODY, HOLD_END))             # this cell continues a hold
    pair = ~pad[:-1] & ~pad[1:]
    bad[1:] |= pair & (needs[1:] != opens[:-1])

    if closed:
        bad[0] |= needs[0] & ~pad[0]
        rows = np.flatnonzero(~pad_row)
        if rows.size:
            bad[rows[-1]] |= opens[rows[-1]]
    return np.argwhere(bad)


# ---------------------------------------------------------------------------
# encode
# ---------------------------------------------------------------------------

def encode(chart: Chart, sr: float = math.nan, *, audio_ms: float | None = None,
           max_cells: int = MAX_CELLS) -> tuple[np.ndarray, list[ChunkMeta], EncodeStats]:
    """Chart -> tokens [n_chunks, L, K] int8, ChunkMeta per chunk, EncodeStats.

    audio_ms: if given, the token range also covers the audio from 0 ms to
    audio_ms (intro before the first red line, outro after the last note), and
    PAD starts where the audio ends. If None, the range is set by the notes
    alone (PAD after the last note). Which one preprocessing uses must be fixed
    before the mel cache is built, because it moves the chunk boundaries.
    """
    if chart.key_count != K:
        raise ValueError(f"expected {K}K chart, got {chart.key_count}K")
    notes = chart.notes
    n = len(notes)
    lanes = np.fromiter((nt.lane for nt in notes), dtype=np.int64, count=n)
    if n and (lanes.min() < 0 or lanes.max() >= K):
        raise ValueError(f"lane out of range 0..{K - 1}")
    lane_notes = np.bincount(lanes, minlength=K)
    if n == 0 and audio_ms is None:
        zeros = np.zeros(K, dtype=np.int64)
        return (np.zeros((0, L, K), dtype=np.int8), [],
                EncodeStats(0, 0, 0, 0, 0, 0, 0, zeros, zeros.copy(), zeros.copy()))

    grid = from_chart(chart)                               # raises if there is no red line
    t_on = np.fromiter((nt.time_ms for nt in notes), dtype=np.float64, count=n)
    t_off = np.fromiter((nt.end_ms if nt.end_ms is not None else nt.time_ms for nt in notes),
                        dtype=np.float64, count=n)
    is_hold = np.fromiter((nt.end_ms is not None for nt in notes), dtype=bool, count=n)
    start = np.asarray(grid.cell_index(t_on, D), dtype=np.int64)
    end = np.maximum(np.asarray(grid.cell_index(t_off, D), dtype=np.int64), start)
    n_demoted = int(np.sum(is_hold & (end == start)))

    # --- collision policy: onsets win, releases give way (see module docstring) ---
    keep = np.ones(n, dtype=bool)
    last = [-1] * K                                        # last KEPT note per lane
    n_shortened = n_shortened_to_tap = 0
    for i in np.lexsort((lanes, t_on)):                    # time order, then lane
        k = lanes[i]
        p = last[k]
        if p >= 0:
            if start[i] == end[p] and end[p] > start[p]:   # onset on the previous hold's end
                end[p] -= 1
                n_shortened += 1
                n_shortened_to_tap += int(end[p] == start[p])
            elif start[i] <= end[p]:                       # duplicate / overlapping object
                keep[i] = False
                continue
        last[k] = i

    # --- token range ---
    ranges = []
    if keep.any():
        ranges.append((int(start[keep].min()), int(end[keep].max())))
    if audio_ms is not None:
        if not audio_ms > 0:
            raise ValueError("audio_ms must be positive")
        a_hi = int(grid.cell_index(float(audio_ms), D))
        while grid.time_from_cell(a_hi, D) >= audio_ms:   # last cell that starts inside the audio
            a_hi -= 1
        ranges.append((int(grid.cell_index(0.0, D)), a_hi))
    lo = min(r[0] for r in ranges)
    hi = max(r[1] for r in ranges)
    cell_offset = max(0, -lo)                              # tokens start at grid cell 0 or earlier
    n_cells = hi + cell_offset + 1
    if n_cells > max_cells:
        raise ChartTooLong(f"{n_cells:,} cells > max_cells {max_cells:,}")
    n_cells = max(n_cells, 0)

    # --- write ---
    total = -(-n_cells // L) * L                           # round up to whole chunks
    full = np.full((total, K), EMPTY, dtype=np.int8)
    for i in np.flatnonzero(keep):
        k, s, e = lanes[i], start[i] + cell_offset, end[i] + cell_offset
        if s == e:
            full[s, k] = TAP
        else:
            full[s, k] = HOLD_START
            full[s + 1:e, k] = HOLD_BODY
            full[e, k] = HOLD_END
    full[n_cells:] = PAD
    bad = grammar_violations(full)
    if len(bad):                                           # cannot happen if the policy is right
        raise RuntimeError(f"encode bug: {len(bad)} grammar violations, first {tuple(bad[0])}")

    onsets = (full == TAP) | (full == HOLD_START)
    lane_onsets = onsets.sum(axis=0).astype(np.int64)
    stats = EncodeStats(
        n_notes=n,
        n_onsets=int(lane_onsets.sum()),
        n_dropped=int((~keep).sum()),
        n_shortened=n_shortened,
        n_shortened_to_tap=n_shortened_to_tap,
        n_demoted=n_demoted,
        n_cells=n_cells,
        lane_notes=lane_notes,
        lane_onsets=lane_onsets,
        lane_dropped=np.bincount(lanes[~keep], minlength=K),
    )

    tps = list(chart.timing_points)
    metas = []
    for c in range(total // L):
        start_cell = c * L - cell_offset
        span = grid.time_from_cell(start_cell + L, D) - grid.time_from_cell(start_cell, D)
        metas.append(ChunkMeta(c, start_cell, cell_offset, float(span / BEATS_PER_CHUNK),
                               float(sr), tps))
    return full.reshape(-1, L, K), metas, stats


# ---------------------------------------------------------------------------
# decode
# ---------------------------------------------------------------------------

def decode(tokens: np.ndarray, metas: list[ChunkMeta]) -> Chart:
    """Tokens [n_chunks, L, K] + their ChunkMeta -> Chart (times rounded to whole ms).

    tokens: every chunk of one song, in order. They must be grammatical
    (ValueError otherwise): model outputs need
    constrained sampling or a repair step first. encode(decode(tokens)) gives
    the same tokens back when encode ran with audio_ms=None.
    """
    x = np.asarray(tokens)
    if x.ndim != 3 or x.shape[1:] != (L, K):
        raise ValueError(f"tokens must be [n_chunks, {L}, {K}], got {x.shape}")
    if len(metas) != x.shape[0]:
        raise ValueError(f"{x.shape[0]} chunks but {len(metas)} ChunkMeta")
    if not metas:
        return Chart(key_count=K, audio_filename="")
    offset = metas[0].cell_offset
    tps = [tuple(tp) for tp in metas[0].timing_points]    # tuples even if metas came from JSON
    for i, m in enumerate(metas):
        if (m.chunk_idx, m.start_cell, m.cell_offset) != (i, i * L - offset, offset) \
                or [tuple(tp) for tp in m.timing_points] != tps:
            raise ValueError(f"inconsistent ChunkMeta at chunk {i}")
    full = x.reshape(-1, K)
    bad = grammar_violations(full)
    if len(bad):
        raise ValueError(f"{len(bad)} grammar violations, first at (cell, lane) {tuple(bad[0])}")

    grid = from_timing_points(tps)

    def to_ms(token_idx: np.ndarray) -> np.ndarray:
        return np.rint(grid.time_from_cell(token_idx - offset, D)).astype(np.int64)

    notes = []
    for k in range(K):
        col = full[:, k]
        taps = np.flatnonzero(col == TAP)
        starts = np.flatnonzero(col == HOLD_START)         # the grammar pairs these in order
        ends = np.flatnonzero(col == HOLD_END)
        notes += [Note(int(t), k) for t in to_ms(taps)]
        notes += [Note(int(s), k, int(e)) for s, e in zip(to_ms(starts), to_ms(ends), strict=True)]
    notes.sort(key=lambda nt: (nt.time_ms, nt.lane))
    return Chart(key_count=K, audio_filename="", timing_points=list(tps), notes=notes)


# ---------------------------------------------------------------------------
# inspection
# ---------------------------------------------------------------------------

def to_ascii(chunk: np.ndarray, n_rows: int = 96, times_ms: np.ndarray | None = None,
             beat_rows: np.ndarray | None = None) -> str:
    """One line per cell, one column per lane.  . o [ | ] ? #  =  empty tap start body end mask pad
    With beat_rows (bool per row), a '-' after the row number marks the first cell of a beat."""
    sym = ".o[|]?#"
    lines = []
    for i, row in enumerate(chunk[:n_rows]):
        t = f" {times_ms[i]:10.1f}ms" if times_ms is not None else ""
        mark = "-" if beat_rows is not None and beat_rows[i] else " "
        lines.append(f"{i:4d}{mark}{t}  " + "".join(sym[v] for v in row))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Encode one .osu chart and print it next to its notes.")
    ap.add_argument("osu", type=Path)
    ap.add_argument("--chunk", type=int, default=0)
    ap.add_argument("--rows", type=int, default=96, help="cells to print (96 = 2 bars of 4/4)")
    ap.add_argument("--notes", type=int, default=16, help="source notes to list")
    a = ap.parse_args(argv)

    chart = parse_osu(a.osu)
    tokens, metas, st = encode(chart)
    grid = from_chart(chart)
    offset = metas[0].cell_offset if metas else 0
    print(f"{a.osu.name}: {st.n_notes} notes -> {len(metas)} chunks, {st.n_cells} cells, "
          f"cell_offset {offset}")
    print(f"  onsets {st.n_onsets}, dropped {st.n_dropped}, shortened {st.n_shortened} "
          f"(to tap {st.n_shortened_to_tap}), demoted {st.n_demoted}, balanced {st.balanced}")
    print(f"  grammar violations {len(grammar_violations(tokens))}")

    print(f"\nfirst {a.notes} notes: .osu time -> token cell -> time back")
    for nt in chart.notes[:a.notes]:
        cell = grid.cell_index(nt.time_ms, D)
        chunk, row = divmod(cell + offset, L)
        back = grid.time_from_cell(cell, D)
        hold = f"hold to {nt.end_ms}" if nt.end_ms is not None else "tap"
        print(f"  {nt.time_ms:8d} ms  lane {nt.lane}  {hold:>14}  ->  chunk {chunk} row {row:3d}"
              f"  ->  {back:10.1f} ms  ({back - nt.time_ms:+.1f})")

    if 0 <= a.chunk < len(metas):
        rows = min(a.rows, L)
        cells = metas[a.chunk].start_cell + np.arange(rows)
        starts = grid.cell_starts(D)                       # beats restart at every red line
        section = np.clip(np.searchsorted(starts, cells, side="right") - 1, 0, None)
        beat_rows = (cells - starts[section]) % D == 0
        print(f"\nchunk {a.chunk}, rows 0..{rows - 1}  ('-' = first cell of a beat)")
        print(to_ascii(tokens[a.chunk], rows, grid.time_from_cell(cells, D), beat_rows))

    tokens2, metas2, _ = encode(decode(tokens, metas))
    same = np.array_equal(tokens, tokens2) and \
        [m.start_cell for m in metas] == [m.start_cell for m in metas2]
    print(f"\nencode(decode(tokens)) == tokens: {same}")
    return 0 if same and st.balanced else 1


if __name__ == "__main__":
    raise SystemExit(main())
