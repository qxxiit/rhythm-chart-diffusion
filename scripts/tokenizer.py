from dataclasses import dataclass

import numpy as np

from src.data.beat_grid import from_chart
from src.data.chart_parser import Chart

D, L, K = 12, 384, 4                 # 비트당 칸, 청크당 칸(32비트), 레인
BEATS_PER_CHUNK = L // D             # 32
EMPTY, TAP, HOLD_START, HOLD_BODY, HOLD_END, MASK, PAD = range(7)


@dataclass
class ChunkMeta:
    chunk_idx: int
    start_cell: int
    cell_offset: int
    beat_len_ms: float
    sr: float
    timing_points: list[tuple[int, float]]


@dataclass
class EncodeStats:             # 오늘의 '끝났다' 기준이자 다음 주 round-trip 테스트
    n_notes: int               # == parser n_notes 여야 함
    n_collisions: int          # ≈ 0 여야 함
    n_demoted_holds: int       # 1칸 미만 롱노트 → tap
    lane_hist: np.ndarray      # parser 레인 분포와 일치


def encode(chart: Chart, sr: float) -> tuple[np.ndarray, list[ChunkMeta], EncodeStats]:
    """Chart → tokens [n_chunks, L, K] int8, ChunkMeta 리스트, 통계."""
    if chart.key_count != K:
        raise ValueError(f"expected {K}K chart, got {chart.key_count}K")

    lane_hist = np.bincount(
        [note.lane for note in chart.notes], minlength=K
    ).astype(np.int64)
    if not chart.notes:
        return (
            np.zeros((0, L, K), dtype=np.int8),
            [],
            EncodeStats(0, 0, 0, lane_hist),
        )

    grid = from_chart(chart)
    start_times = np.asarray([note.time_ms for note in chart.notes], dtype=np.float64)
    start_cells = np.asarray(grid.cell_index(start_times, D), dtype=np.int64)
    cell_offset = max(0, -int(start_cells.min()))

    encoded_notes: list[tuple[int, int, int, bool]] = []
    max_cell = 0
    for note, start_cell in zip(chart.notes, start_cells):
        end_cell = start_cell
        if note.end_ms is not None:
            end_cell = int(grid.cell_index(note.end_ms, D))
        start_cell = int(start_cell) + cell_offset
        end_cell += cell_offset
        encoded_notes.append((start_cell, end_cell, note.lane, note.end_ms is not None))
        max_cell = max(max_cell, end_cell)

    total = ((max_cell // L) + 1) * L
    full = np.full((total, K), EMPTY, dtype=np.int8)
    n_collisions = 0
    n_demoted_holds = 0

    def write(cell: int, lane: int, token: int) -> None:
        nonlocal n_collisions
        if full[cell, lane] == EMPTY:
            full[cell, lane] = token
        else:
            n_collisions += 1

    for start_cell, end_cell, lane, is_hold in encoded_notes:
        if start_cell == end_cell:
            if is_hold:
                n_demoted_holds += 1
            write(start_cell, lane, TAP)
            continue

        write(start_cell, lane, HOLD_START)
        for cell in range(start_cell + 1, end_cell):
            write(cell, lane, HOLD_BODY)
        write(end_cell, lane, HOLD_END)

    full[max_cell + 1:] = PAD
    tokens = full.reshape(-1, L, K)
    metas = []
    for chunk_idx in range(tokens.shape[0]):
        start_cell = chunk_idx * L - cell_offset
        beat_len_ms = (
            grid.time_from_cell(start_cell + L, D)
            - grid.time_from_cell(start_cell, D)
        ) / BEATS_PER_CHUNK
        metas.append(ChunkMeta(
            chunk_idx=chunk_idx,
            start_cell=start_cell,
            cell_offset=cell_offset,
            beat_len_ms=float(beat_len_ms),
            sr=sr,
            timing_points=list(chart.timing_points),
        ))
    return tokens, metas, EncodeStats(
        len(chart.notes), n_collisions, n_demoted_holds, lane_hist
    )


def decode(tokens: np.ndarray, metas: list[ChunkMeta]) -> Chart:
    raise NotImplementedError    # 다음 주 수요일 블록


def to_ascii(chunk: np.ndarray, n_rows: int = 96) -> str:
    """한 줄 = 칸, 열 = 레인. 기호: . o [ | ] #  (empty tap start body end pad)"""
    sym = ".o[|]?#"
    return "\n".join(f"{i:4d} " + "".join(sym[v] for v in row)
                     for i, row in enumerate(chunk[:n_rows]))


if __name__ == "__main__":
    # .osu 하나 → parse → encode → 첫 청크 ASCII + stats 출력
    # 출력의 앞 두 마디를 .osu HitObjects 첫 줄들과 ms 기준으로 손으로 대조
    ...