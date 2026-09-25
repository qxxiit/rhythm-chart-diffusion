# Decision Log

> One-line decisions. When something non-obvious is chosen, write it down.

| Date | Decision | Reason |
|------|----------|--------|
| 2026-05-XX | Baseline: Yi et al. (ISMIR 2023) Beat-Aligned Transformer | Most recent, clean reproducible target |
| 2026-05-XX | Primary dataset: osu!mania 4K ranked | Largest quality-controlled corpus, format simplest |
| 2026-05-XX | 5th course of 2nd semester: light CS elective (not probability) | UGRP is top priority; math load already covered by 'Advanced Probability for ML' |
| 2026-09-25 | Re-originate the integer cell grid at each red line with recursively accumulated section offsets; boundary cells belong to the next section | Cumulative global cells reduce 1/12, 5 ms note coverage from 97.36% to 90.81%; recursive offsets avoid 3,420 observed one-cell gaps or overlaps |
| 2026-09-25 | Preserve cells before the first red line with a per-chart nonnegative `cell_offset` in `ChunkMeta` | 16,782 notes across 3,928 charts have negative global/cell coordinates under osu! timing extrapolation |
| 2026-09-26 | Onsets win cell collisions: an onset on the previous hold's end cell (same lane) shortens that hold by one cell, to a tap if it reaches zero length; any other onset inside the previous note drops the later note | F1 scores onsets only (the release-aware mAP@tIoU is deferred). Without it the encoder lost the later onset and wrote hold_body after hold_end |
| 2026-09-26 | Audio frames sit on the token cells: frame r*c+j = t(c) + (j/r)(t(c+1) - t(c)), via `BeatGrid.frame_times`; `time_from_cell` rejects fractional cells | A separately re-originated 1/48 grid drifts from the 1/12 cells after an off-beat red line (cell_starts(48) != 4 * cell_starts(12)); fractional cells were truncated silently |

260806 "방학 스프린트 중엔 conda base 사용, 환경 정식화(environment.yml 갱신)는 vast.ai 세팅 시점(8월 중순)에"

260808 다운로더의 완료 판정은 .osz가 아니라 data/raw/{sid} 기준 — 추출기가 .osz를 삭제하므로.
