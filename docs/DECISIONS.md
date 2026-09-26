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
| 2026-09-26 | Onset F1 uses greedy one-to-one matching (closest pairs first, same lane unless stated) at ±20 and ±50 ms, and every F1 is reported next to the tokenizer's own ceiling F1(decode(encode(human)), human) | Greedy as in the design doc; the ceiling separates model error from quantization |
| 2026-09-26 | SR for evaluation is computed locally with rosu-pp-py 4.0.2 (pinned) | Generated charts have no API SR. SR error is reported against the API SR (the training condition) and against the local SR of the human chart (same calculator on both sides) until check_sr.py shows how far the two calculators agree |
| 2026-09-26 | Row 0 of every chart is a bar line: `cell_offset` is rounded up to a multiple of 48 cells (4/4) | osu! opens a measure at every red line. Without this, the 3,159 charts with notes before the first red line started chunks mid-bar, and an audio-based token range would do so for every chart whose first red line is after 0 ms |
| 2026-09-26 | Split by `md5(audio_key)` into buckets 0-89 / 90-94 / 95-99 (train / val / test); cache files named by `md5(path)[:16]` (`scripts/build_manifest.py`) | A pure function of the song: re-uploads and all difficulties stay together, and the split never moves when charts are added |
| 2026-09-26 | Denoiser deviations from §4.10: 5-class head; learned positional embedding on the audio memory; pre-LN; PAD rows are not attention keys; loss divided by N = 1,536 | Same distributions as the doc's -inf masking; cross-attention cannot align frames to cells without positions; standard stable residual form; PAD must not be attended; the loss then reads as a per-position NLL bound |
| 2026-09-26 | Audio memory row i is also added to cell i (`audio_add`, on by default); cross-attention kept for context | Cell i and frames 4i..4i+3 are aligned by construction. On a toy set whose "audio" spells out the answer, a 2-layer model with cross-attention only was at masked CE 0.72 after 600 steps; with the addition, 0.09 after 200. `--no-audio-add` gives the doc's model (실험 큐 5) |
| 2026-09-26 | Audio frames sit on the token cells: frame r*c+j = t(c) + (j/r)(t(c+1) - t(c)), via `BeatGrid.frame_times`; `time_from_cell` rejects fractional cells | A separately re-originated 1/48 grid drifts from the 1/12 cells after an off-beat red line (cell_starts(48) != 4 * cell_starts(12)); fractional cells were truncated silently |

260806 "방학 스프린트 중엔 conda base 사용, 환경 정식화(environment.yml 갱신)는 vast.ai 세팅 시점(8월 중순)에"

260808 다운로더의 완료 판정은 .osz가 아니라 data/raw/{sid} 기준 — 추출기가 .osz를 삭제하므로.
