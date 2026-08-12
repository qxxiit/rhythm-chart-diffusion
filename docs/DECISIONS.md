# Decision Log

> One-line decisions. When something non-obvious is chosen, write it down.

| Date | Decision | Reason |
|------|----------|--------|
| 2026-05-XX | Baseline: Yi et al. (ISMIR 2023) Beat-Aligned Transformer | Most recent, clean reproducible target |
| 2026-05-XX | Primary dataset: osu!mania 4K ranked | Largest quality-controlled corpus, format simplest |
| 2026-05-XX | 5th course of 2nd semester: light CS elective (not probability) | UGRP is top priority; math load already covered by 'Advanced Probability for ML' |

260806 "방학 스프린트 중엔 conda base 사용, 환경 정식화(environment.yml 갱신)는 vast.ai 세팅 시점(8월 중순)에"

260808 다운로더의 완료 판정은 .osz가 아니라 data/raw/{sid} 기준 — 추출기가 .osz를 삭제하므로.
