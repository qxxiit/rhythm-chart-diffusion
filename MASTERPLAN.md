# UGRP 2026: ActFusion-inspired Diffusion for Rhythm Game Chart Generation

> **마스터플랜 v2** · 2026.05.21 ~ 2027.01.31
> 팀 공유용 / 본 문서는 변동 가능. 주요 체크포인트 결과에 따라 재조정.

---

## 0. 한눈에 보기

### 프로젝트 한 문장
음원(.mp3/.wav)을 입력받아 4~8키 리듬 게임 채보를 자동 생성하는 ActFusion 영감 기반 audio-conditioned discrete diffusion 모델을 개발하고, Yi et al. (ISMIR 2023)의 autoregressive Transformer baseline과 정량 비교한다.

### 진짜 데드라인
- **2026년 11월 말**: POSTECH CVLab(조민수 교수님) 27년 겨울학기 연참 지원 (정확한 마감일 별도 확인 필요)
- **2027년 1월 말**: 외부 워크숍 투고 (ISMIR LBD / AIIDE / IEEE CoG 등) + UGRP 공식 마무리

### 1순위 / 양보 가능한 것
| 순위 | 항목 | 양보 가능성 |
|---|---|---|
| 1 | UGRP 프로젝트 (연참 목표) | 양보 불가 |
| 2 | 학교 5전공 (학점) | 양보 불가 |
| 3 | 8개월 DL 자율 학습 계획 | 축소/연기 가능 |

→ **자율 학습 계획에서 Convex Optimization, Reinforcement Learning은 27년 봄/여름으로 연기.**
→ **2학기에는 컴퓨터비전개론/인공지능/고급확률이론이 자율 학습의 핵심을 대체.**

---

## 1. 팀 및 역할

| 이름 | 학과 | 주 역할 | 보조 역할 |
|---|---|---|---|
| 문정현 (팀장) | 무은재학부 | 모델 (Transformer/Diffusion) | 실험 설계, paper 작성 |
| 지현우 | 무은재학부 | 데이터 파이프라인, Unity 통합 | 평가 진행 |
| 표영복 | 화학과 | 실험 운영, 평가 지표 | 데이터 보조 |

> 역할은 가이드일 뿐, 매 phase 시작 시 팀 미팅에서 재조정.

**멘토**: 김원화 교수님 (컴퓨터공학과, ML/CV/Medical Imaging)

**소통 채널**:
- Slack / KakaoTalk: 일상 소통
- Notion: 회의록, 진행 상황
- GitHub: 코드, issue, PR
- 주간 미팅: 매주 1회 (요일/시간 킥오프에서 확정)
- 멘토 미팅: 월 1회 (메일로 일정 잡기)

---

## 2. 기술 스택

| 영역 | 도구 |
|---|---|
| 모델/학습 | PyTorch, Hydra (config), wandb (logging) |
| 데이터 | osu! API, librosa, NumPy |
| 평가 | 자체 구현 (F1, mAP@tIoU, n-gram diversity) |
| GPU | vast.ai (RTX 4090 / A100 80GB) |
| 게임 통합 | Unity (기존 자체 개발 4~8키 게임) |
| 협업 | GitHub, Notion, Slack |

---

## 3. 데이터셋

| 데이터셋 | 용도 | 규모 |
|---|---|---|
| osu!mania 4K ranked | Phase 1~2 main | 약 10,000~20,000곡 |
| osu!mania 5~8K ranked | Phase 3 multi-key | 키별 수천 곡 |
| Fraxtil, ITG (StepMania) | 보조 비교 (선택적) | 수백~수천 곡 |

**수집 채널**: osu! 공식 API, NerinyanMirror 또는 Beatconnect mirror.
**저장**: 4TB 외장 SSD, GitHub LFS는 사용 안 함.

---

## 4. 전체 일정 (Bird's-eye View)

```
2026
05      06        07         08         09         10         11         12      | 2027.01
 |휴면|준비|여행|----Phase 1----|--Phase 2--|--Phase 3--|Phase4|Phase 5|Phase 6|
       5d   16d    6주            7주          6.5주     2주    4주     4주
```

| 시기 | Phase | 산출물 |
|---|---|---|
| 5/21~6/11 | 휴면기 (시험 + 다른 일정) | 팀 킥오프, 멘토 메일 |
| 6/12~6/16 | 준비 (여행 전) | 데이터 수집 가동, 환경 셋업 |
| 6/17~7/2 | 가족 여행 | (작업 없음, 데이터 수집 자동) |
| 7/3~8/11 | **Phase 1 — Baseline 재현** | Yi 2023 재현 모델 + F1 |
| 8/12~9/30 | **Phase 2 — Diffusion 구현** | ActFusion-inspired diffusion 모델 |
| 10/1~11/15 | **Phase 3 — Ablation + Multi-key** | 정량 결과 표 모음 |
| 11/16~11/30 | **Phase 4 — Demo + 사용자 평가 + 연참 지원** | Demo 영상, 평가 결과, 연참 지원 |
| 12/1~12/31 | **Phase 5 — Paper draft** | Mini-paper draft |
| 27/1/1~1/31 | **Phase 6 — 투고 + 마무리** | 워크숍 투고, public release |

---

## 5. 결정적 체크포인트 (7개)

이 7개의 milestone은 반드시 확보. 못 하면 후속 단계 조정.

| 날짜 | 체크포인트 | 실패 시 대응 |
|---|---|---|
| **6/16** | 데이터 수집 가동 + 환경 셋업 완료 | 여행 중 자동 수집 안 됨, 7월 일주일 손실 |
| **8/11** | Baseline F1 재현 (Yi 2023 ±2%p 이내) | Phase 2 시작 1~2주 지연 |
| **9/30** | Diffusion 모델 학습/평가, Track A/B 결정 | Track A 전환 (Diffusion 포기, AR 위주) |
| **11/15** | 모든 정량 결과 확보 (ablation + multi-key) | 일부 ablation 제외, 핵심 표만 유지 |
| **11/30** | 연참 지원 완료 (지원서 + demo + repo) | 27년 여름학기로 연참 미루기 |
| **12/31** | UGRP 학교 최종 보고서 + paper draft 80% | UGRP 마감 우선, 투고는 1월에 |
| **1/31** | 워크숍 투고 + 공개 release | 투고는 다음 deadline으로, release만 진행 |

---

## 6. Phase별 상세 일정

### Phase 0 — 휴면기 (5/21 ~ 6/11, 3주)

**상황**: 본인 시험 등으로 시간 못 냄. 작업 최소화.

**This Phase Goals**: 팀 인프라 셋업, 멘토 동의, 데이터 수집 준비.

| 작업 | 담당 | 예상 시간 | 상태 |
|---|---|---|---|
| 팀 킥오프 미팅 1회 (방향 공유, 역할 분담) | 전원 | 2h | ☐ |
| GitHub repo 생성, 디렉토리 구조 셋업 | 위임 가능 | 1h | ☐ |
| Notion 워크스페이스 셋업 | 위임 가능 | 30min | ☐ |
| Slack/KakaoTalk 채널 정함 | 위임 가능 | 10min | ☐ |
| 멘토 교수님께 방향 변경 메일 + 1차 미팅 일정 | 본인 | 30min | ☐ |
| vast.ai 계정 + 결제 수단 등록 | 위임 가능 | 30min | ☐ |
| osu! API 키 신청 | 위임 가능 | 10min | ☐ |
| Colab Pro / Pro+ 결제 검토 (선택) | 위임 가능 | — | ☐ |

**본인 부담**: 약 2.5h. 시험 영향 거의 없음.

---

### Phase 0.5 — 여행 전 준비 (6/12 ~ 6/16, 5일)

**상황**: 본인 시험 끝, 여행 전. 5일 동안 집중 작업.

**This Phase Goals**: 데이터 수집을 백그라운드로 가동, 환경 셋업 완료.

| 작업 | 담당 | 예상 시간 | 상태 |
|---|---|---|---|
| 데이터 수집 스크립트 v1 작성 (osu! API, ranked 4K 메타데이터) | 데이터 담당 | 4~6h | ☐ |
| 채보 + 음원 다운로드 스크립트 | 데이터 담당 | 3~4h | ☐ |
| 수집 시작 (백그라운드로 클라우드 또는 한 팀원 PC에서 가동) | 데이터 담당 | 1h | ☐ |
| 본인 노트북에 PyTorch + CUDA/MPS 환경 셋업 | 전원 | 2h | ☐ |
| Yi 2023 논문 본인 1차 정독 | 본인 | 2~3h | ☐ |
| 팀원 한 명이 Donahue 2017 (DDC) 정독 — 분야 기초 | 1명 | 2~3h | ☐ |

**본인 부담**: 약 4~5h (정독 + 환경 셋업).

**🎯 체크포인트 6/16**: 데이터 수집이 자동으로 돌아가고 있어야 함. 본인 환경에서 PyTorch import + GPU 인식 확인.

---

### Phase 0.7 — 가족 여행 (6/17 ~ 7/2, 16일)

**작업 없음.** 데이터 수집만 자동 가동.

**선택적 (무리하지 말 것)**:
- 이동 중 Yi 2023 또는 ActFusion 가볍게 한 번 더 훑기 (총 1~2시간)
- 혁펜하임 Easy DL 한두 강의

**진짜로 쉬세요.** 7~8월 메인 작업에 체력 필요.

---

### Phase 1 — Baseline 재현 (7/3 ~ 8/11, 6주)

**상황**: 여행 복귀, 여름방학 전반부. 주당 30~35h 투입 가능.

**This Phase Goals**: Yi et al. (ISMIR 2023) Beat-Aligned Transformer를 osu!mania 4K에서 재현. F1 score를 논문 보고치 ±2%p 이내로.

#### Week 1~2 (7/3 ~ 7/14): 전처리 + 데이터 파이프라인

| 작업 | 담당 | 산출물 |
|---|---|---|
| 데이터 수집 결과 검수, 품질 필터링 (broken file 제거 등) | 데이터 | 정제된 데이터셋 |
| Log-Mel spectrogram 추출 (80 mel bins, FFT 512, hop = 1/48 beat) | 모델 | 전처리 모듈 |
| Beat alignment 전처리 구현 | 모델 | beat-aligned 데이터 |
| Chart tokenization 스키마 확정 + 구현 | 모델 | tokenizer 모듈 |
| PyTorch Dataset / DataLoader 클래스 | 데이터 | dataset.py |
| Train/val/test split (song-level, leakage 방지) | 데이터 | split 파일 |

#### Week 3~4 (7/15 ~ 7/28): 모델 구현 + 첫 학습

| 작업 | 담당 | 산출물 |
|---|---|---|
| Transformer encoder-decoder 구현 (3-layer, dim 256) | 모델 | model.py |
| 학습 loop, optimizer, scheduler | 모델 | train.py |
| wandb logging 셋업 | 모델 | wandb config |
| **단일 곡 overfit 테스트** (학습 코드 sanity check) | 모델 | overfit log |
| 첫 풀 학습 (vast.ai RTX 4090) | 모델 | checkpoint v1 |

#### Week 5~6 (7/29 ~ 8/11): 평가 + 디버깅 + 완료

| 작업 | 담당 | 산출물 |
|---|---|---|
| F1 metric (Donahue 2017 protocol, ±30ms tolerance) | 평가 | metrics.py |
| mAP@tIoU (TAL style) | 평가 | metrics.py |
| Pattern n-gram diversity | 평가 | metrics.py |
| Test set 평가, 논문 보고치와 비교 | 평가 | 비교 표 |
| F1 gap이 있으면 디버깅 (beat alignment, tokenization 의심) | 모델 | 수정된 코드 |
| **Phase 1 완료 보고서 (1~2 페이지)** | 전원 | report_p1.md |

**🎯 체크포인트 8/11**: Baseline F1이 논문 보고치 ±2%p 이내. 안 되면 1주 추가 디버깅.

**팀 회의**: 8월 11일 phase 1 review, Phase 2 준비도 점검.

---

### Phase 2 — ActFusion-inspired Diffusion (8/12 ~ 9/30, 7주)

**상황**: 여름방학 후반부 + 학기 시작 첫 4주. 8월은 주 35h, 9월부터 주 12~15h.

**This Phase Goals**: Discrete diffusion 기반 chart generation 구현, baseline과 정량 비교.

#### Week 1 (8/12 ~ 8/18): ActFusion 정독 1~2단계

- 본인 ActFusion 정독 가이드 1단계 (1시간 큰 그림) → 2단계 (사전지식 보충)
- 팀 세미나 1회: 한 명이 ActFusion 발표 (30~45분), 모두 질문
- D3PM 논문 *abstract + Figure 1*만 읽기 (사전 노출)

#### Week 2 (8/19 ~ 8/25): ActFusion 정독 3단계 + D3PM 보조

- Method Q1~Q6 답변 작성
- D3PM 정독 (discrete diffusion 수학적 기초)
- ActFusion 공식 코드 받아서 환경 셋업, 학습 1 iter 돌리기

#### Week 3 (8/26 ~ 9/1): Diffusion 모델 설계

- Discrete diffusion forward process 수학적 정의 (우리 task)
- Denoiser architecture 설계 (audio cross-attention 포함)
- 코드 skeleton 작성 (model_diffusion.py)
- 설계 문서 작성, 팀 review

#### Week 4 (9/2 ~ 9/15): Diffusion 구현 — **학기 시작, 속도 감소**

- Diffusion 모델 구현 완료
- 단일 곡 overfit 테스트 (sanity check)
- Sampling/inference 코드 구현
- 작은 데이터셋으로 디버깅

#### Week 5~6 (9/16 ~ 9/30): 첫 풀 학습 + Track 결정

- 첫 풀 학습 (vast.ai A100 80GB 권장)
- 학습 곡선 모니터링, 첫 생성 결과 정성적 확인
- Test set 평가, baseline 대비 F1/mAP 비교
- **🎯 체크포인트 9/30**: Track A/B 결정 팀 미팅

**Track A (Diffusion이 baseline 이하)**: Diffusion 포기, Autoregressive baseline + multi-key + 풍부한 ablation으로 전환. Diffusion은 "preliminary experiment" 1페이지로 포함.

**Track B (Diffusion이 baseline 이상)**: 메인 contribution으로 진행. Phase 3 ablation을 diffusion 중심으로.

---

### Phase 3 — Ablation + Multi-key Generalization (10/1 ~ 11/15, 6.5주)

**상황**: 학기 중. 주 12~15h. 중간고사(10월 중하순) 영향.

**This Phase Goals**: 풍부한 ablation으로 "왜 작동하는가" 답변 가능하게, multi-key 통합.

#### Week 1~2 (10/1 ~ 10/14): Ablation A — Diffusion design

| 실험 | 비교 |
|---|---|
| Noise schedule | linear vs cosine vs mutual-info |
| Sampling step | 50 / 100 / 250 |
| Masking strategy | absorbing vs random |

- 실험을 Hydra config로 명세, 한 번에 큐잉
- 학기 중엔 코드 작성보다 결과 모니터링 + 분석

#### Week 3~4 (10/15 ~ 10/28): Ablation B — Conditioning (중간고사 영향)

| 실험 | 비교 |
|---|---|
| Conditioning level | audio-only vs +difficulty vs +CFG |
| Guidance scale | 1.0 / 1.5 / 3.0 / 5.0 |

- 중간고사 주는 실험만 큐잉, 코딩 최소화

#### Week 5~6 (10/29 ~ 11/15): Ablation C + Multi-key

| 실험 | 비교 |
|---|---|
| Anticipative masking | ON / OFF |
| Multi-key strategy | joint training vs 4K pretrain + finetune |
| Per-key 분석 | F1, lane balance entropy by key mode |

- 시각화: loss curve, attention map, 생성 샘플
- **🎯 체크포인트 11/15**: 모든 정량 결과 표/그래프로 정리, paper에 들어갈 figure 후보 확정

---

### Phase 4 — Demo + 사용자 평가 + 연참 지원 (11/16 ~ 11/30, 2주)

**상황**: 학기 중, 연참 마감 임박. **빡센 2주.**

#### Week 1 (11/16 ~ 11/22): Unity 통합 + Demo + 사용자 평가 시작

| 작업 | 담당 |
|---|---|
| Python 모델 → Unity JSON 변환 모듈 | Unity 담당 |
| 다양한 장르 음원에서 inference 테스트 (K-Pop, J-Pop, EDM, 클래식) | 전원 |
| Demo 영상 촬영 (1~2분, 자동 채보 → Unity 게임플레이) | Unity 담당 |
| 사용자 평가 설계 (Likert 설문, 곡 선정, 익명화 절차) | 평가 담당 |
| 평가 진행 (15~20명 모집, 블라인드 비교: baseline vs diffusion vs 인간) | 평가 담당 |

#### Week 2 (11/23 ~ 11/30): Paper draft 50% + 연참 지원

| 작업 | 담당 |
|---|---|
| Paper draft Abstract, Introduction, Related Work | 본인 |
| Method 섹션 초안 | 본인 |
| 사용자 평가 통계 분석 | 평가 |
| GitHub repo 정리 (README, reproduce 명령어, wandb public link) | 모델 |
| 연참 지원서 작성 (paper draft에서 narrative 발췌, demo + repo link 첨부) | 본인 |
| **🎯 연참 지원 제출** | 본인 |

---

### Phase 5 — Paper Draft 완성 (12/1 ~ 12/31, 4주)

**상황**: 학기 말 + 기말고사. 12월 중순 시험 기간 보호.

| 주차 | 작업 |
|---|---|
| 12/1 ~ 12/7 | Paper draft 80%: Experiments, Discussion, Conclusion |
| 12/8 ~ 12/14 | 멘토 교수님께 검토 요청, 1차 피드백 반영 |
| 12/15 ~ 12/24 (기말고사) | 시험 집중, paper는 마이너 수정만 |
| 12/25 ~ 12/31 | 종강 후: paper 최종 다듬기, UGRP 학교 양식 최종 보고서 작성 |

**🎯 체크포인트 12/31**: UGRP 학교 보고서 제출, paper draft 80~90%.

---

### Phase 6 — 투고 + 공개 (1/1 ~ 1/31, 4주)

**상황**: 겨울방학, 자유로움. 연참 결과가 나왔을 수도 있음.

| 주차 | 작업 |
|---|---|
| 1/1 ~ 1/14 | Paper 영문화, camera-ready 형식 변환 |
| 1/15 ~ 1/24 | 외부 review 요청 (멘토, 가능하면 외부 인사) → 수정 |
| 1/25 ~ 1/31 | **워크숍 투고**, GitHub v1.0 release, 데모 페이지/블로그 글 (선택), UGRP 공식 마무리 |

**투고 후보**:
- ISMIR 2027 Late-Breaking Demo (마감 ~ 27년 7월)
- AIIDE 2027 (마감 ~ 27년 4~5월)
- IEEE CoG 2027 (마감 ~ 27년 2~3월)
- 또는 27년 가능한 가까운 워크숍

---

## 7. 학습 계획 조정 (자율 학습)

8개월 학습 계획은 다음과 같이 압축:

| 시기 | 학습 항목 | 처리 |
|---|---|---|
| 5월 ~ 6월 중순 | Phase 1 시작 (Easy DL, 신호및시스템) | 시간 거의 없음, 여행 후 가볍게 |
| 7~8월 (여름방학) | Phase 2 (혁펜하임 DL + PyTorch + CS231n 전반부) | **PyTorch 실습은 UGRP와 동시 진행 (시너지)**. CS231n 전반부는 가능한 만큼만. 혁펜하임 DL은 영상이라도. |
| 9~12월 (2학기) | 원래 Phase 3 (CS231n 후반부, Convex Opt) | **컴퓨터비전개론 + 인공지능 + 고급확률이론으로 자동 대체**. Convex Opt **연기**. |
| 27년 1~2월 (겨울방학) | 원래 Phase 4 (RL) | **연기 → 27년 봄/여름** |
| 27년 봄/여름 | 연기된 Convex Opt + RL | 그때 재계획 |

**5번째 과목 결정**: 가벼운 CS 전공 (확통 X). 이미 고급확률이론이 있어서 수학 부담 분산.

---

## 8. 예산

| 항목 | 금액 | 비고 |
|---|---|---|
| 클라우드 GPU (vast.ai) | 1,000,000원 | A100/H100, 학습 + ablation 약 200~400시간 |
| 사용자 평가 사례비 | 225,000원 | 15명 × 15,000원 |
| 외장 SSD 4TB | 200,000원 | 데이터셋 + 체크포인트 |
| 포스터/인쇄 | 120,000원 | 발표자료 |
| 참고도서 | 100,000원 | DL/MIR |
| **합계** | **1,645,000원** | UGRP 200만원 한도 내 |

GPU 회계 처리(시험분석료 vs 용역비)는 교육혁신센터 / 멘토 교수님께 사전 확인.

---

## 9. 리스크와 대응

| 리스크 | 확률 | 대응 |
|---|---|---|
| 8/11 baseline 재현 실패 | 25% | 1주 추가 디버깅, Phase 2 1~2주 지연 |
| 9/30 diffusion 학습 실패 | 50% | Track A 전환 (AR + ablation 강화) |
| 11/15 ablation 미완성 | 40% | 핵심 ablation만 유지, 추가 실험 제외 |
| 11/30 사용자 평가 인원 부족 | 30% | 10명으로 축소, 정성 피드백 강화 |
| 팀원 한 명 중도 이탈 | 20% | 모듈식 역할 분리로 인계 가능하게 매주 commit |
| 본인 학기 중 번아웃 | 30% | UGRP 주말로만 제한, 평일은 학교 우선 |

---

## 10. 즉시 할 일 (this week)

- [ ] **5번째 과목을 가벼운 CS로 확정** (확통 X)
- [ ] **팀 킥오프 미팅 일정 잡기** (6/11 전)
- [ ] **멘토 교수님께 방향 변경 메일** (ActFusion-based diffusion)
- [ ] **GitHub repo + Notion 워크스페이스 셋업** (팀원에게 위임)
- [ ] **vast.ai 계정 + 결제 수단** (팀원에게 위임)
- [ ] **osu! API 키 신청** (며칠 걸림, 위임 가능)
- [ ] **조민수 교수님 연구실 27년 겨울 연참 정확한 마감일 확인** (CVLab 홈페이지 / 메일 문의)

---

## 11. 참고문헌

- [1] C. Donahue, Z. C. Lipton, J. McAuley. *Dance Dance Convolution.* ICML 2017.
- [2] A. Takada et al. *GenéLive! Generating Rhythm Actions in Love Live!* AAAI 2023.
- [3] J. Yi, S. Lee, K. Lee. *Beat-Aligned Spectrogram-to-Sequence Generation of Rhythm-Game Charts.* arXiv:2311.13687, 2023.
- [4] D. Gong, S. Kwak, M. Cho. *ActFusion: a Unified Diffusion Model for Action Segmentation and Anticipation.* NeurIPS 2024.
- [5] J. Austin et al. *Structured Denoising Diffusion Models in Discrete State-Spaces.* (D3PM) NeurIPS 2021.
- [6] E. Halina, M. Guzdial. *TaikoNation.* FDG 2021.

**저장소 링크 (TBD)**: GitHub / Notion / wandb

---

*Last updated: 2026.05.21 · v2*
