# MASTERPLAN

**Project**: Audio-Conditioned Discrete Diffusion for Rhythm Game Chart Generation
**Program**: POSTECH UGRP 2026 (research track)
**Team**: 문정현 (lead), 지현우, 표영복 · **Mentor**: 김원화 교수님 (POSTECH CSE)
**Last updated**: 2026-10-02 (lane patterns: sampler first; §2)

---

## 1. Research question

Existing learning-based chart generators produce notes token by token (autoregressive). That makes two things hard:

- **Structural consistency** — a chorus that appears twice in a song should get corresponding patterns, but a left-to-right decoder has no mechanism to enforce that.
- **Controllability** — difficulty, pattern style, and per-section constraints are awkward to impose during sequential decoding.

We ask whether **discrete diffusion** — generating the entire note sequence at once and refining it over denoising steps, conditioned on audio — addresses these limitations, and at what cost in accuracy relative to a strong autoregressive baseline.

**Contributions targeted**
1. First application of discrete diffusion to audio-conditioned chart generation
2. Quantitative comparison against a reproduced Yi et al. (ISMIR 2023) baseline
3. A measurable definition of "structural consistency" for generated charts
4. Multi-key (4–8K) generalization within a single model

**Priority (team meeting, Sep 30).** The first goal is now a generator whose charts people want to play; the paper-style comparison comes second. In practice:
- Work that makes the generated charts better goes first: pattern clarity (2x its chance level against 6.7x in human charts), song-level repetition (e.g. inpainting a repeated section from its first occurrence), the ~5% extra notes, playability checks.
- The AR baselines (AR-4, AR-32) are on hold. AR-32 may come back as a check that diffusion is worth it.
- Ablations are run when they show how to make charts better, not to fill a table.
- What a good chart is gets settled by playing: blind playtests (`scripts/playtest_pack.py`) and the criteria each member brings to the Oct 7 meeting.

---

## 2. Current status

| Area | State |
|---|---|
| Data pipeline | ✅ Complete — 18,452 4K charts from 5,882 beatmapsets, 0 parse failures |
| Chart parser | ✅ Complete with tests and piano-roll visualization |
| Tokenization | ✅ 1/12-beat grid re-originated at red lines, validated on every chart (Sep 26) |
| Problem formulation & math | ✅ Design doc §1–3 and derivations (Sep 24) |
| Split, SR labels, filter | ✅ Split by song, local SR labels, 18,191 charts kept (Sep 26) |
| Audio pipeline | ✅ Decoded at osu!'s 0 ms, log-Mel per audio file, alignment checked on 2,000 songs (Sep 29) |
| Baseline model | ⏸ On hold (AR-4 / AR-32), generator first (Sep 30) |
| Diffusion model | 🟡 D-32 on the full training set (full-v1, val CE 0.072, Sep 29); sampler random T = 128, then a left-to-right lane pass and two refinement sweeps: pattern clarity at the human level (Oct 3). Gaps now: phrase repetition (whole-bar copies 1.4% vs 14.5%; bar copies in the sampler, Oct 3), long single-note trills, jacks. full-v2 with genre and mapper inputs training (Oct 3) |
| Generator | 🟡 Any audio file → playable .osz (`scripts/generate.py`, timing given or estimated; style by genre and mapper with full-v2); blind playtest packs (`scripts/playtest_pack.py`, in pairs from Oct 3) |
| Compute | 🟡 MacBook (MPS, ~86-91 chunks/s; full-v1 60k steps in ~3 h, 27.8 GB log-Mel cache); lab GPU expected ~October |

---

## 3. Constraints shaping the plan

**Compute.** No dedicated GPU until roughly October, when lab machines are expected to free up. Until then the priority is not large training runs but **designing experiments that are cheap to run and learning to train efficiently on a small budget** — subset iteration, mixed precision, gradient accumulation, small-model-first scaling, and one hypothesis per experiment rather than blind sweeps. A prioritized experiment queue should be ready the day GPUs become available.

**Budget.** The approved proposal (Apr 23) allocates ₩520,000 with no compute or storage line item. The revised plan's cloud-GPU and SSD figures were never approved, and external SSD prices have roughly doubled since the proposal was written. Any spend requires a budget amendment; lab hardware is the preferred path.

**Sequencing.** Per mentor guidance, **formulation precedes implementation**. Loss function, optimization scheme, preprocessing, and encoding are to be derived and written down before model code is committed.

---

## 4. Phases

### Phase 0 — Setup & data pipeline ✅ *(Jul–Aug 2026, complete)*
Repository structure, environment, osu! API v2 client, metadata collection, filtered download with mirror fallback and resume, extraction, catalog, chart parser with tests.

### Phase 1 — Problem formulation & baseline *(Aug–Sep 2026)*
- **Aug 12–29: design document** (see §5) — the current deliverable
- Tokenization scheme fixed and implemented, validated by round-trip reconstruction
- log-Mel preprocessing (80 mel bins, FFT window 512, hop = 1/48 beat), beat-aligned
- Baseline implementation: 3-layer encoder–decoder Transformer, hidden dim 256
- Evaluation code: F1, mAP@tIoU, pattern n-gram diversity — modularized for reuse in Phase 2
- Small-scale sanity checks (overfit on a subset); full baseline training deferred to GPU availability

### Phase 2 — Discrete diffusion *(Sep–Oct 2026)*
- Forward process: absorbing (mask) state diffusion over chart tokens, D3PM-style
- Reverse process: audio-conditioned Transformer denoiser, audio injected via cross-attention
- Loss derived from the discrete ELBO; training loop and sampling procedure
- Direct comparison against the Phase 1 baseline on identical data and metrics

### Phase 3 — Ablation & multi-key *(Oct–Nov 2026)*
- Diffusion design: noise schedule, number of steps, masking strategy
- Conditioning: audio-only / +difficulty embedding / classifier-free guidance
- Multi-key generalization — **requires collecting 5K–8K charts** (pipeline reusable, filter change only)
- Three seeds per configuration, mean and variance reported

### Phase 4 — Integration & evaluation *(Nov–Dec 2026)*
- Export to the team's Unity rhythm game JSON format, end-to-end verification
- Blind user study (15–20 participants): timing accuracy, pattern naturalness, difficulty appropriateness, overall enjoyment
- Inference time and beat-alignment accuracy across genres

### Phase 5 — Writing *(Dec 2026 – Jan 2027)*
Mini-paper (8–10 pages), code and checkpoint release, workshop submission considered (ISMIR Late-Breaking Demo, AIIDE).

---

## 5. Immediate deliverable — design document (due Aug 29)

1. **Problem definition** — formal input/output, chunking decision (whole song vs. bar-level)
2. **Encoding** — token vocabulary $\mathcal{V}$, sequence length $L$, representation of chords and hold notes, justified by dataset statistics
3. **Preprocessing** — log-Mel parameters, beat-aligned hop, audio–token alignment
4. **Forward process** — $q(x_t \mid x_{t-1})$, absorbing state, noise schedule
5. **Reverse process** — $p_\theta(x_{t-1} \mid x_t, a)$, denoiser architecture, cross-attention injection point
6. **Loss and optimization** — derivation, optimizer and schedule, low-budget training strategy
7. **Evaluation** — F1 and mAP@tIoU definitions, plus a proposed measurable definition of structural consistency
8. **Experiment queue** — prioritized runs to execute once GPUs are available

**2nd mentor meeting target: Aug 31.** Document sent Aug 29 (D-2).

---

## 6. Milestones

| Date | Milestone |
|---|---|
| Aug 11 | 1st mentor meeting — direction endorsed ✅ |
| Aug 15 | Tokenization scheme fixed |
| Aug 25 | Forward/reverse/loss derivable from blank paper |
| Aug 29 | Design document sent |
| Aug 31 | 2nd mentor meeting |
| Sep | Semester begins — baseline implementation |
| Oct | GPU access — experiment queue executed |
| Nov | Ablations complete, user study |
| Dec | Draft complete |
| Jan 2027 | Submission |

---

## 7. Known technical debt

| Item | Detail | When |
|---|---|---|
| Environment | Running on conda base; `environment.yml` not regenerated | Before GPU setup |
| Downloader completion check | Uses `.osz` presence, but the extractor deletes archives — re-running re-downloads everything. Should check `data/raw/{sid}` | Before next data run |
| Log buffering | No `flush()`; progress appears stale during long runs | Before next long run |
| badzip | `1011337.osz` not yet re-fetched | Any time |
| Parser validation doc | 10-chart visual comparison partially done | With design doc |

---

## 8. Working conventions

- Branch `dev`; conventional commit prefixes (`feat:`, `chore:`, `docs:`)
- Data never committed — `/data/` is gitignored; datasets are rebuilt from scripts
- Decisions recorded in `docs/DECISIONS.md`, experiments in `docs/EXPERIMENTS.md`
- Mentor meetings: roughly monthly during the break, cadence to be fixed at the 2nd meeting; email between meetings for blocking questions
- Every meeting followed by a summary email within 24 hours
