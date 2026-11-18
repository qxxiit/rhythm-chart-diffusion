# Experiment log

> The entries live in one file a month under [`docs/experiments/`](experiments/), oldest first;
> this page is their index. Each run script is in [`experiments/runs/`](../experiments/runs/)
> with a one-line outcome; the decisions are in [DECISIONS.md](DECISIONS.md).

## Models

| Date | Run ID | Phase | Model | Key Config | Val F1 | Test F1 | wandb | Notes |
|------|--------|-------|-------|------------|--------|---------|-------|-------|
| 2026-11-16 | stems-v2 | 2 | D-32, its audio input the mix and htdemucs's 4 stems (144 channels) | full-v4's recipe exactly (60k steps, lr 3e-4), `--stems`, MPS, 3.6 h | F1@50 0.377 (raw draw, first 120 val songs) | - | - | val CE 0.07269 (full-v4 0.07396, -1.7%); raw draws level with full-v4's (2026-11-16 (2)); not taken |
| 2026-11-15 | stems-v1 / ft-v1 | 2 | full-v4 widened to the stems with zero weights / without | `--init full-v4 (--stems)`, 20k steps at lr 1e-4 | - | - | - | val CE 0.07209 / 0.07260: the gain mostly the extra training (2026-11-15 (1)); not taken |
| 2026-10-06 | full-v4 | 2 | D-32 + the chart stats (long-note share, jack and trill rate) as inputs | full-v1's recipe, `--row-mask 0.5 --chart-stats`, 60k steps | F1@50 0.376 (raw draw, first 120 val songs) | - | - | val CE 0.07396; **the system's model** (DECISIONS 2026-11-17) |
| 2026-10-04 | full-v2 | 2 | D-32 + genre / mapper embeddings, 6.93M | full-v1's recipe, labels dropped 0.1 / 0.1, `--style data/style.csv`, MPS, ~3.5 h | F1@50 0.377 (240 val songs, fwd + ref2, own labels) | - | - | val CE 0.0732 with labels = 0.0732 without (full-v1 0.0717): the labels go unused; differences to full-v1 not attributable to them |
| 2026-10-04 | full-v1 + lane passes + bar copies | 2 | D-32, 6.87M | `--copy-bias 0` on the saved fwd + ref2 charts (no training) | F1@50 0.375 (+0.001) | - | - | whole-bar copies about 15% of bars (human 14.5%), rho_in 0.219 → 0.231 (human 0.228) |
| 2026-10-03 | full-v1 + lane passes | 2 | D-32, 6.87M, audio_add | sampler: random T128, then forward_lanes and refine_lanes 2 at lane temperature 0.5 (no training) | F1@50 0.373 (240 val songs) | - | - | pattern coverage 0.047 (human 0.042), runs 7.6 (7.5), motion_pred 0.122 (0.132); jacks a quarter of the human rate |
| 2026-09-29 | full-v1 | 2 | D-32, 6.87M, audio_add | all train songs (16,410 charts, 218,142 chunks), 60k steps (4.4 epochs), batch 16, MPS, ~3 h | F1@50 0.373 (240 val songs, random T128) | - | - | val CE 0.072; vs subset: F1 same, SR error 0.375→0.236, rho at the human level, pattern coverage still 2x chance; sampler fixed to random T128 |
| 2026-09-27 | subset600-v1 | 2 | D-32, 6.87M, audio_add | 600 train songs (2,110 charts, 28,004 chunks), 20k steps, batch 16, MPS, 1 h | F1@50 0.37 (240 val songs, every sampler) | - | - | val CE 0.087, no overfitting; sampler moves density 0.81-1.16; without audio onset CE +79% |
| 2026-09-26 | 20260926-060904 | 2 | D-32, 6.87M, audio_add | `--overfit 10`, fake mel, 3000 steps, batch 16, MPS | - | - | - | Plumbing check; sampler order matters (see Phase 2) |
| TBD  | -      | 1     | -     | -          | -      | -       | -     | First baseline run |

## How to log

1. Before a run: an entry at the end of this month's file, `### YYYY-MM-DD (n) · what is asked (predictions)`,
   with why, the run script (`experiments/runs/run_MMDD.sh`) and the predictions, written before it runs.
2. After it: the results under the same entry (`Results, run_MMDD (...)`, each prediction marked ✓ / ✗),
   a reading, `(predictions, results)` in its title, a row in DECISIONS.md if something changes,
   and its line in `experiments/runs/README.md`.
3. A new model: a row in the table above.
4. Everything aggregate: per-song rows of playtest songs are not read (EVALUATION.md).

Phase 1 (the AR baselines) is on hold since 2026-09-30 (MASTERPLAN).

## Entries


### [September 2026](experiments/2026-09.md)

- 2026-09-26 · Overfit check and sampler order (fake mel)
- 2026-09-27 · Audio alignment (data check, before training)
- 2026-09-27 · Training (Try #1)
- 2026-09-27 · Sampling settings (queue 9, first pass)
- 2026-09-28 · Training on the full set (full-v1)
- 2026-09-29 · Audio alignment on the full cache
- 2026-09-29 · Sampling settings (second pass, full-v1)
- 2026-09-30 · Sampling settings (third pass: T = 256)
- 2026-09-28 · Audio ablation (subset600-v1)
- 2026-09-29 · Audio ablation (full-v1)

### [October 2026](experiments/2026-10.md)

- 2026-10-01 · Patterns in a playtest song (PT08, xi - Rebellion Trigger)
- 2026-10-01 · Long notes in human charts (data check)
- 2026-10-02 · Lane choice: probe, refinement and a replicate (full-v1)
- 2026-10-02 · Lane passes on PT08 (playtest check, one song)
- 2026-10-02/03 · Lane passes on 240 val songs (overnight)
- 2026-10-03 · Jack bias (60 val songs)
- 2026-10-03 · Why trills break and bars do not repeat (240 val songs, rescored)
- 2026-10-03 · Bar copies (`sampler.copy_bars`), first check (12 val songs, CPU)
- 2026-10-03 · full-v2: genre and mapper as inputs (prediction, written before the run)
- 2026-10-04 · Blind pairs playtest and what it found (16 songs; 240 val songs measured)
- 2026-10-05 · Long-note passes, loudness bias, full-v3, and why the style is averaged (240 val songs)
- 2026-10-06 · full-v4: chart stats as inputs; the quiet-side loudness bias (240 / 60 val songs)
- 2026-10-06/07 · Lane guidance towards the chart stats (full-v4, first 60 val songs)
- 2026-10-06 · Playtest of full-v4 (10 of 16 songs) and the quiet sections
- 2026-10-06/07 · The onset gate and a stronger quiet-bar bias (60 val songs), and a fix
- 2026-10-07 · The onset gate on note starts only; the candidate on 240 val songs
- 2026-10-07 night · Choices for a whole bar: rests, long-note bars, the SR offset (predictions)
- 2026-10-08 · The gate tapered by SR, stream holes, rests and long-note bars without a chart (predictions)
- 2026-10-08 night · The path of training on the loss surface (notebook 6.5 at 6.9M parameters; predictions)
- 2026-10-10 · Rhythms carried over, long notes among taps, the long-note share from the audio (predictions)
- 2026-10-11 · Copies from further back; the candidate on a tapered base (predictions)
- 2026-10-11 · Long-note bars in the rhythm carry (predictions)
- 2026-10-11 night · The carry's biases under keep_notes; sources with no fewer onset rows (predictions)
- 2026-10-12 · What players praise and blame, measured; sections that come back, jacks in chords, chords over the cap (predictions)
- 2026-10-12 · Sections either way; jack taps moved, by grade (predictions)
- 2026-10-12 · Bars that sound the same; chords on the strong sounds (predictions)
- 2026-10-12 · Long notes up to the share asked; light bars in Easy and Normal (predictions)
- 2026-10-12 · Repeats thinned alike; jack windows in Insane and Expert (predictions)
- 2026-10-12 · The default to play: a last check and the playtest of 10-12 (predictions)
- 2026-10-15 · Where the notes are; rows asked again (predictions)
- 2026-10-15 · A judge: telling the charts from the human ones, 8 bars at a time (predictions)

### [November 2026](experiments/2026-11.md)

- 2026-11-05 · The checks before a number counts; F without each family; M3; the sampler's settings (predictions)
- 2026-11-07 · run_1105 meta and the judges of 10-15: results; the diagnostics as a judge
- 2026-11-07 (2) · The families, M3 and the sweep under the judges; a judge is read for its own run; run_1107 (predictions)
- 2026-11-08 · run_1107: the control, the sweep, the ladder of own judges; cosine on all songs (predictions)
- 2026-11-08 (2) · run_1108: cosine on all songs; what the judge reads; views (predictions)
- 2026-11-08 (3) · run_1109: the views, and the SR they were shown with; again with the SR as shown (predictions)
- 2026-11-08 (4) · run_1110: what the judge reads, with the SR as shown; the rhythm against the music; alike bars (predictions)
- 2026-11-08 (5) · Windows chosen by a judge (predictions)
- 2026-11-08 (6) · run_1111 and run_1112: alike bars at 6 nats; windows chosen by a judge; 8 draws on all songs (predictions)
- 2026-11-09 (1) · run_1113: 8 draws on all songs; the lanes chosen too (predictions)
- 2026-11-09 (2) · run_1114: the lanes chosen by a judge; what is left of the chosen run's tell
- 2026-11-09 (3) · The lanes' and the long notes' settings on the chosen draw (predictions)
- 2026-11-09 (4) · run_1115: the lanes' and the long notes' settings, none taken
- 2026-11-09 (5) · The lanes at temperature 1 in the taps view; the long notes drawn by the model (predictions)
- 2026-11-09 (6) · run_1116: the hold bias at 0.5 overshoots; at the human share (predictions)
- 2026-11-09 (7) · The windows chosen by a judge that sees the long notes (predictions)
- 2026-11-10 (1) · run_1118 and run_1117: the long notes at the human share, chosen with them; the system on 240 songs (predictions)
- 2026-11-11 (1) · Where the system's tell is left; its lanes at 0.7 and its hold bias at 0.1 (predictions)
- 2026-11-11 (2) · run_1119: the system on 240 songs, the playtest gate passed; run_1120 results
- 2026-11-14 (1) · The windows chosen by the rhythm judge and the holds judge together (predictions)
- 2026-11-14 (2) · The pilot's tells: empty stretches, mirrored answers; the rhythm drawn beat by beat (predictions)
- 2026-11-15 (1) · full-v4 taught the stems (predictions)
- 2026-11-15 (2) · The pilot's tells in the passes: no long empty stretch from the rest pass, a beat answered (predictions)
- 2026-11-16 (1) · The stems learned from the start; the answers made stricter (predictions, results)
- 2026-11-16 (2) · stems-v2 or the system around it: the judges' trunk, and a raw draw of each model (predictions, results)
- 2026-11-17 (1) · The patterns followed: a unit's first row drawn freely, the rest lower (predictions, results)
- 2026-11-17 (2) · Short long notes where the chart is busy become taps (predictions, results)
- 2026-11-17 (3) · Only the repeats become taps (predictions, results)
- 2026-11-18 (1) · Chords and releases where the human chart has them (predictions, results)
