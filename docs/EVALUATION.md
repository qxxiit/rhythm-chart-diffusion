# Evaluation charter

> Written 2026-11-05, before run_1105 and before the 10-15 run_judge results were read.
> It changes only by a dated row in DECISIONS.md that says what changed and why; results
> obtained before the change stay reported under the old rule.

## The question

On the same song and grade, can a player tell a generated chart from the human one better
than chance?

Every number below either answers that (primary), stands in for it between playtests
(secondary) or says where the charts give themselves away (diagnostics). Nothing else
decides a default.

## Endpoints

1. **Primary: single-stimulus playtest.** One judgment = one chart played (human or AI, mixed
   50:50 in a random order; the player is not told the share), answered "human" or "AI" with a
   confidence 1-5. Reported: the accuracy with a 90% interval from a bootstrap over players and
   songs. Indistinguishable = the whole interval inside [40%, 60%]. About 400 judgments (20
   players x 20 charts) give a half-width near ±4 points before clustering, so the observed
   accuracy has to fall within about 44-56%; clustering by player and song narrows that.
   Fixed now, so that a miss still says something: the accuracy itself, and its difference
   from the control run on the same songs (an outside model such as Mapperatorinator, or M0).
   Packs: `scripts/playtest_pack.py --single --from-run R --songs 20 --versions 2` (one
   chart per song from the frozen system's saved run or the human one, each song human in one
   version and AI in the other; `--score` gives the accuracy with that interval, the share
   taken for human per source and an auc of the answers weighted by sureness). A pilot with
   2-3 players before the freeze tries the protocol and gives a first player-against-judge
   reading; it decides nothing.
   Pairs ("which of the two is human") stay as a format for quick checks: the share of AI
   charts taken for human with a Wilson interval, per player and per system, never pooled
   across players or systems (the 0/33 of 10-04..10-12 is one player and three systems).
2. **Secondary: the judge** (`scripts/judge.py`, a classifier two-sample test). auc and paired
   on held-out songs (5 folds by audio), by grade, with intervals over songs. It counts only
   after the judge checks below. The judge that picks charts (best-of-N, resampling) is never
   the one that scores them: another seed and fold split, and the last word is the playtest.
   A judge is read for the run it learned (added 11-07, DECISIONS): another run scored by it
   only shows how much of that run's tells it has, so runs are compared by their own judges'
   auc, or by the diagnostics judge, on the same songs.
3. **Diagnostics: where the charts give themselves away**, never a success criterion. Six to
   eight metrics, each reported as one number: the share of the AI charts inside the human
   band, the 10th-90th percentile of the human charts of the same grade (80% for human
   charts). Candidates: new_onsets; flow among chords (jumptrills) and single-note flow
   (stairs, trills); the longest run in one lane; 3+ chords; section_rhythm and section_copy;
   hold_share. A metric stays only if it passes the metric checks below. Also all of them at
   once (added 11-07, DECISIONS): `scripts/diag_judge.py`, a classifier on every diagnostic of
   the run's chart against the human chart's, 5 folds by song (auc, paired, the metrics it
   leans on), because a run can sit inside every band and still be told apart.

## Data

- **val, 240 songs:** development. Anything may be tuned on it.
- **test, 286 songs:** once, after the freeze (11-29), for the frozen systems only.
- **Oracle inputs** (from the human chart being scored: `--stats oracle`, `--hold-share
  oracle`, `--style oracle`) are reported only next to the same system without them
  (`--stats sample`, no hold share asked). The human chart's SR as the difficulty asked for
  is allowed: a user asks for a difficulty too.
- **Playtest songs:** songs not yet played stay blind. Only aggregate numbers are looked at,
  no per-song output (PT01-PT07, PT14, PT15 as of 11-05).

## Systems

Named and frozen. A result names its system, the git commit and the flags (`run_1105.sh`).

| Name | What |
|---|---|
| M0 | the model and the default sampler (random T=128), no pass |
| M1 | M0 + lane passes (forward + refine 2 at 0.5) |
| M2 = F | the playable default (generate.py since patch 0051), `--stats oracle` |
| M3 | F without oracle inputs (`--stats sample`) |

Pass families for the leave-one-family-out ablation on F: density and rests (rest, light,
fill holes); rhythm reuse and copies (carry, copy bias, sections, alike); long notes (ln tidy,
hold share); chords (strong chords, chord cap); jacks (untangle). A family stays in the default
only if taking it out makes the judge or a diagnostic worse, or breaks a guideline check.

## Checks before a number counts

- **Judge:** trained with shuffled labels, auc 0.40-0.60 pooled over the folds (a check of the
  pipeline: on random labels each fold's judge ranks by some arbitrary direction, so one fold
  may stray further); a human chart scored as a run (another mapset's chart of the same
  audio, where there is one), 0.40-0.60; human charts with the
  corruptions below, scores rising with the dose; M0, T_ref, F in the order the passes were
  meant to move them.
- **Metrics** (`scripts/corrupt_test.py`): on the human val charts, 4 corruptions x 3 doses:
  timing (taps moved one cell, 10/20/40% of them), lanes (the taps of onset rows in other
  lanes, 20/40/80% of the rows), new positions (taps on empty rows, 2/4/8% more onset rows),
  jumptrills (pairs of single taps made alternating chords, 20/40/80%); long notes are left
  alone. A metric is sensitive to a corruption when its median within-song Spearman with the
  dose is 0.8 or more in size, 80%+ of the songs agree with its sign and the change at the
  largest dose is 0.2+ of the human charts' spread; invariant when that median is under 0.2
  or the change under 0.05 of the spread. Sensitive to none = blind: it leaves the
  diagnostics. In the spirit of ChartGenEval (Lin, arXiv 2607.12857), which checks its
  metrics the same way and has no human correlation either.
- **What a judge reads** (`judge.py --view`, 11-08): a judge trained on every chart with part of
  it taken away alike (taps: long notes as taps; rhythm: taps on lanes drawn at random), each
  chart shown with rosu's SR of the chart as shown (`--sr view`), next to the full view the
  same way; read as nested, each view removing one more thing (`judge_table.py`: the steps on
  the songs they share). Not the human chart's SR for all: it is the SR the run was asked for,
  and the judge reads the miss (+0.043 auc on the same songs, 11-08). Its rows (rows.npz) cannot
  say this: they repeat the window's verdict on every row (11-08).

## Rules

- Predictions go in EXPERIMENTS.md before a run, as since 10-10.
- With them, when the run may stop early (a judge's `--stop-bounds`, a chain's skips), and
  it is cut there (11-08).
- One results table: a row per (system, setting) with the same columns: primary (when there
  is a playtest), judge auc and paired with intervals, the band shares of the diagnostics, SR
  error, F1@50 as a sanity check.
- No new pass and no new metric before the current ones are checked (week of 11-05).
- val CE is for training. Sampler settings and loss weights are chosen by the judge on
  generated charts, never by val CE.
