# Experiment runs

Each `run_*.sh` is one experiment as it was run on the MacBook (MPS): the question and the
predictions are in the [experiment log](../../docs/EXPERIMENTS.md) entry it names, the decision
in [DECISIONS](../../docs/DECISIONS.md). They are kept as a record: run one from the repository
root, one phase at a time (`sh experiments/runs/run_1129.sh passes`), and it writes under
`outputs/` only. The older ones name patches, judges and runs of their day, which a fresh
checkout does not have.

The current system in one command is [`scripts/system.sh`](../../scripts/system.sh).

| run | when | what it asked | what came of it | log entry |
|---|---|---|---|---|
| `run_1005` | 10/4-10/5 | long-note passes and the loudness bias on saved charts; full-v3 (row masks) | `refine_holds` not a default; the loudness bias on the quiet side only; row masks join the recipe | 2026-10-05 |
| `run_1006` | 10/5-10/6 | full-v4 (the chart's long-note share, jack and trill rate as inputs); the quiet-side loudness bias | full-v4's chart stats stay; playable charts thin quiet bars (`--loud-bias 0.1`) | 2026-10-06 |
| `run_quiet` | 10/5-10/6 night | a pairs playtest pack with lane guidance 2; the onset-strength gate and the quiet-section metrics | the gate, tried again in `run_gate` | 2026-10-06 (quiet sections) |
| `run_1007` | 10/6-10/7 | a pairs pack with full-v4; lane guidance towards the chart stats | lane guidance 2 | 2026-10-06/07 |
| `run_gate` | 10/6-10/7 night | the pack of the playable candidate; the onset gate on note starts | playable charts start notes less where nothing in the music starts (`--onset-bias 1`) | 2026-10-06/07 (the onset gate and a fix) |
| `run_rest` | 10/7-10/8 night | choices for a whole bar: rests, long-note bars, the SR offset | the rest pass (`--rest 1`); the gate tapered by SR, no flat offset | 2026-10-07 night |
| `run_1105` | from 11/5 | the checks before a number counts; F without each pass family; M3; the sampler's settings | the evaluation charter (docs/EVALUATION.md); the diagnostics judge | 2026-11-05, 2026-11-07 |
| `run_1107` | 11/7 | the control, the sweep, a ladder of own judges; the cosine schedule | a judge is read for the run it learned | 2026-11-07 (2), 2026-11-08 |
| `run_1108` | 11/8 | the cosine schedule on all 240 val songs with F's passes | cosine is the base | 2026-11-08, 2026-11-08 (2) |
| `run_1109` | 11/8 evening | what of the charts the judge reads (the views) | the views: full, rhythm, taps, holds | 2026-11-08 (2), (3) |
| `run_1110` | 11/8 night | the views with the SR of the chart as shown | judges show the SR as shown (`--sr view`), 2 epochs | 2026-11-08 (3), (4) |
| `run_1111` | 11/8 night | bars that sound alike aligned at 6 nats, not 2 | not taken (easier to tell apart) | 2026-11-08 (6) |
| `run_1112` | 11/8-11/9 | windows chosen by a judge among 4 draws | windows are chosen by a judge (`--select-judge`) | 2026-11-08 (5), (6) |
| `run_1113` | 11/8-11/9 night | 8 draws a window on all 240 songs | 4 draws, not 8 | 2026-11-08 (6), 2026-11-09 (1) |
| `run_1114` | 11/9 | the lanes chosen by a judge too | not by default | 2026-11-09 (1), (2) |
| `run_1115` | 11/9 | the lanes' and the long notes' settings on the chosen draw | none taken | 2026-11-09 (3), (4) |
| `run_1116` | 11/9 evening | the long notes drawn by the model (hold bias 0.5); the lanes at temperature 1 | the lanes at 1 go on; the hold bias 0.5 overshoots | 2026-11-09 (5), (6) |
| `run_1118` | 11/9 night | the long notes at the human share (hold bias 0.2) | hold bias 0.2 | 2026-11-09 (6), 2026-11-10 (1) |
| `run_1117` | 11/9-11/10 | the windows chosen by a judge that sees the long notes | the system: chosen among 4 by sel_holds, hold bias 0.2, lanes at 1, F's passes | 2026-11-09 (7), 2026-11-10 (1) |
| `run_1119` | 11/10 | the system on all 240 val songs, the playtest gate | the gate passed | 2026-11-11 (2) |
| `run_1120` | 11/11 night | where the system's tell is left; its lanes at 0.7, its hold bias at 0.1 | neither taken | 2026-11-11 (1), (2) |
| `run_1121` | (not run) | the windows chosen by the rhythm and the holds judges together | postponed | 2026-11-14 (1) |
| `run_1122` | 11/14 night | the rhythm drawn in block order; every audio file's stems | block order not taken; the stems extracted | 2026-11-14 (2) |
| `run_1123` | 11/15 | full-v4 widened to the stems (stems-v1) | not taken: its val_ce gain was the extra training | 2026-11-15 (1) |
| `run_1124` | 11/16 night | the rest guard; every beat offered the last one's rhythm | the rest guard a candidate; the answers not taken | 2026-11-15 (2) |
| `run_1125` | 11/16 | the stems from the start (stems-v2); the answers made stricter | stems-v2 not taken through the system as tuned; the answers dropped | 2026-11-16 (1) |
| `run_1126` | 11/16-11/17 | the judges' trunk; a raw draw of each model | each judge favours its own trunk's model; the models level; full-v4 stays | 2026-11-16 (2) |
| `run_1127` | 11/17 | the lanes in units (each unit's first row at 1, the rest at 0.5) | the pattern numbers nearer the human charts; decided by a blind play | 2026-11-17 (1) |
| `run_1128` | 11/17 | short long notes where the chart is busy become taps | `--dense-taps 0.15` joins the system (at the edge) | 2026-11-17 (2) |
| `run_1129` | 11/17 | only the repeats become taps | `--dense-taps 0.2 --dense-repeats` replaces 0.15: the holds view's cost halved | 2026-11-17 (3) |
| `run_1130` | 11/18 | the chord share to the human chart's; releases snapped to the next note | `--snap-releases 2` joins the system; the chord share not taken (chords too exactly on the accents) | 2026-11-18 (1) |
| `run_1131` | 11/19 | the chord share half way in a drawn order; bars mirrored where the rhythm repeats | `--mirror-bars` joins the system; the soft chord share not taken (the taps view +0.09) | 2026-11-19 (1) |
| `run_1132` | 11/19 | the stair share and the long notes' length as chart stats (pat-v1, against ft-v1) | not taken: the draws hardly follow the new stats (stair correlation +0.14 -> +0.19) | 2026-11-19 (2) |
| `run_1133` | 11/20 | six chart stats from the first step (pat-v2, against full-v4); a probe of pat-v1 | pat-v2 a candidate, not taken: long notes' length and chords follow the human chart better, stairs not; the judges level | 2026-11-20 (1) |
| `run_1134` | 11/20 | the mirrored bars at the shares counted again; the same bar again 4 bars later | `--mirror-shares 0.034,0.049,0.062` joins the system; the repeats not taken (the taps view +0.014) | 2026-11-20 (2) |
| `run_1135` | 11/21 | the system drawn with pat-v2 (a choosing judge of its own), against the system | not taken: each trunk's judge favours its own (averages +0.008, +0.012); full-v4 stays through the freeze | 2026-11-21 (1) |
| `run_1136` | 11/22 | the human charts corrupted the ways players call bad, and another human chart, scored by judges trained as now | the judges see rhythm, placement, chords, long notes; trills, jumptrills, minijacks and lanes hardly: those by playing | 2026-11-22 (1) |
| `run_1137` | 11/22 | a judge of the lanes alone: the human rhythm, the lanes drawn again (at 1, 0.7, in units) | lanes at 1 nearly a mapper's (0.559); 0.7 and units easier to tell; the rhythm is where the lanes go wrong | 2026-11-22 (2) |
| `run_1138` | 11/22 | the system's lane passes (untangle, mirrored bars) and all its passes on run_1137's lt1, by the lane-only judge | both stay (-0.003, -0.010, across 0); all the passes cost a human chart +0.016: the draw's rhythm is what is left | 2026-11-22 (3) |
