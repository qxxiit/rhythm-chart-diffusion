"""Generate charts for held-out songs and score them against the human charts (§4.11).

    python scripts/evaluate.py --ckpt outputs/<run>/best.pt --split val --n 50
    python scripts/evaluate.py --ckpt ... --mode independent --order confidence --steps 64
    python scripts/evaluate.py --ckpt ... --per-song --n 0 --order noisy --temperature 2
    python scripts/evaluate.py --ckpt ... --per-song --n 0 --lanes forward --refine 2
    python scripts/evaluate.py --ckpt ... --per-song --n 20 --steps 0     # the ceiling
    python scripts/evaluate.py --ckpt ... --per-song --n 0 --lanes forward --refine 2 --copy-bias 4
    python scripts/evaluate.py --ckpt outputs/full-v2/best.pt --per-song --n 0 --lanes forward \
        --refine 2 --style oracle                  # each chart's own genre and mapper
    python scripts/evaluate.py --ckpt ... --per-song --n 0 --copy-bias 0 \
        --from-charts outputs/full-v1/eval_val_songs_continue_random_T128_fwd_ref2t0.5

--per-song takes one chart per song (audio_key, a seeded random difficulty), so the
N rows are N different songs; without it the first N charts of the split are
scored, which are a handful of songs at several difficulties. --n 0 = all.
--seed picks the charts and seeds the sampler; --sample-seed reseeds only the
sampler, so the same charts are generated again with other random draws (a
replicate: how far two runs of one setting differ by chance).
summary.json also holds 95% bootstrap intervals over the rows (ci95_*) for the
main numbers. Compare runs song by song with scripts/compare_runs.py.

For each chart (see --per-song) with SR, tokens and mel, generate at the
human chart's SR with the song's real timing, then score:
    f1@20 / f1@50         onsets, same lane, greedy matching (metrics.onset_f1)
    f1@50_any_lane        lanes ignored
    violation_rate        grammar violations per position (0 for the constrained sampler)
    sr_gen, sr_error      local SR of the generated chart, |sr_gen - s| (s = API SR)
    sr_error_vs_local     |sr_gen - local SR of the human chart|: the same calculator on
                          both sides, in case rosu-pp and the API disagree (check_sr.py)
    density_ratio         generated onsets / human onsets
    rho_all/in/cross/far  structure (structure.py), human_rho_* for the human chart
    coverage, run_length, breaks_per_100, motion_pred
                          pattern clarity (patterns.py), human_* likewise; coverage_p1 /
                          _p2 / _p3plus split coverage into jacks, trills and longer motifs
                          (whether lane passes overdo one kind); move_stair / _trill /
                          _jack / _leap, chord_share: the kinds of lane motion
    hold_share, hold_beats, short_holds, quick_regrab
                          long notes (holds.py), human_* likewise
    grade, bpm            for splitting results by SR grade and tempo (§4.11-4)
    lone_chord, bar_rhythm_repeat, bar_lane_repeat
                          chords that cut single-note streams, and bars that repeat an
                          earlier bar's rhythm / also its lanes (patterns.py)
    passes, seconds       forward passes and wall time spent on the song (the cost)
    copied_bars           with --copy-bias: bars the copy pass replaced (sampler.copy_bars)
    genre, mapper_known   the song's genre (data/style.csv, if there) and, for a model with
                          style inputs, whether the chart's mapper is in its vocab
--from-charts DIR scores the charts an earlier run saved (charts.npz) instead of
sampling, after the copy pass if --copy-bias is given (sampler.copy_song): minutes
instead of hours. Use the same --per-song / --n / --seed as that run; the sampler
options are ignored, and passes / seconds are the copy pass's. Writes to DIR_cp<bias>.
--style oracle (a model trained with --style) generates every chart with the genre
and mapper of the human chart it is scored against, --style none without labels;
--style-guidance W adds classifier-free guidance towards that style.
and the tokenizer's own ceiling: the same F1 for decode(encode(human)).
Writes <ckpt dir>/eval_<split>_<mode>_<order>_T<steps>/per_song.csv, summary.json and
charts.npz (the generated tokens by chart key, so new metrics need no new sampling).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from src.data.cache import read_manifest
from src.data.chart_parser import parse_osu
from src.data.mel import open_mel
from src.data.style import GENRES, encode, read_style
from src.data.tokenizer import HOLD_START, TAP, K, L, decode, make_metas
from src.evaluation.holds import hold_stats
from src.evaluation.metrics import onset_f1, violation_rate
from src.evaluation.patterns import summarize
from src.evaluation.sr import ROSU_VERSION, star_rating, star_rating_file
from src.evaluation.structure import structure_scores
from src.models.diffusion import load_denoiser, pick_device
from src.models.sampler import LANE_PASSES, ORDERS, STATS, copy_song, generate_song, steps_name

GRADES = [("Easy", 0.0, 2.0), ("Normal", 2.0, 2.7), ("Hard", 2.7, 4.0),
          ("Insane", 4.0, 5.3), ("Expert", 5.3, 6.5), ("Expert+", 6.5, float("inf"))]
PATTERN_KEYS = ("coverage", "coverage_chance", "coverage_p1", "coverage_p2", "coverage_p3plus",
                "run_length", "breaks_per_100", "motion_pred",
                "move_stair", "move_trill", "move_jack", "move_leap", "chord_share",
                "lone_chord", "bar_rhythm_repeat", "bar_lane_repeat")


def structure_and_patterns(gen_tokens, human_tokens, mel, n_cells: int, far_k: int) -> dict:
    row = {}
    for who, tokens in (("", gen_tokens), ("human_", human_tokens)):
        flat = tokens.reshape(-1, tokens.shape[-1])[:n_cells]
        s = structure_scores(flat, mel, n_cells, far_k=far_k)
        row.update({f"{who}rho_{k}": s[f"rho_{k}"] for k in ("all", "in", "cross", "far")})
        p = summarize(flat, chance_seeds=3)
        row.update({f"{who}{k}": p[k] for k in PATTERN_KEYS})
        row.update({f"{who}{k}": v for k, v in hold_stats(flat).items()})
    return row


def score(gen, ref, back, tokens, sr: float, sr_gen: float, sr_human: float) -> dict:
    row = {}
    for tol in (20, 50):
        row[f"f1@{tol}"] = onset_f1(gen, ref, tol)["f1"]
    row["f1@50_any_lane"] = onset_f1(gen, ref, 50, lanes=False)["f1"]
    p = onset_f1(gen, ref, 50)
    row["precision@50"], row["recall@50"] = p["precision"], p["recall"]
    row["ceiling_f1@20"] = onset_f1(back, ref, 20)["f1"]
    row["violation_rate"] = violation_rate(tokens)
    row["sr_target"], row["sr_gen"], row["sr_human_local"] = sr, sr_gen, sr_human
    row["sr_error"] = abs(sr_gen - sr)
    row["sr_error_vs_local"] = abs(sr_gen - sr_human)
    row["density_ratio"] = len(gen.notes) / max(len(ref.notes), 1)
    return row


CI_KEYS = ("f1@20", "f1@50", "f1@50_any_lane", "precision@50", "recall@50", "density_ratio",
           "motion_pred", "hold_share",
           "sr_error", "rho_all", "rho_in", "coverage")


def pick_per_song(rows: list[dict], seed: int) -> list[dict]:
    """One chart per audio_key, a seeded random difficulty, in manifest order of songs."""
    by_song: dict[str, list[dict]] = {}
    for r in rows:
        by_song.setdefault(r["audio_key"], []).append(r)
    rng = np.random.default_rng(seed)
    return [charts[int(rng.integers(len(charts)))] for charts in by_song.values()]


def bootstrap_ci(values, n: int = 2000, seed: int = 0) -> list[float]:
    """95% percentile interval of the mean, resampling rows (NaN rows left out)."""
    v = np.asarray([x for x in values if x == x], dtype=np.float64)
    if len(v) < 2:
        return [float("nan"), float("nan")]
    means = np.random.default_rng(seed).choice(v, size=(n, len(v))).mean(axis=1)
    return [round(float(q), 4) for q in np.percentile(means, [2.5, 97.5])]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=50, help="charts (songs with --per-song); 0 = all")
    ap.add_argument("--per-song", action="store_true", help="one chart per song")
    ap.add_argument("--steps", type=int, default=128,   # DECISIONS 2026-09-29
                    help="forward passes per window; 0 = one cell per pass (the ceiling of "
                         "parallel sampling, about 10x the passes of 128)")
    ap.add_argument("--order", choices=list(ORDERS), default="random")
    ap.add_argument("--spread", action="store_true",
                    help="cells opened together come from different beats (block: rows)")
    ap.add_argument("--temperature", type=float, default=1.0, help="--order noisy")
    ap.add_argument("--refine", type=int, default=0,
                    help="sweeps of lane refinement after sampling (sampler.refine_lanes)")
    ap.add_argument("--lanes", choices=list(LANE_PASSES), default="sampled",
                    help="forward: after sampling, choose every row's lanes again left to "
                         "right with the rhythm known (sampler.forward_lanes), before --refine")
    ap.add_argument("--lane-temp", type=float, default=0.5,
                    help="--lanes forward / --refine: temperature of the lane choice "
                         "(0 = most likely lanes)")
    ap.add_argument("--hold-bias", type=float, default=0.0,
                    help="log-scale bias on starting long notes; -1 roughly a third as many "
                         "start, -inf none (sampler.sample_window)")
    ap.add_argument("--forward-temp", type=float, default=None,
                    help="--lanes forward: its own lane temperature (default: --lane-temp)")
    ap.add_argument("--jack-bias", type=float, default=0.0,
                    help="lane passes: log-score bonus for repeating the previous onset's lanes")
    ap.add_argument("--copy-bias", type=float, default=None,
                    help="after the lane passes, copy earlier bars with similar audio when the "
                         "model scores the copy within this many nats of the bar "
                         "(sampler.copy_bars); default: no copies")
    ap.add_argument("--empty-bias", type=float, default=0.0,
                    help="log-scale bias on EMPTY while sampling: > 0 fewer notes, < 0 more")
    ap.add_argument("--min-hold", type=int, default=None,
                    help="long notes shorter than this many cells (1/12 beat) become taps; "
                         "0 = keep; default by SR (sampler.HOLD_RULES)")
    ap.add_argument("--release-gap", type=int, default=None,
                    help="empty cells required between a release and the next onset in its lane; "
                         "0 = keep; default by SR (sampler.HOLD_RULES)")
    ap.add_argument("--mode", choices=["continue", "independent"], default="continue")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sample-seed", type=int, default=None,
                    help="sampler seed only (default: --seed); the charts stay those of --seed")
    ap.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"))
    ap.add_argument("--root", type=Path, default=Path("data/raw"))
    ap.add_argument("--cache", type=Path, default=Path("data/cache"))
    ap.add_argument("--fake-mel", action="store_true", help="the plumbing-only mel (train.py --fake-mel)")
    ap.add_argument("--far-k", type=int, default=8,
                    help="bar distance for rho_far; fix it from scripts/human_baselines.py")
    ap.add_argument("--style", choices=["none", "oracle"], default="none",
                    help="oracle: generate with the human chart's genre and mapper (a model "
                         "trained with --style); none: without labels")
    ap.add_argument("--style-csv", type=Path, default=Path("data/style.csv"),
                    help="genre and mapper per chart (scripts/build_style.py)")
    ap.add_argument("--style-guidance", type=float, default=0.0,
                    help="--style oracle: classifier-free guidance weight (0 = none)")
    ap.add_argument("--from-charts", type=Path, default=None,
                    help="score the charts.npz of this earlier run (after --copy-bias, if "
                         "given) instead of sampling")
    ap.add_argument("--device", default="auto")
    a = ap.parse_args(argv)

    store = open_mel(a.cache, fake=a.fake_mel)
    rows = [r for r in read_manifest(a.manifest) if r["split"] == a.split and r["sr"]
            and r.get("drop", "") == ""
            and (a.cache / "tokens" / f"{r['key']}.npz").exists()
            and store.has(r["key"])]
    if a.per_song:
        rows = pick_per_song(rows, a.seed)
    if a.n:
        rows = rows[:a.n]
    if not rows:
        print(f"no {a.split} songs with SR, tokens and mel", file=sys.stderr)
        return 2
    model = load_denoiser(a.ckpt, pick_device(a.device))
    labels = read_style(a.style_csv) if a.style_csv.exists() else {}
    vocab = getattr(model, "style", None)
    if a.style == "oracle" and (vocab is None or not labels):
        print("--style oracle needs a model trained with --style and --style-csv",
              file=sys.stderr)
        return 2
    if a.style_guidance and a.style == "none":
        print("--style-guidance guides towards a style: use it with --style oracle",
              file=sys.stderr)
        return 2
    order = a.order if a.order != "noisy" else f"noisy{a.temperature:g}"
    tag = f"{a.split}{'_songs' if a.per_song else ''}_{a.mode}_{order}_T{steps_name(a.steps)}"
    if a.spread:
        tag += "_spread"
    fwd_temp = a.lane_temp if a.forward_temp is None else a.forward_temp
    if a.lanes == "forward":
        tag += "_fwd" + ("" if a.refine and fwd_temp == a.lane_temp else f"t{fwd_temp:g}")
    if a.refine:
        tag += f"_ref{a.refine}t{a.lane_temp:g}"
    if a.hold_bias:
        tag += f"_hb{a.hold_bias:g}"
    if a.empty_bias:
        tag += f"_eb{a.empty_bias:g}"
    if a.jack_bias:
        tag += f"_jb{a.jack_bias:g}"
    if a.copy_bias is not None:
        tag += f"_cp{a.copy_bias:g}"
    if a.style == "oracle":
        tag += "_style" + (f"_sg{a.style_guidance:g}" if a.style_guidance else "")
    if a.min_hold is not None or a.release_gap is not None:
        tag += f"_mh{a.min_hold if a.min_hold is not None else 'a'}" \
               f"rg{a.release_gap if a.release_gap is not None else 'a'}"
    sample_seed = a.seed if a.sample_seed is None else a.sample_seed
    if a.sample_seed is not None:
        tag += f"_s{a.sample_seed}"
    out_dir = a.ckpt.parent / f"eval_{tag}"
    saved = None
    if a.from_charts is not None:
        saved = np.load(a.from_charts / "charts.npz")
        missing = [r["key"] for r in rows if r["key"] not in saved.files]
        if missing:
            print(f"{a.from_charts}: no saved chart for {len(missing)} of the {len(rows)} charts "
                  "(use that run's --per-song / --n / --seed)", file=sys.stderr)
            return 2
        out_dir = a.from_charts.parent / (a.from_charts.name + (
            f"_cp{a.copy_bias:g}" if a.copy_bias is not None else "_rescored"))
    out_dir.mkdir(parents=True, exist_ok=True)

    results, charts = [], {}
    for i, r in enumerate(rows):
        z = np.load(a.cache / "tokens" / f"{r['key']}.npz")
        tps = [(float(t), float(bl)) for t, bl in z["timing_points"]]
        offset, n_cells, sr = int(z["cell_offset"]), int(z["n_cells"]), float(r["sr"])
        mel = store.chart(r["key"], tps, offset, (len(z["tokens"]) + 1) * L)   # + overhang
        label = labels.get(r["key"], {})
        style = {}
        if vocab is not None:
            g, m = encode(vocab, label.get("genre_id"), label.get("mapper_id"))
            if a.style == "oracle":
                style = {"genre": g, "mapper": m, "style_guidance": a.style_guidance}
        passes0, copies0, t0 = STATS["passes"], STATS["copied_bars"], time.perf_counter()
        if saved is not None:
            tokens = saved[r["key"]]
            if a.copy_bias is not None:
                tokens = copy_song(model, tokens, mel, sr, tps, offset, n_cells,
                                   copy_bias=a.copy_bias, min_hold=a.min_hold,
                                   release_gap=a.release_gap,
                                   **{k: v for k, v in style.items() if k != "style_guidance"})
        else:
            tokens = generate_song(model, mel, sr, tps, offset, n_cells, steps=a.steps,
                                   order=a.order, mode=a.mode, seed=sample_seed + i,
                                   temperature=a.temperature, refine=a.refine,
                                   lane_temperature=a.lane_temp,
                                   hold_bias=a.hold_bias, min_hold=a.min_hold,
                                   release_gap=a.release_gap, lanes=a.lanes, spread=a.spread,
                                   empty_bias=a.empty_bias, forward_temperature=a.forward_temp,
                                   jack_bias=a.jack_bias, copy_bias=a.copy_bias, **style)
        cost = {"passes": STATS["passes"] - passes0,
                "seconds": round(time.perf_counter() - t0, 2)}
        if a.copy_bias is not None:
            cost["copied_bars"] = STATS["copied_bars"] - copies0
        charts[r["key"]] = tokens
        metas = make_metas(tps, offset, len(tokens), sr)
        gen = decode(tokens, metas)
        back = decode(z["tokens"], metas)
        ref_path = a.root / r["path"]
        ref = parse_osu(ref_path)
        gen.audio_filename = back.audio_filename = ref.audio_filename
        onset_rows = np.flatnonzero(np.isin(z["tokens"].reshape(-1, K), (TAP, HOLD_START))
                                    .any(axis=1))
        beats = (onset_rows[-1] - onset_rows[0]) / 12 if len(onset_rows) > 1 else 0.0
        times = [n.time_ms for n in ref.notes]
        seconds = (max(times) - min(times)) / 1000 if times else 0.0
        row = {"key": r["key"], "path": r["path"],
               "grade": next(g for g, lo, hi in GRADES if lo <= sr < hi),
               "genre": GENRES.get(label.get("genre_id"), ""),
               "mapper_known": "" if vocab is None or not label.get("mapper_id")
               else int(label["mapper_id"] in vocab["mappers"]),
               "bpm": round(60 * beats / seconds, 2) if seconds > 0 else float("nan"),
               **score(gen, ref, back, tokens, sr, star_rating(gen), star_rating_file(ref_path)),
               **structure_and_patterns(tokens, z["tokens"], mel, n_cells, a.far_k), **cost}
        results.append(row)
        print(f"[{i + 1}/{len(rows)}] f1@50 {row['f1@50']:.3f}  sr {row['sr_gen']:.2f} "
              f"(target {sr:.2f})  {cost['seconds']:.0f} s  {r['path']}", flush=True)

    np.savez_compressed(out_dir / "charts.npz", **charts)
    with open(out_dir / "per_song.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(results)
    numeric = [k for k in results[0] if k not in ("key", "path", "grade", "genre", "mapper_known")]
    summary = {"songs": len(results), "distinct_songs": len({r["audio_key"] for r in rows}),
               "per_song": a.per_song, "split": a.split, "mode": a.mode, "order": a.order,
               "temperature": a.temperature if a.order == "noisy" else None,
               "steps": a.steps, "spread": a.spread, "lanes": a.lanes, "refine": a.refine,
               "hold_bias": a.hold_bias, "empty_bias": a.empty_bias, "jack_bias": a.jack_bias,
               "copy_bias": a.copy_bias, "style": a.style, "style_guidance": a.style_guidance,
               "min_hold": a.min_hold, "release_gap": a.release_gap,
               "lane_temp": a.lane_temp if a.refine or a.lanes != "sampled" else None,
               "forward_temp": fwd_temp if a.lanes == "forward" else None,
               "seed": a.seed, "sample_seed": sample_seed,
               "far_k": a.far_k, "ckpt": str(a.ckpt),
               "rosu_pp_py": ROSU_VERSION,
               **{f"mean_{k}": round(float(np.nanmean([x[k] for x in results])), 4)
                  for k in numeric},
               **{f"ci95_{k}": bootstrap_ci([x[k] for x in results], seed=a.seed)
                  for k in CI_KEYS if k in results[0]}}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
