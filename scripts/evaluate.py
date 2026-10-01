"""Generate charts for held-out songs and score them against the human charts (§4.11).

    python scripts/evaluate.py --ckpt outputs/<run>/best.pt --split val --n 50
    python scripts/evaluate.py --ckpt ... --mode independent --order confidence --steps 64
    python scripts/evaluate.py --ckpt ... --per-song --n 0 --order noisy --temperature 2

--per-song takes one chart per song (audio_key, a seeded random difficulty), so the
N rows are N different songs; without it the first N charts of the split are
scored, which are a handful of songs at several difficulties. --n 0 = all.
--seed picks the charts and seeds the sampler; --sample-seed reseeds only the
sampler, so the same charts are generated again with other random draws (a
replicate: how far two runs of one setting differ by chance).
summary.json also holds 95% bootstrap intervals over the rows (ci95_*) for the
main numbers.

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
                          pattern clarity (patterns.py), human_* likewise
    hold_share, hold_beats, short_holds, quick_regrab
                          long notes (holds.py), human_* likewise
    grade, bpm            for splitting results by SR grade and tempo (§4.11-4)
and the tokenizer's own ceiling: the same F1 for decode(encode(human)).
Writes <ckpt dir>/eval_<split>_<mode>_<order>_T<steps>/per_song.csv and summary.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from src.data.cache import read_manifest
from src.data.chart_parser import parse_osu
from src.data.mel import open_mel
from src.data.tokenizer import HOLD_START, TAP, K, L, decode, make_metas
from src.evaluation.holds import hold_stats
from src.evaluation.metrics import onset_f1, violation_rate
from src.evaluation.patterns import summarize
from src.evaluation.sr import ROSU_VERSION, star_rating, star_rating_file
from src.evaluation.structure import structure_scores
from src.models.diffusion import load_denoiser, pick_device
from src.models.sampler import ORDERS, generate_song

GRADES = [("Easy", 0.0, 2.0), ("Normal", 2.0, 2.7), ("Hard", 2.7, 4.0),
          ("Insane", 4.0, 5.3), ("Expert", 5.3, 6.5), ("Expert+", 6.5, float("inf"))]
PATTERN_KEYS = ("coverage", "coverage_chance", "run_length", "breaks_per_100", "motion_pred")


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
    ap.add_argument("--steps", type=int, default=128)   # DECISIONS 2026-09-29
    ap.add_argument("--order", choices=list(ORDERS), default="random")
    ap.add_argument("--temperature", type=float, default=1.0, help="--order noisy")
    ap.add_argument("--refine", type=int, default=0,
                    help="sweeps of lane refinement after sampling (sampler.refine_lanes)")
    ap.add_argument("--lane-temp", type=float, default=0.5,
                    help="--refine: temperature of the lane choice (0 = most likely lanes)")
    ap.add_argument("--hold-bias", type=float, default=0.0,
                    help="log-scale bias on starting long notes; -1 roughly a third as many "
                         "start, -inf none (sampler.sample_window)")
    ap.add_argument("--min-hold", type=int, default=3,
                    help="long notes shorter than this many cells (1/12 beat) "
                         "become taps; 0 = keep")
    ap.add_argument("--release-gap", type=int, default=2,
                    help="empty cells required between a release and the next onset in its lane; "
                         "0 = keep")
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
    order = a.order if a.order != "noisy" else f"noisy{a.temperature:g}"
    tag = f"{a.split}{'_songs' if a.per_song else ''}_{a.mode}_{order}_T{a.steps}"
    if a.refine:
        tag += f"_ref{a.refine}t{a.lane_temp:g}"
    if a.hold_bias:
        tag += f"_hb{a.hold_bias:g}"
    if (a.min_hold, a.release_gap) != (3, 2):
        tag += f"_mh{a.min_hold}rg{a.release_gap}"
    sample_seed = a.seed if a.sample_seed is None else a.sample_seed
    if a.sample_seed is not None:
        tag += f"_s{a.sample_seed}"
    out_dir = a.ckpt.parent / f"eval_{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for i, r in enumerate(rows):
        z = np.load(a.cache / "tokens" / f"{r['key']}.npz")
        tps = [(float(t), float(bl)) for t, bl in z["timing_points"]]
        offset, n_cells, sr = int(z["cell_offset"]), int(z["n_cells"]), float(r["sr"])
        mel = store.chart(r["key"], tps, offset, (len(z["tokens"]) + 1) * L)   # + overhang
        tokens = generate_song(model, mel, sr, tps, offset, n_cells, steps=a.steps,
                               order=a.order, mode=a.mode, seed=sample_seed + i,
                               temperature=a.temperature, refine=a.refine,
                               lane_temperature=a.lane_temp,
                               hold_bias=a.hold_bias, min_hold=a.min_hold,
                               release_gap=a.release_gap)
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
               "bpm": round(60 * beats / seconds, 2) if seconds > 0 else float("nan"),
               **score(gen, ref, back, tokens, sr, star_rating(gen), star_rating_file(ref_path)),
               **structure_and_patterns(tokens, z["tokens"], mel, n_cells, a.far_k)}
        results.append(row)
        print(f"[{i + 1}/{len(rows)}] f1@50 {row['f1@50']:.3f}  sr {row['sr_gen']:.2f} "
              f"(target {sr:.2f})  {r['path']}", flush=True)

    with open(out_dir / "per_song.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(results)
    numeric = [k for k in results[0] if k not in ("key", "path", "grade")]
    summary = {"songs": len(results), "distinct_songs": len({r["audio_key"] for r in rows}),
               "per_song": a.per_song, "split": a.split, "mode": a.mode, "order": a.order,
               "temperature": a.temperature if a.order == "noisy" else None,
               "steps": a.steps, "refine": a.refine, "hold_bias": a.hold_bias,
               "min_hold": a.min_hold, "release_gap": a.release_gap,
               "lane_temp": a.lane_temp if a.refine else None,
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
