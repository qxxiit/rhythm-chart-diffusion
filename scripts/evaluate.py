"""Generate charts for held-out songs and score them against the human charts (§4.11).

    python scripts/evaluate.py --ckpt outputs/<run>/best.pt --split val --n 50
    python scripts/evaluate.py --ckpt ... --mode independent --order confidence --steps 64

For each song (the first N of the split with SR, tokens and mel), generate at the
human chart's SR with the song's real timing, then score:
    f1@20 / f1@50         onsets, same lane, greedy matching (metrics.onset_f1)
    f1@50_any_lane        lanes ignored
    violation_rate        grammar violations per position (0 for the constrained sampler)
    sr_gen, sr_error      local SR of the generated chart, |sr_gen - s| (s = API SR)
    sr_error_vs_local     |sr_gen - local SR of the human chart|: the same calculator on
                          both sides, in case rosu-pp and the API disagree (check_sr.py)
    density_ratio         generated onsets / human onsets
and the tokenizer's own ceiling: the same F1 for decode(encode(human)).
Writes <ckpt dir>/eval_<split>_<mode>_<order>_T<steps>/per_song.csv and summary.json.
Not here yet: rho (structure) and pattern clarity.
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
from src.data.tokenizer import decode, make_metas
from src.evaluation.metrics import onset_f1, violation_rate
from src.evaluation.sr import ROSU_VERSION, star_rating, star_rating_file
from src.models.diffusion import load_denoiser, pick_device
from src.models.sampler import generate_song


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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=50, help="songs to evaluate")
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--order", choices=["random", "confidence"], default="random")
    ap.add_argument("--mode", choices=["continue", "independent"], default="continue")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"))
    ap.add_argument("--root", type=Path, default=Path("data/raw"))
    ap.add_argument("--cache", type=Path, default=Path("data/cache"))
    ap.add_argument("--device", default="auto")
    a = ap.parse_args(argv)

    rows = [r for r in read_manifest(a.manifest) if r["split"] == a.split and r["sr"]
            and (a.cache / "tokens" / f"{r['key']}.npz").exists()
            and (a.cache / "mel" / f"{r['key']}.npy").exists()][:a.n]
    if not rows:
        print(f"no {a.split} songs with SR, tokens and mel", file=sys.stderr)
        return 2
    model = load_denoiser(a.ckpt, pick_device(a.device))
    out_dir = a.ckpt.parent / f"eval_{a.split}_{a.mode}_{a.order}_T{a.steps}"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for i, r in enumerate(rows):
        z = np.load(a.cache / "tokens" / f"{r['key']}.npz")
        mel = np.load(a.cache / "mel" / f"{r['key']}.npy").astype(np.float32)
        tps = [(float(t), float(bl)) for t, bl in z["timing_points"]]
        offset, n_cells, sr = int(z["cell_offset"]), int(z["n_cells"]), float(r["sr"])
        tokens = generate_song(model, mel, sr, tps, offset, n_cells, steps=a.steps,
                               order=a.order, mode=a.mode, seed=a.seed + i)
        metas = make_metas(tps, offset, len(tokens), sr)
        gen = decode(tokens, metas)
        back = decode(z["tokens"], metas)
        ref_path = a.root / r["path"]
        ref = parse_osu(ref_path)
        gen.audio_filename = back.audio_filename = ref.audio_filename
        row = {"key": r["key"], "path": r["path"],
               **score(gen, ref, back, tokens, sr, star_rating(gen), star_rating_file(ref_path))}
        results.append(row)
        print(f"[{i + 1}/{len(rows)}] f1@50 {row['f1@50']:.3f}  sr {row['sr_gen']:.2f} "
              f"(target {sr:.2f})  {r['path']}", flush=True)

    with open(out_dir / "per_song.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(results)
    numeric = [k for k in results[0] if k not in ("key", "path")]
    summary = {"songs": len(results), "split": a.split, "mode": a.mode, "order": a.order,
               "steps": a.steps, "ckpt": str(a.ckpt), "rosu_pp_py": ROSU_VERSION,
               **{f"mean_{k}": round(float(np.mean([x[k] for x in results])), 4) for k in numeric}}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
