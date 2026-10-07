"""Generate a chart for one cached song and write a playable .osu.

    python scripts/sample.py --ckpt outputs/<run>/best.pt --key <manifest key>
    python scripts/sample.py --ckpt ... --key ... --sr 4.5 --steps 64 --order confidence
    python scripts/sample.py --ckpt outputs/full-v2/best.pt --key ... --style oracle
    python scripts/sample.py --ckpt outputs/full-v4/best.pt --key ... --stats oracle

Uses the song's cached mel and its real timing (BPM and offset are given,
design doc §4.4) and writes <ckpt dir>/samples/<key>_....osu that refers to the
original audio file, and next to it an .osz with the beatmap's own files (audio,
background, the human charts) plus the generated chart: open it with osu!
(lazer: double-click or drag onto the window) to play both side by side. For
osu! stable, the .osu alone can go into the beatmap's folder under Songs (F5). Also prints a quick comparison with the real chart
(onsets in the same cell and lane); proper F1 at ±20/50 ms is for the
evaluation code (§4.11).
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from src.data import chartstats
from src.data.cache import read_manifest
from src.data.chart_parser import _kv, _split_sections, parse_osu
from src.data.chart_writer import write_osu
from src.data.mel import open_mel
from src.data.style import encode, read_style
from src.data.tokenizer import HOLD_START, PAD, TAP, L, decode, grammar_violations, make_metas
from src.models.diffusion import load_denoiser, pick_device
from src.models.sampler import LANE_PASSES, LOUD_SIDES, ORDERS, generate_song, steps_name


def onset_match(pred: np.ndarray, real: np.ndarray) -> tuple[float, float]:
    keep = real != PAD
    p = np.isin(pred, (TAP, HOLD_START)) & keep
    r = np.isin(real, (TAP, HOLD_START)) & keep
    hit = (p & r).sum()
    return hit / max(p.sum(), 1), hit / max(r.sum(), 1)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--key", required=True, help="manifest key of the song")
    ap.add_argument("--sr", type=float, default=None,
                    help="target SR (default: the chart's label in the manifest)")
    ap.add_argument("--steps", type=int, default=128,   # DECISIONS 2026-09-29
                    help="forward passes per window; 0 = one cell per pass")
    ap.add_argument("--order", choices=list(ORDERS), default="random")
    ap.add_argument("--temperature", type=float, default=1.0, help="--order noisy")
    ap.add_argument("--refine", type=int, default=2,   # DECISIONS 2026-10-03
                    help="sweeps of lane refinement after sampling (sampler.refine_lanes)")
    ap.add_argument("--lane-temp", type=float, default=0.5,
                    help="--lanes forward / --refine: temperature of the lane choice "
                         "(0 = most likely lanes)")
    ap.add_argument("--lanes", choices=list(LANE_PASSES), default="forward",   # 2026-10-03
                    help="forward: choose every row's lanes again left to right with the "
                         "rhythm known (sampler.forward_lanes), before --refine")
    ap.add_argument("--spread", action="store_true",
                    help="cells opened together come from different beats (block: rows)")
    ap.add_argument("--forward-temp", type=float, default=None,
                    help="--lanes forward: its own lane temperature (default: --lane-temp)")
    ap.add_argument("--jack-bias", type=float, default=0.0,
                    help="lane passes: log-score bonus for repeating the previous onset's lanes")
    ap.add_argument("--copy-bias", type=float, default=0.0,   # 2026-10-04
                    help="after the lane passes, copy earlier bars with similar audio when the "
                         "model scores the copy within this many nats (sampler.copy_bars)")
    ap.add_argument("--no-copy", action="store_true", help="no bar copies")
    ap.add_argument("--refine-holds", action="store_true",
                    help="decide tap or long note and the release again with the whole chart "
                         "in view (sampler.refine_holds)")
    ap.add_argument("--loud-bias", type=float, default=0.1,   # 2026-10-06
                    help="fewer notes in quiet bars (sampler.loudness_bias); 0 = off")
    ap.add_argument("--onset-bias", type=float, default=1.0,   # 2026-10-07
                    help="fewer notes where nothing in the music starts: log penalty of up to X "
                         "on starting a note on rows with weak audio onsets (sampler.onset_gate); "
                         "0 = off")
    ap.add_argument("--loud-side", choices=list(LOUD_SIDES), default="quiet",
                    help="--loud-bias: quiet bars only, or both (also more notes in loud bars)")
    ap.add_argument("--hold-share", type=float, default=None,
                    help="this share of the onsets become long notes, where the model expects "
                         "them most (0: no long notes; implies --refine-holds)")
    ap.add_argument("--empty-bias", type=float, default=0.0,
                    help="log-scale bias on EMPTY while sampling: > 0 fewer notes, < 0 more")
    ap.add_argument("--hold-bias", type=float, default=0.0,
                    help="log-scale bias on starting long notes; -1 roughly a third as many "
                         "start, -inf none (sampler.sample_window)")
    ap.add_argument("--min-hold", type=int, default=None,
                    help="long notes shorter than this many cells (1/12 beat) become taps; "
                         "0 = keep; default by SR (sampler.HOLD_RULES)")
    ap.add_argument("--release-gap", type=int, default=None,
                    help="empty cells required between a release and the next onset in its lane; "
                         "0 = keep; default by SR (sampler.HOLD_RULES)")
    ap.add_argument("--mode", choices=["continue", "independent"], default="continue")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"))
    ap.add_argument("--root", type=Path, default=Path("data/raw"))
    ap.add_argument("--cache", type=Path, default=Path("data/cache"))
    ap.add_argument("--fake-mel", action="store_true", help="the plumbing-only mel (train.py --fake-mel)")
    ap.add_argument("--out", type=Path, default=None, help="default: <ckpt dir>/samples")
    ap.add_argument("--style", choices=["none", "oracle"], default="none",
                    help="oracle: the genre and mapper of this chart (a model trained with "
                         "--style); none: without labels")
    ap.add_argument("--style-csv", type=Path, default=Path("data/style.csv"))
    ap.add_argument("--style-guidance", type=float, default=0.0,
                    help="--style oracle: classifier-free guidance weight")
    ap.add_argument("--stats", default=None,
                    help="chart stats (model trained with --chart-stats): oracle (this chart's "
                         "own), or values like ln=0.4,jack=0.05,trill=0.15")
    ap.add_argument("--lane-guidance", type=float, default=None,
                    help="the lane passes' own guidance weight towards the style / chart stats "
                         "(default: --style-guidance); long notes are decided before them")
    ap.add_argument("--device", default="auto")
    a = ap.parse_args(argv)
    if a.no_copy:
        a.copy_bias = None

    row = next((r for r in read_manifest(a.manifest) if r["key"] == a.key), None)
    if row is None:
        print(f"{a.key} is not in {a.manifest}", file=sys.stderr)
        return 2
    z = np.load(a.cache / "tokens" / f"{a.key}.npz")
    tps = [(float(t), float(bl)) for t, bl in z["timing_points"]]
    cell_offset, n_cells = int(z["cell_offset"]), int(z["n_cells"])
    store = open_mel(a.cache, fake=a.fake_mel)
    if not store.has(a.key):
        print(f"no mel for {a.key}: run scripts/preprocess_data.py --mel", file=sys.stderr)
        return 2
    mel = store.chart(a.key, tps, cell_offset, (len(z["tokens"]) + 1) * L)   # + overhang
    sr = a.sr if a.sr is not None else float(row["sr"] or z["sr"])

    model = load_denoiser(a.ckpt, pick_device(a.device))
    style = {}
    if a.style == "oracle":
        vocab = getattr(model, "style", None)
        label = read_style(a.style_csv).get(a.key, {}) if a.style_csv.exists() else {}
        if vocab is None or not label:
            print("--style oracle needs a model trained with --style and this chart in "
                  "--style-csv", file=sys.stderr)
            return 2
        genre, mapper = encode(vocab, label.get("genre_id"), label.get("mapper_id"))
        style = {"genre": genre, "mapper": mapper, "style_guidance": a.style_guidance}
    buckets = None
    if a.stats is not None:
        spec = getattr(model, "chart_stats", None)
        if spec is None:
            print("--stats needs a model trained with --chart-stats", file=sys.stderr)
            return 2
        try:
            wanted = (chartstats.chart_stats(z["tokens"].reshape(-1, z["tokens"].shape[-1])
                                             [:n_cells])
                      if a.stats == "oracle" else chartstats.parse(a.stats))
        except ValueError as e:
            print(e, file=sys.stderr)
            return 2
        buckets = chartstats.encode(spec, wanted)
        print("chart stats: " + ", ".join(f"{chartstats.SHORT[n]} {v:.3f}"
                                          for n, v in wanted.items()))
        if a.style_guidance and not style:
            style = {"style_guidance": a.style_guidance}          # guidance towards the stats
    tokens = generate_song(model, mel, sr, tps, cell_offset, n_cells, steps=a.steps,
                           order=a.order, mode=a.mode, seed=a.seed, temperature=a.temperature,
                           refine=a.refine, lane_temperature=a.lane_temp,
                           hold_bias=a.hold_bias, min_hold=a.min_hold,
                           release_gap=a.release_gap, lanes=a.lanes, spread=a.spread,
                           empty_bias=a.empty_bias, forward_temperature=a.forward_temp,
                           jack_bias=a.jack_bias, copy_bias=a.copy_bias,
                           holds=a.refine_holds, hold_share=a.hold_share, loud_bias=a.loud_bias,
                           loud_side=a.loud_side, stats=buckets, onset_bias=a.onset_bias,
                           lane_guidance=a.lane_guidance, **style)
    bad = len(grammar_violations(tokens))
    chart = decode(tokens, make_metas(tps, cell_offset, len(tokens), sr))

    src = a.root / row["path"]
    meta = _kv(_split_sections(src.read_text(encoding="utf-8-sig", errors="replace"))
               .get("Metadata", []))
    chart.audio_filename = parse_osu(src).audio_filename
    out_dir = a.out or a.ckpt.parent / "samples"
    order = a.order if a.order != "noisy" else f"noisy{a.temperature:g}"
    if a.spread:
        order += "-spread"
    if a.lanes == "forward":
        order += "-fwd" + (f"t{a.forward_temp:g}" if a.forward_temp is not None else "")
    if a.refine:
        order += f"-ref{a.refine}t{a.lane_temp:g}"
    if a.empty_bias:
        order += f"-eb{a.empty_bias:g}"
    if a.jack_bias:
        order += f"-jb{a.jack_bias:g}"
    if a.loud_bias:
        order += f"-lb{'q' if a.loud_side == 'quiet' else ''}{a.loud_bias:g}"
    if a.onset_bias:
        order += f"-og{a.onset_bias:g}"
    if a.refine_holds or a.hold_share is not None:
        order += "-hr" + (f"{a.hold_share:g}" if a.hold_share is not None else "")
    if a.copy_bias is not None:
        order += f"-cp{a.copy_bias:g}"
    if "genre" in style:
        order += "-style"
    if style.get("style_guidance"):
        order += f"-sg{a.style_guidance:g}"
    if a.lane_guidance is not None:
        order += f"-lg{a.lane_guidance:g}"
    if a.stats is not None:
        order += "-st" + "".join(ch for ch in a.stats.replace("=", "").replace(",", "")
                                 if ch.isalnum() or ch in ".-")
    steps = steps_name(a.steps)
    out = out_dir / f"{a.key}_{a.mode}_{order}_T{steps}_s{sr:.2f}.osu"
    write_osu(out, chart, title=meta.get("Title", ""), artist=meta.get("Artist", ""),
              version=f"diffusion s={sr:.2f} {a.mode}/{order}/T{steps}")

    real = z["tokens"]
    same = real.shape == tokens.shape
    precision, recall = onset_match(tokens, real) if same else (float("nan"), float("nan"))
    print(f"{row['path']}\n  -> {out}")
    print(f"  {len(chart.notes)} notes (real chart {int(np.isin(real, (TAP, HOLD_START)).sum())}), "
          f"grammar violations {bad}, same-cell onset precision {precision:.3f} "
          f"recall {recall:.3f}")
    osz = out.with_suffix(".osz")
    with zipfile.ZipFile(osz, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(src.parent.rglob("*")):
            if f.is_file():
                zf.write(f, f.relative_to(src.parent).as_posix())
        zf.write(out, out.name)
    print(f"  {osz.name}: the beatmap with the generated chart added; open it with osu!")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
