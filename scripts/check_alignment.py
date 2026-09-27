"""Do the notes sit on the audio's onsets? (design doc §4.5, 정합 검증)

    python scripts/check_alignment.py                   # 200 train songs
    python scripts/check_alignment.py --n 500 --plots 10
    python scripts/check_alignment.py --cached-only   # only songs whose log-Mel is cached (for 2.)

One chart per audio file (the one with the most notes), train split only.

1. Audio: spectral flux of a fine log-Mel (hop 32 = 1.45 ms) of decode_osu's
   output, averaged around every note start from -60 to +60 ms. The lag where
   the average peaks is where the audio's onsets sit relative to the chart.
   Every group of files (mp3 lame / xing / none / vbri, ogg, ...) should peak at
   the same lag. A group about +12 ms away (529 samples) or +26 ms (1,152) was
   started differently by osu! than src/data/audio.py assumes.
2. Grid: the same average on the token grid, from the cached log-Mel exactly as
   the Dataset reads it, in frames (4 per cell) around the onset frame. It
   should peak at the onset frame or one before it, within about a frame of 1.
   Also printed: the share of chunks faster than 215 BPM, where 1/48 beat is
   shorter than the 5.8 ms hop and the grid interpolates between hops (§4.5).

The absolute lag in 1. is not 0: flux rises as soon as the 23 ms window reaches
an onset (a click in silence peaks about 11 ms early), and mappers time by ear.
Only the differences between groups, and between 1. and 2., carry information.
Writes outputs/alignment/: per_song.csv, profiles.png and --plots images of the
log-Mel as the model sees it, with the note rows drawn on it.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from scripts.preprocess_data import find_audio
from src.data.audio import SAMPLE_RATE, load_osu
from src.data.beat_grid import from_timing_points
from src.data.cache import read_manifest
from src.data.chart_parser import parse_osu
from src.data.mel import HOP, log_mel, open_mel
from src.data.tokenizer import BAR, HOLD_START, TAP

FINE_HOP = 32
LAGS_MS = np.arange(-60.0, 60.25, 0.5)
REACH = 8                                        # grid frames on each side
N_TICK = 72                                      # lane ticks drawn from this band up
_STORE = None


def _init(cache: Path) -> None:
    global _STORE
    _STORE = open_mel(cache)


def flux(logmel: np.ndarray) -> np.ndarray:
    """Positive spectral difference summed over bands, z-scored over the song."""
    f = np.concatenate([[0.0], np.maximum(np.diff(logmel, axis=0), 0.0).sum(axis=1)])
    return (f - f.mean()) / (f.std() + 1e-9)


def peak(curve: np.ndarray, step: float) -> float:
    """Lag of the maximum with a parabolic refinement, in units of step."""
    k = int(np.argmax(curve))
    if 0 < k < len(curve) - 1:
        a, b, c = curve[k - 1], curve[k], curve[k + 1]
        den = a - 2 * b + c
        return (k + (0.5 * (a - c) / den if den else 0.0)) * step
    return k * step


def group_of(info: dict) -> str:
    return f"mp3 {info['mp3_tag']}" if "mp3_tag" in info else info["codec"]


def one(job: tuple) -> dict | None:
    row, root, cache, audio = job
    chart = parse_osu(root / row["path"])
    onsets = np.unique([n.time_ms for n in chart.notes]).astype(np.float64)
    try:
        y, info = load_osu(root / audio)
    except Exception as e:
        return {"key": row["key"], "audio": audio, "error": f"{type(e).__name__}: {e}"}
    env = flux(log_mel(y, hop=FINE_HOP))
    t_env = np.arange(len(env)) * (1000.0 * FINE_HOP / SAMPLE_RATE)
    inside = onsets[(onsets + LAGS_MS[0] >= 0) & (onsets + LAGS_MS[-1] <= t_env[-1])]
    curve = np.array([np.interp(inside + d, t_env, env).mean() for d in LAGS_MS])
    out = {"key": row["key"], "audio": audio, "group": group_of(info), "error": "",
           "trim": info.get("mp3_trim", ""), "onsets": len(inside),
           "peak_ms": round(LAGS_MS[0] + peak(curve, 0.5), 2), "peak_z": round(float(curve.max()), 3),
           "curve": curve}
    tok = cache / "tokens" / f"{row['key']}.npz"
    if _STORE is not None and _STORE.has(row["key"]) and tok.exists():
        z = np.load(tok)
        rows = z["tokens"].reshape(-1, z["tokens"].shape[-1])
        mel = _STORE.chart(row["key"], [tuple(tp) for tp in z["timing_points"]],
                           int(z["cell_offset"]), len(rows))
        f = flux(mel)
        on = np.flatnonzero(np.isin(rows, (TAP, HOLD_START)).any(axis=1))
        on = on[(4 * on - REACH >= 1) & (4 * on + REACH < len(f))]
        grid = np.mean([f[4 * r - REACH:4 * r + REACH + 1] for r in on], axis=0)
        hop_ms = 1000.0 * HOP / SAMPLE_RATE
        out.update(grid_peak_frames=round(peak(grid, 1.0) - REACH, 2),       # 1/48 beat frames
                   frame_ms=round(float(np.median(z["beat_len_ms"])) / 48, 3), grid=grid,
                   first_bar=int(on[0]) // BAR,
                   fast_chunks=float(np.mean(z["beat_len_ms"] / 48 < hop_ms)))
    return out


def overlay(store, cache: Path, key: str, first_bar: int, path: Path, title: str = "",
            bars: int = 4) -> None:
    """The log-Mel the model reads for `bars` bars, with every note row drawn on it."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    z = np.load(cache / "tokens" / f"{key}.npz")
    rows = z["tokens"].reshape(-1, z["tokens"].shape[-1])
    start, n = first_bar * BAR, bars * BAR
    grid = from_timing_points([tuple(tp) for tp in z["timing_points"]])
    mel = store.frames(key, grid, int(z["cell_offset"]), start, n)
    part = rows[start:start + n]
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.imshow(mel.T, origin="lower", aspect="auto", cmap="magma", interpolation="nearest")
    for row in np.flatnonzero(np.isin(part, (TAP, HOLD_START)).any(axis=1)):
        ax.axvline(4 * row, color="cyan", lw=0.6, alpha=0.45)
    for lane in range(part.shape[1]):
        for row in np.flatnonzero(np.isin(part[:, lane], (TAP, HOLD_START))):
            ax.plot([4 * row, 4 * row], [N_TICK + 2 * lane, N_TICK + 2 * lane + 1.6], color="cyan", lw=2)
    for bar in range(0, n, BAR):
        ax.axvline(4 * bar - 0.5, color="white", lw=0.5, alpha=0.6)
    ax.set_title(f"{title}  rows {start}-{start + n}: note rows in cyan (frame 4r), lanes at the top",
                 fontsize=8)
    ax.set_xlabel("frame on the token grid (4 per cell, 1/48 beat)")
    ax.set_ylabel("mel band (song-standardized)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"))
    ap.add_argument("--root", type=Path, default=Path("data/raw"))
    ap.add_argument("--cache", type=Path, default=Path("data/cache"))
    ap.add_argument("--out", type=Path, default=Path("outputs/alignment"))
    ap.add_argument("--split", default="train")
    ap.add_argument("--n", type=int, default=200, help="songs")
    ap.add_argument("--plots", type=int, default=10, help="overlay images to draw")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cached-only", action="store_true",
                    help="only songs with cached log-Mel, so every song also gets check 2")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    a = ap.parse_args(argv)

    rows = [r for r in read_manifest(a.manifest) if r["split"] == a.split
            and r.get("drop", "") == "" and r["sr"] != ""]
    with Pool(a.workers) as pool:
        found = dict(pool.imap(find_audio, [(r, a.root) for r in rows], chunksize=64))
    best: dict[str, dict] = {}
    for r in rows:
        audio = found.get(r["key"])
        if audio and int(r["n_notes"] or 0) > int(best.get(audio, {}).get("n_notes") or -1):
            best[audio] = r
    if a.cached_only:
        store = open_mel(a.cache)
        best = {k: r for k, r in best.items() if store.has(r["key"])}
    songs = sorted(best)
    random.Random(a.seed).shuffle(songs)
    jobs = [(best[s], a.root, a.cache, s) for s in songs[:a.n]]
    print(f"{len(jobs)} songs ({len(best):,} audio files in {a.split})", flush=True)
    with Pool(a.workers, initializer=_init, initargs=(a.cache,)) as pool:
        recs = [r for r in pool.imap(one, jobs) if r]

    ok = [r for r in recs if not r["error"]]
    a.out.mkdir(parents=True, exist_ok=True)
    fields = ["key", "audio", "group", "trim", "onsets", "peak_ms", "peak_z",
              "grid_peak_frames", "frame_ms", "error"]
    with open(a.out / "per_song.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        w.writerows(recs)
    if not ok:
        print("no song could be checked", file=sys.stderr)
        return 2

    overall = float(np.median([r["peak_ms"] for r in ok]))
    pooled_all = np.mean([r["curve"] for r in ok], axis=0)
    print("\n1. audio onsets vs note times, peak lag in ms (median over songs; pooled curve):")
    print(f"   {'group':12s} {'songs':>5s} {'median':>7s} {'IQR':>13s} {'pooled':>7s} {'>10 ms off':>10s}")
    groups: dict[str, list] = {}
    for r in ok:
        groups.setdefault(r["group"], []).append(r)
    for g, rs in [*sorted(groups.items(), key=lambda kv: -len(kv[1])), ("all", ok)]:
        p = np.array([r["peak_ms"] for r in rs])
        pooled = LAGS_MS[0] + peak(np.mean([r["curve"] for r in rs], axis=0), 0.5)
        q1, q3 = np.percentile(p, [25, 75])
        print(f"   {g:12s} {len(rs):5d} {np.median(p):7.2f} {q1:6.2f}..{q3:6.2f} {pooled:7.2f} "
              f"{np.mean(np.abs(p - overall) > 10):10.1%}")
    worst = sorted(ok, key=lambda r: -abs(r["peak_ms"] - overall))[:8]
    print("   farthest from the median: " + "; ".join(f"{r['peak_ms']:+.1f} ms {r['audio']}"
                                                     for r in worst[:4]))

    gridded = [r for r in ok if "grid" in r]
    if gridded:
        g = np.mean([r["grid"] for r in gridded], axis=0)
        frames = np.median([r["grid_peak_frames"] for r in gridded])
        ms = np.median([r["grid_peak_frames"] * r["frame_ms"] for r in gridded])
        print(f"\n2. token grid, {len(gridded)} songs with cached log-Mel: pooled peak at "
              f"{peak(g, 1.0) - REACH:+.2f} frames, median {frames:+.2f} frames = {ms:+.1f} ms "
              f"(1. says {overall:+.1f} ms)")
        print("   frames " + " ".join(f"{d:+d}:{v:.2f}" for d, v in zip(range(-REACH, REACH + 1),
                                                               g / g.max(), strict=True)))
        print(f"   chunks faster than 215 BPM (1/48 beat < hop): "
              f"{np.mean([r['fast_chunks'] for r in gridded]):.1%} of these songs' chunks")
    else:
        print("\n2. no cached log-Mel for these songs: run scripts/preprocess_data.py --mel")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return 0
    fig, ax = plt.subplots(figsize=(8, 4))
    for gname, rs in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        ax.plot(LAGS_MS, np.mean([r["curve"] for r in rs], axis=0), label=f"{gname} ({len(rs)})")
    ax.plot(LAGS_MS, pooled_all, "k--", lw=1, label="all")
    ax.axvline(0, color="grey", lw=0.5)
    ax.set_xlabel("audio time - note time (ms)")
    ax.set_ylabel("mean onset strength (z)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(a.out / "profiles.png", dpi=120)
    plt.close(fig)

    store = open_mel(a.cache, norm="song")
    for r in gridded[:a.plots]:
        overlay(store, a.cache, r["key"], r["first_bar"], a.out / f"overlay_{r['key']}.png",
                title=r["audio"])
    print(f"\nwrote {a.out}/per_song.csv, profiles.png and {min(a.plots, len(gridded))} overlays")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
