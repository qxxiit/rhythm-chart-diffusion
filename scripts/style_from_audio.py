"""How much of a new song's chart style its audio tells: the long-note share, jack and
trill rate of every val chart, guessed from train charts of about the same SR.

    python scripts/style_from_audio.py                  # outputs/style_from_audio.json
    python scripts/style_from_audio.py --width 0.5 --dims 16 --k 10 30 --draws 20

For a new song, --stats sample (chartstats.sample) takes all three stats from a random
train chart within chartstats.SAMPLE_SR of the target SR: whether the chart is a tap
chart or a long-note chart is then a coin the audio has no say in, and the playtest of
10-07 found long-note charts for songs a mapper would chart with taps and the other way
round. The guesses compared, for each val chart:
    rand          a random train chart in the SR window, as chartstats.sample (--draws)
    sr_mean       the mean of the train charts in the window
    knn<k>        the mean of the k train songs in the window whose audio is nearest
    knn<k>_draw   one of those k charts at random, all stats from the same chart (--draws)
    knn<k>_bpm    knn<k> with the main tempo (log BPM) as one more feature
Audio: the per-band mean and std of the song's log-Mel (logmel/<audio>.json, the numbers
MelStore normalises with), z-scored over the train songs and projected on their first
--dims principal components; a train song counts once (its chart nearest the target SR)
and never the val chart's own audio file. Fewer than --min-pool charts in the window: the
--min-pool nearest by SR, as chartstats.sample.
Scores per stat: r (Pearson) and mae against the val chart's own value; for the long-note
share also tap_as_ln (of the val charts under TAP_MAX, the share given LN_MIN or more),
ln_as_tap (of those at LN_MIN or more, given TAP_MAX or less) and auc (a long-note chart
given more than a tap chart; ties half). Numpy only: no model, no audio decoding.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from src.data.beat_grid import from_timing_points
from src.data.chartstats import NAMES, SAMPLE_SR, read_stats

TAP_MAX = 0.1               # a tap chart: long-note share at most this ...
LN_MIN = 0.3                # ... a long-note chart: at least this (EXPERIMENTS 2026-10-05)
MIN_POOL = 50               # fewer train charts in the SR window: the nearest this many


def audio_features(cache: Path, audio_ids) -> dict[str, np.ndarray]:
    """audio id -> [2 * bands]: the per-band mean and std of its log-Mel (logmel/<id>.json)."""
    out = {}
    for aid in sorted(set(audio_ids)):
        path = cache / "logmel" / f"{aid}.json"
        if path.exists():
            meta = json.loads(path.read_text(encoding="utf-8"))
            out[aid] = np.concatenate([np.asarray(meta["mean"], dtype=np.float64),
                                       np.asarray(meta["std"], dtype=np.float64)])
    return out


def main_bpm(cache: Path, key: str) -> float:
    """BPM of the section covering the most time (BeatGrid.bpm_main); nan without tokens."""
    path = cache / "tokens" / f"{key}.npz"
    if not path.exists():
        return float("nan")
    z = np.load(path)
    return float(from_timing_points([tuple(tp) for tp in z["timing_points"]]).bpm_main)


def embedding(x_train: np.ndarray, dims: int):
    """x -> its first dims principal components, fitted on x_train (z-scored per column)."""
    mu, sd = x_train.mean(axis=0), x_train.std(axis=0) + 1e-9
    z = (x_train - mu) / sd
    centre = z.mean(axis=0)
    _, _, vt = np.linalg.svd(z - centre, full_matrices=False)
    p = vt[:dims]

    def project(x: np.ndarray) -> np.ndarray:
        return ((np.asarray(x, dtype=np.float64) - mu) / sd - centre) @ p.T
    return project


def nearest(pool: np.ndarray, dist: np.ndarray, sr_gap: np.ndarray, audio: np.ndarray,
            n: int) -> list[int]:
    """The n train charts of pool nearest by dist, one per audio file (of a song's charts,
    the one nearest the SR: smallest sr_gap); dist and sr_gap are over pool."""
    seen, out = set(), []
    for i in pool[np.lexsort((sr_gap, dist))]:
        if audio[i] not in seen:
            seen.add(audio[i])
            out.append(int(i))
            if len(out) == n:
                break
    return out


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(a positive scores above a negative), ties half; nan if either is empty."""
    pos, neg = np.asarray(pos, dtype=np.float64), np.asarray(neg, dtype=np.float64)
    if not len(pos) or not len(neg):
        return float("nan")
    d = pos[:, None] - neg[None, :]
    return float((d > 0).mean() + 0.5 * (d == 0).mean())


def scores(true: np.ndarray, guess: np.ndarray, hold: bool) -> dict:
    """r and mae of the guesses; for the long-note share also tap_as_ln, ln_as_tap, auc."""
    true, guess = np.asarray(true, dtype=np.float64), np.asarray(guess, dtype=np.float64)
    r = float(np.corrcoef(true, guess)[0, 1]) if guess.std() > 0 and true.std() > 0 \
        else float("nan")
    out = {"r": r, "mae": float(np.abs(true - guess).mean())}
    if hold:
        tap, ln = true < TAP_MAX, true >= LN_MIN
        out["tap_as_ln"] = float((guess[tap] >= LN_MIN).mean()) if tap.any() else float("nan")
        out["ln_as_tap"] = float((guess[ln] <= TAP_MAX).mean()) if ln.any() else float("nan")
        out["auc"] = auc(guess[ln], guess[tap])
    return out


def mean_scores(runs: list[dict]) -> dict:
    """Per-key mean over draws (nan-aware)."""
    return {k: float(np.nanmean([r[k] for r in runs])) for k in runs[0]}


def guesses(stats: dict, aid_of: dict, feats: dict, bpm_of: dict, *, width: float = SAMPLE_SR,
            dims: int = 16, ks=(10, 30), draws: int = 20, seed: int = 0,
            min_pool: int = MIN_POOL) -> tuple[dict, dict]:
    """({method: {stat: [guess per val chart]}} with draws as lists of such lists,
    {stat: [true value per val chart]}) for the val charts with stats, audio features
    and train charts to compare with."""
    def usable(k: str) -> bool:
        v = stats[k]
        return k in aid_of and aid_of[k] in feats and np.isfinite(v["sr"]) \
            and all(np.isfinite(v[n]) for n in NAMES)
    train = sorted(k for k in stats if stats[k]["split"] == "train" and usable(k))
    val = sorted(k for k in stats if stats[k]["split"] == "val" and usable(k))
    t_aid = np.array([aid_of[k] for k in train])
    t_sr = np.array([stats[k]["sr"] for k in train])
    t_y = {n: np.array([stats[k][n] for k in train]) for n in NAMES}
    songs = sorted(set(t_aid))
    project = embedding(np.stack([feats[a] for a in songs]), dims)
    t_z = project(np.stack([feats[a] for a in t_aid]))
    t_bpm = np.log(np.array([bpm_of.get(k, np.nan) for k in train], dtype=np.float64))
    has_bpm = np.isfinite(t_bpm)
    bpm_sd = float(t_bpm[has_bpm].std()) if has_bpm.sum() > 1 else 0.0
    bpm_scale = float(t_z[:, 0].std()) / bpm_sd if bpm_sd > 0 else 0.0   # as wide as PC 1
    rng = np.random.default_rng(seed)
    methods: dict[str, dict] = {"sr_mean": {n: [] for n in NAMES}}
    for k in ks:
        methods[f"knn{k}"] = {n: [] for n in NAMES}
        methods[f"knn{k}_bpm"] = {n: [] for n in NAMES}
    drawn: dict[str, list] = {"rand": [], f"knn{ks[0]}_draw": []}   # [val][draw] -> index
    true = {n: [] for n in NAMES}
    kept = []
    for key in val:
        s = stats[key]["sr"]
        pool = np.flatnonzero((np.abs(t_sr - s) <= width) & (t_aid != aid_of[key]))
        if len(pool) < min_pool:
            pool = np.argsort(np.abs(t_sr - s) + 1e9 * (t_aid == aid_of[key]))[:min_pool]
        z = project(feats[aid_of[key]][None])[0]
        bpm = np.log(bpm_of.get(key, np.nan))
        gap = np.abs(t_sr[pool] - s)
        d_audio = np.linalg.norm(t_z[pool] - z, axis=1)
        near = nearest(pool, d_audio, gap, t_aid, max(ks))
        near_bpm = near
        if np.isfinite(bpm):
            d_bpm = np.where(has_bpm[pool], np.abs(t_bpm[pool] - bpm) * bpm_scale, 0.0)
            near_bpm = nearest(pool, np.sqrt(d_audio ** 2 + d_bpm ** 2), gap, t_aid, max(ks))
        for n in NAMES:
            true[n].append(stats[key][n])
            methods["sr_mean"][n].append(float(t_y[n][pool].mean()))
            for k in ks:
                methods[f"knn{k}"][n].append(float(t_y[n][near[:k]].mean()))
                methods[f"knn{k}_bpm"][n].append(float(t_y[n][near_bpm[:k]].mean()))
        drawn["rand"].append(rng.choice(pool, size=draws))
        drawn[f"knn{ks[0]}_draw"].append(rng.choice(near[:ks[0]], size=draws))
        kept.append(key)
    for name, idx in drawn.items():
        idx = np.array(idx).reshape(len(kept), draws)
        methods[name] = [{n: t_y[n][idx[:, j]].tolist() for n in NAMES} for j in range(draws)]
    return methods, {"keys": kept, **true}


def evaluate(methods: dict, true: dict) -> dict:
    """{method: {stat: scores}}; draws averaged."""
    out = {}
    for name, g in methods.items():
        runs = g if isinstance(g, list) else [g]
        out[name] = {n: mean_scores([scores(np.array(true[n]), np.array(r[n]), n == "hold_share")
                                     for r in runs]) for n in NAMES}
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--stats-csv", type=Path, default=Path("data/chart_stats.csv"))
    ap.add_argument("--cache", type=Path, default=Path("data/cache"))
    ap.add_argument("--out", type=Path, default=Path("outputs/style_from_audio.json"))
    ap.add_argument("--width", type=float, default=SAMPLE_SR, help="SR window of the train charts")
    ap.add_argument("--dims", type=int, default=16, help="principal components of the audio")
    ap.add_argument("--k", type=int, nargs="+", default=[10, 30],
                    help="neighbours (the first also for knn<k>_draw)")
    ap.add_argument("--draws", type=int, default=20, help="draws of the random guesses")
    ap.add_argument("--min-pool", type=int, default=MIN_POOL)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    if not a.stats_csv.exists():
        print(f"no {a.stats_csv} (scripts/build_chart_stats.py)", file=sys.stderr)
        return 2
    index = a.cache / "logmel" / "index.csv"
    if not index.exists():
        print(f"no {index}", file=sys.stderr)
        return 2
    stats = read_stats(a.stats_csv)
    with open(index, newline="", encoding="utf-8") as f:
        aid_of = {r["key"]: r["audio_id"] for r in csv.DictReader(f) if r["key"] in stats}
    feats = audio_features(a.cache, aid_of.values())
    bpm_of = {k: main_bpm(a.cache, k) for k in aid_of}
    methods, true = guesses(stats, aid_of, feats, bpm_of, width=a.width, dims=a.dims,
                            ks=tuple(a.k), draws=a.draws, seed=a.seed, min_pool=a.min_pool)
    if not true["keys"]:
        print("no val chart with stats and audio features", file=sys.stderr)
        return 2
    result = evaluate(methods, true)
    h = np.array(true["hold_share"])
    summary = {"val_charts": len(true["keys"]), "audio_files": len(feats),
               "tap_charts": int((h < TAP_MAX).sum()), "ln_charts": int((h >= LN_MIN).sum()),
               "width": a.width, "dims": a.dims, "k": a.k, "draws": a.draws, "seed": a.seed,
               "scores": result}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(f"{len(true['keys'])} val charts ({summary['tap_charts']} tap, "
          f"{summary['ln_charts']} long-note), SR window {a.width:g}")
    print(f"{'':14s}" + "".join(f"{n:>20s}" for n in NAMES) + "   tap_as_ln ln_as_tap   auc")
    for name, sc in result.items():
        cells = "".join(f"   r {sc[n]['r']:+.3f} mae {sc[n]['mae']:.3f}" for n in NAMES)
        hs = sc["hold_share"]
        print(f"{name:14s}{cells}   {hs['tap_as_ln']:9.3f} {hs['ln_as_tap']:9.3f} "
              f"{hs['auc']:5.3f}")
    print(f"-> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
