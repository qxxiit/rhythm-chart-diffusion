"""Does the model know where humans rest and where they hold? (EXPERIMENTS 2026-10-07 night)

    python scripts/probe_bars.py --ckpt outputs/full-v4/best.pt --stats oracle
    python scripts/probe_bars.py --ckpt outputs/full-v4/best.pt --stats oracle \
        --charts outputs/full-v4/eval_.../charts.npz       # the context: those charts
    python scripts/probe_bars.py --ckpt outputs/full-v4/best.pt --stats oracle \
        --context none                                     # no chart: audio, SR, stats only

The val songs of evaluate.py --per-song (same --seed, --n), or with --charts the charts in
that file. For every bar inside a song
(between the human chart's first and last note start), with the bar MASK and the rest of
the chart in view, as sampler.rest_bars asks (the human chart, or with --charts the saved
generated charts): the expected note starts in the bar (the sum of p(TAP) + p(HOLD_START)
over its cells) and the expected long-note starts (p(HOLD_START)). Scored against the
human chart:

    rest     the human chart starts no note in the bar (0.85% of the val bars)
    ln bar   4+ note starts in the human bar, at least 80% of them long notes

ROC AUC of the expected starts (fewer = rest) against the audio's loudness (the bar's level
in dB below the song's 95th-percentile bar, and the loudness z of dynamics / rest_bars'
note); of the expected long-note share (long-note starts / starts) for ln bars against the
chart's long-note share (the chart-stat input). For thresholds of the expected starts: the
share of bars below, of those how many humans leave empty (precision), of the human rests
how many are below (recall), and the human note starts in them (what a rest pass on the
human chart would remove). The *_within AUCs pair bars of the same song only (the
chart-level share is no help there): can the model place them inside a song? --context
none masks the whole chart: what the audio, the SR and the stats alone tell (could rests
and long-note bars be decided before sampling?). Writes <out>.csv, one row per bar, and
<out>.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from evaluate import pick_per_song

from src.data import chartstats
from src.data.cache import read_manifest
from src.data.mel import open_mel
from src.data.tokenizer import BAR, HOLD_START, MASK, PAD, TAP, K, L
from src.models.diffusion import Styled, load_denoiser, pick_device
from src.models.sampler import _beat_len_fn, _song_frames, bar_probs

LN_BAR, LN_BAR_MIN = 0.8, 4          # as structure.bar_kinds


def auc(score: np.ndarray, label: np.ndarray) -> float:
    """P(score of a positive > score of a negative), ties half (Mann-Whitney)."""
    pos, neg = score[label], score[~label]
    if not len(pos) or not len(neg):
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order))
    allv = np.concatenate([pos, neg])[order]
    i = 0
    while i < len(allv):                       # average ranks over ties
        j = i
        while j + 1 < len(allv) and allv[j + 1] == allv[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    r_pos = ranks[:len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def auc_within(score: np.ndarray, label: np.ndarray, group: np.ndarray) -> float:
    """auc over the pairs of bars of the same group (song), weighted by their pairs."""
    num = den = 0.0
    for g in np.unique(group):
        m = group == g
        npos, nneg = int(label[m].sum()), int((~label[m]).sum())
        if npos and nneg:
            num += auc(score[m], label[m]) * npos * nneg
            den += npos * nneg
    return num / den if den else float("nan")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=0, help="songs; 0 = all")
    ap.add_argument("--seed", type=int, default=0, help="evaluate.py --per-song's --seed")
    ap.add_argument("--stats", choices=["none", "oracle"], default="none",
                    help="oracle: the human chart's long-note share, jack and trill rate")
    ap.add_argument("--charts", type=Path, default=None,
                    help="charts.npz of an evaluate.py run: those charts as the context")
    ap.add_argument("--context", choices=["chart", "none"], default="chart",
                    help="none: the whole chart MASK (audio, SR and stats alone)")
    ap.add_argument("--thresholds", default="0.5,1,2,3")
    ap.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"))
    ap.add_argument("--cache", type=Path, default=Path("data/cache"))
    ap.add_argument("--fake-mel", action="store_true", help="the plumbing-only mel (tests)")
    ap.add_argument("--out", type=Path, default=None,
                    help="default <ckpt dir>/probe_bars[_<charts run>]")
    ap.add_argument("--device", default="auto")
    a = ap.parse_args(argv)

    store, raw = open_mel(a.cache, fake=a.fake_mel), open_mel(a.cache, fake=a.fake_mel, norm="none")
    rows = [r for r in read_manifest(a.manifest) if r["split"] == a.split and r["sr"]
            and r.get("drop", "") == "" and (a.cache / "tokens" / f"{r['key']}.npz").exists()
            and store.has(r["key"])]
    saved = np.load(a.charts) if a.charts is not None else None
    rows = [r for r in rows if r["key"] in saved.files] if saved is not None \
        else pick_per_song(rows, a.seed)
    if a.n:
        rows = rows[:a.n]
    base = load_denoiser(a.ckpt, pick_device(a.device))
    spec = getattr(base, "chart_stats", None)
    if a.stats == "oracle" and spec is None:
        print("--stats oracle needs a model trained with --chart-stats", file=sys.stderr)
        return 2
    out = a.out or a.ckpt.parent / ("probe_bars" + (f"_{a.charts.parent.name}" if a.charts else "")
                                    + ("_none" if a.context == "none" else ""))
    out.parent.mkdir(parents=True, exist_ok=True)

    table = []
    for i, r in enumerate(rows):
        z = np.load(a.cache / "tokens" / f"{r['key']}.npz")
        tps = [(float(t), float(bl)) for t, bl in z["timing_points"]]
        offset, n_cells, sr = int(z["cell_offset"]), int(z["n_cells"]), float(r["sr"])
        human = z["tokens"].reshape(-1, K)[:n_cells].astype(np.int64)
        context = (saved[r["key"]] if saved is not None else z["tokens"]).reshape(-1, K)
        song = np.concatenate([context.astype(np.int64), np.full((L, K), PAD, np.int64)])
        song[n_cells:] = PAD
        if a.context == "none":
            song[:n_cells] = MASK
        n_rows = (len(z["tokens"]) + 1) * L
        mel = store.chart(r["key"], tps, offset, n_rows)
        lev = np.asarray(raw.chart(r["key"], tps, offset, n_rows), dtype=np.float64)
        stats = None
        if a.stats == "oracle":
            stats = chartstats.encode(spec, chartstats.chart_stats(human))
        model = Styled(base, stats=stats) if stats is not None else base
        if len(song) < n_rows:
            song = np.concatenate([song, np.full((n_rows - len(song), K), PAD, np.int64)])
        frames, _ = _song_frames(model, mel, len(song))
        beat_len = _beat_len_fn(tps, offset)
        n_bars = n_cells // BAR
        starts = np.isin(human[:n_bars * BAR], (TAP, HOLD_START)).reshape(n_bars, BAR, K)
        h_on = starts.sum(axis=(1, 2))
        h_ln = (human[:n_bars * BAR] == HOLD_START).reshape(n_bars, BAR, K).sum(axis=(1, 2))
        c_on = np.isin(song[:n_bars * BAR], (TAP, HOLD_START)).reshape(n_bars, -1).sum(axis=1)
        full = np.flatnonzero(h_on)
        if len(full) < 2:
            continue
        bar_lev = lev[:n_bars * BAR * 4].mean(axis=1).reshape(n_bars, -1).mean(axis=1)
        db = 4.343 * (bar_lev - np.percentile(bar_lev, 95))
        loud = np.asarray(mel[:n_bars * BAR * 4], dtype=np.float64).reshape(n_bars, -1).mean(axis=1)
        lz = (loud - loud.mean()) / (loud.std() + 1e-9)
        share = float(h_ln.sum() / max(h_on.sum(), 1))
        for b in range(full[0], full[-1] + 1):
            p = bar_probs(model, song, frames, sr, beat_len, b * BAR)
            e_on = float((p[..., TAP] + p[..., HOLD_START]).sum())
            e_ln = float(p[..., HOLD_START].sum())
            table.append({"key": r["key"], "bar": b, "human_starts": int(h_on[b]),
                          "human_long": int(h_ln[b]), "context_starts": int(c_on[b]),
                          "expected_starts": round(e_on, 4), "expected_long": round(e_ln, 4),
                          "db": round(float(db[b]), 2), "loud_z": round(float(lz[b]), 3),
                          "chart_hold_share": round(share, 4)})
        print(f"[{i + 1}/{len(rows)}] {r['path']}", flush=True)

    if not table:
        print("no bars to probe (no songs, or none with two bars of notes)", file=sys.stderr)
        return 2
    with open(out.parent / (out.name + ".csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(table)
    col = {k: np.array([t[k] for t in table], dtype=np.float64) for k in table[0] if k != "key"}
    song_id = np.array([t["key"] for t in table])
    rest = col["human_starts"] == 0
    busy = col["human_starts"] >= LN_BAR_MIN
    ln_bar = busy & (col["human_long"] >= LN_BAR * col["human_starts"])
    ln_share = col["expected_long"] / np.maximum(col["expected_starts"], 1e-9)
    summary = {
        "songs": len({t["key"] for t in table}), "bars": len(table),
        "context": "none" if a.context == "none" else str(a.charts) if a.charts else "human",
        "stats": a.stats,
        "rest_share": float(rest.mean()),
        "auc_rest_expected_starts": auc(-col["expected_starts"], rest),
        "auc_rest_db": auc(-col["db"], rest),
        "auc_rest_loud_z": auc(-col["loud_z"], rest),
        "auc_rest_expected_starts_within": auc_within(-col["expected_starts"], rest, song_id),
        "auc_rest_db_within": auc_within(-col["db"], rest, song_id),
        "ln_bar_share_of_busy": float(ln_bar[busy].mean()) if busy.any() else float("nan"),
        "auc_ln_bar_expected_share": auc(ln_share[busy], ln_bar[busy]),
        "auc_ln_bar_chart_share": auc(col["chart_hold_share"][busy], ln_bar[busy]),
        "auc_ln_bar_expected_share_within": auc_within(ln_share[busy], ln_bar[busy],
                                                       song_id[busy]),
        "thresholds": {},
    }
    total = col["human_starts"].sum()
    for t in (float(x) for x in a.thresholds.split(",")):
        below = col["expected_starts"] < t
        summary["thresholds"][f"{t:g}"] = {
            "bars_below": float(below.mean()),
            "precision": float(rest[below].mean()) if below.any() else float("nan"),
            "recall": float(below[rest].mean()) if rest.any() else float("nan"),
            "human_starts_in_them": float(col["human_starts"][below].sum() / max(total, 1)),
            "bars_below_with_context_notes": float((below & (col["context_starts"] > 0)).mean()),
        }
    (out.parent / (out.name + ".json")).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
