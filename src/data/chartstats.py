"""Chart stats as inputs: how much of a chart is long notes, jacks and trills (EXPERIMENTS 2026-10-05).

Human charts commit to a style per chart, and the sampled ones do not: over the 240 val
songs the AI's long-note share, jack rate and trill rate vary half as much from song to
song as the human charts' (per-song SD of the long-note share 0.09 against 0.18, jacks
0.02 against 0.06), and they barely follow the human chart of the same song
(correlation 0.12 / 0.01 / 0.00; chords 0.71). The song does not decide these by itself:
two charts of the same song from different mapsets agree on the long-note share at 0.25
(chords 0.71), and genre and SR explain 7% of it. They are the mapper's choice, like
the SR, so the model gets them as inputs, like the SR, and sampling chooses them.

    data/chart_stats.csv   key, split, sr, hold_share, move_jack, move_trill
                           (scripts/build_chart_stats.py, from the token cache)
    hold_share             long notes / onsets (evaluation.holds.hold_stats)
    move_jack, move_trill  evaluation.patterns.move_shares: of the single-note moves within
                           a beat, the same lane again; of two lane-changing single-note
                           moves in a row, straight back (1-3-1)
    spec                   what a model was trained with (saved in its checkpoint):
                           {"names": NAMES, "bins": BINS, "edges": [[...] per name]}, the
                           edges the quantiles of the train charts; a value goes to bucket
                           searchsorted(edges, value, "left"), 0 .. bins - 1 (a value on an
                           edge goes below it, so the charts with none, e.g. no jacks at all,
                           an eighth of them, have bucket 0 to themselves), and a missing one
                           (nan, or dropped) to the null bucket, index bins
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from src.evaluation.holds import hold_stats
from src.evaluation.patterns import events, move_shares

NAMES = ("hold_share", "move_jack", "move_trill")
SHORT = {"hold_share": "ln", "move_jack": "jack", "move_trill": "trill"}   # spec strings
BINS = 8
FIELDS = ["key", "split", "sr", *NAMES]
SAMPLE_SR = 0.3            # "sample": a train chart within this SR of the target


def chart_stats(tokens: np.ndarray) -> dict[str, float]:
    """{name: value} of one chart ([..., K] tokens); nan where a chart has no such moves."""
    x = np.asarray(tokens).reshape(-1, np.asarray(tokens).shape[-1])
    rows, masks = events(x)
    moves = move_shares(rows, masks)
    out = {"hold_share": hold_stats(x)["hold_share"]}
    out.update({n: moves.get(n, float("nan")) for n in ("move_jack", "move_trill")})
    return {n: float(out[n]) for n in NAMES}


def read_stats(path: Path) -> dict[str, dict]:
    """key -> {"split", "sr", name: float (nan if empty)}."""
    out = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out[r["key"]] = {"split": r["split"], "sr": float(r["sr"]) if r["sr"] else float("nan"),
                             **{n: float(r[n]) if r[n] != "" else float("nan") for n in NAMES}}
    return out


def build_spec(stats: dict[str, dict], train_keys, bins: int = BINS) -> dict:
    """Quantile edges of each stat over the train charts (equal shares per bucket, ties
    merged: a stat with many equal values gets fewer buckets)."""
    edges = []
    for n in NAMES:
        v = np.array([stats[k][n] for k in train_keys if k in stats], dtype=np.float64)
        v = v[np.isfinite(v)]
        q = np.quantile(v, np.arange(1, bins) / bins) if len(v) else np.array([])
        edges.append([float(e) for e in np.unique(q)])
    return {"names": list(NAMES), "bins": bins, "edges": edges}


def null_index(spec: dict) -> int:
    return int(spec["bins"])


def encode(spec: dict, values) -> tuple[int, ...]:
    """Buckets of {name: value} (or a sequence in spec order); nan / None / missing -> null."""
    if isinstance(values, dict):
        values = [values.get(n) for n in spec["names"]]
    out = []
    for v, e in zip(values, spec["edges"], strict=True):
        ok = v is not None and np.isfinite(v)
        out.append(int(np.searchsorted(e, v, side="left")) if ok else null_index(spec))
    return tuple(out)


def parse(text: str) -> dict[str, float]:
    """"ln=0.3,jack=0.05" (or the full names) -> {name: value}; names left out stay null."""
    by_short = {v: k for k, v in SHORT.items()}
    out = {}
    for part in filter(None, (p.strip() for p in text.split(","))):
        k, _, v = part.partition("=")
        name = by_short.get(k.strip(), k.strip())
        if name not in NAMES or not v:
            raise ValueError(f"bad chart stat {part!r}: name=value with names "
                             + ", ".join(f"{SHORT[n]} ({n})" for n in NAMES))
        out[name] = float(v)
    return out


def sample(stats: dict[str, dict], sr: float, rng: np.random.Generator,
           split: str = "train", width: float = SAMPLE_SR) -> tuple[str, dict[str, float]]:
    """(key, its stats) of a random chart of the split within width of SR sr (the nearest
    ones if none is): a style a human chose for a chart of that difficulty, all stats from
    the same chart, so they go together as they do in human charts."""
    keys = sorted(k for k, v in stats.items() if v["split"] == split and np.isfinite(v["sr"]))
    srs = np.array([stats[k]["sr"] for k in keys])
    near = np.flatnonzero(np.abs(srs - sr) <= width)
    if not len(near):
        near = np.argsort(np.abs(srs - sr))[:50]
    k = keys[int(rng.choice(near))]
    return k, {n: stats[k][n] for n in NAMES}
