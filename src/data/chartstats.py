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
    MORE                   move_stair, hold_beats (2026-11-19 (2): the user's play found each
                           human chart keeping to its own patterns; over the 240 val songs a
                           song's stair share follows its human chart's at 0.17, and the
                           long notes' length varies half as much; build_chart_stats --more,
                           train.py --stats-names)
    hold_share             long notes / onsets (evaluation.holds.hold_stats)
    move_jack, move_trill  evaluation.patterns.move_shares: of the single-note moves within
                           a beat, the same lane again; of two lane-changing single-note
                           moves in a row, straight back (1-3-1)
    move_stair             the same: two lane-changing moves in a row the same step (1-2-3)
    hold_beats             the long notes' mean length in beats (nan without any)
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

NAMES = ("hold_share", "move_jack", "move_trill")                 # full-v4's
MORE = ("move_stair", "hold_beats")
ALL = NAMES + MORE
SHORT = {"hold_share": "ln", "move_jack": "jack", "move_trill": "trill",   # spec strings
         "move_stair": "stair", "hold_beats": "lnlen"}
BINS = 8
FIELDS = ["key", "split", "sr", *NAMES]
FIELDS_ALL = ["key", "split", "sr", *ALL]
SAMPLE_SR = 0.3            # "sample": a train chart within this SR of the target


def chart_stats(tokens: np.ndarray) -> dict[str, float]:
    """{name: value} of one chart ([..., K] tokens), every name of ALL; nan where a chart has
    no such moves (or no long notes, for hold_beats)."""
    x = np.asarray(tokens).reshape(-1, np.asarray(tokens).shape[-1])
    rows, masks = events(x)
    moves = move_shares(rows, masks)
    holds = hold_stats(x)
    out = {"hold_share": holds["hold_share"], "hold_beats": holds.get("hold_beats", float("nan"))}
    out.update({n: moves.get(n, float("nan")) for n in ("move_jack", "move_trill", "move_stair")})
    return {n: float(out[n]) for n in ALL}


def read_stats(path: Path) -> dict[str, dict]:
    """key -> {"split", "sr", name: float (nan if empty)}, the names of ALL the file has."""
    out = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        names = [n for n in ALL if n in (reader.fieldnames or [])]
        for r in reader:
            out[r["key"]] = {"split": r["split"], "sr": float(r["sr"]) if r["sr"] else float("nan"),
                             **{n: float(r[n]) if r[n] != "" else float("nan") for n in names}}
    return out


def _edges(stats: dict[str, dict], train_keys, name: str, bins: int) -> list[float]:
    v = np.array([stats[k].get(name, np.nan) for k in train_keys if k in stats], dtype=np.float64)
    v = v[np.isfinite(v)]
    q = np.quantile(v, np.arange(1, bins) / bins) if len(v) else np.array([])
    return [float(e) for e in np.unique(q)]


def build_spec(stats: dict[str, dict], train_keys, bins: int = BINS, names=NAMES) -> dict:
    """Quantile edges of each stat over the train charts (equal shares per bucket, ties
    merged: a stat with many equal values gets fewer buckets)."""
    for n in names:
        if n not in ALL:
            raise ValueError(f"unknown chart stat {n!r}: one of {', '.join(ALL)}")
    return {"names": list(names), "bins": bins,
            "edges": [_edges(stats, train_keys, n, bins) for n in names]}


def extend_spec(spec: dict, stats: dict[str, dict], train_keys, names) -> dict:
    """spec with the names of names it does not have added after its own (their edges from
    the train charts, spec's bins); its own names and edges stay, so a model trained on spec
    reads its buckets as before (train.py --init --stats-names)."""
    if list(names)[:len(spec["names"])] != list(spec["names"]):
        raise ValueError(f"{list(names)} does not start with the model's {spec['names']}")
    new = [n for n in names if n not in spec["names"]]
    extra = build_spec(stats, train_keys, spec["bins"], new)
    return {"names": [*spec["names"], *new], "bins": spec["bins"],
            "edges": [*spec["edges"], *extra["edges"]]}


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
        if name not in ALL or not v:
            raise ValueError(f"bad chart stat {part!r}: name=value with names "
                             + ", ".join(f"{SHORT[n]} ({n})" for n in ALL))
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
    return k, {n: stats[k][n] for n in ALL if n in stats[k]}
