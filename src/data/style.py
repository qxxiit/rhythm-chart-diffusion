"""Style labels for conditioning: the song's genre and the chart's mapper (EXPERIMENTS 2026-10-03).

Human charts of one song differ by who made them as much as by the music: on the
train split 1,146 mappers made the 16,442 kept charts, and the model's charts
over-pattern pop, anime and rock songs where the human ones do not. With the
mapper and genre as inputs (each dropped at random in training), one model
learns every style and also the style-free distribution, and sampling picks one.

    data/style.csv      key, set_id, beatmap_id, genre_id, mapper_id, mapper_name
                        (scripts/build_style.py, from data/metadata/beatmapsets.jsonl)
    genre_id            the set's genre on osu! (GENRES); one per song
    mapper_id           the user id on the beatmap: the guest mapper for a guest
                        difficulty; mapper_name only where that user hosts the set
    vocab               what a model was trained with (saved in its checkpoint):
                        "genres": the genre ids, one index each, then the null index
                        "mappers": user ids of mappers with at least min_charts train
                        charts, then "other" (every other mapper), then null
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

GENRES = {1: "unspecified", 2: "video game", 3: "anime", 4: "rock", 5: "pop", 6: "other",
          7: "novelty", 9: "hip hop", 10: "electronic", 11: "metal", 12: "classical",
          13: "folk", 14: "jazz"}
FALLBACK_GENRE = 1                      # unknown genre ids count as "unspecified"
MIN_MAPPER_CHARTS = 20                  # 215 mappers, 74% of the train charts (2026-10-03)
FIELDS = ["key", "set_id", "beatmap_id", "genre_id", "mapper_id", "mapper_name"]


def read_style(path: Path) -> dict[str, dict]:
    """key -> {"genre_id": int | None, "mapper_id": int | None, "mapper_name": str}."""
    out = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out[r["key"]] = {"genre_id": int(r["genre_id"]) if r["genre_id"] else None,
                             "mapper_id": int(r["mapper_id"]) if r["mapper_id"] else None,
                             "mapper_name": r.get("mapper_name", "")}
    return out


def build_vocab(style: dict[str, dict], train_keys, min_charts: int = MIN_MAPPER_CHARTS) -> dict:
    """The vocab from the train charts: mappers with >= min_charts of them, most charts first."""
    counts = Counter(style[k]["mapper_id"] for k in train_keys
                     if k in style and style[k]["mapper_id"] is not None)
    mappers = sorted((m for m, n in counts.items() if n >= min_charts),
                     key=lambda m: (-counts[m], m))
    names = {}
    for k in train_keys:
        s = style.get(k)
        if s and s["mapper_id"] in counts and s["mapper_name"]:
            names.setdefault(str(s["mapper_id"]), s["mapper_name"])
    return {"genres": sorted(GENRES), "mappers": mappers, "min_charts": min_charts,
            "mapper_charts": [counts[m] for m in mappers],
            "mapper_names": {str(m): names.get(str(m), "") for m in mappers}}


def sizes(vocab: dict) -> tuple[int, int]:
    """(n_genres, n_mappers) for DenoiserConfig: the real indices; null is one past them."""
    return len(vocab["genres"]), len(vocab["mappers"]) + 1          # + "other"


def encode(vocab: dict, genre_id: int | None, mapper_id: int | None) -> tuple[int, int]:
    """Vocab indices of a chart's style; unknown -> null (genre) / "other" (a mapper not in
    the vocab); None -> null."""
    n_genres, n_mappers = sizes(vocab)
    if genre_id is None:
        g = n_genres
    else:
        g = vocab["genres"].index(genre_id if genre_id in vocab["genres"] else FALLBACK_GENRE)
    if mapper_id is None:
        m = n_mappers
    else:
        m = vocab["mappers"].index(mapper_id) if mapper_id in vocab["mappers"] else n_mappers - 1
    return g, m


def genre_index(vocab: dict, text: str) -> int:
    """--genre: a genre id or name (GENRES) -> vocab index."""
    gid = int(text) if text.isdigit() else next((i for i, n in GENRES.items() if n == text.lower()),
                                               None)
    if gid not in vocab["genres"]:
        raise ValueError(f"unknown genre {text!r}: one of " +
                         ", ".join(f"{i} {GENRES[i]}" for i in vocab["genres"]))
    return vocab["genres"].index(gid)


def mapper_index(vocab: dict, text: str) -> int:
    """--mapper: a user id or a name from the vocab, or "other" -> vocab index."""
    if text.lower() == "other":
        return len(vocab["mappers"])
    if text.isdigit() and int(text) in vocab["mappers"]:
        return vocab["mappers"].index(int(text))
    by_name = {n.lower(): int(m) for m, n in vocab["mapper_names"].items() if n}
    if text.lower() in by_name:
        return vocab["mappers"].index(by_name[text.lower()])
    raise ValueError(f"mapper {text!r} is not in this model's vocab (generate.py --list-styles)")
