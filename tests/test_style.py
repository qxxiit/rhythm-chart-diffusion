"""Style labels for conditioning (src/data/style.py)."""

import pytest

from src.data.style import build_vocab, encode, genre_index, mapper_index, sizes


def test_vocab_and_encoding() -> None:
    style = {f"k{i}": {"genre_id": [10, 3, 99][i % 3],
                       "mapper_id": 7 if i < 5 else 8 if i < 8 else 9,
                       "mapper_name": "Seven" if i < 5 else ""} for i in range(10)}
    style["k10"] = {"genre_id": None, "mapper_id": None, "mapper_name": ""}
    vocab = build_vocab(style, [f"k{i}" for i in range(11)], min_charts=3)
    assert vocab["mappers"] == [7, 8] and vocab["mapper_charts"] == [5, 3]
    assert vocab["mapper_names"] == {"7": "Seven", "8": ""}
    n_genres, n_mappers = sizes(vocab)
    assert n_mappers == 3                                     # 7, 8 and "other"
    assert encode(vocab, 10, 7) == (vocab["genres"].index(10), 0)
    assert encode(vocab, 99, 9) == (vocab["genres"].index(1), 2)   # unknown genre; other
    assert encode(vocab, None, None) == (n_genres, n_mappers)       # no label
    assert genre_index(vocab, "Electronic") == genre_index(vocab, "10")
    assert mapper_index(vocab, "seven") == 0 and mapper_index(vocab, "8") == 1
    assert mapper_index(vocab, "other") == 2
    with pytest.raises(ValueError):
        mapper_index(vocab, "9")                              # too few charts: not in the vocab
    with pytest.raises(ValueError):
        genre_index(vocab, "polka")
