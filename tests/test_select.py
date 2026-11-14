"""Charts chosen by a judge: k draws of a window together (sampler.sample_windows), the one the
score likes kept (generate_song's select), the lanes chosen so too (sampler.choose_lanes,
relane_song), and the watch that stops a choosing that moves little
(scripts/select_watch.py); EXPERIMENTS 2026-11-08 (5), 2026-11-09 (1)."""

import json
import subprocess
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.data.tokenizer import (  # noqa: E402
    HOLD_BODY,
    HOLD_END,
    HOLD_START,
    MASK,
    PAD,
    TAP,
    K,
    L,
    grammar_violations,
)
from src.models.diffusion import Denoiser  # noqa: E402
from src.models.sampler import (  # noqa: E402
    LANE_SEAM,
    STATS,
    _free_lanes,
    choose_lanes,
    forward_lanes,
    forward_lanes_k,
    generate_song,
    hold_rules,
    lane_spans,
    refine_lanes,
    refine_lanes_k,
    relane_song,
    sample_window,
    sample_windows,
)
from tests.test_model import TINY, song_fixture  # noqa: E402


def onsets(x: np.ndarray) -> int:
    return int(np.isin(x, (TAP, HOLD_START)).sum())


def test_windows_drawn_together_keep_the_grammar_and_differ() -> None:
    torch.manual_seed(0)
    model = Denoiser(TINY)
    x = np.full((L, K), MASK, dtype=np.int64)
    x[300:] = PAD
    mel = torch.randn(L * 4, 80)
    draw = lambda seed, k=3: sample_windows(model, x, mel, 3.0, 400.0, k, steps=6,  # noqa: E731
                                            rng=np.random.default_rng(seed), schedule="cosine")
    outs, again = draw(0), draw(0)
    assert len(outs) == 3 and all(np.array_equal(a, b) for a, b in zip(outs, again, strict=True))
    for o in outs:
        assert not (o == MASK).any() and (o[300:] == PAD).all()
        assert len(grammar_violations(o)) == 0
    assert not np.array_equal(outs[0], outs[1])
    alone = sample_window(model, x, mel, 3.0, 400.0, steps=6, rng=np.random.default_rng(5),
                          schedule="cosine")
    assert np.array_equal(draw(5, k=1)[0], alone)                 # one draw: sample_window's


def test_the_window_kept_is_the_one_the_score_likes() -> None:
    chart, tokens, metas, stats, _ = song_fixture()
    torch.manual_seed(0)
    model = Denoiser(TINY)
    mel = np.random.default_rng(1).normal(size=(len(tokens) * L * 4, 80))
    args = (model, mel, 3.0, chart.timing_points, metas[0].cell_offset, stats.n_cells)
    kw = {"steps": 6, "order": "random", "mode": "continue", "seed": 2}
    windows0, gain0 = STATS["select_windows"], STATS["select_gain"]
    few = generate_song(*args, **kw, select_k=4,
                        select=lambda cands, mel, b: [onsets(c) for c in cands])
    windows = STATS["select_windows"] - windows0
    many = generate_song(*args, **kw, select_k=4,
                         select=lambda cands, mel, b: [-onsets(c) for c in cands])
    assert windows >= 1 and STATS["select_gain"] >= gain0
    assert onsets(few) < onsets(many) and len(grammar_violations(few)) == 0
    plain = generate_song(*args, **kw)
    assert np.array_equal(generate_song(*args, **kw, select=None, select_k=4), plain)
    assert np.array_equal(generate_song(*args, **kw, select=lambda c, m, b: [0], select_k=1),
                          plain)


def kept_parts(x: np.ndarray) -> tuple[np.ndarray, ...]:
    """What the lane passes keep: the onset rows, the long notes, the taps of each row."""
    x = np.asarray(x)
    holds = np.isin(x, (HOLD_START, HOLD_BODY, HOLD_END))
    return (np.isin(x, (TAP, HOLD_START)).any(axis=-1), np.where(holds, x, -1),
            (x == TAP).sum(axis=-1))


def same_parts(a: np.ndarray, b: np.ndarray) -> bool:
    return all(np.array_equal(p, q) for p, q in zip(kept_parts(a), kept_parts(b), strict=True))


def lane0(x: np.ndarray) -> int:
    return int((np.asarray(x)[..., 0] == TAP).sum())


def two_chunks():
    """A tiny model and the first two chunks of the fixture song, with the PAD chunk after
    them that generate_song keeps for the overhang (cut there, a long note may be left
    open: the grammar is checked with closed=False)."""
    chart, tokens, metas, _, mel = song_fixture()
    torch.manual_seed(0)
    model = Denoiser(TINY)
    song = np.concatenate([tokens[:2].reshape(-1, K).astype(np.int64),
                           np.full((L, K), PAD, dtype=np.int64)])
    frames = torch.as_tensor(mel[:len(song) * 4], dtype=torch.float32)
    return model, song, frames, chart, tokens[:2], metas[0].cell_offset, mel


def test_lane_versions_drawn_together_keep_the_rhythm_and_the_holds() -> None:
    model, song, frames, *_ = two_chunks()
    n_cells, beat = 2 * L, (lambda row0: 400.0)
    songs = np.repeat(song[None], 3, axis=0)
    forward_lanes_k(model, songs, frames, 3.0, beat, n_cells, rng=np.random.default_rng(0),
                    release_gap=2)
    refine_lanes_k(model, songs, frames, 3.0, beat, n_cells, sweeps=1,
                   rng=np.random.default_rng(1), release_gap=2)
    for v in songs:
        assert same_parts(v, song) and len(grammar_violations(v, closed=False)) == 0
    assert not np.array_equal(songs[0], songs[1])                      # lanes of their own
    one = song[None].copy()                                             # one version: the
    forward_lanes_k(model, one, frames, 3.0, beat, n_cells, rng=np.random.default_rng(5),
                    release_gap=2)                                      # plain passes
    alone = forward_lanes(model, song.copy(), frames, 3.0, beat, n_cells,
                          rng=np.random.default_rng(5), release_gap=2)
    assert np.array_equal(one[0], alone)
    refine_lanes_k(model, one, frames, 3.0, beat, n_cells, sweeps=1, rng=np.random.default_rng(6),
                   release_gap=2)
    refine_lanes(model, alone, frames, 3.0, beat, n_cells, sweeps=1,
                 rng=np.random.default_rng(6), release_gap=2)
    assert np.array_equal(one[0], alone)


def test_the_lanes_kept_are_those_the_score_likes_and_the_seams_are_drawn_again() -> None:
    model, song, frames, *_ = two_chunks()
    n_cells, beat = 2 * L, (lambda row0: 400.0)
    assert lane_spans(1000) == [(0, 192), (192, 576), (576, 960), (960, 1000)]
    assert lane_spans(100) == [(0, 100)] and lane_spans(193) == [(0, 192), (192, 193)]
    assert lane_spans(n_cells) == [(0, 192), (192, 576), (576, 768)]

    def chosen(score, k=4):
        x = song.copy()
        choose_lanes(model, x, frames, 3.0, beat, n_cells, score, k, refine=1,
                     rng=np.random.default_rng(2), release_gap=2)
        return x
    windows0 = STATS["lane_select_windows"]
    few = chosen(lambda cands, mel, b: [lane0(c) for c in cands])
    many = chosen(lambda cands, mel, b: [-lane0(c) for c in cands])
    assert STATS["lane_select_windows"] - windows0 == 6
    assert same_parts(few, song) and same_parts(many, song) and lane0(few) < lane0(many)
    assert len(grammar_violations(few, closed=False)) == 0
    # the versions as choose_lanes draws them: span 0 and 2 from version 0, span 1 from 1
    versions = np.repeat(song[None], 2, axis=0)
    rng = np.random.default_rng(2)
    forward_lanes_k(model, versions, frames, 3.0, beat, n_cells, rng=rng, release_gap=2)
    refine_lanes_k(model, versions, frames, 3.0, beat, n_cells, sweeps=1, rng=rng, release_gap=2)
    calls = []

    def alternate(cands, mel, b):
        calls.append(len(calls))
        return [0.0 if i == (len(calls) - 1) % 2 else 1.0 for i in range(len(cands))]
    seams0 = STATS["lane_seams"]
    out = chosen(alternate, k=2)
    assert STATS["lane_seams"] - seams0 == 2 and len(calls) == 3
    h = LANE_SEAM
    assert np.array_equal(out[:192 - h], versions[0, :192 - h])
    assert np.array_equal(out[192 + h:576 - h], versions[1, 192 + h:576 - h])
    assert np.array_equal(out[576 + h:n_cells], versions[0, 576 + h:n_cells])
    assert same_parts(out, song) and len(grammar_violations(out, closed=False)) == 0


def test_lanes_drawn_again_on_a_saved_chart() -> None:
    model, _, _, chart, tokens, offset, mel = two_chunks()
    n_cells, gap = 2 * L, hold_rules(3.0)[1]
    shuffled = tokens.reshape(-1, K).astype(np.int64).copy()       # the same chart, other lanes
    rng = np.random.default_rng(9)
    for row in range(n_cells):
        free, n = _free_lanes(shuffled, row, gap)
        if 0 < n < len(free):
            shuffled[row, free] = 0
            shuffled[row, list(rng.choice(free, n, replace=False))] = TAP
    shuffled = shuffled.reshape(tokens.shape).astype(np.int8)
    assert not np.array_equal(shuffled, tokens) and same_parts(shuffled, tokens)
    args = (model, mel, 3.0, chart.timing_points, offset, n_cells)
    a = relane_song(*args[:1], tokens, *args[1:], refine=1, seed=3)
    b = relane_song(*args[:1], shuffled, *args[1:], refine=1, seed=3)
    assert np.array_equal(a, b)                       # the lanes it had never enter
    assert a.shape == tokens.shape and a.dtype == np.int8 and same_parts(a, tokens)
    windows0 = STATS["lane_select_windows"]
    c = relane_song(*args[:1], tokens, *args[1:], refine=1, seed=3, select_k=3,
                    select=lambda cands, mel, b: [lane0(x) for x in cands])
    assert STATS["lane_select_windows"] - windows0 == 3
    assert same_parts(c, tokens) and lane0(c) < lane0(a)


def test_a_song_drawn_with_its_lanes_chosen() -> None:
    chart, tokens, metas, _, _ = song_fixture()
    torch.manual_seed(0)
    model = Denoiser(TINY)
    mel = np.random.default_rng(1).normal(size=(len(tokens) * L * 4, 80))
    args = (model, mel, 3.0, chart.timing_points, metas[0].cell_offset, 2 * L)
    kw = {"steps": 4, "order": "random", "mode": "continue", "seed": 2, "lanes": "forward",
          "refine": 1}
    plain = generate_song(*args, **kw)
    assert np.array_equal(generate_song(*args, **kw, lane_select=None, lane_select_k=4), plain)
    windows0 = STATS["lane_select_windows"]
    few = generate_song(*args, **kw, lane_select_k=3,
                        lane_select=lambda cands, mel, b: [lane0(c) for c in cands])
    many = generate_song(*args, **kw, lane_select_k=3,
                         lane_select=lambda cands, mel, b: [-lane0(c) for c in cands])
    assert STATS["lane_select_windows"] - windows0 == 6
    assert same_parts(few, plain) and same_parts(many, plain)     # the same draw, other lanes
    assert lane0(few) < lane0(many) and len(grammar_violations(few)) == 0


def test_several_judges_choose_together() -> None:
    from src.models.select import combined_scorer
    up = lambda cands, mel, b: [1.0, 2.0, 3.0]                      # noqa: E731
    down = lambda cands, mel, b: [3.0, 2.0, 1.0]                    # noqa: E731
    both = combined_scorer([up, down], [1.0, 2.0])                  # each over its own gap
    assert np.allclose(both(None, None, 0.0), [2.5, 3.0, 3.5])
    assert combined_scorer([up], [5.0]) is up                       # one judge: its logits


def log_lines(n: int, gain: float, field: str = "select") -> str:
    return "".join(f"[{i}/120] f1@50 0.400  sr 3.50 (target 3.40)  100 s  {field} gain "
                   f"{gain:.3f}  1/a.osu\n" for i in range(1, n + 1))


def test_the_watch_stops_a_choosing_that_moves_little(tmp_path) -> None:
    from scripts import select_watch
    (tmp_path / "models.json").write_text(json.dumps({"gap": 3.0}))
    small, big = tmp_path / "s0.log", tmp_path / "s1.log"
    small.write_text("other line\n" + log_lines(30, 0.1))
    big.write_text(log_lines(40, 0.6))
    run = ["--judge", str(tmp_path), "--after", "30", "--every", "0"]
    assert select_watch.main(["--logs", str(small), *run]) == 0
    assert "under 10%" in (tmp_path / "s0.log.STOPPED").read_text()    # 0.1 of 3: 3%
    assert select_watch.main(["--logs", str(big), *run]) == 0
    assert not (tmp_path / "s1.log.STOPPED").exists()                 # 20%: it goes on
    assert select_watch.decide([0.2] * 30, 3.0, 30, 0.1)[0]
    assert not select_watch.decide([0.4] * 30, 3.0, 30, 0.1)[0]
    done = subprocess.Popen([sys.executable, "-c", "pass"])
    done.wait()
    few = tmp_path / "s2.log"
    few.write_text(log_lines(5, 0.1))
    assert select_watch.main(["--logs", str(few), *run, "--pids", str(done.pid)]) == 0
    assert not (tmp_path / "s2.log.STOPPED").exists()                 # ended before 30 songs
    lanes = tmp_path / "s3.log"                                       # the lanes' gain
    lanes.write_text(log_lines(30, 0.1).replace("  1/a.osu", "  lane gain 0.900  1/a.osu"))
    assert select_watch.main(["--logs", str(lanes), *run, "--field", "lane"]) == 0
    assert not (tmp_path / "s3.log.STOPPED").exists()                 # 30% of the gap
    assert select_watch.gains([lanes], "lane") == [0.9] * 30
    assert select_watch.gains([lanes]) == [0.1] * 30
