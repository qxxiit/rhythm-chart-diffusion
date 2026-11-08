"""Charts chosen by a judge: k draws of a window together (sampler.sample_windows), the one the
score likes kept (generate_song's select), and the watch that stops a choosing that moves
little (scripts/select_watch.py); EXPERIMENTS 2026-11-08 (5)."""

import json
import subprocess
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.data.tokenizer import HOLD_START, MASK, PAD, TAP, K, L, grammar_violations  # noqa: E402
from src.models.diffusion import Denoiser  # noqa: E402
from src.models.sampler import STATS, generate_song, sample_window, sample_windows  # noqa: E402
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


def log_lines(n: int, gain: float) -> str:
    return "".join(f"[{i}/120] f1@50 0.400  sr 3.50 (target 3.40)  100 s  select gain "
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
