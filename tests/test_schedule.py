"""How many cells the sampler opens at each step (sampler.open_share, DECISIONS 2026-11-05)."""

from itertools import pairwise

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.data.tokenizer import MASK, PAD, K, L  # noqa: E402
from src.models.sampler import SCHEDULES, generate_song, open_share, sample_window  # noqa: E402
from tests.test_model import Oracle, song_fixture  # noqa: E402


def opened(steps: int, schedule: str) -> list[float]:
    """Expected share of the window opened at each step, first step first."""
    left, out = 1.0, []
    for t in range(steps, 0, -1):
        share = open_share(t, steps, schedule)
        out.append(left * share)
        left *= 1 - share
    return out


def test_both_schedules_open_every_cell_and_linear_is_one_over_t() -> None:
    for steps in (8, 128):
        for schedule in SCHEDULES:
            assert sum(opened(steps, schedule)) == pytest.approx(1.0)
        assert [open_share(t, steps) for t in (steps, 2, 1)] == [1 / steps, 0.5, 1.0]
        assert opened(steps, "linear") == pytest.approx([1 / steps] * steps)
    with pytest.raises(ValueError):
        open_share(3, 8, "sqrt")


def test_cosine_opens_few_cells_first_and_more_at_the_end() -> None:
    share = opened(128, "cosine")
    assert sum(share[:16]) < 0.02 and sum(share[-16:]) > 0.18
    assert all(a <= b + 1e-12 for a, b in pairwise(share))


@pytest.mark.parametrize("order", ["random", "confidence", "noisy"])
def test_an_oracle_rebuilds_the_song_under_the_cosine_schedule(order: str) -> None:
    chart, tokens, metas, stats, mel = song_fixture()
    out = generate_song(Oracle(tokens), mel, 3.0, chart.timing_points, metas[0].cell_offset,
                        stats.n_cells, steps=16, order=order, mode="continue",
                        schedule="cosine")
    assert np.array_equal(out, tokens)


class Count(torch.nn.Module):
    """Uniform logits; counts the forward passes and the cells still MASK at each."""

    def __init__(self):
        super().__init__()
        self.masked = []
        self.anchor = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x, mel, s, b):
        self.masked.append(int((x == MASK).sum()))
        return torch.zeros(*x.shape, 5)


def test_cosine_spends_every_pass_and_opens_slowly_at_first() -> None:
    x = np.full((L, K), MASK)
    x[300:] = PAD
    mel = torch.zeros(L * 4, 80)
    for schedule in SCHEDULES:
        model = Count()
        out = sample_window(model, x, mel, 3.0, 400.0, steps=32, schedule=schedule,
                            rng=np.random.default_rng(0))
        assert not (out == MASK).any() and len(model.masked) == 32
        assert all(a > b for a, b in pairwise(model.masked))
        first = model.masked[0] - model.masked[4]           # cells opened in the first 4 passes
        assert first < 60 if schedule == "cosine" else first > 100
    with pytest.raises(ValueError):
        sample_window(Count(), x, mel, 3.0, 400.0, steps=4, schedule="sqrt")
