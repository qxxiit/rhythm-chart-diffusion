"""Denoiser, forward process, loss weighting and the constrained sampler (design doc §4.7-4.10)."""

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from src.data.chart_parser import Chart, Note  # noqa: E402
from src.data.tokenizer import (  # noqa: E402
    MASK,
    PAD,
    K,
    L,
    decode,
    encode,
    grammar_violations,
    make_metas,
)
from src.models.diffusion import (  # noqa: E402
    N_CLASSES,
    Denoiser,
    DenoiserConfig,
    diffusion_loss,
    mask_tokens,
    n_params,
    sample_gamma,
)
from src.models.sampler import allowed, generate_song, sample_window  # noqa: E402

TINY = DenoiserConfig(d_model=64, lane_dim=16, n_layers=2, n_heads=2, d_ff=128)


def batch(n: int = 4, pad_from: int = 300):
    g = torch.Generator().manual_seed(0)
    x0 = torch.randint(0, N_CLASSES, (n, L, K), generator=g)
    x0[-1, pad_from:] = PAD
    mel = torch.randn(n, L * 4, 80, generator=g)
    return x0, mel, torch.full((n,), 3.0), torch.full((n,), 400.0)


class Fixed(nn.Module):
    """Returns the given logits whatever the input (for testing the loss)."""

    def __init__(self, logits):
        super().__init__()
        self.logits = logits
        self.config = DenoiserConfig()

    def forward(self, x_t, mel, s, b):
        return self.logits


def test_parameter_count_matches_the_design() -> None:
    model = Denoiser()
    blocks = sum(p.numel() for p in model.blocks.parameters())
    assert blocks == 6_320_640                                   # 6 x 1,053,440 as in §4.10
    assert 6_800_000 < n_params(model) < 6_900_000


def test_forward_shapes_and_every_parameter_learns() -> None:
    for audio_add in (True, False):
        model = Denoiser(DenoiserConfig(**{**TINY.__dict__, "audio_add": audio_add}))
        x0, mel, s, b = batch()
        assert model(x0, mel, s, b).shape == (4, L, K, N_CLASSES)
        loss, _ = diffusion_loss(model, x0, mel, s, b)
        loss.backward()
        assert all(p.grad is not None for p in model.parameters())


def test_mask_rate_is_gamma_and_pad_is_never_masked() -> None:
    x0, *_ = batch(n=8)
    gamma = torch.full((8,), 0.3)
    x_t, masked = mask_tokens(x0, gamma, torch.Generator().manual_seed(1))
    real = x0 != PAD
    assert masked[real].float().mean() == pytest.approx(0.3, abs=0.01)
    assert not masked[~real].any() and torch.equal(x_t[~real], x0[~real])
    assert torch.equal(x_t == MASK, masked)


def test_gamma_is_stratified() -> None:
    gamma = sample_gamma(8, generator=torch.Generator().manual_seed(2)).sort().values
    assert torch.allclose(gamma.diff(), torch.full((7,), 1 / 8), atol=1e-6)
    assert gamma.min() > 0 and gamma.max() <= 1


def test_loss_weighting() -> None:
    # A predictor that knows nothing scores log 5 on every scored position. The
    # 1/gamma weight makes the expected loss exactly log 5 x (non-PAD share).
    n = 2000
    x0 = torch.zeros(n, L, K, dtype=torch.long)
    x0[:, 288:] = PAD                                            # a quarter PAD
    blind = Fixed(torch.zeros(n, L, K, N_CLASSES))
    loss, _ = diffusion_loss(blind, x0, None, None, None,
                             generator=torch.Generator().manual_seed(3))
    assert float(loss) == pytest.approx(math.log(5) * 0.75, rel=0.02)
    # a predictor that knows the answer scores 0
    oracle = Fixed(nn.functional.one_hot(x0.clamp(max=4), N_CLASSES).float() * 50)
    loss, _ = diffusion_loss(oracle, x0, None, None, None)
    assert float(loss) < 1e-6


# --- sampler ---------------------------------------------------------------

class Oracle(nn.Module):
    """Knows the whole song; finds the window from mel[0, 0, 0] (frame 4r carries r)."""

    def __init__(self, song: np.ndarray):
        super().__init__()
        self.song = torch.as_tensor(song.reshape(-1, K).astype(np.int64)).clamp(max=4)
        self.config = DenoiserConfig()
        self.anchor = nn.Parameter(torch.zeros(1))

    def forward(self, x, mel, s, b):
        row0 = round(float(mel[0, 0, 0]))
        target = self.song[row0:row0 + L]
        target = torch.cat([target, torch.zeros(L - len(target), K, dtype=torch.long)])
        return nn.functional.one_hot(target, N_CLASSES).float()[None] * 20


def song_fixture():
    rng = np.random.default_rng(0)
    bl, notes = 400.0, []
    for k in range(K):
        t = rng.uniform(-bl, bl)
        while t < 90_000:
            is_ln = rng.random() < 0.3
            dur = bl * rng.choice([0.5, 1, 2, 3])
            notes.append(Note(round(float(t)), k, round(float(t + dur)) if is_ln else None))
            t += (dur if is_ln else 0) + bl * rng.choice([0.25, 0.5, 1])
    notes.sort(key=lambda n: (n.time_ms, n.lane))
    chart = Chart(4, "a.mp3", [(0, bl), (40_123, bl * 0.8)], notes)
    tokens, metas, stats = encode(chart, 3.0)
    mel = np.zeros(((len(tokens) + 1) * L * 4, 80), np.float32)
    mel[:, 0] = np.arange(len(mel)) // 4
    return chart, tokens, metas, stats, mel


@pytest.mark.parametrize("order", ["random", "confidence"])
def test_continuation_with_an_oracle_rebuilds_the_song(order: str) -> None:
    chart, tokens, metas, stats, mel = song_fixture()
    out = generate_song(Oracle(tokens), mel, 3.0, chart.timing_points, metas[0].cell_offset,
                        stats.n_cells, steps=16, order=order, mode="continue")
    assert np.array_equal(out, tokens)


@pytest.mark.parametrize("mode", ["continue", "independent"])
@pytest.mark.parametrize("order", ["random", "confidence"])
def test_untrained_model_still_writes_grammatical_charts(mode: str, order: str) -> None:
    chart, tokens, metas, stats, _ = song_fixture()
    torch.manual_seed(0)
    model = Denoiser(TINY)
    mel = np.random.default_rng(1).normal(size=(len(tokens) * L * 4, 80))
    out = generate_song(model, mel, 3.0, chart.timing_points, metas[0].cell_offset,
                        stats.n_cells, steps=6, order=order, mode=mode, seed=2)
    flat = out.reshape(-1, K)
    assert out.shape == tokens.shape
    assert len(grammar_violations(out)) == 0
    assert np.all(flat[stats.n_cells:] == PAD) and not np.any(flat[:stats.n_cells] == PAD)
    decode(out, make_metas(chart.timing_points, metas[0].cell_offset, len(out), 3.0))


def test_sample_window_keeps_fixed_cells() -> None:
    torch.manual_seed(0)
    model = Denoiser(TINY)
    _, tokens, *_ = song_fixture()
    x = tokens[1].astype(np.int64)
    x[96:] = MASK                                   # the first 2 bars are a fixed prefix
    out = sample_window(model, x, torch.randn(L * 4, 80), 3.0, 400.0, steps=4,
                        rng=np.random.default_rng(0), right_closed=False)
    assert np.array_equal(out[:96], tokens[1][:96])
    assert not np.any(out == MASK)
    assert len(grammar_violations(out, closed=False)) == 0


def test_the_grammar_never_leaves_a_cell_without_a_choice() -> None:
    values = [None, 0, 1, 2, 3, 4]
    assert all(allowed(left, right).any() for left in values for right in values)
