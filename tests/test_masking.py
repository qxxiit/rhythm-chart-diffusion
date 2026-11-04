"""Loss weights and masks of the training objective (DECISIONS 2026-11-05): count_weight,
block_masks, complementary masks, and the default path left as it was."""

import math

import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from src.data.tokenizer import MASK, PAD, K, L  # noqa: E402
from src.models.diffusion import (  # noqa: E402
    BAR_ROWS,
    N_CLASSES,
    Denoiser,
    DenoiserConfig,
    block_masks,
    count_weight,
    diffusion_loss,
    mask_tokens,
    sample_gamma,
)

TINY = DenoiserConfig(d_model=64, lane_dim=16, n_layers=2, n_heads=2, d_ff=128)


def batch(n: int = 4, pad_from: int = 300):
    g = torch.Generator().manual_seed(0)
    x0 = torch.randint(0, N_CLASSES, (n, L, K), generator=g)
    x0[-1, pad_from:] = PAD
    mel = torch.randn(n, L * 4, 80, generator=g)
    return x0, mel, torch.full((n,), 3.0), torch.full((n,), 400.0)


class Uniform(nn.Module):
    """Uniform logits the size of its input (CE log 5 everywhere); keeps the last x_t."""

    def __init__(self):
        super().__init__()
        self.config = DenoiserConfig()
        self.seen = None

    def forward(self, x_t, mel, s, b, **style):
        self.seen = x_t.clone()
        return torch.zeros(*x_t.shape, N_CLASSES)


def test_count_weight_has_the_expectation_of_one_over_gamma_and_less_variance() -> None:
    # n positions, gamma ~ U(0, 1), each masked with probability gamma; the loss of a masked
    # position grows with how much is masked, as a model's does. Both weights estimate
    # sum_k S(k) / k = sum_{k=1..n} (0.1 + k / n) = 0.1 n + (n + 1) / 2.
    g = torch.Generator().manual_seed(0)
    n, draws = 12, 400_000
    gamma = torch.rand(draws, generator=g).clamp_min(1e-3)          # as sample_gamma clips
    masked = (torch.rand(draws, n, generator=g) < gamma[:, None]).view(draws, n, 1)
    k = masked.sum(dim=(1, 2)).float()
    summed = k * (0.1 + k / n)
    exact = 0.1 * n + (n + 1) / 2
    by_count = summed * count_weight(masked, torch.ones_like(masked))
    by_gamma = summed / gamma
    assert float(by_count.mean()) == pytest.approx(exact, rel=0.01)
    assert float(by_gamma.mean()) == pytest.approx(exact, rel=0.01)
    assert float(by_count.var()) < 0.85 * float(by_gamma.var())     # 17.8 against 24.1


def test_count_weight_given_k_is_the_mean_of_one_over_gamma() -> None:
    g = torch.Generator().manual_seed(1)
    n, draws = 8, 400_000
    gamma = torch.rand(draws, generator=g)
    k = (torch.rand(draws, n, generator=g) < gamma[:, None]).sum(dim=1)
    for kk in (2, 4, 8):                         # k = 1: E[1/gamma] is finite, the variance not
        assert float((1 / gamma[k == kk]).mean()) == pytest.approx((n + 1) / kk, rel=0.03)


def test_count_weight_counts_rows_in_row_masked_chunks() -> None:
    masked = torch.zeros(2, L, K, dtype=torch.bool)
    masked[:, :3] = True                                     # 3 whole rows = 12 cells
    valid = torch.ones(2, L, K, dtype=torch.bool)
    valid[:, 100:] = False                                   # 100 rows, 400 cells
    w = count_weight(masked, valid, rows=torch.tensor([True, False]))
    assert float(w[0]) == pytest.approx(101 / 3)
    assert float(w[1]) == pytest.approx(401 / 12)
    assert float(count_weight(torch.zeros_like(masked), valid)[0]) == 0.0


def test_block_masks_are_bars_or_all_but_two_bars_and_never_pad() -> None:
    x0 = torch.zeros(64, L, K, dtype=torch.long)
    x0[-1, 300:] = PAD
    m = block_masks(x0, torch.Generator().manual_seed(2))
    assert m.shape == x0.shape and not (m & (x0 == PAD)).any()
    assert torch.equal(m, block_masks(x0, torch.Generator().manual_seed(2)))
    kinds = []
    for i in range(64):
        bars = m[i, :BAR_ROWS * (L // BAR_ROWS)].view(L // BAR_ROWS, -1)
        whole = bool(bars.all(dim=1).any())
        prefix = not m[i, :2 * BAR_ROWS].any() and bool(m[i, 2 * BAR_ROWS:].any())
        assert whole or prefix
        kinds.append(whole)
    assert 10 < sum(kinds) < 54


def test_complementary_masks_cover_every_position_once() -> None:
    x0, mel, s, b = batch()
    model = Uniform()
    loss, info = diffusion_loss(model, x0, mel, s, b, complementary=True,
                                generator=torch.Generator().manual_seed(3))
    first, second = model.seen[:4] == MASK, model.seen[4:] == MASK
    assert not (first & second).any()
    assert torch.equal(first | second, x0 != PAD)
    assert math.isfinite(float(loss)) and info["mask_rate"] == pytest.approx(0.5)


def test_count_weight_with_a_uniform_predictor_is_exact() -> None:
    # CE log 5 on each masked position: (n + 1) / k * k log 5 = (n + 1) log 5 per chunk
    x0, mel, s, b = batch()
    loss, info = diffusion_loss(Uniform(), x0, mel, s, b, gamma=torch.full((4,), 0.3),
                                weight="count", generator=torch.Generator().manual_seed(4))
    n = (x0 != PAD).sum(dim=(1, 2)).float()
    assert float(loss) == pytest.approx(float(((n + 1) * math.log(5)).mean()) / (L * K))
    assert info["masked_ce_onset_rows"] == pytest.approx(math.log(5))
    assert info["masked_ce_other_rows"] == pytest.approx(math.log(5))


def test_the_default_loss_is_the_old_one_draw_for_draw() -> None:
    torch.manual_seed(0)
    model = Denoiser(TINY)
    x0, mel, s, b = batch()
    for row_mask in (0.0, 0.5):
        loss, _ = diffusion_loss(model, x0, mel, s, b, row_mask=row_mask,
                                 generator=torch.Generator().manual_seed(5))
        g = torch.Generator().manual_seed(5)                     # the old code, step by step
        gamma = sample_gamma(4, generator=g)
        rows = torch.rand(4, generator=g) < row_mask if row_mask > 0 else None
        x_t, masked = mask_tokens(x0, gamma, g, rows)
        logits = model(x_t, mel, s, b).float()
        target = torch.where(x0 == PAD, torch.zeros_like(x0), x0)
        ce = nn.functional.cross_entropy(logits.reshape(-1, N_CLASSES), target.reshape(-1),
                                         reduction="none").view_as(x0)
        old = ((ce * masked).sum(dim=(1, 2)) / gamma).mean() / (L * K)
        assert torch.equal(loss, old)


@pytest.mark.parametrize("weight", ["gamma", "count"])
@pytest.mark.parametrize("complementary", [False, True])
def test_every_option_trains(weight: str, complementary: bool) -> None:
    torch.manual_seed(0)
    model = Denoiser(TINY)
    x0, mel, s, b = batch()
    loss, info = diffusion_loss(model, x0, mel, s, b, row_mask=0.5, weight=weight,
                                complementary=complementary, block_mask=0.5,
                                generator=torch.Generator().manual_seed(6))
    loss.backward()
    assert math.isfinite(loss.item()) and all(p.grad is not None for p in model.parameters())
    assert info["weight_max"] > 0 and info["chunk_spread"] >= 1
    with pytest.raises(ValueError):
        diffusion_loss(model, x0, mel, s, b, weight="half")
