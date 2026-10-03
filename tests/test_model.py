"""Denoiser, forward process, loss weighting and the constrained sampler (design doc §4.7-4.10)."""

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from src.data.chart_parser import Chart, Note  # noqa: E402
from src.data.tokenizer import (  # noqa: E402
    EMPTY,
    MASK,
    PAD,
    TAP,
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
from src.models.sampler import allowed, generate_song, refine_lanes, sample_window  # noqa: E402

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


@pytest.mark.parametrize("order", ["random", "confidence", "noisy"])
def test_continuation_with_an_oracle_rebuilds_the_song(order: str) -> None:
    chart, tokens, metas, stats, mel = song_fixture()
    out = generate_song(Oracle(tokens), mel, 3.0, chart.timing_points, metas[0].cell_offset,
                        stats.n_cells, steps=16, order=order, mode="continue")
    assert np.array_equal(out, tokens)


@pytest.mark.parametrize("mode", ["continue", "independent"])
@pytest.mark.parametrize("order", ["random", "confidence", "noisy"])
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


def test_noisy_order_spans_confidence_and_random() -> None:
    """temperature 0 is the confidence order exactly; a high one opens cells in an
    order unrelated to confidence, the way the random order does."""
    torch.manual_seed(0)
    model = Denoiser(TINY)
    x = np.full((L, K), MASK, dtype=np.int64)
    mel = torch.randn(L * 4, 80)
    run = {o: sample_window(model, x, mel, 3.0, 400.0, steps=8, order=o, temperature=tau,
                            rng=np.random.default_rng(5))
           for o, tau in (("confidence", 1.0), ("noisy", 0.0))}
    assert np.array_equal(run["confidence"], run["noisy"])
    hot = sample_window(model, x, mel, 3.0, 400.0, steps=8, order="noisy", temperature=50.0,
                        rng=np.random.default_rng(5))
    assert not np.array_equal(hot, run["confidence"]) and not np.any(hot == MASK)


def _shuffle_taps(song: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Same rhythm, chord sizes and holds; the taps of every row moved to random free lanes."""
    out = song.copy()
    for row in range(len(out)):
        free = [k for k in range(K) if out[row, k] in (EMPTY, TAP)]
        n = int(sum(out[row, k] == TAP for k in free))
        if 0 < n < len(free):
            out[row, free] = EMPTY
            out[row, rng.choice(free, n, replace=False)] = TAP
    return out


def test_refine_lanes_restores_the_lanes_a_model_knows() -> None:
    """With a model that knows the song, one sweep at temperature 0 moves shuffled taps
    back; holds, chord sizes and the rhythm never change."""
    _, tokens, _, stats, mel = song_fixture()
    flat = tokens.reshape(-1, K).astype(np.int64)
    song = np.concatenate([flat, np.full((L, K), PAD, np.int64)])
    shuffled = _shuffle_taps(song, np.random.default_rng(3))
    assert not np.array_equal(shuffled, song)
    oracle = Oracle(tokens)
    frames = torch.as_tensor(mel[:len(song) * 4])
    out = refine_lanes(oracle, shuffled.copy(), frames, 3.0, lambda row0: 400.0, stats.n_cells,
                       sweeps=1, temperature=0.0, rng=np.random.default_rng(0))
    assert np.array_equal(out, song)


def test_refine_keeps_rhythm_holds_and_grammar() -> None:
    chart, tokens, metas, stats, _ = song_fixture()
    torch.manual_seed(0)
    model = Denoiser(TINY)
    mel = np.random.default_rng(1).normal(size=(len(tokens) * L * 4, 80))
    args = (model, mel, 3.0, chart.timing_points, metas[0].cell_offset, stats.n_cells)
    plain = generate_song(*args, steps=6, seed=2)
    refined = generate_song(*args, steps=6, seed=2, refine=2, lane_temperature=0.5)
    assert len(grammar_violations(refined)) == 0
    def onsets_per_row(x):                                   # TAP or HOLD_START
        return np.isin(x.reshape(-1, K), (TAP, 2)).sum(axis=1)

    def holds(x):
        return np.where(np.isin(x, (2, 3, 4)), x, 0)

    assert np.array_equal(onsets_per_row(plain), onsets_per_row(refined))
    assert np.array_equal(holds(plain), holds(refined))
    assert not np.array_equal(plain, refined)


def test_row_masking_masks_all_lanes_of_a_row_together() -> None:
    x0, *_ = batch(n=4, pad_from=300)
    g = torch.Generator().manual_seed(1)
    rows = torch.tensor([True, False, True, False])
    _, masked = mask_tokens(x0, torch.full((4,), 0.5), g, rows)
    non_pad = (x0 != PAD)
    for i in (0, 2):                                   # whole rows: all lanes alike
        m = masked[i][non_pad[i].all(dim=1)]
        assert torch.all(m.all(dim=1) | ~m.any(dim=1))
        assert 0.4 < m.all(dim=1).float().mean() < 0.6
    m = masked[1]
    assert not torch.all(m.all(dim=1) | ~m.any(dim=1))   # cell by cell: rows get split


def test_hold_bias_and_clean_up() -> None:
    from src.evaluation.holds import hold_stats
    chart, tokens, metas, stats, _ = song_fixture()
    torch.manual_seed(0)
    model = Denoiser(TINY)
    mel = np.random.default_rng(1).normal(size=(len(tokens) * L * 4, 80))
    args = (model, mel, 3.0, chart.timing_points, metas[0].cell_offset, stats.n_cells)
    raw = generate_song(*args, steps=6, seed=2, min_hold=0, release_gap=0)
    clean = generate_song(*args, steps=6, seed=2, refine=1)               # defaults: 3 cells, gap 2
    none = generate_song(*args, steps=6, seed=2, hold_bias=float("-inf"))
    assert hold_stats(raw)["short_holds"] > 0                             # an untrained model
    h = hold_stats(clean)
    assert h["short_holds"] == 0 and h["quick_regrab"] == 0
    assert len(grammar_violations(clean)) == 0
    assert not np.isin(none, (2, 3, 4)).any() and np.isin(none, TAP).any()


def test_hold_rules_follow_the_target_sr() -> None:
    from src.evaluation.holds import hold_spans, release_gaps
    from src.models.sampler import hold_rules
    assert hold_rules(1.5) == (6, 5) and hold_rules(2.3) == (5, 2) and hold_rules(6.0) == (3, 2)
    chart, tokens, metas, stats, _ = song_fixture()
    torch.manual_seed(0)
    model = Denoiser(TINY)
    mel = np.random.default_rng(1).normal(size=(len(tokens) * L * 4, 80))
    easy = generate_song(model, mel, 1.5, chart.timing_points, metas[0].cell_offset,
                         stats.n_cells, steps=6, seed=2)
    spans = hold_spans(easy)
    assert spans and min(e - s for _, s, e in spans) >= 6
    gaps = release_gaps(easy, spans)
    assert np.all((gaps == -1) | (gaps >= 6))


# --- orders and lane passes (EXPERIMENTS 2026-10-02) ------------------------

class Spy(nn.Module):
    """Wraps a model and keeps every input it was called with."""

    def __init__(self, model):
        super().__init__()
        self.model, self.config, self.seen = model, model.config, []

    def forward(self, x, mel, s, b):
        self.seen.append(x[0].cpu().numpy().copy())
        return self.model(x, mel, s, b)


@pytest.mark.parametrize("order,steps", [("block", 16), ("block", 0), ("random", 0)])
def test_new_orders_with_an_oracle_rebuild_the_song(order: str, steps: int) -> None:
    chart, tokens, metas, stats, mel = song_fixture()
    out = generate_song(Oracle(tokens), mel, 3.0, chart.timing_points, metas[0].cell_offset,
                        stats.n_cells, steps=steps, order=order, mode="continue")
    assert np.array_equal(out, tokens)


def test_block_order_fills_beats_left_to_right() -> None:
    from src.data.tokenizer import D
    torch.manual_seed(0)
    spy = Spy(Denoiser(TINY))
    x = np.full((L, K), MASK, dtype=np.int64)
    x[:96] = EMPTY                                     # a fixed prefix, as in continuation
    out = sample_window(spy, x, torch.randn(L * 4, 80), 3.0, 400.0, steps=48, order="block",
                        rng=np.random.default_rng(0))
    assert not np.any(out == MASK) and len(grammar_violations(out, closed=False)) == 0
    assert 48 <= len(spy.seen) <= 48 + 24              # about steps, at least one per beat
    for seen in spy.seen:
        beat = np.flatnonzero((seen == MASK).any(axis=1))[0] // D     # the beat being filled
        assert not np.any(seen[:beat * D] == MASK)                   # every earlier beat done
        assert np.all(seen[(beat + 1) * D:] == MASK)                 # every later beat untouched


def test_sequential_steps_open_one_cell_per_pass() -> None:
    from src.models.sampler import steps_name
    assert steps_name(0) == "seq" and steps_name(128) == "128"
    torch.manual_seed(0)
    for order in ("random", "block", "confidence"):
        spy = Spy(Denoiser(TINY))
        x = np.full((L, K), MASK, dtype=np.int64)
        x[:300] = EMPTY
        sample_window(spy, x, torch.randn(L * 4, 80), 3.0, 400.0, steps=0, order=order,
                      rng=np.random.default_rng(0))
        n_mask = [int((seen == MASK).sum()) for seen in spy.seen]
        assert n_mask == list(range(84 * K, 0, -1))


def test_spread_opens_cells_of_different_beats_together() -> None:
    from src.data.tokenizer import D

    def groupings(spread: bool) -> list[tuple[int, int, int]]:
        torch.manual_seed(0)
        spy = Spy(Denoiser(TINY))
        x = np.full((L, K), MASK, dtype=np.int64)
        sample_window(spy, x, torch.randn(L * 4, 80), 3.0, 400.0, steps=64, spread=spread,
                      rng=np.random.default_rng(1))
        out = []
        for before, after in zip(spy.seen, spy.seen[1:], strict=False):
            new = np.argwhere((before == MASK) & (after != MASK))
            open_beats = len(np.unique(np.flatnonzero((before == MASK).any(axis=1)) // D))
            out.append((len(new), len(np.unique(new[:, 0] // D)), open_beats))
        return out

    spread = groupings(True)
    assert all(beats == min(n, open_beats) for n, beats, open_beats in spread)
    assert any(beats < min(n, open_beats) for n, beats, open_beats in groupings(False))


def test_empty_bias_moves_the_note_count() -> None:
    chart, tokens, metas, stats, _ = song_fixture()
    torch.manual_seed(0)
    model = Denoiser(TINY)
    mel = np.random.default_rng(1).normal(size=(len(tokens) * L * 4, 80))
    args = (model, mel, 3.0, chart.timing_points, metas[0].cell_offset, stats.n_cells)
    onsets = {eb: int(np.isin(generate_song(*args, steps=6, seed=2, empty_bias=eb),
                              (TAP, 2)).sum()) for eb in (-2.0, 0.0, 2.0)}
    assert onsets[2.0] < onsets[0.0] < onsets[-2.0]


def test_forward_lanes_restores_the_lanes_a_model_knows() -> None:
    from src.models.sampler import forward_lanes
    _, tokens, _, stats, mel = song_fixture()
    flat = tokens.reshape(-1, K).astype(np.int64)
    song = np.concatenate([flat, np.full((L, K), PAD, np.int64)])
    shuffled = _shuffle_taps(song, np.random.default_rng(3))
    frames = torch.as_tensor(mel[:len(song) * 4])
    out = forward_lanes(Oracle(tokens), shuffled.copy(), frames, 3.0, lambda row0: 400.0,
                        stats.n_cells, temperature=0.0, rng=np.random.default_rng(0))
    assert np.array_equal(out, song)


def test_forward_lanes_hides_later_lanes_but_not_the_rhythm() -> None:
    from src.models.sampler import forward_lanes, lane_choice_rows
    _, tokens, _, stats, mel = song_fixture()
    flat = tokens.reshape(-1, K).astype(np.int64)
    song = np.concatenate([flat, np.full((L, K), PAD, np.int64)])
    todo = lane_choice_rows(song, stats.n_cells)
    later = np.zeros(song.shape, dtype=bool)
    for row, free, _ in todo:
        later[row, free] = True
    torch.manual_seed(0)
    spy = Spy(Denoiser(TINY))
    frames = torch.as_tensor(mel[:len(song) * 4])
    out = forward_lanes(spy, song.copy(), frames, 3.0, lambda row0: 400.0, stats.n_cells,
                        temperature=0.5, rng=np.random.default_rng(0))
    assert len(spy.seen) == len(todo)
    for (row, free, _), seen in zip(todo, spy.seen, strict=True):
        w0 = int(np.clip(row - (L - L // 4), 0, len(song) - L))
        hidden = seen == MASK
        expect = later[w0:w0 + L].copy()
        expect[:row - w0] = False                      # rows before it: lanes as chosen
        assert np.array_equal(hidden, expect)
        assert hidden[row - w0, free].all() and hidden[row - w0].sum() == len(free)
    def onsets_per_row(x):
        return np.isin(x, (TAP, 2)).sum(axis=1)

    assert np.array_equal(onsets_per_row(out), onsets_per_row(song))
    assert np.array_equal(np.where(np.isin(out, (2, 3, 4)), out, 0),
                          np.where(np.isin(song, (2, 3, 4)), song, 0))


def test_forward_lanes_in_generate_song_keeps_rhythm_holds_and_grammar() -> None:
    chart, tokens, metas, _, _ = song_fixture()
    torch.manual_seed(0)
    model = Denoiser(TINY)
    mel = np.random.default_rng(1).normal(size=(len(tokens) * L * 4, 80))
    args = (model, mel, 3.0, chart.timing_points, metas[0].cell_offset, 700)   # 3 windows
    plain = generate_song(*args, steps=6, seed=2)
    forward = generate_song(*args, steps=6, seed=2, lanes="forward", refine=1)
    assert len(grammar_violations(forward)) == 0
    assert np.array_equal(np.isin(plain, (TAP, 2)).sum(axis=-1),
                          np.isin(forward, (TAP, 2)).sum(axis=-1))
    assert np.array_equal(np.where(np.isin(plain, (2, 3, 4)), plain, 0),
                          np.where(np.isin(forward, (2, 3, 4)), forward, 0))
    assert not np.array_equal(plain, forward)
    with pytest.raises(ValueError):
        generate_song(*args, steps=6, lanes="backward")


def test_probe_hides_what_forward_lanes_hides() -> None:
    """pattern_probe's past+rhythm view and forward_lanes mask the same cells."""
    from scripts.pattern_probe import lane_choice_cells, one_row_view
    from src.models.sampler import lane_choice_rows
    _, tokens, *_ = song_fixture()
    for x0 in tokens:
        x0 = x0.astype(np.int64)
        cells = np.zeros(x0.shape, dtype=bool)
        for row, free, _ in lane_choice_rows(x0, L):
            cells[row, free] = True
        assert np.array_equal(cells, lane_choice_cells(x0))
        choice_rows = np.flatnonzero(cells.any(axis=1))
        if len(choice_rows) == 0:
            continue
        row = int(choice_rows[len(choice_rows) // 2])
        view = one_row_view(x0, row, "past+rhythm")
        assert np.array_equal(view[:row], x0[:row]) and np.all(view[row] == MASK)
        assert np.array_equal(view[row + 1:] == MASK, cells[row + 1:])
        past = one_row_view(x0, row, "past")
        assert np.all(past[row:][x0[row:] != PAD] == MASK) and np.array_equal(past[:row], x0[:row])


def test_previous_single_tap() -> None:
    from scripts.pattern_probe import previous_single
    x = np.full((L, K), EMPTY, dtype=np.int64)
    x[10, 2] = TAP
    assert previous_single(x, 13) == 2 and previous_single(x, 22) == 2
    assert previous_single(x, 23) is None                  # more than a beat later
    x[12, 0] = 2                                           # a hold start counts as an onset
    assert previous_single(x, 14) == 0
    x[12, 1] = TAP                                         # a chord is not a single tap
    assert previous_single(x, 14) is None and previous_single(x, 5) is None


def test_jack_bias_repeats_the_previous_lanes() -> None:
    from src.models.sampler import _lane_sets, jack_bonus
    song = np.full((40, K), EMPTY, dtype=np.int64)
    song[10, 3] = TAP
    assert jack_bonus(song, 13, 0.0) is None and jack_bonus(song, 30, 1.0) is None
    assert list(jack_bonus(song, 13, 1.5)) == [0, 0, 0, 1.5]
    p_tap = np.array([0.5, 0.2, 0.2, 0.1])
    rng = np.random.default_rng(0)
    assert _lane_sets([0, 1, 2, 3], 1, p_tap, 1 - p_tap, 0.0, rng) == (0,)
    assert _lane_sets([0, 1, 2, 3], 1, p_tap, 1 - p_tap, 0.0, rng, jack_bonus(song, 13, 3.0)) == (3,)
