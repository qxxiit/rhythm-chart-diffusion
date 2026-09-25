import numpy as np
import pytest

from src.data.beat_grid import from_chart
from src.data.chart_parser import Chart, Note
from src.data.tokenizer import (
    EMPTY,
    HOLD_BODY,
    HOLD_END,
    HOLD_START,
    MASK,
    PAD,
    TAP,
    ChartTooLong,
    D,
    K,
    L,
    decode,
    encode,
    grammar_violations,
)

BL = 500.0                                     # 120 BPM: one cell = 41.67 ms


def make_chart(notes, tps=((0, BL),)) -> Chart:
    return Chart(4, "song.mp3", list(tps), sorted(notes, key=lambda n: (n.time_ms, n.lane)))


def lane(tokens: np.ndarray, k: int, a: int, b: int) -> str:
    return "".join(".o[|]?#"[v] for v in tokens.reshape(-1, K)[a:b, k])


def round_trips(tokens, metas) -> bool:
    tokens2, metas2, _ = encode(decode(tokens, metas))
    return np.array_equal(tokens, tokens2) and \
        [m.start_cell for m in metas] == [m.start_cell for m in metas2]


# --- encode basics ---------------------------------------------------------

def test_encode_uses_reoriginated_cells_and_preserves_negative_cells() -> None:
    chart = make_chart(
        [Note(-250, 0), Note(500, 1, 510), Note(5185, 2, 5185 + round(2 * 400 / D))],
        tps=[(0, 500.0), (5185, 400.0)],
    )

    tokens, metas, stats = encode(chart, sr=4.5)
    full = tokens.reshape(-1, K)

    assert tokens.shape == (1, L, K)
    assert metas[0].cell_offset == 6
    assert metas[0].start_cell == -6
    assert metas[0].sr == 4.5
    assert full[0, 0] == TAP                   # -250 ms, before the first red line
    assert full[18, 1] == TAP                  # a 10 ms hold rounds to one cell -> tap
    assert (full[130, 2], full[131, 2], full[132, 2]) == (HOLD_START, HOLD_BODY, HOLD_END)
    assert np.all(full[133:] == PAD)
    assert (stats.n_notes, stats.n_onsets, stats.n_dropped, stats.n_demoted) == (3, 3, 0, 1)
    assert stats.balanced
    np.testing.assert_array_equal(stats.lane_onsets, [1, 1, 1, 0])


def test_empty_chart() -> None:
    tokens, metas, stats = encode(make_chart([]))

    assert tokens.shape == (0, L, K) and metas == [] and stats.balanced
    assert decode(tokens, metas).notes == []


# --- collision policy: onsets win, releases give way -----------------------

def test_onset_on_previous_hold_end_shortens_the_hold() -> None:
    # hold 1000-1500 ms = cells 24-36; the next hold starts at 1510 ms = cell 36.
    # Before the policy this came out as [|||]|||] : hold_body right after hold_end.
    chart = make_chart([Note(1000, 0, 1500), Note(1510, 0, 2000)])

    tokens, metas, stats = encode(chart)

    assert lane(tokens, 0, 22, 51) == "..[||||||||||][|||||||||||]##"
    assert (stats.n_shortened, stats.n_dropped) == (1, 0)
    assert stats.balanced and len(grammar_violations(tokens)) == 0
    assert round_trips(tokens, metas)


def test_onset_on_previous_hold_end_keeps_the_tap() -> None:
    tokens, _, stats = encode(make_chart([Note(1000, 0, 1500), Note(1510, 0)]))

    assert lane(tokens, 0, 34, 38) == "|]o#"
    assert stats.n_onsets == 2 and stats.balanced


def test_one_cell_hold_shortened_to_zero_becomes_a_tap() -> None:
    # hold cells 24-25, next tap at 1050 ms = cell 25
    tokens, _, stats = encode(make_chart([Note(1000, 0, 1042), Note(1050, 0)]))

    assert lane(tokens, 0, 23, 27) == ".oo#"
    assert (stats.n_shortened, stats.n_shortened_to_tap) == (1, 1)


def test_duplicate_onset_is_dropped_and_counted() -> None:
    tokens, _, stats = encode(make_chart([Note(0, 0), Note(0, 0)]))

    assert tokens[0, 0, 0] == TAP
    assert (stats.n_notes, stats.n_onsets, stats.n_dropped) == (2, 1, 1)
    assert stats.balanced


def test_policy_compares_with_the_last_kept_note() -> None:
    # B duplicates A's start and is dropped; C then lands on A's end cell.
    # C must be compared with A (kept), not with B (dropped).
    chart = make_chart([Note(1000, 0, 1500), Note(1000, 0), Note(1510, 0)])

    tokens, _, stats = encode(chart)

    assert lane(tokens, 0, 23, 38) == ".[||||||||||]o#"
    assert (stats.n_dropped, stats.n_shortened) == (1, 1)
    assert stats.balanced


# --- grammar ---------------------------------------------------------------

def song(*lane0: int) -> np.ndarray:
    x = np.full((len(lane0), K), EMPTY, dtype=np.int8)
    x[:, 0] = lane0
    return x


def test_grammar_accepts_a_valid_song() -> None:
    assert len(grammar_violations(song(TAP, HOLD_START, HOLD_BODY, HOLD_END, EMPTY))) == 0
    assert len(grammar_violations(song(HOLD_START, HOLD_END, HOLD_START, HOLD_END))) == 0


def test_grammar_rejects_body_after_end() -> None:
    bad = grammar_violations(song(HOLD_START, HOLD_END, HOLD_BODY, HOLD_END))
    assert [tuple(b) for b in bad] == [(2, 0)]


def test_grammar_closed_song_vs_open_window() -> None:
    starts_inside = song(HOLD_BODY, HOLD_END, EMPTY)
    left_open = song(EMPTY, HOLD_START, HOLD_BODY)

    assert [tuple(b) for b in grammar_violations(starts_inside)] == [(0, 0)]
    assert [tuple(b) for b in grammar_violations(left_open)] == [(2, 0)]
    assert len(grammar_violations(starts_inside, closed=False)) == 0
    assert len(grammar_violations(left_open, closed=False)) == 0


def test_grammar_rejects_mask_and_misplaced_pad() -> None:
    assert len(grammar_violations(song(EMPTY, MASK, EMPTY))) == 1
    pad_in_middle = song(TAP, EMPTY, EMPTY)
    pad_in_middle[1] = PAD
    assert len(grammar_violations(pad_in_middle)) > 0
    mixed_row = song(TAP, PAD)                         # PAD in lane 0 only
    assert len(grammar_violations(mixed_row)) > 0
    pad_suffix = song(TAP, EMPTY, EMPTY)
    pad_suffix[1:] = PAD
    assert len(grammar_violations(pad_suffix)) == 0


def test_decode_refuses_bad_grammar() -> None:
    tokens, metas, _ = encode(make_chart([Note(1000, 0, 1500)]))
    tokens[0, 30, 0] = EMPTY                           # cut the hold body

    with pytest.raises(ValueError):
        decode(tokens, metas)


# --- decode / round trip ---------------------------------------------------

def test_hold_across_a_chunk_boundary_round_trips() -> None:
    start_ms, end_ms = round(380 * BL / D), round(390 * BL / D)
    tokens, metas, _ = encode(make_chart([Note(start_ms, 1, end_ms)]))

    assert tokens.shape[0] == 2
    assert tokens[0, 380, 1] == HOLD_START and tokens[1, 390 - L, 1] == HOLD_END
    assert round_trips(tokens, metas)
    note = decode(tokens, metas).notes[0]
    assert abs(note.time_ms - start_ms) <= 1 and abs(note.end_ms - end_ms) <= 1


def test_decode_returns_grid_times() -> None:
    chart = make_chart([Note(-250, 0), Note(1003, 1), Note(5185, 2), Note(5230, 3, 5700)],
                       tps=[(0, 500.0), (5185, 400.0)])
    tokens, metas, _ = encode(chart)

    back = decode(tokens, metas)
    grid = from_chart(chart)

    assert [(n.time_ms, n.lane) for n in back.notes] == \
        [(round(grid.time_from_cell(grid.cell_index(n.time_ms, D), D)), n.lane) for n in chart.notes]
    assert round_trips(tokens, metas)


def random_chart(rng: np.random.Generator) -> Chart:
    """LN-heavy chart: holds followed by the next note 1/16 - 1/2 beat later."""
    bl = 60000 / rng.uniform(120, 260)
    tps = [(0, bl)]
    if rng.random() < 0.4:                            # off-beat red line with a tempo change
        tps.append((int(rng.uniform(20_000, 30_000)), bl * rng.uniform(0.7, 1.3)))
    notes = []
    for k in range(K):
        t = rng.uniform(-bl, bl)
        while t < 60_000:
            is_ln = rng.random() < 0.6
            dur = bl * rng.choice([0.25, 0.5, 1.0, 1.5])
            notes.append(Note(round(float(t)), k, round(float(t + dur)) if is_ln else None))
            t += (dur if is_ln else 0) + bl * rng.choice([1 / 16, 1 / 12, 1 / 8, 1 / 4, 1 / 2])
    if rng.random() < 0.3:                            # a duplicated object
        notes.append(Note(notes[0].time_ms, notes[0].lane))
    return make_chart(notes, tps)


def test_random_charts_keep_every_invariant() -> None:
    rng = np.random.default_rng(0)
    shortened = 0
    for _ in range(150):
        tokens, metas, stats = encode(random_chart(rng))   # raises on a grammar violation

        assert stats.balanced
        assert round_trips(tokens, metas)
        shortened += stats.n_shortened
    assert shortened > 0                               # the policy was actually exercised


# --- token range -----------------------------------------------------------

def test_audio_range_covers_intro_and_outro() -> None:
    # first red line at 1500 ms: audio time 0 is cell -36; audio ends at 10 s = cell 204
    chart = make_chart([Note(2000, 0), Note(3000, 1)], tps=[(1500, BL)])

    by_notes, m1, _ = encode(chart)
    by_audio, m2, s2 = encode(chart, audio_ms=10_000)

    assert m1[0].cell_offset == 0 and m1[0].start_cell == 0
    assert np.all(by_notes.reshape(-1, K)[37:] == PAD)          # PAD right after the last note
    assert m2[0].cell_offset == 36 and m2[0].start_cell == -36   # row 0 is audio time 0
    assert s2.n_cells == 36 + 203 + 1                            # cell 203 is the last inside
    full = by_audio.reshape(-1, K)
    assert full[36 + 12, 0] == TAP and full[36 + 36, 1] == TAP
    assert np.all(full[: s2.n_cells] != PAD) and np.all(full[s2.n_cells:] == PAD)
    assert len(grammar_violations(by_audio)) == 0


def test_extreme_bpm_section_hits_the_guard() -> None:
    chart = make_chart([Note(3000, 0)], tps=[(0, BL), (1000, 0.01), (2000, BL)])

    with pytest.raises(ChartTooLong):
        encode(chart)
