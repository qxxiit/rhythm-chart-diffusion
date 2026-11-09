"""What a judge is shown (judge.py --view), when it may stop (--stop-bounds) and how the songs
are shared among processes (evaluate.py --shard); EXPERIMENTS 2026-11-08 (2)."""

import numpy as np

from src.data.tokenizer import EMPTY, HOLD_BODY, HOLD_END, HOLD_START, PAD, TAP, K


def chart() -> np.ndarray:
    x = np.full((12, K), EMPTY, dtype=np.int8)
    x[0, 0] = TAP
    x[1, [1, 2]] = TAP                                   # a chord
    x[2, 3] = HOLD_START
    x[3:5, 3] = HOLD_BODY
    x[5, 3] = HOLD_END
    x[4, 0] = TAP                                        # a tap beside the hold's body
    x[6, :] = TAP                                        # four at once
    x[10:, :] = PAD                                      # past the end
    return x


def test_each_view_keeps_what_it_says() -> None:
    from scripts.judge import view_tokens
    x = chart()
    assert view_tokens(x, "full") is x
    taps = view_tokens(x, "taps")
    assert not np.isin(taps, (HOLD_START, HOLD_BODY, HOLD_END)).any()
    assert taps[2, 3] == TAP and (taps[3:6, 3] == EMPTY).all()     # the head stays, as a tap
    assert (taps[:, :3] == x[:, :3]).all() and (taps[10:] == PAD).all()
    onsets = np.isin(x, (TAP, HOLD_START)).sum(axis=1)
    for seed in range(5):
        r = view_tokens(x, "rhythm", (seed, 1, 2))
        assert ((r == TAP).sum(axis=1) == onsets).all()           # the rows and their counts
        assert set(np.unique(r[:10])) <= {EMPTY, TAP} and (r[10:] == PAD).all()
    a, b = view_tokens(x, "rhythm", (0, 1, 2)), view_tokens(x, "rhythm", (0, 1, 2))
    assert (a == b).all()                                          # the same seed, the same draw
    lanes = {tuple(np.flatnonzero(view_tokens(x, "rhythm", (s, 1, 2))[0] == TAP))
             for s in range(40)}
    assert len(lanes) == K                                         # a lone note goes anywhere
    assert (x == chart()).all()                                    # the chart itself untouched


def test_the_holds_view_keeps_the_long_notes_whole() -> None:
    from scripts.judge import view_tokens
    from src.data.tokenizer import grammar_violations
    x = chart()
    onsets = np.isin(x, (TAP, HOLD_START)).sum(axis=1)
    heads = (x == HOLD_START).sum(axis=1)
    for seed in range(8):
        h = view_tokens(x, "holds", (seed, 1, 2))
        assert (np.isin(h, (TAP, HOLD_START)).sum(axis=1) == onsets).all()
        assert ((h == HOLD_START).sum(axis=1) == heads).all()
        assert len(grammar_violations(h[:10])) == 0 and (h[10:] == PAD).all()
        lane = int(np.flatnonzero(h[2] == HOLD_START)[0])          # the long note, whole
        assert (h[3:5, lane] == HOLD_BODY).all() and h[5, lane] == HOLD_END
        assert (h[4] == TAP).sum() == 1 and h[4, lane] == HOLD_BODY   # the tap beside it
    lanes = {int(np.flatnonzero(view_tokens(x, "holds", (s, 1, 2))[2] == HOLD_START)[0])
             for s in range(40)}
    assert len(lanes) == K                                         # the long note goes anywhere
    tail = view_tokens(x[3:], "holds", (0, 1, 2))                  # a window inside it
    assert ((tail == HOLD_BODY).sum(), (tail == HOLD_END).sum()) == (2, 1)
    lane = int(np.flatnonzero(tail[0] == HOLD_BODY)[0])
    assert tail[1, lane] == HOLD_BODY and tail[2, lane] == HOLD_END
    assert (view_tokens(x, "holds", (3,)) == view_tokens(x, "holds", (3,))).all()
    assert (x == chart()).all()                                    # the chart itself untouched


def test_stopping_needs_the_interval_clear_of_every_bound() -> None:
    from scripts.judge import auc_ci, decided
    assert decided(0.70, 0.80, [0.65, 0.86]) and decided(0.87, 0.95, [0.65, 0.86])
    assert not decided(0.84, 0.90, [0.65, 0.86]) and not decided(float("nan"), 0.9, [0.5])
    rng = np.random.default_rng(0)
    far = {(f"s{i}", c): (rng.normal(2.0 if c == "ai" else 0.0, 1.0, 12),)
           for i in range(40) for c in ("ai", "human")}
    keys = [f"s{i}" for i in range(40)]
    point, lo, hi = auc_ci(far, keys)
    assert 0.85 < lo <= point <= hi < 0.99
    near = {(k, c): (rng.normal(0.0, 1.0, 12),) for k in keys for c in ("ai", "human")}
    point, lo, hi = auc_ci(near, keys)
    assert lo < 0.5 < hi and hi - lo < 0.2


def test_shards_share_the_songs_by_length(tmp_path) -> None:
    from scripts.evaluate import shard_of
    (tmp_path / "tokens").mkdir()
    sizes = [900, 100, 800, 120, 700, 300, 50, 640, 60, 400]
    rows = []
    for i, n in enumerate(sizes):
        np.savez(tmp_path / "tokens" / f"k{i}.npz", n_cells=np.int64(n))
        rows.append({"key": f"k{i}", "sr": str(1.0 + i)})
    got = shard_of(rows, tmp_path, 3)
    assert got == shard_of(rows, tmp_path, 3) and sorted(set(got)) == [0, 1, 2]
    load = [sum(n for n, s in zip(sizes, got, strict=True) if s == k) for k in range(3)]
    by_index = [sum(n for i, n in enumerate(sizes) if i % 3 == k) for k in range(3)]
    assert max(load) - min(load) <= 60 < max(by_index) - min(by_index)
    skipped = shard_of(rows, tmp_path, 2, sr_min=6.0)              # songs below 6 weigh nothing
    assert skipped[5:] == shard_of(rows[5:], tmp_path, 2)


def test_the_judge_table_compares_on_the_songs_all_judges_have(tmp_path) -> None:
    import csv
    import json

    from scripts import judge_table
    rng = np.random.default_rng(1)
    for name, songs, shift in (("full", 10, 2.0), ("taps", 4, 0.0)):
        d = tmp_path / name
        d.mkdir()
        with open(d / "windows.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["key", "grade", "fold", "start", "chart", "logit", "onset_rows"])
            for i in range(songs):
                for st in range(0, 4 * 384, 384):
                    w.writerow([f"k{i}", "Hard", 0, st, "human", rng.normal(), 3])
                    w.writerow([f"k{i}", "Hard", 0, st, "ai", rng.normal(shift), 3])
        view = {"view": name, "sr": "human", "no_audio": False, "folds": 5}
        (d / "summary.json").write_text(json.dumps({
            "ai": {"songs": songs, "auc": 0.9 if shift else 0.5, "auc_ci": [0.8, 0.95],
                   "paired": 0.9}, "folds_run": 5 if songs == 10 else 2, "folds": 5,
            "by_epoch": [{"epoch": 1, "auc": 0.88}], "args": view, "seconds": 600}))
    out = tmp_path / "t.md"
    assert judge_table.main([str(tmp_path / "full"), str(tmp_path / "taps"),
                             str(tmp_path / "none"), "--out", str(out)]) == 0
    got = json.loads(out.with_suffix(".json").read_text())
    assert got["shared_songs"] == 4 and [r["folds"] for r in got["judges"]] == ["5/5", "2/5"]
    full, taps = got["judges"]
    assert full["shared_auc"] > 0.8 and abs(taps["shared_auc"] - 0.5) < 0.2
    assert full["step"] is None
    d, lo, hi = taps["step"]                                     # what one more removal costs
    assert lo <= d <= hi < 0 and abs(d - (taps["shared_auc"] - full["shared_auc"])) < 1e-9
    assert "auc on the 4 shared songs" in out.read_text()
    assert judge_table.main([str(tmp_path / "none")]) == 2
