"""Does the model know which lanes a human chart uses? Lane choice with full and thinned context.

    python scripts/pattern_probe.py --ckpt outputs/full-v1/best.pt
    python scripts/pattern_probe.py --ckpt ... --split val --batches 100

On human chunks, rows that hold taps only (1-3 taps, no hold cells) are hidden:
every 8th row at a time (the rows sampler.refine_lanes re-chooses together), all
four lanes MASK, the rest of the chunk as the human chart has it. For a row with
n taps the model's lane set is the n lanes with the highest log p(TAP) -
log p(EMPTY), the best set under its independent cells (refine_lanes at
temperature 0). Scored per row: the exact lane set.

Context around the hidden rows:
    full      every other cell visible
    thin50    half of the other cells hidden as well (random cells)
    thin90    90% hidden: about what the sampler has when it first chooses lanes
Baselines on the same rows: chance (1 / C(4, n)); "previous", the lane set of the
previous onset row; "best of 8", right if any of the previous 8 onset rows has
the same lane set (an oracle that knows which earlier event the pattern repeats).

Reading: full context far above the baselines but thin90 near chance means the
model knows patterns and sampling commits lanes before the context is there
(refine_lanes is the fix). Full context near chance means the model has not
learned them (train.py --row-mask, a larger model, or the representation).

One hidden row per chunk at a time (up to 8 rows per chunk, rows with at least
2 bars of the chunk before them), the views of the sampler's left-to-right passes:
    full (1 row)  everything else visible (the reference for the two below)
    past+rhythm   the rows before as the human chart has them; every later row with
                  a lane choice (some but not all of its tap-or-empty cells are taps)
                  has those cells hidden, so where later notes are not stays visible:
                  what sampler.forward_lanes gives the model
    past          everything after the row hidden: what --order block gives it
    chance (1 row) 1 / C(4, n) on these rows
Single-tap rows right after a single tap (at most a beat earlier), by how the human
chart moves there (full context; the rows of the first table):
    move: jack / step / skip / leap   exact lane for moves of 0 / 1 / 2 / 3 lanes
    jack rate (human / model)         how often the human row repeats the lane, and
                                      how often the model's pick does (the lane passes
                                      take the model's pick at temperature 0)
Reading: a model pick rate far below the human rate means the model itself
under-predicts jacks with the whole chart in view, so the lane passes inherit it.

Reading: past+rhythm close to full (1 row) means the lanes can be chosen left to
right without the later lanes, which sampling draws with little context
(--lanes forward); far below it means the later lanes carry the pattern and
refine_lanes' both-sided view is needed.
"""

from __future__ import annotations

import argparse
import json
import sys
from math import comb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dataset import ChunkDataset
from src.data.tokenizer import BAR, EMPTY, HOLD_START, MASK, PAD, TAP, K, L
from src.models.diffusion import load_denoiser, pick_device

STRIDE = 8
EDGE = 8                                           # rows at the chunk edges are not probed
CONTEXTS = {"full": 0.0, "thin50": 0.5, "thin90": 0.9}
ROUNDS = 8                                         # one-row contexts: rows per chunk
PAST_MIN = 2 * BAR                                 # ... at least 2 bars into the chunk
ROW_CONTEXTS = ("full (1 row)", "past+rhythm", "past")
MOVE_KEYS = ("move: jack", "move: step", "move: skip", "move: leap")


def previous_single(x0: np.ndarray, row: int) -> int | None:
    """Lane of the onset row before `row` if it is a single tap at most a beat earlier."""
    onset = np.isin(x0[:row], (TAP, HOLD_START))
    rows = np.flatnonzero(onset.any(axis=1))
    if len(rows) == 0 or row - rows[-1] > 12 or onset[rows[-1]].sum() != 1:
        return None
    return int(np.flatnonzero(onset[rows[-1]])[0])


def probe_rows(x0: np.ndarray) -> np.ndarray:
    """Bool [L]: rows with 1-3 taps and nothing but taps and empty cells."""
    taps = (x0 == TAP).sum(axis=1)
    only = np.isin(x0, (EMPTY, TAP)).all(axis=1)
    ok = only & (taps >= 1) & (taps < K)
    ok[:EDGE] = ok[L - EDGE:] = False
    return ok


def lane_choice_cells(x0: np.ndarray) -> np.ndarray:
    """Bool [L, K]: the tap-or-empty cells of rows with a lane choice (0 < taps < such
    cells), the cells sampler.forward_lanes hides in the rows it has not reached yet."""
    free = np.isin(x0, (EMPTY, TAP))
    taps = (x0 == TAP).sum(axis=1)
    return free & ((taps > 0) & (taps < free.sum(axis=1)))[:, None]


def one_row_view(x0: np.ndarray, row: int, context: str) -> np.ndarray:
    """The chunk as a ROW_CONTEXTS context shows it, with all lanes of `row` hidden."""
    x = x0.astype(np.int64).copy()
    if context == "past+rhythm":
        later = lane_choice_cells(x0)
        later[:row] = False
        x[later] = MASK
    elif context == "past":
        x[row:][x0[row:] != PAD] = MASK
    x[row] = MASK
    return x


def baselines(x0: np.ndarray, row: int, truth: int) -> tuple[bool, bool]:
    onset = np.isin(x0, (TAP, HOLD_START))
    rows = np.flatnonzero(onset[:row].any(axis=1))[-8:]
    masks = [(onset[r] * (1 << np.arange(K))).sum() for r in rows]
    return (bool(masks) and masks[-1] == truth), truth in masks


@torch.no_grad()
def run(model, loader, device, batches: int, seed: int) -> dict:
    stats = {c: {} for c in (*CONTEXTS, "chance", "previous", "best of 8")}

    def add(name: str, n: int, value: float) -> None:
        tot = stats[name].setdefault(n, [0.0, 0])
        tot[0] += value
        tot[1] += 1

    def score_row(logp: np.ndarray, x0: np.ndarray, row: int) -> tuple[int, float]:
        truth = x0[row] == TAP
        n = int(truth.sum())
        pick = np.zeros(K, dtype=bool)
        pick[np.argsort(-(logp[row, :, TAP] - logp[row, :, EMPTY]))[:n]] = True
        return n, float(np.array_equal(pick, truth))

    stats |= {c: {} for c in (*ROW_CONTEXTS, "chance (1 row)")}
    stats |= {c: {} for c in (*MOVE_KEYS, "jack rate (human)", "jack rate (model)")}
    gen = torch.Generator().manual_seed(seed)
    for bi, batch in enumerate(loader):
        if bi >= batches:
            break
        x0 = batch["x0"].numpy()
        mel, s, b = (batch[k].to(device) for k in ("mel", "s", "b"))
        probe = np.stack([probe_rows(x) for x in x0])                       # [B, L]
        for i, j in zip(*np.nonzero(probe), strict=True):                   # baselines once
            truth = int(((x0[i, j] == TAP) * (1 << np.arange(K))).sum())
            n = int((x0[i, j] == TAP).sum())
            prev, best = baselines(x0[i], j, truth)
            add("chance", n, 1 / comb(K, n))
            add("previous", n, prev)
            add("best of 8", n, best)
        for name, thin in CONTEXTS.items():
            for res in range(STRIDE):
                hide = probe & (np.arange(L) % STRIDE == res)[None]         # [B, L]
                x = x0.astype(np.int64).copy()
                if thin:
                    other = torch.rand(x.shape, generator=gen).numpy() < thin
                    x[other & (x != PAD)] = MASK
                x[hide] = MASK
                logits = model(torch.as_tensor(x, device=device), mel, s, b).float()
                logp = torch.log_softmax(logits, dim=-1).cpu().numpy()
                score = logp[..., TAP] - logp[..., EMPTY]                    # [B, L, K]
                for i, j in zip(*np.nonzero(hide), strict=True):
                    truth = x0[i, j] == TAP
                    n = int(truth.sum())
                    pick = np.zeros(K, dtype=bool)
                    pick[np.argsort(-score[i, j])[:n]] = True
                    add(name, n, float(np.array_equal(pick, truth)))
                    if name == "full" and n == 1:
                        prev = previous_single(x0[i], j)
                        if prev is not None:
                            lane, guess = int(np.flatnonzero(truth)[0]), int(np.flatnonzero(pick)[0])
                            add(MOVE_KEYS[abs(lane - prev)], 1, float(lane == guess))
                            add("jack rate (human)", 1, float(lane == prev))
                            add("jack rate (model)", 1, float(guess == prev))

        # one row per chunk at a time (own generator: the numbers above stay as they were)
        pick_rng = np.random.default_rng([seed, bi])
        rows = []
        for i in range(len(x0)):
            ok = np.flatnonzero(probe[i] & (np.arange(L) >= PAST_MIN))
            rows.append(pick_rng.permutation(ok)[:ROUNDS])
        for k in range(ROUNDS):
            chosen = [(i, int(r[k])) for i, r in enumerate(rows) if len(r) > k]
            if not chosen:
                break
            for i, row in chosen:
                add("chance (1 row)", int((x0[i, row] == TAP).sum()),
                    1 / comb(K, int((x0[i, row] == TAP).sum())))
            for name in ROW_CONTEXTS:
                x = x0.astype(np.int64).copy()
                for i, row in chosen:
                    x[i] = one_row_view(x0[i], row, name)
                logits = model(torch.as_tensor(x, device=device), mel, s, b).float()
                logp = torch.log_softmax(logits, dim=-1).cpu().numpy()
                for i, row in chosen:
                    add(name, *score_row(logp[i], x0[i], row))
    return {name: {n: v[0] / v[1] for n, v in sorted(d.items())} | {
        "all": sum(v[0] for v in d.values()) / max(sum(v[1] for v in d.values()), 1),
        "rows": sum(v[1] for v in d.values())} for name, d in stats.items()}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"))
    ap.add_argument("--cache", type=Path, default=Path("data/cache"))
    ap.add_argument("--batches", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args(argv)

    ds = ChunkDataset(a.manifest, a.cache, (a.split,))
    if len(ds) == 0:
        print(f"no {a.split} chunks with log-Mel", file=sys.stderr)
        return 2
    device = pick_device(a.device)
    model = load_denoiser(a.ckpt, device)
    loader = DataLoader(ds, batch_size=a.batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(a.seed))
    res = run(model, loader, device, a.batches, a.seed)

    print(f"{a.ckpt}: tap-only rows of {min(a.batches, len(loader)) * a.batch_size:,} "
          f"{a.split} chunks, exact lane set")
    print(f"  {'':12s} {'1 tap':>7s} {'2 taps':>7s} {'3 taps':>7s} {'all':>7s}")
    def line(name: str, d: dict) -> None:
        label = name.removesuffix(" (1 row)")
        print(f"  {label:12s} " + " ".join(f"{d.get(n, float('nan')):7.3f}" for n in (1, 2, 3))
              + f" {d['all']:7.3f}")

    one_row = (*ROW_CONTEXTS, "chance (1 row)")
    moves = (*MOVE_KEYS, "jack rate (human)", "jack rate (model)")
    for name, d in res.items():
        if name not in one_row and name not in moves:
            line(name, d)
    print(f"  rows probed: {res['full']['rows']:,} per context")
    print(f"  one hidden row per chunk at a time, rows {PAST_MIN}+ of the chunk:")
    for name in one_row:
        line(name, res[name])
    print(f"  rows probed: {res['past']['rows']:,} per context")
    print("  single taps after a single tap, full context, by the human move:")
    for name in (*MOVE_KEYS, "jack rate (human)", "jack rate (model)"):
        d = res[name]
        print(f"  {name:18s} {d.get(1, float('nan')):7.3f}   ({d['rows']:,} rows)")
    out = a.ckpt.parent / f"pattern_probe_{a.split}.json"
    out.write_text(json.dumps({k: {str(n): round(v, 4) for n, v in d.items()}
                               for k, d in res.items()}, indent=1))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
