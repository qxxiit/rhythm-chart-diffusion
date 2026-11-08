"""Human or AI (실험 2026-10-15): a judge trained to tell a run's charts from the human charts of
the same songs, 8 bars at a time, the songs split into folds so that every song is judged by a
model that never saw it. The machine's version of the blind playtest, and a measure of how far
the charts still are from the human ones that does not need a player.

    python scripts/judge.py --ckpt outputs/full-v4/best.pt --ai <eval dir> \
        [--score <eval dir> ...] [--no-audio] [--folds 5] [--epochs 3] [--name F]

--ai and --score are evaluate.py runs (charts.npz and per_song.csv); the judge trains on
every song of --ai, and a --score run over fewer of them is scored on the songs it has. Each
fold starts a Judge (src/models/judge.py) from the denoiser of --ckpt and trains it on pairs,
the human chart's window and the run's at the same rows, windows every 4 bars, lanes mirrored at
random, for --epochs; it then scores the held-out songs' windows every 8 bars: the run's, the
human chart's and every --score run's. A window where neither the human chart nor the run has an
onset is left out. The judge sees the chart, its audio (silence with --no-audio), the chart's
own SR (rosu: sr_gen for a run, sr_human_local for the human chart) and the beat length. Folds
group the songs by audio (the manifest's audio_key), as the splits do.

Writes outputs/judge/<name>/: windows.csv (key, grade, fold, start, chart, logit, onset rows),
rows.npz (per-row scores of the held-out windows, "<key>|<chart>" -> [windows, L] and
"<key>|starts"), log.csv (loss and accuracy every --log-every steps), summary.json; prints

  auc       P(a run window scores above a human window), over the held-out windows
  paired    share of windows where the run's chart scores above the human chart's at the
            same rows (the playtest's question, asked of 8 bars; ties count half)
  songs     the same with each song's windows averaged
by grade, and for every --score run the same against the human charts, and its mean logit
minus the judged run's per song (95% interval over songs): below 0, it looks more human to
this judge.

Checks before a judge counts (docs/EVALUATION.md): --shuffle-labels trains on pairs whose
labels are swapped at random (the auc pooled over the folds should land at 0.40-0.60; each
fold's judge ranks by an arbitrary direction, so one fold may stray; 0.7+ is a bug), and
--score with the runs of scripts/corrupt_test.py --write-runs (human charts corrupted at
three doses: the score should rise with the dose for the corruptions the judge can see).
scripts/judge_vs_diag.py relates the judged run's per-song scores to its diagnostics.

What the judge is shown (EXPERIMENTS 2026-11-08 (2), (3)): --view taps or rhythm shows every
chart (the human one, the run's, the scored runs') with part of it taken away alike, so a judge
trained on that can only use what is left (view_tokens). A view needs --sr view: every chart
with rosu's SR of the chart as shown (shown_sr; what a player would see of it). The SR of the
whole chart (own) tells some of what the view hides, and the human chart's SR for all (human)
is the SR the run was asked for: the judge then tells "not as hard as it says" (run_1109, the
full view on the same 96 songs: own 0.901, human 0.944).

Time (EXPERIMENTS 2026-11-08 (2)): --stop-bounds ends the run after a fold (from --min-folds
on) once the auc so far and its 95% interval over songs lie between the same two bounds (the
question asked is answered; the songs of the folds run are reported, the summary says where it
stopped). --eval-epochs scores the held-out songs after every epoch as well (by_epoch in the
summary; the training is the same), to learn how many epochs a judge needs.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import math
import sys
import time
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch

from src.data.beat_grid import from_timing_points
from src.data.cache import read_manifest
from src.data.mel import open_mel
from src.data.tokenizer import (
    BAR,
    BEATS_PER_CHUNK,
    EMPTY,
    HOLD_BODY,
    HOLD_END,
    HOLD_START,
    PAD,
    TAP,
    D,
    K,
    L,
)
from src.evaluation.playability import grade_of
from src.models.diffusion import load_denoiser, pick_device
from src.models.judge import Judge

TRAIN_STRIDE = 4 * BAR                 # training windows every 4 bars (half overlapping)
GRADES = ("Easy", "Normal", "Hard", "Insane", "Expert", "Expert+")
VIEWS = ("full", "taps", "rhythm")


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ckpt", type=Path, required=True, help="the denoiser to start from")
    ap.add_argument("--ai", type=Path, required=True, help="evaluate.py run to judge")
    ap.add_argument("--score", type=Path, nargs="*", default=[],
                    help="more evaluate.py runs over the same songs, scored by the same folds")
    ap.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"))
    ap.add_argument("--cache", type=Path, default=Path("data/cache"))
    ap.add_argument("--fake-mel", action="store_true", help="plumbing tests only")
    ap.add_argument("--out", type=Path, default=Path("outputs/judge"))
    ap.add_argument("--name", default=None, help="run name (default: time stamp)")
    ap.add_argument("--no-audio", action="store_true", help="the judge hears silence")
    ap.add_argument("--shuffle-labels", action="store_true",
                    help="the null check: each training pair's labels swapped at random, so "
                         "there is nothing to learn: the held-out auc pooled over the folds "
                         "should be 0.40-0.60 (docs/EVALUATION.md); 0.7+ is a bug")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-pairs", type=int, default=8, help="pairs a step (2x windows)")
    ap.add_argument("--lr", type=float, default=1e-4, help="the trunk's")
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--mirror", type=float, default=0.5, help="chance a window is mirrored")
    ap.add_argument("--limit", type=int, default=None, help="first N songs (smoke runs)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--view", choices=VIEWS, default="full",
                    help="what of every chart the judge sees (view_tokens): full; taps (long "
                         "notes as taps); rhythm (taps on lanes drawn at random)")
    ap.add_argument("--sr", choices=("own", "view", "human"), default="own",
                    help="the SR each chart is shown with: its own (rosu, of the whole chart); "
                         "rosu's of the chart as --view shows it, which a view needs; or the "
                         "human chart's for all (the SR the run was asked for: a tell)")
    ap.add_argument("--stop-bounds", default=None, metavar="B[,B...]",
                    help="stop after a fold once the auc so far and its 95%% interval over "
                         "songs lie between the same two of these, e.g. 0.65,0.86")
    ap.add_argument("--min-folds", type=int, default=2, help="folds before --stop-bounds")
    ap.add_argument("--eval-epochs", action="store_true",
                    help="score the held-out songs after every epoch too (summary by_epoch)")
    return ap.parse_args(argv)


def view_tokens(tokens: np.ndarray, view: str, seed=(0,)) -> np.ndarray:
    """A chart [rows, K] as --view shows it; the same for every chart of a song, so a judge can
    only use what is kept.

        full    the chart
        taps    long notes as taps at their heads (bodies and tails empty): nothing of which
                notes are held, nor for how long
        rhythm  taps, and each onset row's notes on lanes drawn at random (as many as it had,
                seed: the song and the chart): which rows have notes and how many, nothing of
                the lanes
    """
    if view == "full":
        return tokens
    x = np.array(tokens, copy=True)
    x[x == HOLD_START] = TAP
    x[(x == HOLD_BODY) | (x == HOLD_END)] = EMPTY
    if view == "rhythm":
        n = (x == TAP).sum(axis=1)
        rows = np.flatnonzero(n)
        perm = np.argsort(np.random.default_rng(list(seed)).random((len(rows), K)), axis=1)
        x[rows] = EMPTY
        x[np.repeat(rows, n[rows]), perm[np.arange(K)[None, :] < n[rows][:, None]]] = TAP
    return x


def stable_hash(s: str) -> int:
    return zlib.crc32(s.encode("utf-8"))


def shown_sr(tokens: np.ndarray, timing_points, cell_offset: int) -> float:
    """rosu's SR of a chart [rows, K] as it is shown (--sr view), decoded on the song's grid as
    evaluate.py decodes a run's chart for sr_gen."""
    from src.data.tokenizer import decode, make_metas
    from src.evaluation.sr import star_rating  # rosu, only when asked for

    x = np.asarray(tokens).reshape(-1, L, K)
    return float(star_rating(decode(x, make_metas(timing_points, cell_offset, len(x),
                                                  math.nan))))


def read_run(path: Path) -> tuple[dict, dict]:
    """(key -> per_song row, key -> tokens [rows, K]) of an evaluate.py run."""
    with open(path / "per_song.csv", newline="", encoding="utf-8") as f:
        rows = {r["key"]: r for r in csv.DictReader(f)}
    charts = np.load(path / "charts.npz")
    return rows, {k: charts[k].reshape(-1, K) for k in rows if k in charts.files}


def label_of(path: Path, ai: Path) -> str:
    """A short name for a scored run: what it adds to the judged run's name, or its own."""
    name = path.name
    if name.startswith(ai.name) and len(name) > len(ai.name):
        return "+" + name[len(ai.name):].lstrip("_")
    return name


def as_float(v, default: float) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return default
    return x if math.isfinite(x) else default


def onset_rows(tokens: np.ndarray, lo: int, hi: int) -> int:
    return int(np.isin(tokens[lo:hi], (TAP, HOLD_START)).any(axis=1).sum())


def window(tokens: np.ndarray, start: int) -> np.ndarray:
    """Rows [start, start + L) of a chart, PAD past its end."""
    x = np.full((L, K), PAD, dtype=np.int64)
    part = tokens[start:start + L]
    x[:len(part)] = part
    return x


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(a random pos > a random neg), ties half."""
    pos, neg = np.sort(np.asarray(pos, float)), np.sort(np.asarray(neg, float))
    if not len(pos) or not len(neg):
        return float("nan")
    lo = np.searchsorted(neg, pos, side="left")
    hi = np.searchsorted(neg, pos, side="right")
    return float(np.mean((lo + 0.5 * (hi - lo)) / len(neg)))


def paired(a: np.ndarray, b: np.ndarray) -> float:
    """Share of pairs where a > b, ties half."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.mean((a > b) + 0.5 * (a == b))) if len(a) else float("nan")


def boot_mean(values: np.ndarray, n: int = 2000, seed: int = 0) -> tuple[float, float]:
    v = np.asarray(values, float)
    if len(v) < 2:
        return float("nan"), float("nan")
    idx = np.random.default_rng(seed).integers(len(v), size=(n, len(v)))
    lo, hi = np.percentile(v[idx].mean(axis=1), [2.5, 97.5])
    return float(lo), float(hi)


def auc_ci(scores: dict, keys: list[str], chart: str = "ai", n: int = 1000,
           seed: int = 0) -> tuple[float, float, float]:
    """(auc, lo, hi): the chart's windows against the human ones over the songs keys, and its
    95% interval with the songs drawn again (a song's windows go together)."""
    c = [scores[(k, chart)][0] for k in keys]
    h = [scores[(k, "human")][0] for k in keys]
    if not keys:
        return float("nan"), float("nan"), float("nan")
    point = auc(np.concatenate(c), np.concatenate(h))
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n):
        idx = rng.integers(len(keys), size=len(keys))
        draws.append(auc(np.concatenate([c[i] for i in idx]), np.concatenate([h[i] for i in idx])))
    lo, hi = np.nanpercentile(draws, [2.5, 97.5])
    return point, float(lo), float(hi)


def decided(lo: float, hi: float, bounds: list[float]) -> bool:
    """The interval [lo, hi] lies between the same two bounds (no bound inside it)."""
    return math.isfinite(lo) and math.isfinite(hi) and not any(lo <= b <= hi for b in bounds)


class Songs:
    """The songs both the human cache and every run have, with what the windows need."""

    def __init__(self, a: argparse.Namespace):
        man = {r["key"]: r for r in read_manifest(a.manifest)}
        ai_rows, ai_charts = read_run(a.ai)
        runs = [("ai", ai_rows, ai_charts)]
        self.labels = ["ai"]
        for path in a.score:
            if not ((path / "per_song.csv").exists() and (path / "charts.npz").exists()):
                print(f"--score {path}: no per_song.csv and charts.npz there, left out",
                      file=sys.stderr)
                continue
            rows, charts = read_run(path)
            runs.append((label_of(path, a.ai), rows, charts))
            self.labels.append(runs[-1][0])
        self.store = open_mel(a.cache, fake=a.fake_mel)
        keys = [k for k in ai_rows if k in ai_charts and k in man                # every song of
                and (a.cache / "tokens" / f"{k}.npz").exists() and self.store.has(k)]  # the run
        if a.limit:
            keys = keys[:a.limit]
        self.songs = []
        for k in keys:
            z = np.load(a.cache / "tokens" / f"{k}.npz")
            n = int(z["n_cells"])
            human = z["tokens"].reshape(-1, K)
            tps = [(float(t), float(bl)) for t, bl in z["timing_points"]]
            sr_label = as_float(man[k]["sr"], 0.0)
            song = {"key": k, "group": man[k].get("audio_key") or man[k]["set_id"],
                    "grade": ai_rows[k].get("grade") or grade_of(sr_label), "n": n,
                    "grid": from_timing_points(tps), "offset": int(z["cell_offset"]),
                    "charts": {"human": human}, "sr": {"human": as_float(
                        ai_rows[k].get("sr_human_local"), sr_label)}}
            for label, rows, charts in runs:          # a --score run is scored on the songs
                if k in charts:                        # it has; the judge trains on them all
                    song["charts"][label] = charts[k]
                    song["sr"][label] = as_float(rows[k].get("sr_gen"), sr_label)
            view = getattr(a, "view", "full")
            song["charts"] = {c: view_tokens(t, view, (a.seed, stable_hash(k), stable_hash(c)))
                              for c, t in song["charts"].items()}   # onset rows stay
            mode = getattr(a, "sr", "own")
            if mode == "human":
                song["sr"] = {c: song["sr"]["human"] for c in song["sr"]}
            elif mode == "view":
                for c, t in song["charts"].items():
                    try:
                        song["sr"][c] = shown_sr(t, tps, song["offset"])
                    except ValueError as e:                  # not decodable: its own SR stays
                        print(f"{k} {c}: no SR as shown ({e}); its own", file=sys.stderr)
            song["train"] = self.starts(song, TRAIN_STRIDE)
            song["test"] = self.starts(song, L)
            self.songs.append(song)

    @staticmethod
    def starts(song: dict, stride: int) -> list[int]:
        """Window starts every stride rows where the human chart or the run has an onset."""
        out = []
        for start in range(0, max(song["n"], 1), stride):
            hi = min(start + L, song["n"])
            if onset_rows(song["charts"]["human"], start, hi) \
                    or onset_rows(song["charts"]["ai"], start, hi):
                out.append(start)
        return out

    def beat_len(self, song: dict, start: int) -> float:
        cell = start - song["offset"]
        grid = song["grid"]
        return float(grid.time_from_cell(cell + L, D) - grid.time_from_cell(cell, D)) \
            / BEATS_PER_CHUNK

    def mel(self, song: dict, start: int) -> np.ndarray:
        return np.ascontiguousarray(self.store.frames(song["key"], song["grid"], song["offset"],
                                                      start, L), dtype=np.float32)

    def batch(self, items: list[tuple[dict, int, str]], device, rng=None,
              p_mirror: float = 0.0) -> tuple[torch.Tensor, ...]:
        """(x, mel, s, b) of the windows (song, start, chart)."""
        xs, mels, ss, bs = [], [], [], []
        for song, start, chart in items:
            x = window(song["charts"][chart], start)
            if rng is not None and rng.random() < p_mirror:
                x = x[:, ::-1].copy()
            xs.append(x)
            mels.append(self.mel(song, start))
            ss.append(song["sr"][chart])
            bs.append(self.beat_len(song, start))
        return (torch.as_tensor(np.stack(xs), device=device),
                torch.as_tensor(np.stack(mels), device=device),
                torch.tensor(ss, dtype=torch.float32, device=device),
                torch.tensor(bs, dtype=torch.float32, device=device))


def folds_of(songs: list[dict], n_folds: int, seed: int) -> list[int]:
    groups = sorted({s["group"] for s in songs})
    order = np.random.default_rng(seed).permutation(len(groups))
    fold = {groups[g]: i % n_folds for i, g in enumerate(order)}
    return [fold[s["group"]] for s in songs]


def train_fold(a, data: Songs, train: list[dict], device, fold: int, log_path: Path,
               on_epoch=None):
    """The fold's judge; on_epoch(judge, epoch) after every epoch but the last (--eval-epochs:
    it only scores, so the training is the same)."""
    trunk = load_denoiser(a.ckpt, device)
    judge = Judge(trunk, audio=not a.no_audio).to(device)
    judge.train()
    opt = torch.optim.AdamW([{"params": judge.trunk.parameters(), "lr": a.lr},
                             {"params": judge.score.parameters(), "lr": a.head_lr}],
                            weight_decay=a.weight_decay)
    pairs = [(s, start) for s in train for start in s["train"]]
    rng = np.random.default_rng([a.seed, fold])
    steps = max(1, a.epochs * math.ceil(len(pairs) / a.batch_pairs))
    warm = max(1, steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(1.0, (i + 1) / warm) * 0.5 * (
        1 + math.cos(math.pi * min(1.0, max(0, i - warm) / max(1, steps - warm)))))
    use_amp = device.type == "cuda"
    step, run_loss, run_acc, n_run = 0, 0.0, 0.0, 0
    for epoch in range(a.epochs):
        if epoch and on_epoch is not None:
            judge.eval()
            on_epoch(judge, epoch)
            judge.train()
        order = rng.permutation(len(pairs))
        for j in range(0, len(order), a.batch_pairs):
            chosen = [pairs[i] for i in order[j:j + a.batch_pairs]]
            items = [(s, st, "human") for s, st in chosen] + [(s, st, "ai") for s, st in chosen]
            x, mel, s_, b = data.batch(items, device, rng, a.mirror)
            y = torch.cat([torch.zeros(len(chosen)), torch.ones(len(chosen))]).to(device)
            if a.shuffle_labels:
                swap = torch.as_tensor(rng.random(len(chosen)) < 0.5, dtype=torch.float32)
                y = torch.cat([swap, 1 - swap]).to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_amp \
                    else contextlib.nullcontext():
                logit = judge(x, mel, s_, b)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logit.float(), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(judge.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            run_loss += loss.item()
            run_acc += float(((logit > 0).float() == y).float().mean())
            n_run += 1
            if step % a.log_every == 0 or step == steps:
                new = not log_path.exists()
                with open(log_path, "a", newline="") as f:
                    w = csv.writer(f)
                    if new:
                        w.writerow(["fold", "step", "loss", "acc", "lr"])
                    w.writerow([fold, step, f"{run_loss / n_run:.4f}", f"{run_acc / n_run:.3f}",
                                f"{sched.get_last_lr()[0]:.2e}"])
                print(f"  fold {fold} step {step}/{steps}  loss {run_loss / n_run:.4f}  "
                      f"acc {run_acc / n_run:.3f}", flush=True)
                run_loss, run_acc, n_run = 0.0, 0.0, 0
    return judge.eval()


@torch.no_grad()
def score_fold(judge: Judge, data: Songs, test: list[dict], device, batch: int = 16,
               charts: tuple[str, ...] | None = None):
    """{(key, chart): (logits [windows], rows [windows, L])} for the held-out songs (every
    chart, or those given)."""
    out = {}
    for song in test:
        starts = song["test"]
        for chart in charts or ("human", *data.labels):
            if chart not in song["charts"]:
                continue
            logits, rows = [], []
            for j in range(0, len(starts), batch):
                items = [(song, st, chart) for st in starts[j:j + batch]]
                x, mel, s_, b = data.batch(items, device)
                r = judge.rows(x, mel, s_, b).float()
                keep = ~(x == PAD).all(dim=-1)
                logits.append(((r * keep).sum(1) / keep.sum(1).clamp_min(1)).cpu().numpy())
                rows.append(torch.where(keep, r, torch.zeros_like(r)).cpu().numpy())
            out[(song["key"], chart)] = (np.concatenate(logits) if logits else np.zeros(0),
                                         np.concatenate(rows) if rows else np.zeros((0, L)))
    return out


def report(songs: list[dict], scores: dict, labels: list[str]) -> tuple[dict, list[str]]:
    lines, summary = [], {}
    keys = [s["key"] for s in songs if len(s["test"])]
    grade = {s["key"]: s["grade"] for s in songs}

    def block(chart: str, ks: list[str]) -> dict:
        h = np.concatenate([scores[(k, "human")][0] for k in ks]) if ks else np.zeros(0)
        c = np.concatenate([scores[(k, chart)][0] for k in ks]) if ks else np.zeros(0)
        hs = np.array([scores[(k, "human")][0].mean() for k in ks])
        cs = np.array([scores[(k, chart)][0].mean() for k in ks])
        return {"windows": len(c), "songs": len(ks), "auc": auc(c, h), "paired": paired(c, h),
                "songs_paired": paired(cs, hs), "mean_logit": float(c.mean()) if len(c) else
                float("nan"), "human_mean_logit": float(h.mean()) if len(h) else float("nan")}

    for chart in labels:
        ks = [k for k in keys if (k, chart) in scores]          # a --score run's own songs
        b = block(chart, ks)
        if chart == "ai":
            _, lo, hi = auc_ci(scores, ks)
            b["auc_ci"] = [lo, hi]
        b["by_grade"] = {g: block(chart, [k for k in ks if grade[k] == g])
                         for g in GRADES if any(grade[k] == g for k in ks)}
        if chart != "ai" and ks:
            d = np.array([scores[(k, chart)][0].mean() - scores[(k, "ai")][0].mean()
                          for k in ks])
            lo, hi = boot_mean(d)
            b["minus_ai"] = {"mean": float(d.mean()), "lo": lo, "hi": hi,
                             "paired_vs_ai": paired(
                                 np.concatenate([scores[(k, chart)][0] for k in ks]),
                                 np.concatenate([scores[(k, "ai")][0] for k in ks]))}
        summary[chart] = b
        ci = f" [{b['auc_ci'][0]:.3f}, {b['auc_ci'][1]:.3f}]" if "auc_ci" in b else ""
        lines.append(f"{chart}: {b['windows']} windows of {b['songs']} songs  auc {b['auc']:.3f}"
                     f"{ci}  paired {b['paired']:.3f}  songs {b['songs_paired']:.3f}  "
                     f"mean logit {b['mean_logit']:+.3f} (human {b['human_mean_logit']:+.3f})")
        if "minus_ai" in b:
            m = b["minus_ai"]
            lines.append(f"  minus the judged run, per song: {m['mean']:+.3f} "
                         f"[{m['lo']:+.3f}, {m['hi']:+.3f}]; windows above it "
                         f"{m['paired_vs_ai']:.3f}")
        for g, gb in b["by_grade"].items():
            lines.append(f"  {g:8s} {gb['windows']:5d} windows {gb['songs']:3d} songs  "
                         f"auc {gb['auc']:.3f}  paired {gb['paired']:.3f}  "
                         f"songs {gb['songs_paired']:.3f}")
    return summary, lines


def main(argv=None) -> int:
    a = parse_args(argv)
    bounds = []
    if a.stop_bounds:
        try:
            bounds = [float(x) for x in a.stop_bounds.split(",")]
        except ValueError:
            print(f"--stop-bounds {a.stop_bounds}: numbers with commas, e.g. 0.65,0.86",
                  file=sys.stderr)
            return 2
    if a.view != "full" and a.sr == "own":
        print(f"--view {a.view} with each chart's own SR: the SR of the whole chart tells the "
              "judge some of what the view hides (--sr view)", file=sys.stderr)
    if a.sr == "human":
        print("--sr human: the run's charts are shown with the SR they were asked for, so the "
              "judge can tell 'not as hard as it says' (a tell a player does not have)",
              file=sys.stderr)
    torch.manual_seed(a.seed)
    device = pick_device(a.device)
    data = Songs(a)
    songs = [s for s in data.songs if s["test"]]
    groups = {s["group"] for s in songs}
    if len(groups) < 2:
        print(f"{len(songs)} songs from {len(groups)} audio files: need two or more to hold "
              "songs out", file=sys.stderr)
        return 2
    n_folds = min(a.folds, len(groups))
    fold_of = folds_of(songs, n_folds, a.seed)
    run_dir = a.out / (a.name or time.strftime("%Y%m%d-%H%M%S"))
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "log.csv"
    if log_path.exists():
        log_path.unlink()
    n_pairs = sum(len(s["train"]) for s in songs)
    print(f"{run_dir}: {len(songs)} songs, {n_folds} folds, {n_pairs:,} training pairs "
          f"(every {TRAIN_STRIDE} rows), {sum(len(s['test']) for s in songs):,} held-out windows "
          f"a chart; charts: human, {', '.join(data.labels)}; audio {'off' if a.no_audio else 'on'}"
          f"; view {a.view}, SR {a.sr}; {device}", flush=True)
    t0 = time.time()
    scores, fold_by_key, by_epoch, stopped = {}, {}, {}, None
    for fold in range(n_folds):
        train = [s for s, f in zip(songs, fold_of, strict=True) if f != fold]
        test = [s for s, f in zip(songs, fold_of, strict=True) if f == fold]

        def at_epoch(judge, epoch, test=test):
            done = score_fold(judge, data, test, device, charts=("human", "ai"))
            by_epoch.setdefault(epoch, {}).update({k: (v[0],) for k, v in done.items()})

        judge = train_fold(a, data, train, device, fold, log_path,
                           at_epoch if a.eval_epochs else None)
        scores.update(score_fold(judge, data, test, device))
        fold_by_key.update({s["key"]: fold for s in test})
        del judge
        keys = [s["key"] for s in songs if (s["key"], "human") in scores]
        so_far, lo, hi = auc_ci(scores, keys)
        print(f"fold {fold}: {len(train)} songs trained, {len(test)} held out, "
              f"{time.time() - t0:.0f}s; auc so far {so_far:.3f} [{lo:.3f}, {hi:.3f}] over "
              f"{len(keys)} songs", flush=True)
        if bounds and fold + 1 >= a.min_folds and fold + 1 < n_folds and decided(lo, hi, bounds):
            stopped = {"after_folds": fold + 1, "of": n_folds, "bounds": bounds,
                       "auc": so_far, "ci": [lo, hi]}
            print(f"decided after {fold + 1} of {n_folds} folds: [{lo:.3f}, {hi:.3f}] holds no "
                  f"bound of {bounds}; the other folds are not run", flush=True)
            break
    scored = [s for s in songs if (s["key"], "human") in scores]
    with open(run_dir / "windows.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["key", "grade", "fold", "start", "chart", "logit", "onset_rows"])
        for s in scored:
            for chart in ("human", *data.labels):
                if (s["key"], chart) not in scores:             # not in that --score run
                    continue
                for st, lg in zip(s["test"], scores[(s["key"], chart)][0], strict=True):
                    w.writerow([s["key"], s["grade"], fold_by_key[s["key"]], st, chart,
                                f"{lg:.4f}", onset_rows(s["charts"][chart], st,
                                                        min(st + L, s["n"]))])
    rows = {f"{k}|{c}": v[1].astype(np.float16) for (k, c), v in scores.items()}
    rows.update({f"{s['key']}|starts": np.asarray(s["test"], dtype=np.int64) for s in scored})
    np.savez_compressed(run_dir / "rows.npz", **rows)
    summary, lines = report(scored, scores, data.labels)
    summary["folds_run"], summary["folds"], summary["stopped"] = (
        len({fold_by_key[s["key"]] for s in scored}), n_folds, stopped)
    if a.eval_epochs:
        keys = [s["key"] for s in scored]
        summary["by_epoch"] = []
        for epoch, sc in [*sorted(by_epoch.items()), (a.epochs, scores)]:
            ks = [k for k in keys if (k, "human") in sc and (k, "ai") in sc]
            if not ks:
                continue
            c = np.concatenate([sc[(k, "ai")][0] for k in ks])
            h = np.concatenate([sc[(k, "human")][0] for k in ks])
            summary["by_epoch"].append({"epoch": epoch, "songs": len(ks), "auc": auc(c, h),
                                        "paired": paired(c, h)})
        lines.append("by epoch: " + ", ".join(f"{e['epoch']}: auc {e['auc']:.3f}"
                                              for e in summary["by_epoch"]))
    if stopped:
        lines.append(f"stopped after {stopped['after_folds']} of {n_folds} folds "
                     f"(--stop-bounds {a.stop_bounds})")
    summary["args"] = {k: (str(v) if isinstance(v, Path) else
                           [str(x) for x in v] if isinstance(v, list) else v)
                       for k, v in vars(a).items()}
    summary["labels"] = {"ai": str(a.ai), **{label_of(p, a.ai): str(p) for p in a.score
                                             if label_of(p, a.ai) in data.labels}}
    summary["seconds"] = round(time.time() - t0, 1)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
