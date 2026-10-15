"""Human or AI (실험 2026-10-15): a judge trained to tell a run's charts from the human charts of
the same songs, 8 bars at a time, the songs split into folds so that every song is judged by a
model that never saw it. The machine's version of the blind playtest, and a measure of how far
the charts still are from the human ones that does not need a player.

    python scripts/judge.py --ckpt outputs/full-v4/best.pt --ai <eval dir> \
        [--score <eval dir> ...] [--no-audio] [--folds 5] [--epochs 3] [--name F]

--ai and --score are evaluate.py runs (charts.npz and per_song.csv) over the same songs. Each
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
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch

from src.data.beat_grid import from_timing_points
from src.data.cache import read_manifest
from src.data.mel import open_mel
from src.data.tokenizer import BAR, BEATS_PER_CHUNK, HOLD_START, PAD, TAP, D, K, L
from src.evaluation.playability import grade_of
from src.models.diffusion import load_denoiser, pick_device
from src.models.judge import Judge

TRAIN_STRIDE = 4 * BAR                 # training windows every 4 bars (half overlapping)
GRADES = ("Easy", "Normal", "Hard", "Insane", "Expert", "Expert+")


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
    return ap.parse_args(argv)


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
        keys = [k for k in ai_rows if all(k in c for _, _, c in runs) and k in man
                and (a.cache / "tokens" / f"{k}.npz").exists() and self.store.has(k)]
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
            for label, rows, charts in runs:
                song["charts"][label] = charts[k]
                song["sr"][label] = as_float(rows[k].get("sr_gen"), sr_label)
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


def train_fold(a, data: Songs, train: list[dict], device, fold: int, log_path: Path):
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
    for _ in range(a.epochs):
        order = rng.permutation(len(pairs))
        for j in range(0, len(order), a.batch_pairs):
            chosen = [pairs[i] for i in order[j:j + a.batch_pairs]]
            items = [(s, st, "human") for s, st in chosen] + [(s, st, "ai") for s, st in chosen]
            x, mel, s_, b = data.batch(items, device, rng, a.mirror)
            y = torch.cat([torch.zeros(len(chosen)), torch.ones(len(chosen))]).to(device)
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
def score_fold(judge: Judge, data: Songs, test: list[dict], device, batch: int = 16):
    """{(key, chart): (logits [windows], rows [windows, L])} for the held-out songs."""
    out = {}
    for song in test:
        starts = song["test"]
        for chart in ("human", *data.labels):
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
        b = block(chart, keys)
        b["by_grade"] = {g: block(chart, [k for k in keys if grade[k] == g])
                         for g in GRADES if any(grade[k] == g for k in keys)}
        if chart != "ai":
            d = np.array([scores[(k, chart)][0].mean() - scores[(k, "ai")][0].mean()
                          for k in keys])
            lo, hi = boot_mean(d)
            b["minus_ai"] = {"mean": float(d.mean()), "lo": lo, "hi": hi,
                             "paired_vs_ai": paired(
                                 np.concatenate([scores[(k, chart)][0] for k in keys]),
                                 np.concatenate([scores[(k, "ai")][0] for k in keys]))}
        summary[chart] = b
        lines.append(f"{chart}: {b['windows']} windows of {b['songs']} songs  auc {b['auc']:.3f}  "
                     f"paired {b['paired']:.3f}  songs {b['songs_paired']:.3f}  "
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
          f"; {device}", flush=True)
    t0 = time.time()
    scores, fold_by_key = {}, {}
    for fold in range(n_folds):
        train = [s for s, f in zip(songs, fold_of, strict=True) if f != fold]
        test = [s for s, f in zip(songs, fold_of, strict=True) if f == fold]
        judge = train_fold(a, data, train, device, fold, log_path)
        scores.update(score_fold(judge, data, test, device))
        fold_by_key.update({s["key"]: fold for s in test})
        print(f"fold {fold}: {len(train)} songs trained, {len(test)} held out, "
              f"{time.time() - t0:.0f}s", flush=True)
        del judge
    with open(run_dir / "windows.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["key", "grade", "fold", "start", "chart", "logit", "onset_rows"])
        for s in songs:
            for chart in ("human", *data.labels):
                for st, lg in zip(s["test"], scores[(s["key"], chart)][0], strict=True):
                    w.writerow([s["key"], s["grade"], fold_by_key[s["key"]], st, chart,
                                f"{lg:.4f}", onset_rows(s["charts"][chart], st,
                                                        min(st + L, s["n"]))])
    rows = {f"{k}|{c}": v[1].astype(np.float16) for (k, c), v in scores.items()}
    rows.update({f"{s['key']}|starts": np.asarray(s["test"], dtype=np.int64) for s in songs})
    np.savez_compressed(run_dir / "rows.npz", **rows)
    summary, lines = report(songs, scores, data.labels)
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
