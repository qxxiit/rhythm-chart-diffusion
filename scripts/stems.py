"""Per-stem features for every cached audio file: htdemucs's four stems, each as a 16-band
log-Mel on the mel cache's own frames (11/14).

    PYTORCH_ENABLE_MPS_FALLBACK=1 python scripts/stems.py
    python scripts/stems.py --shard 0/2 --limit 50 --device cpu

Every audio file of data/cache/logmel/index.csv once, the val and test songs' first (about
2 h on an M-series GPU), then the train songs' (about 18 h): a model can be tried on val
before the train set is done. For each: decode as the mel cache does (src.data.audio:
sample 0 is osu!'s 0 ms), separate at 44.1 kHz with htdemucs (drums, bass, other, vocals;
the mono mix as stereo, shifts 1, overlap 0.25), resample each stem to 22,050 Hz, take its
log-Mel (src.data.mel.log_mel: hop 128, 80 bands) and pool it to 16 bands (the power of 5
neighbouring bands summed). Written to data/cache/stems/:

    {audio_id}.npy    float16 [frames, 4 * 16]: drums, bass, other, vocals; the same frames
                      as the mixture's logmel (padded or cut to its count)
    {audio_id}.json   written last (it marks the file done): the sources, bands, frames,
                      per-column mean and std (the song normalization of the model input),
                      the seconds it took

Files already done are skipped, so it can stop and start again. It stops when the disk has
less than --min-free GB left (the whole cache is about 22 GB). Separated audio is not kept.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import shutil
import sys
import time
import types
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.audio import SAMPLE_RATE, decode_osu, resample
from src.data.mel import log_mel, song_stats

SOURCES = ("drums", "bass", "other", "vocals")
BANDS = 16                       # per stem
POOL = 5                         # 80 mel bands / 16
SPLIT_ORDER = {"val": 0, "test": 1, "train": 2}


def pool_bands(logmel: np.ndarray, pool: int = POOL) -> np.ndarray:
    """[frames, 80] log power -> [frames, 80 / pool]: the power of `pool` neighbouring bands
    summed, as a log again."""
    x = np.asarray(logmel, dtype=np.float32)
    n, b = x.shape
    x = x.reshape(n, b // pool, pool)
    top = x.max(axis=2, keepdims=True)
    return (top[..., 0] + np.log(np.exp(x - top).sum(axis=2))).astype(np.float32)


def stem_features(stems: np.ndarray, rate: int, frames: int) -> np.ndarray:
    """[4, n] mono stems at `rate` -> float32 [frames, 4 * BANDS] on the mel cache's frames."""
    cols = []
    for st in stems:
        lm = pool_bands(log_mel(resample(np.asarray(st, dtype=np.float32), rate, SAMPLE_RATE)))
        if len(lm) < frames:                              # resampling can be a frame short
            lm = np.concatenate([lm, np.repeat(lm[-1:], frames - len(lm), axis=0)])
        cols.append(lm[:frames])
    return np.concatenate(cols, axis=1)


def openunmix_shim() -> None:
    """demucs needs only openunmix.filtering.wiener, but openunmix's __init__ imports
    torchaudio, which this .venv need not be able to load: register the package without
    running its __init__."""
    if "openunmix" in sys.modules:
        return
    spec = importlib.util.find_spec("openunmix")
    if spec is None or not spec.submodule_search_locations:
        raise SystemExit("openunmix is not installed (outputs/patches_1108/start_stems_speed.sh "
                         "installs demucs and it)")
    pkg = types.ModuleType("openunmix")
    pkg.__path__ = list(spec.submodule_search_locations)
    sys.modules["openunmix"] = pkg


def demucs_separator(device: str):
    """(separate(mono, rate) -> ([4, n] mono stems, rate), the model's sources)."""
    import torch
    openunmix_shim()
    from demucs.apply import apply_model
    from demucs.pretrained import get_model

    model = get_model("htdemucs")
    model.eval()
    if device == "mps" and not torch.backends.mps.is_available():
        print("mps is not available: cpu", flush=True)
        device = "cpu"

    def separate(mono: np.ndarray, rate: int) -> tuple[np.ndarray, int]:
        wav = resample(mono, rate, model.samplerate)
        x = torch.from_numpy(np.stack([wav, wav]).astype(np.float32))
        ref = x.mean(0)
        mean, std = ref.mean(), ref.std() + 1e-8
        with torch.no_grad():
            y = apply_model(model, ((x - mean) / std)[None], device=device, shifts=1,
                            split=True, overlap=0.25, progress=False)[0]
        out = (y * std + mean).mean(1).cpu().numpy()
        if device == "mps":
            torch.mps.empty_cache()
        return out, model.samplerate

    return separate, tuple(model.sources)


def todo(cache: Path, manifest: Path, splits: tuple[str, ...]) -> list[tuple[str, str, str]]:
    """(audio_id, audio path under data/raw, split) once per audio file of the logmel index,
    the splits in SPLIT_ORDER, then by audio_id."""
    split_of = {}
    with open(manifest, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            split_of[r["key"]] = r["split"]
    seen: dict[str, tuple[str, str]] = {}
    with open(cache / "logmel" / "index.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            sp = split_of.get(r["key"])
            if sp not in splits:
                continue
            old = seen.get(r["audio_id"])
            if old is None or SPLIT_ORDER.get(sp, 9) < SPLIT_ORDER.get(old[1], 9):
                seen[r["audio_id"]] = (r["audio"], sp)
    return sorted(((a, rel, sp) for a, (rel, sp) in seen.items()),
                  key=lambda t: (SPLIT_ORDER.get(t[2], 9), t[0]))


def process(aid: str, rel: str, raw: Path, cache: Path, out: Path, separate,
            sources: tuple[str, ...]) -> float:
    """Features of one audio file into out/{aid}.npy and .json; seconds it took."""
    t0 = time.time()
    meta = json.loads((cache / "logmel" / f"{aid}.json").read_text(encoding="utf-8"))
    d = decode_osu(raw / rel)
    stems, rate = separate(d.samples, d.sample_rate)
    feats = stem_features(stems, rate, int(meta["frames"]))
    mean, std = song_stats(feats)
    tmp = out / f"{aid}.tmp.npy"
    np.save(tmp, feats.astype(np.float16))
    tmp.replace(out / f"{aid}.npy")
    seconds = time.time() - t0
    record = {"audio": rel, "sources": list(sources), "bands": BANDS, "pool": POOL,
              "frames": len(feats), "mean": [round(float(v), 4) for v in mean],
              "std": [round(float(v), 4) for v in std], "seconds": round(seconds, 2)}
    (out / f"{aid}.json").write_text(json.dumps(record), encoding="utf-8")
    return seconds


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cache", type=Path, default=Path("data/cache"))
    ap.add_argument("--raw", type=Path, default=Path("data/raw"))
    ap.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"))
    ap.add_argument("--splits", nargs="+", default=["val", "test", "train"])
    ap.add_argument("--shard", default="0/1", help="k/n: every n-th file from the k-th")
    ap.add_argument("--limit", type=int, default=None, help="at most this many files to do")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--min-free", type=float, default=15.0, help="GB left on the disk to stop at")
    a = ap.parse_args(argv)

    k, n = (int(v) for v in a.shard.split("/"))
    out = a.cache / "stems"
    out.mkdir(parents=True, exist_ok=True)
    files = [t for i, t in enumerate(todo(a.cache, a.manifest, tuple(a.splits))) if i % n == k]
    left = [t for t in files if not (out / f"{t[0]}.json").exists()]
    n_done = len(files) - len(left)
    if a.limit is not None:
        left = left[:a.limit]
    by_split = {s: sum(1 for t in left if t[2] == s) for s in a.splits}
    print(f"stems: {len(files)} audio files in shard {a.shard}, {n_done} done, "
          f"{len(left)} to do now {by_split}", flush=True)
    if not left:
        return 0
    separate, sources = demucs_separator(a.device)
    t_start, done, failed = time.time(), 0, 0
    for i, (aid, rel, sp) in enumerate(left):
        free = shutil.disk_usage(out).free / 1e9
        if free < a.min_free:
            print(f"stopped: {free:.1f} GB free on the disk (--min-free {a.min_free})", flush=True)
            return 3
        try:
            s = process(aid, rel, a.raw, a.cache, out, separate, sources)
            done += 1
        except Exception as e:                            # one bad file does not stop the run
            failed += 1
            print(f"[{i + 1}/{len(left)}] {aid} ({sp}) failed: {type(e).__name__}: {e}", flush=True)
            continue
        el = time.time() - t_start
        eta = el / (i + 1) * (len(left) - i - 1) / 3600
        print(f"[{i + 1}/{len(left)}] {aid} ({sp}) {s:.1f} s; {el / 3600:.2f} h so far, "
              f"about {eta:.1f} h left", flush=True)
    print(f"stems: {done} done, {failed} failed, {(time.time() - t_start) / 3600:.2f} h", flush=True)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
