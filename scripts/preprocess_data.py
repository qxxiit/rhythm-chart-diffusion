"""Caches for the charts in the manifest (design doc §4.5). Layout: src/data/cache.py.

    python scripts/preprocess_data.py                   # tokens: data/cache/tokens/{key}.npz
    python scripts/preprocess_data.py --mel             # log-Mel of every kept chart's audio
    python scripts/preprocess_data.py --mel --splits train val --train-groups 600
                                                        # subset: 600 train songs + all of val
    python scripts/preprocess_data.py --fake-mel --limit 30   # mel made FROM THE TOKENS
    python scripts/preprocess_data.py --audio-range     # token range covers the audio

--mel decodes every audio file once, the way osu! does (src/data/audio.py), and
writes data/cache/logmel/{audio_id}.npy (fixed hop) and {audio_id}.json (decoder,
mp3 tag, song statistics), then maps each chart to its audio in
data/cache/logmel/index.csv. The token grid is applied when a window is read
(src/data/mel.py), so the charts of one set share one file. Only kept charts
(manifest drop == "") with an SR label are included. --train-groups N takes the
first N train songs (audio_key groups) in md5 order, the same subset every time.
Rerunning skips audio files that are done, so an interrupted run resumes; files
decoded under an older audio.AUDIO_RULE are redone.

--fake-mel writes mel that encodes the answer (cache.oracle_mel) to
data/cache/fake_mel/, read only with train.py/evaluate.py/sample.py --fake-mel.
Training on it only shows that the code runs end to end: never report its numbers.
Existing files are kept unless --overwrite.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
from functools import lru_cache
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from src.data.audio import AUDIO_RULE, SAMPLE_RATE, load_osu
from src.data.cache import oracle_mel, read_manifest, save_tokens
from src.data.chart_parser import parse_osu
from src.data.mel import HOP, log_mel, song_stats
from src.data.tokenizer import ChartTooLong, encode

_AUDIO_LINE = re.compile(r"^\s*AudioFilename\s*:\s*(.*?)\s*$", re.MULTILINE)


def audio_ms(chart_path: Path, audio_filename: str) -> float:
    import soundfile  # only needed for --audio-range
    return 1000.0 * soundfile.info(str(chart_path.parent / audio_filename)).duration


def process(job: tuple) -> str:
    row, root, cache, fake_mel, audio_range, overwrite = job
    tok_path = cache / "tokens" / f"{row['key']}.npz"
    mel_path = cache / "fake_mel" / f"{row['key']}.npy"
    if tok_path.exists() and not overwrite and (not fake_mel or mel_path.exists()):
        return "kept"
    try:
        path = root / row["path"]
        chart = parse_osu(path)
        sr = float(row["sr"]) if row["sr"] else math.nan
        span = audio_ms(path, chart.audio_filename) if audio_range else None
        tokens, metas, stats = encode(chart, sr, audio_ms=span)
    except ChartTooLong:
        return "too_long"
    except Exception as e:
        return f"error: {row['path']}: {type(e).__name__}: {e}"
    save_tokens(tok_path, tokens, metas, stats)
    if fake_mel:
        np.save(mel_path, oracle_mel(tokens, seed=int(row["key"][:8], 16)))
    return "ok"


# ---------- real log-Mel ----------

def audio_id(relpath: str) -> str:
    """Cache name of an audio file: md5 of its path under data/raw, like chart keys."""
    return hashlib.md5(relpath.encode("utf-8")).hexdigest()[:16]


@lru_cache(maxsize=4096)
def _files(folder: Path) -> tuple[Path, ...]:
    return tuple(p for p in folder.iterdir() if p.is_file()) if folder.is_dir() else ()


def find_audio(job: tuple) -> tuple[str, str | None]:
    """(chart key, path of its audio under root or None). AudioFilename is matched
    case-insensitively, as osu! on Windows and macOS does, and the path returned
    uses the name on disk, so the same file gets the same audio_id on any system."""
    row, root = job
    chart = root / row["path"]
    m = _AUDIO_LINE.search(chart.read_text(encoding="utf-8-sig", errors="replace"))
    if not m or not m.group(1):
        return row["key"], None
    target = chart.parent / m.group(1).replace("\\", "/")
    names = _files(target.parent)
    hit = next((p for p in names if p.name == target.name), None) \
        or next((p for p in names if p.name.lower() == target.name.lower()), None)
    return row["key"], hit.relative_to(root).as_posix() if hit else None


def decoder_version() -> str:
    import av
    import soxr
    codec = ".".join(map(str, av.library_versions["libavcodec"]))
    return f"pyav {av.__version__}, libavcodec {codec}, soxr {soxr.__version__}"


def mel_job(job: tuple) -> dict:
    rel, root, out, overwrite, version = job
    aid = audio_id(rel)
    npy, meta = out / f"{aid}.npy", out / f"{aid}.json"
    if npy.exists() and meta.exists() and not overwrite \
            and json.loads(meta.read_text(encoding="utf-8")).get("audio_rule") == AUDIO_RULE:
        return {"audio_id": aid, "audio": rel, "status": "kept"}
    try:
        samples, info = load_osu(root / rel)
        if len(samples) < SAMPLE_RATE:
            raise ValueError(f"only {len(samples)} samples")
        logmel = log_mel(samples)
        mean, std = song_stats(logmel)
    except Exception as e:
        return {"audio_id": aid, "audio": rel, "status": f"error: {type(e).__name__}: {e}"}
    tmp = out / f"{aid}.tmp.npy"
    np.save(tmp, logmel.astype(np.float16))
    tmp.replace(npy)
    record = {"audio": rel, "audio_rule": AUDIO_RULE, **info, "frames": len(logmel), "mel_rate": SAMPLE_RATE, "hop": HOP,
              "decoder": version, "mean": [round(float(v), 4) for v in mean],
              "std": [round(float(v), 4) for v in std]}
    meta.write_text(json.dumps(record), encoding="utf-8")   # written last: marks the audio done
    return {"audio_id": aid, "audio": rel, "status": "ok"}


def pick_rows(rows: list[dict], splits: list[str], train_groups: int) -> list[dict]:
    rows = [r for r in rows if r.get("drop", "") == "" and r["sr"] != "" and r["split"] in splits]
    if train_groups:
        groups = sorted({r["audio_key"] for r in rows if r["split"] == "train"},
                        key=lambda k: hashlib.md5(k.encode("utf-8")).hexdigest())[:train_groups]
        keep = set(groups)
        rows = [r for r in rows if r["split"] != "train" or r["audio_key"] in keep]
    return rows


def build_mel(a) -> int:
    out = a.cache / "logmel"
    out.mkdir(parents=True, exist_ok=True)
    rows = pick_rows(read_manifest(a.manifest), a.splits, a.train_groups)
    if a.limit:
        rows = rows[:a.limit]
    by_split = {s: sum(r["split"] == s for r in rows) for s in a.splits}
    print(f"{len(rows):,} charts (" + ", ".join(f"{s} {n:,}" for s, n in by_split.items()) + ")")

    version = decoder_version()
    results = []
    pool = Pool(a.workers) if a.workers > 1 else None
    try:
        lookups = [(r, a.root) for r in rows]
        found = dict(pool.imap(find_audio, lookups, chunksize=64) if pool
                     else map(find_audio, lookups))
        missing = [k for k, v in found.items() if v is None]
        audios = sorted({v for v in found.values() if v is not None})
        print(f"{len(audios):,} audio files; {len(missing):,} charts without one", flush=True)
        jobs = [(rel, a.root, out, a.overwrite, version) for rel in audios]
        work = pool.imap_unordered(mel_job, jobs, chunksize=1) if pool else map(mel_job, jobs)
        for i, res in enumerate(work, 1):
            results.append(res)
            if i % 200 == 0 or i == len(jobs):
                print(f"  [{i:,}/{len(jobs):,}] audio files", flush=True)
    finally:
        if pool:
            pool.close()
            pool.join()

    done = {r["audio"]: r["audio_id"] for r in results if not r["status"].startswith("error")}
    index_path = out / "index.csv"
    index: dict[str, dict] = {}
    if index_path.exists():
        with open(index_path, newline="", encoding="utf-8") as f:
            index = {r["key"]: r for r in csv.DictReader(f)}
    for key, rel in found.items():
        if rel in done:
            index[key] = {"key": key, "audio_id": done[rel], "audio": rel}
    with open(index_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["key", "audio_id", "audio"], lineterminator="\n")
        w.writeheader()
        w.writerows(index[k] for k in sorted(index))

    counts: dict[str, int] = {}
    for r in results:
        kind = r["status"].split(":")[0]
        counts[kind] = counts.get(kind, 0) + 1
    tags: dict[str, int] = {}
    for aid in done.values():
        info = json.loads((out / f"{aid}.json").read_text(encoding="utf-8"))
        tag = info["codec"] + (f"/{info['mp3_tag']}" if "mp3_tag" in info else "")
        tags[tag] = tags.get(tag, 0) + 1
    size = sum(p.stat().st_size for p in out.glob("*.npy")) / 1e9
    print("audio: " + ", ".join(f"{k} {v:,}" for k, v in sorted(counts.items())))
    print("by codec / mp3 tag: " + ", ".join(f"{k} {v:,}" for k, v in sorted(tags.items())))
    print(f"{index_path}: {len(index):,} charts; {size:.1f} GB of log-Mel in {out}")
    for res in [r for r in results if r["status"].startswith("error")][:10]:
        print(f"  {res['audio']}: {res['status']}")
    for key in missing[:10]:
        print(f"  no audio file for chart {key}")
    return 1 if counts.get("error") else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"))
    ap.add_argument("--root", type=Path, default=Path("data/raw"))
    ap.add_argument("--cache", type=Path, default=Path("data/cache"))
    ap.add_argument("--mel", action="store_true", help="real log-Mel cache (see above)")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"],
                    help="--mel: splits to cover")
    ap.add_argument("--train-groups", type=int, default=0,
                    help="--mel: only the first N train songs in md5 order (0 = all)")
    ap.add_argument("--fake-mel", action="store_true", help="plumbing test only, see above")
    ap.add_argument("--audio-range", action="store_true", help="encode(..., audio_ms=duration)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="only the first N manifest rows")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    a = ap.parse_args(argv)

    if a.mel:
        if a.fake_mel:
            ap.error("--mel and --fake-mel are separate runs")
        return build_mel(a)

    rows = read_manifest(a.manifest)
    if a.fake_mel and not a.limit:
        ap.error("--fake-mel needs --limit: it is for plumbing tests")
    if a.limit:
        rows = rows[:a.limit]
    (a.cache / "tokens").mkdir(parents=True, exist_ok=True)
    if a.fake_mel:
        (a.cache / "fake_mel").mkdir(parents=True, exist_ok=True)
        print("[fake-mel] mel is made from the tokens: plumbing test only")
    jobs = [(r, a.root, a.cache, a.fake_mel, a.audio_range, a.overwrite) for r in rows]
    if a.workers > 1:
        with Pool(a.workers) as pool:
            results = list(pool.imap(process, jobs, chunksize=16))
    else:
        results = [process(j) for j in jobs]

    counts: dict[str, int] = {}
    for res in results:
        kind = res.split(":")[0]
        counts[kind] = counts.get(kind, 0) + 1
    print(f"{len(rows):,} charts: " + ", ".join(f"{k} {v:,}" for k, v in sorted(counts.items())))
    for res in [r for r in results if r.startswith("error")][:10]:
        print("  " + res)
    return 1 if counts.get("error") else 0


if __name__ == "__main__":
    raise SystemExit(main())
