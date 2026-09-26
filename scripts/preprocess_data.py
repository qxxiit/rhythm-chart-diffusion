"""Token cache for every chart in the manifest (design doc §4.5).

    python scripts/preprocess_data.py                  # data/cache/tokens/{key}.npz
    python scripts/preprocess_data.py --fake-mel --limit 30   # + mel made FROM THE TOKENS
    python scripts/preprocess_data.py --audio-range    # token range covers the audio (needs soundfile)

The real mel cache is written by the mel pipeline (design doc §3); the file
layout both sides agree on is in src/data/cache.py.
--fake-mel writes mel that encodes the answer (cache.oracle_mel), so training
on it only shows that the code runs end to end. Never report numbers from it,
and delete data/cache/mel before the real mel is written.
Existing files are kept unless --overwrite.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from src.data.cache import oracle_mel, read_manifest, save_tokens
from src.data.chart_parser import parse_osu
from src.data.tokenizer import ChartTooLong, encode


def audio_ms(chart_path: Path, audio_filename: str) -> float:
    import soundfile  # only needed for --audio-range
    return 1000.0 * soundfile.info(str(chart_path.parent / audio_filename)).duration


def process(job: tuple) -> str:
    row, root, cache, fake_mel, audio_range, overwrite = job
    tok_path = cache / "tokens" / f"{row['key']}.npz"
    mel_path = cache / "mel" / f"{row['key']}.npy"
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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"))
    ap.add_argument("--root", type=Path, default=Path("data/raw"))
    ap.add_argument("--cache", type=Path, default=Path("data/cache"))
    ap.add_argument("--fake-mel", action="store_true", help="plumbing test only, see above")
    ap.add_argument("--audio-range", action="store_true", help="encode(..., audio_ms=duration)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="only the first N manifest rows")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    a = ap.parse_args(argv)

    rows = read_manifest(a.manifest)
    if a.fake_mel and not a.limit:
        ap.error("--fake-mel needs --limit: mel for every chart takes ~55 GB")
    if a.limit:
        rows = rows[:a.limit]
    (a.cache / "tokens").mkdir(parents=True, exist_ok=True)
    if a.fake_mel:
        (a.cache / "mel").mkdir(parents=True, exist_ok=True)
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
