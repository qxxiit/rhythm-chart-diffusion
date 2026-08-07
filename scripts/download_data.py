"""Download osu!mania beatmapsets (.osz) for training.

Phase 1 entry point. Downloads .osz archives for beatmapset IDs listed in
data/metadata/targets.txt (produced by scripts/make_target_list.py, which
applies the mode/status filter at the metadata stage).

Pipeline: fetch_metadata.py -> make_target_list.py -> download_data.py (.osz)
          -> extract_osz.py (data/raw: 4K .osu + audio only)

Usage:
    python scripts/download_data.py --output data/osz [--limit 100]

Requires nothing from the osu! API (mirrors only); no credentials needed here.

Done (was TODO):
- [x] OAuth + cursor pagination  -> scripts/fetch_metadata.py
- [x] Resumable download (skip existing files)
- [x] Rate limiting (sleep + jitter between requests)
- [x] Mode/status filter        -> applied in make_target_list.py
"""

import argparse
import csv
import pathlib
import random
import time
from datetime import datetime

import requests

MIRRORS = [
    "https://catboy.best/d/{sid}",
    "https://api.nerinyan.moe/d/{sid}",
]
MIN_SIZE = 10_000  # bytes; smaller responses are treated as error pages

def download_one(sid: int, out_dir: pathlib.Path, session: requests.Session) -> str:
    if (pathlib.Path("data/raw") / str(sid)).exists():   # 이미 추출 완료
            return "skip"
    out = out_dir / f"{sid}.osz"
    if out.exists() and out.stat().st_size > MIN_SIZE:
        return "skip"
    for base in MIRRORS:
        url = base.format(sid=sid)
        try:
            r = session.get(url, timeout=120)
            if r.status_code == 200 and len(r.content) > MIN_SIZE:
                out.write_bytes(r.content)
                return "ok"
            if r.status_code == 429:
                time.sleep(60)  # this mirror is rate-limiting us; cool down
        except requests.RequestException:
            continue
    return "fail"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", default="mania-4k",
                        choices=["mania-4k", "mania-5k", "mania-7k"],
                        help="Recorded for provenance; filtering already done in targets.txt")
    parser.add_argument("--status", default="ranked",
                        choices=["ranked", "loved", "approved"],
                        help="Recorded for provenance; filtering already done in targets.txt")
    parser.add_argument("--targets", default="data/metadata/targets.txt")
    parser.add_argument("--output", default="data/osz",
                        help=".osz output dir (data/raw is produced by extract_osz.py)")
    parser.add_argument("--limit", type=int, default=None, help="Max beatmapsets to download")
    args = parser.parse_args()

    out_dir = pathlib.Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    targets = [int(line) for line in open(args.targets)]
    if args.limit:
        targets = targets[: args.limit]

    log_path = pathlib.Path("data/metadata/download_log.csv")
    write_header = not log_path.exists()
    session = requests.Session()
    stats = {"ok": 0, "skip": 0, "fail": 0}

    with log_path.open("a", newline="") as lf:
        writer = csv.writer(lf)
        if write_header:
            writer.writerow(["sid", "status", "ts"])
        for i, sid in enumerate(targets, 1):
            status = download_one(sid, out_dir, session)
            stats[status] += 1
            writer.writerow([sid, status, datetime.now().isoformat(timespec="seconds")])
            print(f"[{i}/{len(targets)}] {sid}: {status} | {stats}", end="\r")
            if status == "ok":
                time.sleep(1.0 + random.random())  # be polite to mirrors

    print(f"\ndone: {stats}")


if __name__ == "__main__":
    main()