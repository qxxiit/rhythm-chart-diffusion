"""Download osu!mania beatmaps for training.

Phase 0–1 entry point. Uses osu! API v2 to fetch ranked beatmaps,
saves .osu chart files and associated audio.

Usage:
    python scripts/download_data.py --mode mania-4k --status ranked --output data/raw

Requires OSU_CLIENT_ID and OSU_CLIENT_SECRET in environment (or .env).

TODO (Phase 0.5, week of 6/12):
- [ ] Implement OAuth flow to get access token
- [ ] Implement beatmapset listing with cursor pagination
- [ ] Implement download with resumable state (skip already-downloaded IDs)
- [ ] Add rate limiting (sleep between requests)
- [ ] Filter by mode (4K only initially)
"""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", default="mania-4k", choices=["mania-4k", "mania-5k", "mania-7k"])
    parser.add_argument("--status", default="ranked", choices=["ranked", "loved", "approved"])
    parser.add_argument("--output", default="data/raw", help="Output directory")
    parser.add_argument("--limit", type=int, default=None, help="Max beatmapsets to download")
    args = parser.parse_args()

    raise NotImplementedError(
        f"TODO: implement osu! API download.\n"
        f"  mode={args.mode}, status={args.status}, output={args.output}"
    )


if __name__ == "__main__":
    main()
