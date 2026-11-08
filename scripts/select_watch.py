"""Stop a drawing that chooses by a judge once its choice is seen to move little (EXPERIMENTS
2026-11-08 (5); DECISIONS 2026-11-08: a run is cut where its reading is decided).

    python scripts/select_watch.py --logs L ... --judge J --pids P ... [--after 30]
        [--min-share 0.1] [--every 60] [--field select|lane]

Reads "select gain X" from the per-song lines of the evaluate.py logs (--select-judge: how far,
in the judge's logits, each window's chosen draw lies under the mean of its draws, averaged
over the song), or with --field lane "lane gain X" (--lane-select-judge: the same for the
versions of the lanes, span by span). Once --after songs are done, their mean gain is set against the judge's gap
between the run's windows and the human ones (models.json "gap"): under --min-share of it, the
choosing cannot carry the charts far toward the human ones, so the processes --pids are
stopped (SIGTERM) and STOPPED is written next to the first log (with the numbers); otherwise
the watcher only reports and leaves. It also leaves when the processes end first. Aggregate
only: the gains are not shown per song.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import time
from pathlib import Path

FIELDS = ("select", "lane")


def gain_pattern(field: str = "select") -> re.Pattern:
    return re.compile(rf"^\[\d+/\d+\] .* {field} gain (-?[0-9.]+|nan) ")


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--logs", type=Path, nargs="+", required=True)
    ap.add_argument("--judge", type=Path, required=True, help="the judge.py --save-models run")
    ap.add_argument("--pids", type=int, nargs="*", default=[])
    ap.add_argument("--after", type=int, default=30, help="songs before the reading")
    ap.add_argument("--min-share", type=float, default=0.1,
                    help="the mean gain under this share of the judge's gap stops the run")
    ap.add_argument("--every", type=float, default=60.0, help="seconds between looks")
    ap.add_argument("--field", choices=FIELDS, default="select",
                    help="the gain read: the windows' (select) or the lanes' (lane)")
    return ap.parse_args(argv)


def gains(logs: list[Path], field: str = "select") -> list[float]:
    pattern, out = gain_pattern(field), []
    for p in logs:
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            m = pattern.match(line)
            if m and m.group(1) != "nan":
                out.append(float(m.group(1)))
    return out


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def decide(found: list[float], gap: float, after: int, min_share: float) -> tuple[bool, str]:
    """(stop, line): the mean gain of the first `after` songs against min_share of the gap."""
    first = found[:after]
    mean = sum(first) / len(first)
    share = mean / gap if gap > 0 else float("nan")
    stop = share < min_share
    line = (f"select gain over {len(first)} songs {mean:.3f} logits, {share:.1%} of the judge's "
            f"gap {gap:.2f}: " + (f"under {min_share:.0%}, the choosing moves little; stopped"
                                  if stop else f"{min_share:.0%} or more, the run goes on"))
    return stop, line


def main(argv=None) -> int:
    a = parse_args(argv)
    gap = float(json.loads((a.judge / "models.json").read_text(encoding="utf-8"))["gap"])
    while True:
        found = gains(a.logs, a.field)
        if len(found) >= a.after:
            stop, line = decide(found, gap, a.after, a.min_share)
            print(line, flush=True)
            if stop:
                for pid in a.pids:
                    if alive(pid):
                        os.kill(pid, signal.SIGTERM)
                Path(str(a.logs[0]) + ".STOPPED").write_text(line + "\n", encoding="utf-8")
            return 0
        if a.pids and not any(alive(p) for p in a.pids):
            print(f"the processes ended after {len(found)} songs, before the reading", flush=True)
            return 0
        time.sleep(a.every)


if __name__ == "__main__":
    raise SystemExit(main())
