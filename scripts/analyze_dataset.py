"""Dataset statistics and beat-grid alignment analysis.

Covers the Aug 12 (statistics) and Aug 13 (alignment rate) blocks in one pass.

Outputs
-------
  data/chart_index.parquet     one row per chart
  data/notes_snap.parquet      per-note snap error (for histograms)
  docs/_stats/*.csv            the tables that go into docs/dataset_stats.md

Usage
-----
  python scripts/analyze_dataset.py --out data --docs docs/_stats
  python scripts/analyze_dataset.py --selftest     # synthetic data, no repo deps

ADAPTER: fill in iter_charts() to yield ChartRecord from the existing parser.
Nothing else in this file should need to change.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.beat_grid import (  # noqa: E402
    BeatGrid, TimingPoint, grid_index, snap_error_ms,
)

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

DIVISORS = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48]
TOLERANCES_MS = [3.0, 5.0, 10.0]
CHORD_EPS_MS = 2.0          # notes within this are one chord
COLLISION_DIVISORS = [8, 12, 16, 24, 48]


# --------------------------------------------------------------------------
# Adapter contract
# --------------------------------------------------------------------------

@dataclass
class ChartRecord:
    """Minimal view of one chart. Map the existing parser onto this."""

    map_id: int
    set_id: int
    key_count: int
    artist: str = ""
    title: str = ""
    version: str = ""            # difficulty name
    sr: float = float("nan")     # difficulty_rating from osu! API v2 metadata
    audio_filename: str = ""

    # per-note arrays, all same length, sorted by start_ms
    start_ms: np.ndarray = field(default_factory=lambda: np.zeros(0))
    lane: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    end_ms: np.ndarray = field(default_factory=lambda: np.zeros(0))  # NaN if tap

    timing_points: list[TimingPoint] = field(default_factory=list)


from src.data.chart_parser import parse_osu, _split_sections, _kv  # noqa: E402


def _read_metadata(path: Path) -> dict[str, str]:
    """[Metadata] is dropped by parse_osu, but set_id is what the train/test
    split must group on, so read it back here rather than re-plumbing the
    parser three days before the deadline."""
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    return _kv(_split_sections(text).get("Metadata", []))


def _load_sr(catalog: Path | None) -> dict[int, float]:
    """map_id -> difficulty_rating, from whatever build_catalog.py wrote.
    Column names are guessed; if the mapping comes back empty the run still
    works and `sr` is simply NaN throughout."""
    if catalog is None or not catalog.exists():
        return {}
    df = (pd.read_parquet(catalog) if catalog.suffix == ".parquet"
          else pd.read_json(catalog) if catalog.suffix == ".json"
          else pd.read_csv(catalog))
    id_col = next((c for c in ("map_id", "beatmap_id", "BeatmapID", "id")
                   if c in df.columns), None)
    sr_col = next((c for c in ("sr", "difficulty_rating", "star_rating", "stars")
                   if c in df.columns), None)
    if id_col is None or sr_col is None:
        print(f"  [warn] catalog columns not recognized: {list(df.columns)[:12]}")
        return {}
    return dict(zip(df[id_col].astype(int), df[sr_col].astype(float)))


def iter_charts(root: Path, catalog: Path | None = None, key_count: int | None = 4):
    sr_map = _load_sr(catalog)
    paths = sorted(root.rglob("*.osu"))
    print(f"found {len(paths)} .osu files")
    skipped = {"not_mania": 0, "wrong_keys": 0, "no_timing": 0, "no_notes": 0,
               "parse_error": 0}

    for p in paths:
        try:
            c = parse_osu(p)
        except ValueError:
            skipped["not_mania"] += 1
            continue
        except Exception as e:
            skipped["parse_error"] += 1
            print(f"  [parse_error] {p.name}: {type(e).__name__}: {e}")
            continue

        if key_count is not None and c.key_count != key_count:
            skipped["wrong_keys"] += 1
            continue
        if not c.timing_points:          # BeatGrid needs at least one red line
            skipped["no_timing"] += 1
            continue
        if not c.notes:
            skipped["no_notes"] += 1
            continue

        meta = _read_metadata(p)
        map_id = int(meta.get("BeatmapID") or 0)
        set_id = int(meta.get("BeatmapSetID") or 0)
        # Broken/legacy files carry no IDs. Fall back to the directory name,
        # which is how the downloader lays sets out, so grouped splits still work.
        if set_id <= 0:
            set_id = int(p.parent.name.split()[0]) if p.parent.name.split()[0].isdigit() else abs(hash(p.parent.name)) % 10**8
        if map_id <= 0:
            map_id = abs(hash(str(p))) % 10**9

        notes = c.notes
        yield ChartRecord(
            map_id=map_id,
            set_id=set_id,
            key_count=c.key_count,
            artist=meta.get("ArtistUnicode") or meta.get("Artist", ""),
            title=meta.get("TitleUnicode") or meta.get("Title", ""),
            version=meta.get("Version", ""),
            sr=sr_map.get(map_id, float("nan")),
            audio_filename=c.audio_filename,
            start_ms=np.fromiter((n.time_ms for n in notes), dtype=np.float64,
                                 count=len(notes)),
            lane=np.fromiter((n.lane for n in notes), dtype=np.int64,
                             count=len(notes)),
            end_ms=np.fromiter((n.end_ms if n.end_ms is not None else np.nan
                                for n in notes), dtype=np.float64, count=len(notes)),
            timing_points=[TimingPoint(float(t), float(bl))
                           for t, bl in c.timing_points],
        )

    print(f"skipped: {skipped}")


# --------------------------------------------------------------------------
# Per-chart metrics
# --------------------------------------------------------------------------

def _audio_key(rec: ChartRecord) -> str:
    """Normalized artist+title, for detecting the same song uploaded under
    different beatmapsets. Split policy must group on this, not on set_id
    alone, or the same audio lands in both train and test."""
    s = f"{rec.artist}|{rec.title}".lower()
    s = unicodedata.normalize("NFKD", s)
    s = re.sub(r"\(.*?\)|\[.*?\]", " ", s)          # (TV Size), [Nightcore]
    s = re.sub(r"[^a-z0-9\u3040-\u30ff\u4e00-\u9fff\uac00-\ud7af]+", "", s)
    return s


def _nps_p95(start_ms: np.ndarray, window_ms: float = 1000.0) -> float:
    """95th percentile of the note count in a 1s window anchored at each note.
    Closer to perceived difficulty than the mean, and it is what drives the
    grid-resolution decision."""
    if start_ms.size == 0:
        return 0.0
    right = np.searchsorted(start_ms, start_ms + window_ms, side="left")
    counts = right - np.arange(start_ms.size)
    return float(np.percentile(counts, 95))


def chart_metrics(rec: ChartRecord) -> tuple[dict, pd.DataFrame]:
    n = rec.start_ms.size
    row: dict = {
        "map_id": rec.map_id, "set_id": rec.set_id,
        "key_count": rec.key_count, "sr": rec.sr,
        "artist": rec.artist, "title": rec.title, "version": rec.version,
        "audio_key": _audio_key(rec),
        "n_notes": n,
    }
    if n == 0:
        return row, pd.DataFrame()

    grid = BeatGrid(rec.timing_points)
    t = rec.start_ms
    is_hold = ~np.isnan(rec.end_ms)

    # --- basic ---
    span_s = (t[-1] - t[0]) / 1000.0
    row["duration_s"] = span_s
    row["nps"] = n / span_s if span_s > 0 else 0.0
    row["nps_p95"] = _nps_p95(t)
    row["n_holds"] = int(is_hold.sum())
    row["hold_ratio"] = float(is_hold.mean())
    row["bpm_main"] = grid.bpm_main
    row["n_timing_sections"] = grid.n_sections
    row["offbeat_red_lines"] = grid.offbeat_red_lines()

    # --- beat coordinates ---
    lb = grid.local_beat(t)
    gb = grid.global_beat(t)
    bl = grid.beat_length_at(t)

    # --- snap error per divisor ---
    per_note = {"map_id": np.full(n, rec.map_id), "lane": rec.lane}
    for d in DIVISORS:
        err = snap_error_ms(lb, bl, d)
        per_note[f"err_ms_{d}"] = err
        for tol in TOLERANCES_MS:
            row[f"cov_d{d}_t{int(tol)}"] = float(np.mean(err <= tol))

    # --- chords ---
    # group notes whose start times are within CHORD_EPS_MS
    new_group = np.empty(n, dtype=bool)
    new_group[0] = True
    new_group[1:] = np.diff(t) > CHORD_EPS_MS
    gid = np.cumsum(new_group) - 1
    sizes = np.bincount(gid)
    row["n_chord_groups"] = int(sizes.size)
    for k in range(1, rec.key_count + 1):
        row[f"chord{k}_ratio"] = float(np.mean(sizes == k))
    row["chord_mean"] = float(sizes.mean())

    # --- lanes ---
    lane_counts = np.bincount(rec.lane, minlength=rec.key_count)
    for k in range(rec.key_count):
        row[f"lane{k}_ratio"] = float(lane_counts[k] / n)
    row["lane_imbalance"] = float(lane_counts.max() / max(lane_counts.min(), 1))

    # --- holds, in beats (a hold shorter than 1/d cannot be represented) ---
    if is_hold.any():
        hb = (rec.end_ms[is_hold] - t[is_hold]) / bl[is_hold]
        hb = hb[np.isfinite(hb) & (hb > 0)]
        if hb.size:
            row["hold_beats_min"] = float(hb.min())
            row["hold_beats_p01"] = float(np.percentile(hb, 1))
            row["hold_beats_median"] = float(np.median(hb))

    # --- jacks: consecutive notes in the same lane ---
    jack_min = math.inf
    for k in range(rec.key_count):
        tk = gb[rec.lane == k]
        if tk.size > 1:
            jack_min = min(jack_min, float(np.diff(tk).min()))
    row["jack_beats_min"] = jack_min if math.isfinite(jack_min) else float("nan")

    # --- collisions: two notes in one (cell, lane) at resolution d.
    #     This is exactly the loss the Aug 17 round-trip will report; knowing
    #     it now means the grid decision on Aug 15 is already informed. ---
    for d in COLLISION_DIVISORS:
        cell = grid_index(gb, d)
        key = cell * rec.key_count + rec.lane
        uniq = np.unique(key).size
        row[f"collision_rate_d{d}"] = float((n - uniq) / n)
        row[f"seq_cells_d{d}"] = int(cell.max() - cell.min() + 1)
        row[f"occupancy_d{d}"] = float(uniq / max((cell.max() - cell.min() + 1)
                                                  * rec.key_count, 1))

    return row, pd.DataFrame(per_note)


# --------------------------------------------------------------------------
# Aggregate tables
# --------------------------------------------------------------------------

def coverage_table(df: pd.DataFrame) -> pd.DataFrame:
    """The Aug 13 deliverable: note-weighted cumulative coverage by divisor."""
    w = df["n_notes"].to_numpy(dtype=np.float64)
    rows = []
    for d in DIVISORS:
        r = {"divisor": d}
        for tol in TOLERANCES_MS:
            col = df[f"cov_d{d}_t{int(tol)}"].to_numpy(dtype=np.float64)
            ok = np.isfinite(col)
            r[f"tol_{int(tol)}ms"] = float(np.average(col[ok], weights=w[ok]))
        rows.append(r)
    out = pd.DataFrame(rows).set_index("divisor")
    # Marginal gain of doubling the resolution. Compared against d/2 rather
    # than the previous row, because the divisor list is not a single nested
    # chain -- 1/8 notes do not live on a 1/12 grid, so a plain diff() would
    # print meaningless negative "gains".
    out["gain_vs_half_pp"] = [
        round((out.loc[d, "tol_5ms"] - out.loc[d // 2, "tol_5ms"]) * 100, 2)
        if (d % 2 == 0 and d // 2 in out.index) else float("nan")
        for d in out.index
    ]
    return out.reset_index()


def cost_table(df: pd.DataFrame) -> pd.DataFrame:
    """Cost side of the same decision: sequence length, empty ratio, loss."""
    w = df["n_notes"].to_numpy(dtype=np.float64)
    rows = []
    for d in COLLISION_DIVISORS:
        cells = df[f"seq_cells_d{d}"].to_numpy(dtype=np.float64)
        occ = df[f"occupancy_d{d}"].to_numpy(dtype=np.float64)
        col = df[f"collision_rate_d{d}"].to_numpy(dtype=np.float64)
        rows.append({
            "divisor": d,
            "cells_median": float(np.median(cells)),
            "cells_p95": float(np.percentile(cells, 95)),
            "occupancy_mean": float(np.average(occ, weights=w)),
            "empty_ratio": 1.0 - float(np.average(occ, weights=w)),
            "collision_rate": float(np.average(col, weights=w)),
        })
    return pd.DataFrame(rows)


def split_sanity(df: pd.DataFrame) -> pd.DataFrame:
    """Leakage check. 3+ difficulties per set means a chart-level split puts
    the same audio in train and test. audio_key catches the second-order case:
    the same song uploaded as different beatmapsets."""
    per_set = df.groupby("set_id").size()
    per_audio = df.groupby("audio_key").size()
    sets_per_audio = df.groupby("audio_key")["set_id"].nunique()
    return pd.DataFrame([
        {"metric": "charts", "value": len(df)},
        {"metric": "beatmapsets", "value": int(per_set.size)},
        {"metric": "charts_per_set_mean", "value": round(float(per_set.mean()), 2)},
        {"metric": "distinct_audio_keys", "value": int(per_audio.size)},
        {"metric": "charts_per_audio_mean", "value": round(float(per_audio.mean()), 2)},
        {"metric": "audio_keys_spanning_multiple_sets",
         "value": int((sets_per_audio > 1).sum())},
        {"metric": "charts_affected_by_cross_set_dupes",
         "value": int(df[df["audio_key"].isin(
             sets_per_audio[sets_per_audio > 1].index)].shape[0])},
    ])


def outlier_table(df: pd.DataFrame) -> pd.DataFrame:
    rules = {
        "nps < 0.5": df["nps"] < 0.5,
        "duration < 30s": df["duration_s"] < 30,
        "duration > 400s": df["duration_s"] > 400,
        "hold_ratio > 0.9": df["hold_ratio"] > 0.9,
        "timing_sections > 20": df["n_timing_sections"] > 20,
        "offbeat_red_lines > 0": df["offbeat_red_lines"] > 0,
        "lane_imbalance > 2": df["lane_imbalance"] > 2,
        "hold shorter than 1/12 beat": df.get(
            "hold_beats_min", pd.Series(dtype=float)) < 1 / 12,
    }
    return pd.DataFrame([
        {"rule": k, "n_charts": int(v.sum()), "pct": round(100 * float(v.mean()), 2)}
        for k, v in rules.items() if v is not None
    ])


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def _write(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        try:
            df.to_parquet(path, index=False)
            return
        except Exception:
            path = path.with_suffix(".csv")
    df.to_csv(path, index=False)
    print(f"  wrote {path}")


def run(charts, out_dir: Path, docs_dir: Path, sample_notes: int = 400_000):
    rows, note_frames, seen = [], [], 0
    for i, rec in enumerate(charts):
        try:
            row, notes = chart_metrics(rec)
        except Exception as e:                      # never swallow silently
            rows.append({"map_id": rec.map_id, "set_id": rec.set_id,
                         "error": f"{type(e).__name__}: {e}"})
            continue
        rows.append(row)
        if seen < sample_notes and len(notes):
            note_frames.append(notes)
            seen += len(notes)
        if (i + 1) % 500 == 0:
            print(f"  {i + 1} charts", flush=True)

    df = pd.DataFrame(rows)
    n_err = int(df["error"].notna().sum()) if "error" in df else 0
    df = df[df.get("error").isna()] if "error" in df else df
    print(f"\n{len(df)} charts analyzed, {n_err} errors")

    _write(df, out_dir / "chart_index.parquet")
    if note_frames:
        _write(pd.concat(note_frames, ignore_index=True),
               out_dir / "notes_snap.parquet")

    tables = {
        "coverage": coverage_table(df),
        "cost": cost_table(df),
        "split_sanity": split_sanity(df),
        "outliers": outlier_table(df),
    }
    for name, tbl in tables.items():
        _write(tbl, docs_dir / f"{name}.csv")
        print(f"\n=== {name} ===")
        print(tbl.to_string(index=False))
    return df, tables


# --------------------------------------------------------------------------
# Self-test: synthetic charts with known snap structure
# --------------------------------------------------------------------------

def _synthetic(seed=0, n_charts=40):
    rng = np.random.default_rng(seed)
    for i in range(n_charts):
        bpm = float(rng.uniform(120, 200))
        bl = 60000.0 / bpm
        tps = [TimingPoint(0.0, bl)]
        if i % 5 == 0:                               # BPM change mid-song
            tps.append(TimingPoint(40_000.0 + rng.uniform(0, 137), bl * 0.75))
        grid = BeatGrid(tps)

        beats, lanes = [], []
        # 70% on 1/4, 20% on 1/8, 10% on 1/12 -> coverage at d=12 should be ~1.0
        for b in np.arange(0, 400, 0.25):
            u = rng.random()
            d = 4 if u < 0.70 else (8 if u < 0.90 else 12)
            off = rng.integers(0, d) / d
            beats.append(b + off - (b % 1))
            lanes.append(int(rng.integers(0, 4)))
        gb = np.unique(np.round(np.array(beats), 6))
        lanes = rng.integers(0, 4, size=gb.size)
        t = np.asarray(grid.time_from_global_beat(gb), dtype=np.float64)
        t += rng.normal(0, 1.2, size=t.size)         # human jitter, ms
        order = np.argsort(t)
        t, lanes = t[order], lanes[order]

        end = np.full(t.size, np.nan)
        hold = rng.random(t.size) < 0.15
        end[hold] = t[hold] + bl * rng.choice([0.5, 1.0, 2.0], size=hold.sum())

        yield ChartRecord(
            map_id=1000 + i, set_id=100 + i // 3, key_count=4,
            artist="Artist", title=f"Song {i // 3}", version=f"Lv{i % 3}",
            sr=float(rng.uniform(2, 6)),
            start_ms=t, lane=lanes, end_ms=end, timing_points=tps,
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("data/raw"))
    ap.add_argument("--out", type=Path, default=Path("data"))
    ap.add_argument("--docs", type=Path, default=Path("docs/_stats"))
    ap.add_argument("--catalog", type=Path, default=None,
                    help="build_catalog.py output, for difficulty_rating")
    ap.add_argument("--keys", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    charts = _synthetic() if a.selftest else iter_charts(a.root, a.catalog, a.keys)
    if a.limit:
        import itertools; charts = itertools.islice(charts, a.limit)
    run(charts, a.out, a.docs)


if __name__ == "__main__":
    main()