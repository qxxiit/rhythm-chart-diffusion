"""Flatten beatmapsets.jsonl into a per-beatmap table and join SR onto
chart_index.parquet.

    python scripts/flatten_metadata.py

Writes data/metadata/beatmaps.csv and patches data/chart_index.parquet in
place (adds `sr` and a few API-side columns). Avoids re-running the 27-minute
full analysis just to attach one field.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path("data/metadata/beatmapsets.jsonl")
OUT = Path("data/metadata/beatmaps.csv")
INDEX = Path("data/chart_index.parquet")

KEEP = ["id", "beatmapset_id", "difficulty_rating", "version", "mode_int",
        "cs", "bpm", "total_length", "hit_length", "count_circles",
        "count_sliders", "max_combo", "accuracy", "status"]

rows = []
with SRC.open(encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        s = json.loads(line)
        for b in s.get("beatmaps", []):
            r = {k: b.get(k) for k in KEEP}
            r["set_artist"] = s.get("artist")
            r["set_title"] = s.get("title")
            rows.append(r)

bm = pd.DataFrame(rows)
print(f"{len(bm):,} beatmaps in {SRC.name}")

# mode_int 3 = mania; cs is the key count in mania
mania = bm[bm["mode_int"] == 3]
k4 = mania[mania["cs"] == 4]
print(f"  mania: {len(mania):,}   mania 4K: {len(k4):,}")

bm = bm.rename(columns={"id": "map_id", "beatmapset_id": "set_id"})
OUT.parent.mkdir(parents=True, exist_ok=True)
bm.to_csv(OUT, index=False)
print(f"  wrote {OUT}")

# ---- join onto chart_index -------------------------------------------------
if not INDEX.exists():
    raise SystemExit(f"{INDEX} not found")

df = pd.read_parquet(INDEX)
sr_map = dict(zip(bm["map_id"].astype("Int64"), bm["difficulty_rating"]))
matched = df["map_id"].map(sr_map)
print(f"\nSR matched: {int(matched.notna().sum()):,} / {len(df):,}")

if matched.notna().sum() == 0:
    print("  [!] no overlap. Compare a few ids on both sides:")
    print("      chart_index:", df['map_id'].head(3).tolist())
    print("      beatmaps   :", bm['map_id'].head(3).tolist())
    raise SystemExit(1)

df["sr"] = matched.astype(float)

# API-side fields worth having in §3, and useful as a parser cross-check
extra = bm.set_index("map_id")[["count_circles", "count_sliders", "max_combo",
                                "bpm", "total_length"]]
extra.columns = ["api_taps", "api_holds", "api_max_combo", "api_bpm",
                 "api_length_s"]
df = df.merge(extra, left_on="map_id", right_index=True, how="left")

df.to_parquet(INDEX, index=False)
print(f"  patched {INDEX}")

# ---- SR distribution for §3.1 ---------------------------------------------
sr = df["sr"].dropna().to_numpy()
print("\n=== SR distribution (§3.1) ===")
for q in (0, 25, 50, 75, 100):
    print(f"  p{q:<3}: {np.percentile(sr, q):.2f}")
print(f"  mean: {sr.mean():.2f}")
hist, edges = np.histogram(sr, bins=[0, 2, 3, 4, 5, 6, 7, 100])
for lo, hi, c in zip(edges[:-1], edges[1:], hist):
    hi = "+" if hi == 100 else f"{hi:g}"
    print(f"  {lo:g}-{hi:>2}: {c:6,} ({c / len(sr) * 100:5.2f}%)")

# ---- cross-check: our parser vs the API -----------------------------------
# In mania the API counts hold notes as sliders, so these should line up.
ok = df["api_holds"].notna() & df["n_holds"].notna()
if ok.any():
    d_hold = (df.loc[ok, "n_holds"] - df.loc[ok, "api_holds"]).abs()
    d_note = (df.loc[ok, "n_notes"]
              - (df.loc[ok, "api_taps"] + df.loc[ok, "api_holds"])).abs()
    print("\n=== parser vs API (§3.1 validation) ===")
    print(f"  hold count exact match:  {(d_hold == 0).mean() * 100:.2f}%")
    print(f"  total count exact match: {(d_note == 0).mean() * 100:.2f}%")
    bad = int((d_note > 0).sum())
    if bad:
        print(f"  [!] {bad:,} charts disagree on note count — inspect before"
              f" citing parse reliability in §3")