"""Fill the [확인] placeholders in docs/design/section2_encoding.md.

    python scripts/fill_section2.py

Prints the four missing numbers and writes the snap-error histogram.
"""

from pathlib import Path

import numpy as np
import pandas as pd


def _read(stem: Path) -> pd.DataFrame:
    for suffix in (".parquet", ".csv"):
        p = stem.with_suffix(suffix)
        if p.exists():
            return pd.read_parquet(p) if suffix == ".parquet" else pd.read_csv(p)
    raise FileNotFoundError(f"{stem}.parquet / .csv")


df = _read(Path("data/chart_index"))
print(f"charts: {len(df)}\n")

# ---- 1. chord distribution -------------------------------------------------
# chord*_ratio is a per-chart fraction of chord groups, so the dataset-level
# figure must be weighted by that chart's number of groups, not by note count.
wg = df["n_chord_groups"].to_numpy(dtype=float)
print("=== chord size distribution (group-weighted) ===")
tot3plus = 0.0
for k in (1, 2, 3, 4):
    col = f"chord{k}_ratio"
    if col not in df:
        continue
    v = df[col].to_numpy(dtype=float)
    ok = np.isfinite(v)
    share = float(np.average(v[ok], weights=wg[ok]))
    print(f"  {k}-note chords: {share:7.4f}  ({share * 100:.2f}%)")
    if k >= 3:
        tot3plus += share
print(f"  -> 3+ combined: {tot3plus * 100:.2f}%   <-- §2.4")

# ---- 2. hold ratio ---------------------------------------------------------
wn = df["n_notes"].to_numpy(dtype=float)
hr = df["hold_ratio"].to_numpy(dtype=float)
ok = np.isfinite(hr)
hold_mean = float(np.average(hr[ok], weights=wn[ok]))
print("\n=== hold notes ===")
print(f"  note-weighted hold ratio: {hold_mean:.4f}  ({hold_mean * 100:.2f}%)  <-- §2.7")
print(f"  per-chart median:         {float(np.nanmedian(hr)):.4f}")

# Occupancy including hold_body: every hold covers extra cells, so the true
# non-empty share is higher than the 7.83% computed from note starts alone.
if "hold_beats_median" in df:
    hb = df["hold_beats_median"].to_numpy(dtype=float)
    ok2 = np.isfinite(hb)
    med_len_cells = float(np.average(hb[ok2], weights=wn[ok2])) * 12  # d = 12
    starts = 0.078269  # occupancy_mean at d=12, from docs/_stats/cost.csv
    est = starts * (1 - hold_mean) + starts * hold_mean * (1 + med_len_cells)
    print(f"  mean hold length: {med_len_cells:.2f} cells at d=12")
    print(f"  -> estimated occupancy incl. hold_body: {est * 100:.2f}%"
          f"  (empty {100 - est * 100:.2f}%)  <-- §2.7")

# ---- 3. snap error distribution -------------------------------------------
print("\n=== snap error at d=12 ===")
notes = _read(Path("data/notes_snap"))
e = notes["err_ms_12"].to_numpy(dtype=float)
e = e[np.isfinite(e)]
print(f"  sampled notes: {len(e):,}")
for q in (50, 75, 90, 95, 99):
    print(f"  p{q}: {np.percentile(e, q):6.2f} ms")
for tol in (3, 5, 10, 15):
    print(f"  <= {tol:2d}ms: {(e <= tol).mean() * 100:.2f}%")

out = Path("docs/design/fig_snap_error_d12.png")
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 3.2), dpi=150)
    ax.hist(e[e <= 30], bins=120, color="#3b6ea5")
    for tol, c in ((5, "#c0392b"), (10, "#e67e22")):
        ax.axvline(tol, color=c, ls="--", lw=1,
                   label=f"{tol}ms ({(e <= tol).mean() * 100:.1f}%)")
    ax.set_xlabel("snap error (ms)")
    ax.set_ylabel("notes")
    ax.set_title("Snap error at d=12 (1/12 beat grid)")
    ax.legend()
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print(f"\n  wrote {out}   <-- §2.7")
except ImportError:
    print("\n  [skip] matplotlib not installed: pip install matplotlib")

# ---- 4. sanity checks the doc asserts --------------------------------------
print("\n=== assertions used in §2 ===")
lanes = [f"lane{k}_ratio" for k in range(4) if f"lane{k}_ratio" in df]
print("  lane balance:", ", ".join(
    f"{c[4]}={np.average(df[c].to_numpy(float), weights=wn):.4f}" for c in lanes))
if "jack_beats_min" in df:
    j = df["jack_beats_min"].to_numpy(dtype=float)
    j = j[np.isfinite(j)]
    print(f"  jack gap < 1/12 beat: {(j < 1 / 12).mean() * 100:.2f}% of charts")
print(f"  charts with SR: {int(df['sr'].notna().sum()):,} / {len(df):,}")