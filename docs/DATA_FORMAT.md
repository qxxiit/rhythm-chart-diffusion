# Data Format

How raw osu!mania beatmaps become the structures this project trains on.
Last updated: 2026-08-11 (parser complete; tokenization still in design)

---

## 1. Pipeline overview

```
osu! API v2 ──▶ beatmapsets.jsonl ──▶ targets.txt ──▶ *.osz ──▶ data/raw/{sid}/ ──▶ Chart objects
   metadata        (7,363 sets)        (5,882)      (archives)   .osu + audio        (parser)
```

| Path | Contents |
|---|---|
| `data/metadata/beatmapsets.jsonl` | One JSON object per ranked mania beatmapset, as returned by the API |
| `data/metadata/targets.txt` | Beatmapset IDs containing at least one 4K difficulty |
| `data/metadata/download_log.csv` | `sid, status, ts` — one row per download attempt |
| `data/osz/{sid}.osz` | Downloaded archive (transient; deleted after extraction) |
| `data/raw/{sid}/` | Extracted 4K `.osu` files plus the referenced audio file |
| `data/metadata/catalog.csv` | `sid, artist, title, n_4k_diffs, has_audio` — the filtering reference table |

`data/` is gitignored. The dataset is rebuilt by rerunning the scripts, never committed.

---

## 2. The `.osu` file

A `.osu` file is UTF-8 text (**with BOM** — read as `utf-8-sig`) divided into `[Section]` blocks. Blank lines and `//` comments appear throughout. Only four sections matter here.

### `[General]`
```ini
Mode: 3                    ; 0=std, 1=taiko, 2=catch, 3=mania. Only 3 is kept.
AudioFilename: audio.mp3   ; referenced file inside the archive
```

### `[Difficulty]`
```ini
CircleSize: 4              ; in mania this is the KEY COUNT, not a circle size
```
May appear as `4` or `4.0`, so parse as float then cast to int.

### `[TimingPoints]`
Comma-separated: `time, beatLength, meter, sampleSet, sampleIndex, volume, uninherited, effects`

- `beatLength > 0` → **uninherited** point, a real tempo change. `BPM = 60000 / beatLength`.
- `beatLength < 0` → **inherited** point, a slider-velocity multiplier. Retained in the file but ignored by the parser; may become relevant if scroll speed is ever modeled.

### `[HitObjects]`
Comma-separated: `x, y, time, type, hitSound, objectParams, hitSample`

- **Lane** is encoded in `x`: `lane = floor(x * key_count / 512)`, clamped to `key_count - 1` (x can equal 512 at the boundary).
- **Type** is a bitfield. Bit 7 (`type & 128`) marks a mania hold note; for holds, the first field of `objectParams` is the end time in ms.

```
64,192,1000,1,0,0:0:0:0:            → tap at t=1000ms, lane 0
192,192,1500,128,0,2500:0:0:0:0:    → hold from 1500ms to 2500ms, lane 1
```

---

## 3. Parsed representation

`src/data/chart_parser.py` produces:

```python
@dataclass
class Note:
    time_ms: int             # onset
    lane: int                # 0 .. key_count-1
    end_ms: int | None       # release time for holds; None for taps

@dataclass
class Chart:
    key_count: int
    audio_filename: str
    timing_points: list[tuple[int, float]]   # (time_ms, ms_per_beat), uninherited only
    notes: list[Note]                        # sorted by (time_ms, lane)

    @property
    def bpm(self) -> float                   # from the first uninherited timing point
```

Usage:

```python
from src.data.chart_parser import parse_osu

chart = parse_osu("data/raw/1154776/Se-U-Ra - Tsui no Maihime [Hard].osu")
chart.key_count   # 4
chart.bpm         # 202.0
len(chart.notes)  # 1015
```

`parse_osu` raises `ValueError` for non-mania files. Callers processing the corpus in bulk should catch it and skip rather than abort.

### Parsing notes and edge cases

- Encoding: `utf-8-sig` with `errors="replace"` — some community beatmaps contain invalid bytes in metadata fields.
- Simultaneous notes (chords) share a `time_ms` and differ only in `lane`.
- Only uninherited timing points are kept, so `timing_points` is never empty for a valid chart; a chart without one is malformed.
- Verified across the full corpus: **18,452 charts parsed, 0 failures.**

---

## 4. Dataset statistics

| | |
|---|---|
| Ranked mania beatmapsets surveyed | 7,363 |
| Sets containing 4K difficulties | 5,882 |
| Sets downloaded | 5,882 (0 failures) |
| 4K charts parsed | 18,452 (0 failures) |
| Mean 4K difficulties per set | ≈ 3.1 |

Difficulty distribution, note density, and beat-grid alignment statistics are pending; see `docs/dataset_stats.md` once produced. Those numbers determine the grid resolution chosen for tokenization.

---

## 5. Downstream formats (not yet fixed)

### Audio features
Planned per Yi et al. (ISMIR 2023): log-Mel spectrogram, 80 mel bins, FFT window 512, hop = **1/48 beat**. The beat-relative hop means one frame always spans the same musical duration regardless of tempo, so identical rhythms look identical to the model across songs at different BPMs.

### Chart tokens
Under design. Candidate schemes and the rationale for the final choice will be documented here once fixed (target: mid-August). The leading candidate is a fixed-length beat grid where each position holds the per-lane on/off state, since discrete diffusion assumes a fixed-length sequence and this makes position equivalent to time.

### Unity export
The team's rhythm game consumes a JSON chart format; the converter lives in `src/unity_export/` and is not yet implemented.
