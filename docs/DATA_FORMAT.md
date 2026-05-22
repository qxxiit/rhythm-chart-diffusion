# Data Format Reference

> Reference for raw and processed data formats. Fill in details during Phase 1.

## Raw Data

### `.osu` file format

Text-based, line-oriented. Sections delimited by `[Section]` headers.

Key sections we use:
- `[General]` — audio filename, mode (3 = mania)
- `[Metadata]` — title, artist, difficulty name
- `[Difficulty]` — CircleSize (= key count for mania), OD, HP
- `[TimingPoints]` — BPM changes, hit object positions
- `[HitObjects]` — actual notes

Hit object format (mania):
```
x,y,time,type,hitSound,endTime:hitSample
```
- `x` encodes the lane: `lane = floor(x * keyCount / 512)`
- `time` = onset in ms
- `type & 128` indicates a hold (long note)
- `endTime` is the release time for holds

Full reference: https://osu.ppy.sh/wiki/en/Client/File_formats/osu_%28file_format%29

## Processed Data

### Mel-spectrogram

- 80 mel bins
- FFT size 512
- Hop length: 1/48 of beat duration (BPM-dependent)
- Saved as `.npy` per song: `(T_frames, 80)` float32

### Chart tokens

Tokenized chart as integer sequence.

Token vocabulary (initial design, may revise during Phase 1):
- `[BOS]`, `[EOS]`, `[PAD]` — special
- `[BEAT_<n>]` — beat boundary markers
- `[LANE_<k>_TAP]` — tap note on lane k
- `[LANE_<k>_HOLD_START]`, `[LANE_<k>_HOLD_END]` — long note bounds
- `[DT_<delta>]` — time delta (relative) between events

_TBD: finalize during Phase 1, week of 7/9._

## Train/Val/Test Split

- Song-level split (no song appears in multiple splits).
- Default: 80/10/10 random with fixed seed.
- Stratified by difficulty class to keep distribution similar.
