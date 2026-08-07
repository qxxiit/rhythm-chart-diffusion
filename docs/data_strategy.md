# Data Strategy Notes

Notes verified directly from `sample_response.json`.

## Beatmapset -> difficulty list

- The difficulty list inside a beatmapset is stored under `beatmaps`.
- Example path: `.beatmapsets[0].beatmaps`

## Fields on each difficulty entry

- Key count: `cs`
- Star rating: `difficulty_rating`
- Mode string: `mode`
- Mode integer: `mode_int`
- Difficulty name: `version`

## Sample observed values

From `.beatmapsets[0].beatmaps[0]` in `sample_response.json`:

- `cs`: `4`
- `difficulty_rating`: `4.05338`
- `mode`: `"mania"`
- `mode_int`: `3`

## 260730
총 세트: 7363
4K 포함 세트: 5882
중복: 0

## 260808
5881 sets → catalog.csv