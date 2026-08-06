"""data/raw 스캔 → catalog.csv (이후 모든 필터링의 기준표)."""
import csv, json, pathlib

meta = {}
for l in open("data/metadata/beatmapsets.jsonl", encoding="utf-8"):
    s = json.loads(l); meta[str(s["id"])] = s

rows = []
for d in sorted(pathlib.Path("data/raw").iterdir()):
    if not d.is_dir() or d.name.startswith("_"):
        continue
    s = meta.get(d.name, {})
    osus = list(d.glob("*.osu"))
    audio = [p for p in d.iterdir() if p.suffix in {".mp3", ".ogg", ".wav"}]
    rows.append({
        "sid": d.name,
        "artist": s.get("artist", ""), "title": s.get("title", ""),
        "n_4k_diffs": len(osus), "has_audio": bool(audio),
    })

with open("data/metadata/catalog.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
print(f"{len(rows)} sets → catalog.csv")