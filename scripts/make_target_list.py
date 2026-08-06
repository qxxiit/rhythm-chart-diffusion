"""메타데이터 → 다운로드 대상 beatmapset ID 리스트."""
import json, pathlib

sets = [json.loads(l) for l in open("data/metadata/beatmapsets.jsonl", encoding="utf-8")]

def keep(s):
    beatmaps = s.get("beatmaps", [])
    if not beatmaps:                       # 난이도 정보 없으면 일단 포함
        return True
    return any(b.get("mode_int") == 3 and b.get("cs") == 4 for b in beatmaps)

targets = sorted({s["id"] for s in sets if keep(s)})
out = pathlib.Path("data/metadata/targets.txt")
out.write_text("\n".join(map(str, targets)))
print(f"{len(targets)} targets → {out}")