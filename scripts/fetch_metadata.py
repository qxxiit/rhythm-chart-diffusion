"""ranked mania beatmapset 메타데이터 전체 수집 → JSONL."""
import json, time, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.data.osu_api import get_token, search_beatmapsets

OUT = pathlib.Path("data/metadata"); OUT.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT / "beatmapsets.jsonl"
CURSOR_FILE = OUT / "cursor.txt"          # 중단 시 이어받기용

def main():
    token = get_token()
    cursor = CURSOR_FILE.read_text().strip() if CURSOR_FILE.exists() else None
    mode = "a" if cursor else "w"         # 이어받기면 append
    n_sets, n_pages = 0, 0

    with OUT_FILE.open(mode, encoding="utf-8") as f:
        while True:
            page = search_beatmapsets(token, cursor_string=cursor)
            if page.get("_rate_limited"):
                print("429 rate limited — 60초 대기"); time.sleep(60); continue
            sets = page.get("beatmapsets", [])
            if not sets:
                break
            for s in sets:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
            n_sets += len(sets); n_pages += 1
            cursor = page.get("cursor_string")
            if cursor:
                CURSOR_FILE.write_text(cursor)
            print(f"page {n_pages}: 누적 {n_sets} sets", end="\r")
            if not cursor:
                break
            time.sleep(1)                  # 매너 슬립

    print(f"\n완료: {n_sets} sets → {OUT_FILE}")
    CURSOR_FILE.unlink(missing_ok=True)    # 정상 완료 시 커서 삭제

if __name__ == "__main__":
    main()