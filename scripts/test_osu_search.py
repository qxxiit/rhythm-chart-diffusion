"""beatmapsets/search 첫 페이지 확인: m=3(mania), s=ranked."""
import os, json
import requests
from dotenv import load_dotenv

load_dotenv()

def get_token():
    r = requests.post("https://osu.ppy.sh/oauth/token", json={
        "client_id": os.environ["OSU_CLIENT_ID"],
        "client_secret": os.environ["OSU_CLIENT_SECRET"],
        "grant_type": "client_credentials",
        "scope": "public",
    }, timeout=30)
    r.raise_for_status()
    return r.json()["access_token"]

token = get_token()
r = requests.get(
    "https://osu.ppy.sh/api/v2/beatmapsets/search",
    params={"m": 3, "s": "ranked"},
    headers={"Authorization": f"Bearer {token}"},
    timeout=30,
)
r.raise_for_status()
data = r.json()

print("이 페이지 beatmapset 수:", len(data["beatmapsets"]))
print("다음 페이지 커서:", data.get("cursor_string"))
first = data["beatmapsets"][0]
print("\n첫 번째 세트 미리보기:")
print(json.dumps({k: first[k] for k in ["id", "artist", "title", "status"]},
                 ensure_ascii=False, indent=2))

# 전체 구조를 눈으로 파악하고 싶으면:
with open("sample_response.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
print("\n전체 응답을 sample_response.json에 저장함 — 열어서 구조 훑어볼 것")