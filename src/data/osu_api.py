"""osu! API v2 클라이언트 (client credentials)."""
import os, time
import requests
from dotenv import load_dotenv

load_dotenv()
API = "https://osu.ppy.sh/api/v2"

def get_token() -> str:
    r = requests.post("https://osu.ppy.sh/oauth/token", json={
        "client_id": os.environ["OSU_CLIENT_ID"],
        "client_secret": os.environ["OSU_CLIENT_SECRET"],
        "grant_type": "client_credentials",
        "scope": "public",
    }, timeout=30)
    r.raise_for_status()
    return r.json()["access_token"]

def search_beatmapsets(token: str, mode: int = 3, status: str = "ranked",
                       cursor_string: str | None = None) -> dict:
    params = {"m": mode, "s": status}
    if cursor_string:
        params["cursor_string"] = cursor_string
    r = requests.get(f"{API}/beatmapsets/search", params=params,
                     headers={"Authorization": f"Bearer {token}"}, timeout=30)
    if r.status_code == 429:          # rate limit → 쉬고 재시도 신호
        return {"_rate_limited": True}
    r.raise_for_status()
    return r.json()