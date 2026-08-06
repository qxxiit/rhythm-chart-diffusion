"""osu! API v2 client credentials 토큰 발급 테스트."""
import os
import requests
from dotenv import load_dotenv

load_dotenv()  # repo 루트의 .env 읽기

resp = requests.post(
    "https://osu.ppy.sh/oauth/token",
    json={
        "client_id": os.environ["OSU_CLIENT_ID"],
        "client_secret": os.environ["OSU_CLIENT_SECRET"],
        "grant_type": "client_credentials",
        "scope": "public",
    },
    timeout=30,
)
print("status:", resp.status_code)
resp.raise_for_status()
data = resp.json()
print("token_type:", data["token_type"])
print("expires_in:", data["expires_in"], "sec (약 24시간)")
print("access_token[:20]:", data["access_token"][:20], "...")