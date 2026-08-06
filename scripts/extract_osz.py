"""data/osz/*.osz → data/raw/{sid}/ 에 4K .osu + 음원만 추출."""
import pathlib, re, shutil, zipfile

SRC = pathlib.Path("data/osz")
DST = pathlib.Path("data/raw"); DST.mkdir(parents=True, exist_ok=True)
AUDIO_EXT = {".mp3", ".ogg", ".wav"}

def osu_header(text: str) -> dict:
    """섹션 전체 파싱 전, Mode/CircleSize/AudioFilename만 빠르게 뽑기."""
    out = {}
    for line in text.splitlines()[:80]:    # 헤더는 파일 앞부분에 있음
        m = re.match(r"\s*(Mode|CircleSize|AudioFilename)\s*:\s*(.+)", line)
        if m:
            out[m.group(1)] = m.group(2).strip()
    return out

def process(osz_path: pathlib.Path):
    sid = osz_path.stem
    dst = DST / sid
    if dst.exists():
        return "skip"
    if not zipfile.is_zipfile(osz_path):
        return "badzip"
    kept_audio: set[str] = set()
    tmp = DST / f"_{sid}"; tmp.mkdir(exist_ok=True)
    try:
        with zipfile.ZipFile(osz_path) as z:
            names = z.namelist()
            for name in names:
                if not name.endswith(".osu"):
                    continue
                text = z.read(name).decode("utf-8-sig", errors="replace")
                h = osu_header(text)
                if h.get("Mode") == "3" and h.get("CircleSize", "").startswith("4"):
                    (tmp / pathlib.Path(name).name).write_text(text, encoding="utf-8")
                    if h.get("AudioFilename"):
                        kept_audio.add(h["AudioFilename"])
            if not any(tmp.iterdir()):
                shutil.rmtree(tmp); return "no4k"
            for name in names:             # 필요한 음원만 추출
                if pathlib.Path(name).name in kept_audio:
                    with z.open(name) as fsrc, (tmp / pathlib.Path(name).name).open("wb") as fdst:
                        shutil.copyfileobj(fsrc, fdst)
        tmp.rename(dst)
        osz_path.unlink()
        return "ok"
    except Exception as e:
        shutil.rmtree(tmp, ignore_errors=True)
        return f"err:{type(e).__name__}"

if __name__ == "__main__":
    stats: dict[str, int] = {}
    for p in sorted(SRC.glob("*.osz")):
        s = process(p)
        stats[s] = stats.get(s, 0) + 1
        print(stats, end="\r")
    print("\n", stats)