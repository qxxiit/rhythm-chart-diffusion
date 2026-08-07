"""osu!mania .osu 파일 → 구조화된 채보 데이터."""
from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class Note:
    time_ms: int
    lane: int                  # 0..K-1
    end_ms: int | None = None  # 롱노트면 endTime

@dataclass
class Chart:
    key_count: int
    audio_filename: str
    timing_points: list[tuple[int, float]] = field(default_factory=list)  # (time_ms, ms_per_beat)
    notes: list[Note] = field(default_factory=list)

    @property
    def bpm(self) -> float:    # 대표 BPM (첫 uninherited 포인트 기준)
        return 60000.0 / self.timing_points[0][1] if self.timing_points else 0.0

def _split_sections(text: str) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {}
    current = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("//"):
            continue
        if line.startswith("[") and line.endswith("]"):
            current = line[1:-1]
            sections[current] = []
        elif current:
            sections[current].append(line)
    return sections

def _kv(lines: list[str]) -> dict[str, str]:
    out = {}
    for l in lines:
        if ":" in l:
            k, v = l.split(":", 1)
            out[k.strip()] = v.strip()
    return out

def parse_osu(path: str | Path) -> Chart:
    text = Path(path).read_text(encoding="utf-8-sig", errors="replace")
    sec = _split_sections(text)

    general = _kv(sec.get("General", []))
    diff = _kv(sec.get("Difficulty", []))
    if general.get("Mode") != "3":
        raise ValueError(f"not mania (Mode={general.get('Mode')}): {path}")
    key_count = int(float(diff["CircleSize"]))

    chart = Chart(key_count=key_count,
                  audio_filename=general.get("AudioFilename", ""))

    for l in sec.get("TimingPoints", []):
        parts = l.split(",")
        beat_len = float(parts[1])
        if beat_len > 0:                       # 양수만 실제 BPM (음수는 SV)
            chart.timing_points.append((int(float(parts[0])), beat_len))

    for l in sec.get("HitObjects", []):
        parts = l.split(",")
        x, t, typ = int(parts[0]), int(parts[2]), int(parts[3])
        lane = min(x * key_count // 512, key_count - 1)
        end_ms = None
        if typ & 128:                          # mania 롱노트
            end_ms = int(parts[5].split(":")[0])
        chart.notes.append(Note(time_ms=t, lane=lane, end_ms=end_ms))

    chart.notes.sort(key=lambda n: (n.time_ms, n.lane))
    return chart