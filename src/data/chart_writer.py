"""Chart -> .osu text that osu! (stable and lazer) can open as a mania difficulty.

Put the file in the beatmap folder that holds the audio and refresh song select
(F5). BeatmapID 0 marks it as a local, unsubmitted difficulty.
"""

from __future__ import annotations

from pathlib import Path

from src.data.chart_parser import Chart


def osu_text(chart: Chart, *, mode: int = 3, title: str = "", artist: str = "",
             version: str = "generated", creator: str = "rhythm-chart-diffusion",
             beatmap_id: int = 0, set_id: int = -1, od: float = 8.0, hp: float = 8.0) -> str:
    k = chart.key_count
    lines = [
        "osu file format v14", "",
        "[General]", f"AudioFilename: {chart.audio_filename}", "AudioLeadIn: 0",
        "PreviewTime: -1", "Countdown: 0", "SampleSet: Soft", "StackLeniency: 0.7",
        f"Mode: {mode}", "LetterboxInBreaks: 0", "SpecialStyle: 0", "WidescreenStoryboard: 0", "",
        "[Metadata]", f"Title:{title}", f"TitleUnicode:{title}", f"Artist:{artist}",
        f"ArtistUnicode:{artist}", f"Creator:{creator}", f"Version:{version}", "Source:", "Tags:",
        f"BeatmapID:{beatmap_id}", f"BeatmapSetID:{set_id}", "",
        "[Difficulty]", f"HPDrainRate:{hp:g}", f"CircleSize:{k}", f"OverallDifficulty:{od:g}",
        "ApproachRate:5", "SliderMultiplier:1.4", "SliderTickRate:1", "",
        "[Events]", "//Background and Video events", "//Break Periods", "",
        "[TimingPoints]",
    ]
    lines += [f"{int(t)},{float(bl)!r},4,1,0,100,1,0" for t, bl in chart.timing_points]
    lines += ["", "", "[HitObjects]"]
    for n in chart.notes:
        x = int((n.lane + 0.5) * 512 / k)                  # parser: lane = x * k // 512
        if n.end_ms is None:
            lines.append(f"{x},192,{n.time_ms},1,0,0:0:0:0:")
        else:
            lines.append(f"{x},192,{n.time_ms},128,0,{n.end_ms}:0:0:0:0:")
    return "\n".join(lines) + "\n"


def write_osu(path: Path, chart: Chart, **kwargs) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(osu_text(chart, **kwargs), encoding="utf-8")
