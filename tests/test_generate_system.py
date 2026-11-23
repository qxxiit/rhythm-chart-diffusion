import re
from pathlib import Path

from scripts import generate_system

SYSTEM = Path(__file__).resolve().parents[1] / "scripts" / "system.sh"


def shell_vars(text: str) -> dict[str, list[str]]:
    """NAME="..." assignments of a shell script (backslash continuations joined), as words;
    NAME=${NAME:-...} gives the default."""
    text = text.replace("\\\n", " ")
    out = {}
    for m in re.finditer(r'^([A-Z]+)="([^"]*)"', text, re.M):
        out[m.group(1)] = m.group(2).split()
    for m in re.finditer(r"^([A-Z]+)=\$\{\1:-(.*)\}$", text, re.M):
        out.setdefault(m.group(1), m.group(2).split())
    return out


def test_the_flags_are_the_system_scripts():
    v = shell_vars(SYSTEM.read_text())
    draw = [w for w in v["DRAW"] if w not in ("--sample-seed", "$SEED", "--select-judge", "$JUDGE")]
    assert draw == generate_system.DRAW
    passes = [w for name in ("RESTS", "REUSE", "HOLDS", "CHORDS", "JACKS", "MIRROR", "LAST", "DENSE")
              for w in v[name]]
    i = passes.index("--hold-share")
    assert passes[i + 1] == "oracle"                    # the sampled chart's share in its place
    assert passes[:i] + passes[i + 2:] == generate_system.PASSES
    ev = v["EV"]
    base = ev[ev.index("--lane-guidance"):ev.index("--stats")]
    assert base == generate_system.BASE and ev[ev.index("--stats") + 1] == "oracle"


def test_the_options_are_evaluates():
    d, p = generate_system.system_options(Path("x.pt"), Path("judge"), 0)
    assert d.select_k == 4 and d.schedule == "cosine" and d.hold_bias == 0.2 and d.lane_temp == 1
    assert d.rest is None and d.mirror_bars is False                 # no passes in the draw
    assert p.rest == 1 and p.mirror_bars and p.snap_releases == 2 and p.dense_taps == 0.2
    assert p.hold_share is None and p.stats == "sample"


OSU = """osu file format v14

[General]
AudioFilename: Audio.MP3

[Metadata]
Title:T
Artist:A

[TimingPoints]
0,500,4,2,0,100,1,0
"""


def test_zip_names_in_shift_jis_come_back():
    import zipfile
    info = zipfile.ZipInfo("稲.mp3".encode("cp932").decode("cp437"))   # as zipfile reads it
    info.flag_bits = 0
    assert generate_system.member_name(info) == "稲.mp3"
    flagged = zipfile.ZipInfo("稲.mp3")
    flagged.flag_bits = 0x800
    assert generate_system.member_name(flagged) == "稲.mp3"


def test_from_osz_finds_the_audio_whatever_its_case(tmp_path):
    import zipfile
    osz = tmp_path / "set.osz"
    with zipfile.ZipFile(osz, "w") as zf:
        zf.writestr("song [x].osu", OSU)
        zf.writestr("audio.mp3", b"not really audio")
        zf.writestr("../outside.txt", b"stays out")
    into = tmp_path / "into"
    into.mkdir()
    audio, osu, meta = generate_system.from_osz(osz, into)
    assert audio.name == "audio.mp3" and osu.name == "song [x].osu" and meta["Title"] == "T"
    assert not (tmp_path / "outside.txt").exists()
    assert generate_system.find_audio_file(into, "") == into / "audio.mp3"   # the only one


def test_long_notes_are_capped_by_default():
    a = generate_system.parse_args(["--osz", "x.osz"])
    assert a.hold_down and not generate_system.parse_args(["--osz", "x.osz", "--no-hold-down"]).hold_down
