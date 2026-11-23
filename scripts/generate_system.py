"""The system as decided (scripts/system.sh) on a new song: an audio file and its timing, at
the target SRs, as an .osz to play.

    python scripts/generate_system.py --osz "inputs/arcade/123 Artist - Title.osz" --sr 2.5 3.5 4.5
    python scripts/generate_system.py --audio song.mp3 --timing timed.osu --sr 3.5

--osz takes the audio and the red lines of a beatmap set (any mode, any keys: its charts are
not used); --audio with --timing an .osu whose red lines to use. The two stages of
scripts/system.sh as evaluate.py runs them on a cached song: the draw, every window drawn 4
times and the one the choosing judge (sel_holds) finds most human kept, then the passes. Their
options are read with evaluate.py's own parser from the flags below, so every default is
evaluate.py's; tests/test_generate_system.py checks the flags against system.sh.

A new song has no human chart. The chart stats come from a human train chart of about the
target SR (as evaluate.py --stats sample), drawn again until its long notes are at most
--max-ln of its onsets (players skipped the long-note charts as too hard, 2026-11-23), and the
long-note share the passes aim at (--hold-share oracle in the system) is that chart's, and
by default the long notes over it are made taps (--hold-down, not in the system); the choosing
judge reads the target SR where it read the human chart's. Writes
<out>/<stem>.osz (the audio and one .osu per SR), default outputs/generated_system/<stem>/.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from scripts import evaluate
from scripts.generate import model_input, red_lines, safe, song_range
from src.data import chartstats
from src.data.audio import SAMPLE_RATE, load_osu
from src.data.chart_parser import _kv, _split_sections
from src.data.chart_writer import write_osu
from src.data.tokenizer import L, decode, grammar_violations, make_metas
from src.models.diffusion import load_denoiser, pick_device
from src.models.judge import stable_hash
from src.models.sampler import generate_song, post_song
from src.models.select import WindowJudge

# scripts/system.sh (DECISIONS through 2026-11-20): the flags of both stages, of the draw, and
# of the passes (RESTS REUSE HOLDS CHORDS JACKS MIRROR LAST DENSE). --hold-share oracle is
# HOLDS' (the human chart's share): here the sampled chart's share takes its place.
BASE = ["--lane-guidance", "2", "--lane-temp", "1"]
DRAW = ["--lanes", "forward", "--refine", "2", "--loud-bias", "0.1", "--onset-bias", "1",
        "--onset-taper", "2.7,5,0.35", "--schedule", "cosine", "--select-k", "4",
        "--hold-bias", "0.2"]
PASSES = ["--rest", "1", "--light", "0.5", "--light-alike", "--fill-holes", "0.5",
          "--rest-max-run", "8",
          "--carry-rhythm", "0", "--carry-keep", "--carry-holds", "1", "--copy-bias", "0",
          "--copy-min-lag", "8", "--copy-sections", "4", "--section-keep", "--section-back",
          "--alike", "2", "--alike-keep", "4",
          "--hold-up", "--ln-tidy", "0.5",
          "--strong-chords", "1", "--chord-cap", "2",
          "--untangle", "--untangle-move", "--untangle-grades", "--untangle-insane",
          "--mirror-bars", "--mirror-shares", "0.034,0.049,0.062",
          "--snap-releases", "2",
          "--dense-taps", "0.2", "--dense-repeats"]


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--osz", type=Path, default=None, help="a beatmap set: its audio and timing")
    ap.add_argument("--audio", type=Path, default=None)
    ap.add_argument("--timing", type=Path, default=None, help="with --audio: .osu of the red lines")
    ap.add_argument("--sr", type=float, nargs="+", default=[2.5, 3.5, 4.5])
    ap.add_argument("--ckpt", type=Path, default=Path("outputs/full-v4/best.pt"))
    ap.add_argument("--judge", type=Path, default=Path("outputs/judge/sel_holds"),
                    help="the choosing judge (judge.py --save-models)")
    ap.add_argument("--max-ln", type=float, default=0.3,
                    help="the sampled chart's long notes at most this share of its onsets")
    ap.add_argument("--hold-down", action=argparse.BooleanOptionalAction, default=True,
                    help="long notes over that share made taps (evaluate.py --hold-down; not in "
                         "the system: 11-23's first charts reached 56%% where 28%% was asked)")
    ap.add_argument("--stats-csv", type=Path, default=Path("data/chart_stats.csv"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--title", default=None)
    ap.add_argument("--artist", default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--steps", type=int, default=None,
                    help="forward passes a window in place of the system's (tests)")
    return ap.parse_args(argv)


def system_options(ckpt: Path, judge: Path, seed: int) -> tuple[argparse.Namespace, argparse.Namespace]:
    """evaluate.py's options at the system's flags: (the draw's, the passes')."""
    common = ["--ckpt", str(ckpt), *BASE, "--stats", "sample", "--seed", str(seed)]
    parser = evaluate.make_parser()
    return (parser.parse_args(common + DRAW + ["--select-judge", str(judge)]),
            parser.parse_args(common + PASSES))


AUDIO_SUFFIXES = (".mp3", ".ogg", ".wav", ".flac", ".m4a")


def member_name(info: zipfile.ZipInfo) -> str:
    """A zip member's name as its writer meant it: names without the UTF-8 flag come back from
    zipfile as cp437, and beatmap sets made on Japanese systems are often Shift-JIS (cp932;
    11-23: INARI's audio and .osu)."""
    if info.flag_bits & 0x800:
        return info.filename
    raw = info.filename.encode("cp437", errors="replace")
    for encoding in ("utf-8", "cp932"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return info.filename


def find_audio_file(folder: Path, name: str) -> Path | None:
    """The .osu's AudioFilename in its folder: as written, else ignoring case, else the
    folder's only audio file."""
    if name and (folder / name).is_file():
        return folder / name
    files = [f for f in folder.iterdir() if f.is_file()]
    same = [f for f in files if f.name.lower() == name.lower()] if name else []
    if same:
        return same[0]
    audio = [f for f in files if f.suffix.lower() in AUDIO_SUFFIXES]
    return audio[0] if len(audio) == 1 else None


def from_osz(osz: Path, into: Path) -> tuple[Path, Path, dict]:
    """(audio, an .osu with red lines, its metadata) of a beatmap set, unpacked into `into`."""
    root = into.resolve()
    with zipfile.ZipFile(osz) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            target = (into / member_name(info)).resolve()
            if root not in target.parents:              # nothing written outside `into`
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
    for osu in sorted(into.rglob("*.osu")):
        sections = _split_sections(osu.read_text(encoding="utf-8-sig", errors="replace"))
        name = _kv(sections.get("General", [])).get("AudioFilename", "").strip()
        audio = find_audio_file(osu.parent, name)
        if audio is not None:
            try:
                red_lines(osu)
            except ValueError:
                continue
            return audio, osu, _kv(sections.get("Metadata", []))
    raise ValueError(f"{osz}: no .osu with red lines and its audio file")


def sample_stats(table: dict, sr: float, max_ln: float, seed: int, i: int,
                 song: str = "") -> tuple[str, dict]:
    """A human train chart's stats near SR sr whose long notes are at most max_ln of onsets
    (seeded by the song too, so the songs do not all take the same chart)."""
    for t in range(200):
        key, wanted = chartstats.sample(table, sr,
                                        np.random.default_rng([seed, stable_hash(song), i, t]))
        if np.isfinite(wanted.get("hold_share", np.nan)) and wanted["hold_share"] <= max_ln:
            return key, wanted
    raise ValueError(f"no train chart near SR {sr} with long notes at most {max_ln}")


def main(argv: list[str] | None = None) -> int:
    a = parse_args(argv)
    if (a.osz is None) == (a.audio is None):
        print("give --osz, or --audio with --timing", file=sys.stderr)
        return 2
    if a.audio is not None and a.timing is None:
        print("--audio needs --timing (an .osu with the red lines)", file=sys.stderr)
        return 2
    d, p = system_options(a.ckpt, a.judge, a.seed)
    p.hold_down = a.hold_down
    if a.steps is not None:
        d.steps = a.steps
    tmp = None
    meta: dict = {}
    if a.osz is not None:
        tmp = Path(tempfile.mkdtemp(prefix="osz_"))
        try:
            audio, timing, meta = from_osz(a.osz, tmp)
        except ValueError as e:
            print(e, file=sys.stderr)
            return 2
        stem = safe(f"{meta['Artist']} - {meta['Title']}") if meta.get("Title") \
            and meta.get("Artist") else safe(a.osz.stem)
    else:
        audio, timing, stem = a.audio, a.timing, safe(a.audio.stem)
    title = a.title or meta.get("Title") or stem
    artist = a.artist or meta.get("Artist") or "unknown"

    tps = red_lines(timing)
    samples, _ = load_osu(audio)
    audio_ms = 1000.0 * len(samples) / SAMPLE_RATE
    cell_offset, n_cells = song_range(tps, audio_ms)
    n_rows = (-(-n_cells // L) + 1) * L                     # + overhang chunk
    mel = model_input(samples, tps, cell_offset, n_rows)
    out = a.out or Path("outputs/generated_system") / stem
    out.mkdir(parents=True, exist_ok=True)
    audio_name = "audio" + audio.suffix.lower()
    shutil.copyfile(audio, out / audio_name)
    print(f"{title} - {artist}: {audio_ms / 1000:.1f} s, {60000.0 / tps[0][1]:.2f} BPM"
          f"{f' ({len(tps)} red lines)' if len(tps) > 1 else ''}, {n_cells:,} cells")

    model = load_denoiser(a.ckpt, pick_device(a.device))
    spec = getattr(model, "chart_stats", None)
    if spec is None:
        print(f"{a.ckpt} was trained without chart stats", file=sys.stderr)
        return 2
    table = chartstats.read_stats(a.stats_csv)
    judge = WindowJudge(a.judge, next(model.parameters()).device)
    taper = tuple(float(v) for v in d.onset_taper.split(","))
    mirror_shares = dict(zip((1, 2, 4), map(float, p.mirror_shares.split(",")), strict=True))
    sample_seed = d.seed if d.sample_seed is None else d.sample_seed
    release_temp = 0.5 if d.release_temp is None else d.release_temp
    written = []
    for i, sr in enumerate(a.sr):
        try:
            stat_key, wanted = sample_stats(table, sr, a.max_ln, a.seed, i, stem)
        except ValueError as e:
            print(e, file=sys.stderr)
            return 2
        buckets = chartstats.encode(spec, wanted)
        share = float(wanted["hold_share"])
        select = judge.scorer(stem, sr, draws=d.select_draws, seed=sample_seed + i)
        # the draw: evaluate.py's generate_song call with the draw's options (no passes)
        tokens = generate_song(model, mel, sr + d.sr_offset, tps, cell_offset, n_cells,
                               steps=d.steps, order=d.order, schedule=d.schedule, mode=d.mode,
                               seed=sample_seed + i, temperature=d.temperature,
                               refine=d.refine, lane_temperature=d.lane_temp,
                               hold_bias=d.hold_bias, min_hold=d.min_hold,
                               release_gap=d.release_gap, lanes=d.lanes, spread=d.spread,
                               empty_bias=d.empty_bias, forward_temperature=d.forward_temp,
                               jack_bias=d.jack_bias, copy_bias=d.copy_bias,
                               copy_min_lag=d.copy_min_lag, holds=d.refine_holds,
                               hold_share=None, hold_up=d.hold_up,
                               release_temperature=release_temp, light=d.light,
                               light_alike=d.light_alike, recheck=d.recheck,
                               recheck_add=d.recheck_add, untangle_wide=d.untangle_wide,
                               untangle_insane=d.untangle_insane, loud_bias=d.loud_bias,
                               loud_side=d.loud_side, stats=buckets, onset_bias=d.onset_bias,
                               onset_taper=taper, rest=d.rest, holes=d.fill_holes,
                               carry=d.carry_rhythm, ln_tidy=d.ln_tidy,
                               carry_keep=d.carry_keep, carry_holds=d.carry_holds,
                               carry_rows=d.carry_rows, sections=d.copy_sections,
                               section_keep=d.section_keep, section_back=d.section_back,
                               section_lanes=d.section_lanes, chord_cap=d.chord_cap,
                               untangle=d.untangle, untangle_move=d.untangle_move,
                               untangle_grades=d.untangle_grades, alike=d.alike,
                               alike_keep=d.alike_keep, strong=d.strong_chords,
                               lane_guidance=d.lane_guidance, select=select,
                               select_k=d.select_k if select else 1)
        # the passes: evaluate.py's post_song call with the passes' options
        tokens = post_song(model, tokens, mel, sr, tps, cell_offset, n_cells,
                           rest_max_run=p.rest_max_run, answer=p.answer,
                           dense=p.dense_taps, dense_repeats=p.dense_repeats,
                           chord_share=None, hold_down_to=p.hold_down,
                           snap=p.snap_releases, mirror=p.mirror_bars,
                           mirror_shares=mirror_shares, repeat=p.repeat_bars,
                           chord_step=p.chord_share_step,
                           chord_temperature=p.chord_share_temp,
                           answer_unit=p.answer_unit, answer_lanes=p.answer_lanes,
                           holds=p.refine_holds, hold_share=share, hold_up=p.hold_up,
                           copy_bias=p.copy_bias, light=p.light,
                           recheck=p.recheck, recheck_add=p.recheck_add,
                           light_alike=p.light_alike, untangle_wide=p.untangle_wide,
                           untangle_insane=p.untangle_insane,
                           copy_min_lag=p.copy_min_lag,
                           min_hold=p.min_hold, release_gap=p.release_gap,
                           seed=sample_seed + i, stats=buckets, rest=p.rest,
                           holes=p.fill_holes, carry=p.carry_rhythm, ln_tidy=p.ln_tidy,
                           carry_keep=p.carry_keep, carry_holds=p.carry_holds,
                           carry_rows=p.carry_rows, sections=p.copy_sections,
                           section_keep=p.section_keep, section_back=p.section_back,
                           section_lanes=p.section_lanes, chord_cap=p.chord_cap,
                           untangle=p.untangle, untangle_move=p.untangle_move,
                           untangle_grades=p.untangle_grades, alike=p.alike,
                           alike_keep=p.alike_keep, strong=p.strong_chords,
                           lane_guidance=p.lane_guidance, lane_temperature=p.lane_temp,
                           release_temperature=release_temp)
        bad = len(grammar_violations(tokens))
        chart = decode(tokens, make_metas(tps, cell_offset, len(tokens), sr))
        chart.audio_filename = audio_name
        version = f"AI {sr:g}"
        path = out / f"{stem} [{version}].osu"
        write_osu(path, chart, title=title, artist=artist, version=version,
                  creator="rhythm-chart-diffusion", od=8.0, hp=8.0)
        written.append(path)
        lns = sum(n.end_ms is not None for n in chart.notes)
        print(f"  s={sr:g}: {len(chart.notes):,} notes, long notes "
              f"{lns / max(len(chart.notes), 1):.0%} (asked {share:.0%}, train chart "
              f"{stat_key}), grammar violations {bad} -> {path.name}")
    osz = out / f"{stem}.osz"
    with zipfile.ZipFile(osz, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(out / audio_name, audio_name)
        for path in written:
            z.write(path, path.name)
    if tmp is not None:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"{osz}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
