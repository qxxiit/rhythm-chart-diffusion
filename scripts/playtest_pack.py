"""Blind playtest pack: human and AI charts of the same songs, with the names hidden.

    python scripts/playtest_pack.py --ckpt outputs/full-v1/best.pt --songs 8
    python scripts/playtest_pack.py --ckpt ... --settings random:128 confidence:128 random:128:independent
    python scripts/playtest_pack.py --ckpt ... --settings random:128 random:128:continue:ref2@0.5
    python scripts/playtest_pack.py --ckpt ... --settings random:128:continue:ref2 random:128:continue:fwd+ref2
    python scripts/playtest_pack.py --keys <key> <key> ...          # these charts' songs
    python scripts/playtest_pack.py --score outputs/full-v1/playtest/answers.csv ratings_*.csv

Songs come from --split (val by default: not seen in training), one chart per
song as evaluate.py --per-song, within --sr-range and at most --max-seconds
long, spread evenly over SR. For each song one .osz holds, as difficulties
[A], [B], ... in a random order per song:
    the human chart, put through the tokenizer (decode of its cached tokens): on
        the model's 1/12-beat grid, without hitsounds, SV, storyboard or
        background, which would give it away
    one AI chart per --settings entry, order:steps[:mode[:extras]] (noisy0.3:128
        for the noisy order at temperature 0.3; steps 0 = one cell per pass). extras
        joined by +: refN[@T] for N sweeps of lane refinement at lane temperature T
        (0.5), fwd[@T] for the left-to-right lane pass before it, spread, ebX for an
        EMPTY bias of X; e.g. random:128:continue:fwd+ref2@0.5. At the human chart's SR
all under one creator name and the human chart's OD and HP.

Writes to --out (default <ckpt dir>/playtest/):
    pack/             what the players get: the .osz files, ratings.csv, README.txt
    playtest_pack.zip pack/ as one file to send
    answers.csv       which label is which chart (keep it away from the players)
--score reads answers.csv and the filled-in ratings (one file per player) and
prints, per chart source, the mean rank and how often it was taken for human.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from scripts.preprocess_data import find_audio
from src.data.cache import read_manifest
from src.data.chart_parser import _kv, _split_sections
from src.data.chart_writer import write_osu
from src.data.mel import open_mel
from src.data.tokenizer import L, decode, make_metas
from src.models.sampler import ORDERS, steps_name

CREATOR = "playtest"
RATING_FIELDS = ["song", "title", "label", "rank", "human?", "comment"]

README = """블라인드 플레이 테스트 (rhythm-chart-diffusion)

곡마다 채보가 {n}개 있습니다. 하나는 사람이 만든 채보이고 나머지는 AI 채보입니다.
어느 것이 어느 것인지는 가려 두었습니다 (난이도 이름 {labels}).

1. .osz 파일을 osu!(lazer) 창에 끌어다 놓거나 더블클릭해서 가져옵니다.
   osu! stable: Songs 폴더에 넣고 곡 선택 화면에서 F5.
2. 곡마다 모든 채보를 한 번 이상 플레이합니다 (모드 · 배속 변경 없이).
3. ratings.csv 를 채웁니다.
   rank     곡 안에서의 순위 (1 = 가장 좋은 채보)
   human?   사람이 만든 채보 같으면 y, AI 같으면 n, 모르겠으면 ?
   comment  왜 좋았는지, 왜 별로였는지 자유롭게
4. 파일 이름 끝에 본인 이름을 붙여 돌려주세요 (예: ratings_홍길동.csv).

사람 채보도 AI 와 같은 격자(1/12 비트)로 옮기고 히트사운드 · 배속 변화 · 배경을
지웠습니다. 겉모양으로는 구별되지 않게 하려는 것이니, 채보 내용으로만 판단해 주세요.
"""


def parse_setting(text: str) -> dict:
    """order:steps[:mode[:extras]] -> generate_song arguments; noisyT means order noisy at
    temperature T. extras, joined by +: refN or refN@T, N sweeps of lane refinement at lane
    temperature T (0.5); fwd or fwd@T, the left-to-right lane pass (sampler.forward_lanes)
    at T (default: refinement's); spread; ebX, EMPTY bias X."""
    parts = text.split(":")
    if not 1 <= len(parts) <= 4:
        raise ValueError(f"bad setting {text!r}: order:steps[:mode[:extras]]")
    name = parts[0]
    order, temperature = name, 1.0
    if name.startswith("noisy") and name != "noisy":
        order, temperature = "noisy", float(name[5:])
    if order not in ORDERS:
        raise ValueError(f"unknown order in {text!r}")
    steps = int(parts[1]) if len(parts) > 1 else 128
    mode = parts[2] if len(parts) > 2 else "continue"
    if mode not in ("continue", "independent"):
        raise ValueError(f"unknown mode in {text!r}")
    refine, lane_temp, lanes, spread, empty_bias = 0, 0.5, "sampled", False, 0.0
    fwd_temp = None
    for item in parts[3].split("+") if len(parts) > 3 and parts[3] else []:
        head, _, t = item.partition("@")
        if head == "fwd":
            lanes = "forward"
            fwd_temp = float(t) if t else fwd_temp
        elif head.startswith("ref") and head[3:].isdigit():
            refine = int(head[3:])
            lane_temp = float(t) if t else lane_temp
        elif head == "spread" and not t:
            spread = True
        elif head.startswith("eb") and not t:
            empty_bias = float(head[2:])
        else:
            raise ValueError(f"bad extra {item!r} in {text!r}: refN[@T], fwd[@T], spread, ebX")
    if lanes == "forward" and not refine and fwd_temp is not None:
        lane_temp, fwd_temp = fwd_temp, None              # one lane pass: one temperature
    if fwd_temp == lane_temp:
        fwd_temp = None
    passes = ([f"fwd@{fwd_temp:g}" if fwd_temp is not None else "fwd"] if lanes == "forward"
              else []) + ([f"ref{refine}"] if refine else [])
    if passes and not passes[-1].startswith("fwd@"):    # the lane temperature on the last pass
        passes[-1] += f"@{lane_temp:g}"
    extras = (["spread"] if spread else []) + passes + ([f"eb{empty_bias:g}"] if empty_bias else [])
    name_ = f"ai {name} T{steps_name(steps)} {mode}" + (" " + " ".join(extras) if extras else "")
    return {"order": order, "temperature": temperature, "steps": steps, "mode": mode,
            "refine": refine, "lane_temperature": lane_temp, "lanes": lanes, "spread": spread,
            "empty_bias": empty_bias, "forward_temperature": fwd_temp, "name": name_}


def pick_songs(rows: list[dict], n: int, seed: int, sr_range, max_seconds: float) -> list[dict]:
    """One chart per song (seeded), filtered, then n spread evenly over SR."""
    by_song: dict[str, list[dict]] = {}
    for r in rows:
        by_song.setdefault(r["audio_key"], []).append(r)
    rng = np.random.default_rng(seed)
    one = [charts[int(rng.integers(len(charts)))] for charts in by_song.values()]
    lo, hi = sr_range
    one = [r for r in one if lo <= float(r["sr"]) <= hi
           and (not r.get("duration_s") or float(r["duration_s"]) <= max_seconds)]
    one.sort(key=lambda r: float(r["sr"]))
    if len(one) <= n:
        return one
    idx = sorted(set(np.rint(np.linspace(0, len(one) - 1, n)).astype(int).tolist()))
    return [one[i] for i in idx]


def safe(name: str) -> str:
    return re.sub(r'[\\/:*?"<>|]', "_", name).strip() or "song"


def build(a) -> int:
    settings = [parse_setting(s) for s in a.settings]
    labels = [chr(ord("A") + i) for i in range(len(settings) + 1)]
    store = open_mel(a.cache)
    rows = [r for r in read_manifest(a.manifest) if r["sr"] and r.get("drop", "") == ""
            and (a.cache / "tokens" / f"{r['key']}.npz").exists() and store.has(r["key"])]
    if a.keys:
        wanted = set(a.keys)
        songs = [r for r in rows if r["key"] in wanted]
        if len(songs) != len(wanted):
            missing = wanted - {r["key"] for r in songs}
            print(f"not usable (no SR, tokens or mel): {sorted(missing)}", file=sys.stderr)
            return 2
    else:
        songs = pick_songs([r for r in rows if r["split"] == a.split], a.songs, a.seed,
                           a.sr_range, a.max_seconds)
    if not songs:
        print("no songs to pack", file=sys.stderr)
        return 2

    from src.models.diffusion import load_denoiser, pick_device
    from src.models.sampler import generate_song
    try:
        from src.evaluation.sr import star_rating
    except ImportError:                                   # rosu-pp-py missing: no SR column
        star_rating = None

    model = load_denoiser(a.ckpt, pick_device(a.device))
    out = a.out or a.ckpt.parent / "playtest"
    pack = out / "pack"
    pack.mkdir(parents=True, exist_ok=True)
    answers, ratings = [], []
    for i, row in enumerate(songs, 1):
        song = f"PT{i:02d}"
        src = a.root / row["path"]
        text = src.read_text(encoding="utf-8-sig", errors="replace")
        sec = _split_sections(text)
        meta, diff = _kv(sec.get("Metadata", [])), _kv(sec.get("Difficulty", []))
        od = float(diff.get("OverallDifficulty", 8) or 8)
        hp = float(diff.get("HPDrainRate", 8) or 8)
        title, artist = f"{song} {meta.get('Title', '')}".strip(), meta.get("Artist", "")
        rel = find_audio((row, a.root))[1]                 # the name on disk (case)
        if rel is None:
            print(f"{song}: no audio file for {row['path']}, skipped", file=sys.stderr)
            continue
        audio = Path(rel).name

        z = np.load(a.cache / "tokens" / f"{row['key']}.npz")
        tps = [(float(t), float(bl)) for t, bl in z["timing_points"]]
        offset, n_cells, sr = int(z["cell_offset"]), int(z["n_cells"]), float(row["sr"])
        metas = make_metas(tps, offset, len(z["tokens"]), sr)
        charts = [("human", decode(z["tokens"], metas))]
        mel = store.chart(row["key"], tps, offset, (len(z["tokens"]) + 1) * L)
        for s in settings:
            tokens = generate_song(model, mel, sr, tps, offset, n_cells, steps=s["steps"],
                                   order=s["order"], mode=s["mode"], seed=a.seed + i,
                                   temperature=s["temperature"], refine=s["refine"],
                                   lane_temperature=s["lane_temperature"],
                                   hold_bias=a.hold_bias, min_hold=a.min_hold,
                                   release_gap=a.release_gap, lanes=s["lanes"],
                                   spread=s["spread"], empty_bias=s["empty_bias"],
                                   forward_temperature=s["forward_temperature"])
            charts.append((s["name"], decode(tokens, make_metas(tps, offset, len(tokens), sr))))

        order = np.random.default_rng([a.seed, i]).permutation(len(charts))
        osz = pack / f"{song} {safe(artist)} - {safe(meta.get('Title', ''))}.osz"
        with zipfile.ZipFile(osz, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(a.root / rel, audio)
            for label, j in zip(labels, order, strict=True):
                source, chart = charts[j]
                chart.audio_filename = audio
                name = f"{safe(artist)} - {safe(title)} ({CREATOR}) [{label}].osu"
                tmp = out / ".tmp.osu"
                write_osu(tmp, chart, title=title, artist=artist, version=label,
                          creator=CREATOR, od=od, hp=hp)
                zf.write(tmp, name)
                tmp.unlink()
                answers.append({"song": song, "title": meta.get("Title", ""), "artist": artist,
                                "key": row["key"], "path": row["path"], "sr_target": sr,
                                "label": label, "source": source, "notes": len(chart.notes),
                                "sr": round(star_rating(chart), 2) if star_rating else ""})
                ratings.append({"song": song, "title": meta.get("Title", ""), "label": label,
                                "rank": "", "human?": "", "comment": ""})
        print(f"{song}  s={sr:.2f}  {row['path']}")

    with open(pack / "ratings.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=RATING_FIELDS)
        w.writeheader()
        w.writerows(ratings)
    (pack / "README.txt").write_text(README.format(n=len(labels), labels=", ".join(labels)),
                                     encoding="utf-8")
    with open(out / "answers.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(answers[0]))
        w.writeheader()
        w.writerows(answers)
    with zipfile.ZipFile(out / "playtest_pack.zip", "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(pack.iterdir()):
            zf.write(p, f"playtest/{p.name}")
    print(f"{len(songs)} songs x {len(labels)} charts -> {out / 'playtest_pack.zip'} "
          f"(send this); answers in {out / 'answers.csv'} (keep it)")
    return 0


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def score(paths: list[Path]) -> int:
    """answers.csv first, then one filled ratings.csv per player."""
    truth = {(r["song"], r["label"]): r["source"] for r in read_csv(paths[0])}
    ranks: dict[str, list[float]] = {}
    human_votes: dict[str, list[int]] = {}
    guesses = []
    for path in paths[1:]:
        right = total = 0
        for r in read_csv(path):
            source = truth.get((r["song"], r["label"]))
            if source is None:
                continue
            if r.get("rank", "").strip():
                ranks.setdefault(source, []).append(float(r["rank"]))
            vote = r.get("human?", "").strip().lower()
            if vote in ("y", "n"):
                human_votes.setdefault(source, []).append(vote == "y")
                total += 1
                right += (vote == "y") == (source == "human")
        guesses.append((path.stem, right, total))
    print(f"{'source':32s} {'mean rank':>9s} {'n':>4s} {'taken for human':>16s}")
    for source in sorted(set(truth.values())):
        rk, hv = ranks.get(source, []), human_votes.get(source, [])
        print(f"{source:32s} {np.mean(rk) if rk else float('nan'):9.2f} {len(rk):4d} "
              f"{(np.mean(hv) if hv else float('nan')):16.0%}")
    for who, right, total in guesses:
        print(f"{who}: told human from AI in {right}/{total} answered charts")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ckpt", type=Path, default=Path("outputs/full-v1/best.pt"))
    ap.add_argument("--songs", type=int, default=8)
    ap.add_argument("--keys", nargs="+", default=None, help="pack these charts' songs instead")
    ap.add_argument("--settings", nargs="+", default=["random:128", "confidence:128"],
                    help="AI charts per song: order:steps[:mode[:extras]], extras refN[@T], "
                         "fwd[@T], spread, ebX joined by +")
    ap.add_argument("--split", default="val")
    ap.add_argument("--hold-bias", type=float, default=0.0,
                    help="log-scale bias on starting long notes; -1 roughly a third as many "
                         "start, -inf none (sampler.sample_window)")
    ap.add_argument("--min-hold", type=int, default=None,
                    help="long notes shorter than this many cells (1/12 beat) become taps; "
                         "0 = keep; default by SR (sampler.HOLD_RULES)")
    ap.add_argument("--release-gap", type=int, default=None,
                    help="empty cells required between a release and the next onset in its lane; "
                         "0 = keep; default by SR (sampler.HOLD_RULES)")
    ap.add_argument("--sr-range", type=float, nargs=2, default=(1.5, 4.5))
    ap.add_argument("--max-seconds", type=float, default=150.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"))
    ap.add_argument("--root", type=Path, default=Path("data/raw"))
    ap.add_argument("--cache", type=Path, default=Path("data/cache"))
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--score", type=Path, nargs="+", default=None,
                    help="answers.csv, then the players' ratings files")
    a = ap.parse_args(argv)
    if a.score:
        if len(a.score) < 2:
            ap.error("--score answers.csv ratings.csv [ratings.csv ...]")
        return score(a.score)
    return build(a)


if __name__ == "__main__":
    raise SystemExit(main())
