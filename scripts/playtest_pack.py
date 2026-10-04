"""Blind playtest pack: human and AI charts of the same songs, with the names hidden.

    python scripts/playtest_pack.py --ckpt outputs/full-v1/best.pt --songs 8
    python scripts/playtest_pack.py --ckpt ... --settings random:128 confidence:128 random:128:independent
    python scripts/playtest_pack.py --ckpt ... --settings random:128 random:128:continue:ref2@0.5
    python scripts/playtest_pack.py --ckpt ... --settings random:128:continue:ref2 random:128:continue:fwd+ref2
    python scripts/playtest_pack.py --keys <key> <key> ...          # these charts' songs
    python scripts/playtest_pack.py --pairs --songs 16 --settings \
        random:128:continue:fwd+ref2 random:128:continue:fwd+ref2+cp4
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
        EMPTY bias of X, jbX for a jack bias of X, cpX for bar copies with a bias of X
        nats (sampler.copy_bars), style for the human chart's genre and mapper (a model
        trained with --style; --style-csv), sgW for guidance towards it; e.g.
        random:128:continue:fwd+ref2@0.5. At the human chart's SR
all under one creator name and the human chart's OD and HP.

--pairs: two charts per song instead, [A] and [B]: the human chart and one AI chart,
the --settings taking turns over the songs (in SR order, so each setting gets songs
across the whole range). With one human and two AI charts, players found the human
one by elimination when the two AI charts looked alike (2026-10-03); a pair asks
only "which one is human" (forced choice) and "which one was more fun". Use more
songs (16 or so): each song gives one answer per player.

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
from src.data import chartstats
from src.data.cache import read_manifest
from src.data.chart_parser import _kv, _split_sections
from src.data.chart_writer import write_osu
from src.data.mel import open_mel
from src.data.style import encode, read_style
from src.data.tokenizer import L, decode, make_metas
from src.evaluation.holds import hold_stats
from src.models.sampler import ORDERS, steps_name

CREATOR = "playtest"
RATING_FIELDS = ["song", "title", "label", "rank", "human?", "comment"]
PAIR_FIELDS = ["song", "title", "human", "better", "comment"]

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


README_PAIRS = """블라인드 플레이 테스트, 짝 비교 (rhythm-chart-diffusion)

곡마다 채보가 2개 있습니다 (난이도 이름 A, B). 하나는 사람이 만든 채보이고 하나는 AI
채보입니다. 곡마다 AI 설정이 다를 수 있으니, 곡 안의 두 채보끼리만 비교해 주세요.

1. .osz 파일을 osu!(lazer) 창에 끌어다 놓거나 더블클릭해서 가져옵니다.
   osu! stable: Songs 폴더에 넣고 곡 선택 화면에서 F5.
2. 곡마다 두 채보를 모두 한 번 이상 플레이합니다 (모드 · 배속 변경 없이).
3. ratings.csv 를 채웁니다 (곡마다 한 줄).
   human    사람이 만든 채보 같은 쪽: A 또는 B (헷갈려도 꼭 하나를 고르세요)
   better   더 재밌었던 쪽: A 또는 B, 비슷하면 =
   comment  왜 그렇게 골랐는지 자유롭게
4. 파일 이름 끝에 본인 이름을 붙여 돌려주세요 (예: ratings_홍길동.csv).

사람 채보도 AI 와 같은 격자(1/12 비트)로 옮기고 히트사운드 · 배속 변화 · 배경을
지웠습니다. 겉모양으로는 구별되지 않게 하려는 것이니, 채보 내용으로만 판단해 주세요.
"""


def parse_setting(text: str) -> dict:
    """order:steps[:mode[:extras]] -> generate_song arguments; noisyT means order noisy at
    temperature T. extras, joined by +: refN or refN@T, N sweeps of lane refinement at lane
    temperature T (0.5); fwd or fwd@T, the left-to-right lane pass (sampler.forward_lanes)
    at T (default: refinement's); spread; ebX, EMPTY bias X; jbX, jack bias X; cpX, bar
    copies with bias X (sampler.copy_bars); style, the human chart's genre and mapper;
    sgW, classifier-free guidance W towards it; hr, long notes decided again
    (sampler.refine_holds); lnX, that share of the onsets as long notes (ln alone: the
    human chart's share); lbqX, loudness bias X in quiet bars only (sampler.loudness_bias),
    lbX on both sides; st, the human chart's long-note share, jack and trill rate as chart
    stats (a model trained with --chart-stats), stsample, those of a random train chart
    of about the same SR (--stats-csv)."""
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
    refine, lane_temp, lanes, spread, empty_bias, jack_bias = 0, 0.5, "sampled", False, 0.0, 0.0
    fwd_temp, copy_bias, style, guidance, holds, loud_bias = None, None, False, 0.0, False, 0.0
    hold_share: float | str | None = None
    loud_side, stats = "both", None
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
        elif head.startswith("jb") and not t:
            jack_bias = float(head[2:])
        elif head.startswith("cp") and not t:
            copy_bias = float(head[2:])
        elif head == "style" and not t:
            style = True
        elif head == "hr" and not t:
            holds = True
        elif head.startswith("ln") and not t:
            hold_share = float(head[2:]) if head[2:] else "human"
        elif head.startswith("lbq") and not t:
            loud_bias, loud_side = float(head[3:]), "quiet"
        elif head.startswith("lb") and not t:
            loud_bias, loud_side = float(head[2:]), "both"
        elif head in ("st", "stsample") and not t:
            stats = "human" if head == "st" else "sample"
        elif head.startswith("sg") and not t:
            guidance = float(head[2:])
        else:
            raise ValueError(f"bad extra {item!r} in {text!r}: refN[@T], fwd[@T], spread, ebX, "
                             "jbX, cpX, style, sgW, hr, ln[X], lbX, lbqX, st, stsample")
    if guidance and not style:
        raise ValueError(f"sgW in {text!r} guides towards a style: add style")
    if lanes == "forward" and not refine and fwd_temp is not None:
        lane_temp, fwd_temp = fwd_temp, None              # one lane pass: one temperature
    if fwd_temp == lane_temp:
        fwd_temp = None
    passes = ([f"fwd@{fwd_temp:g}" if fwd_temp is not None else "fwd"] if lanes == "forward"
              else []) + ([f"ref{refine}"] if refine else [])
    if passes and not passes[-1].startswith("fwd@"):    # the lane temperature on the last pass
        passes[-1] += f"@{lane_temp:g}"
    extras = (["spread"] if spread else []) + passes + ([f"eb{empty_bias:g}"] if empty_bias else []) \
        + ([f"jb{jack_bias:g}"] if jack_bias else []) \
        + ([f"lb{'q' if loud_side == 'quiet' else ''}{loud_bias:g}"] if loud_bias else []) \
        + (["hr"] if holds else []) \
        + ([f"ln{hold_share:g}" if isinstance(hold_share, float) else "ln"]
           if hold_share is not None else []) \
        + ([f"cp{copy_bias:g}"] if copy_bias is not None else []) \
        + (["style"] if style else []) + ([f"sg{guidance:g}"] if guidance else []) \
        + ({"human": ["st"], "sample": ["stsample"]}[stats] if stats else [])
    name_ = f"ai {name} T{steps_name(steps)} {mode}" + (" " + " ".join(extras) if extras else "")
    return {"order": order, "temperature": temperature, "steps": steps, "mode": mode,
            "refine": refine, "lane_temperature": lane_temp, "lanes": lanes, "spread": spread,
            "empty_bias": empty_bias, "forward_temperature": fwd_temp, "jack_bias": jack_bias,
            "copy_bias": copy_bias, "style": style, "style_guidance": guidance,
            "holds": holds, "loud_bias": loud_bias, "hold_share": hold_share,
            "loud_side": loud_side, "stats": stats, "name": name_}


def style_args(setting: dict, vocab: dict | None, label: dict) -> dict:
    """generate_song's style arguments for a setting with "style": the human chart's labels."""
    if not setting["style"]:
        return {}
    genre, mapper = encode(vocab, label.get("genre_id"), label.get("mapper_id"))
    return {"genre": genre, "mapper": mapper, "style_guidance": setting["style_guidance"]}


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
    labels = [chr(ord("A") + i) for i in range(2 if a.pairs else len(settings) + 1)]
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
    vocab = getattr(model, "style", None)
    style_labels = read_style(a.style_csv) if a.style_csv.exists() else {}
    if any(s["style"] for s in settings) and (vocab is None or not style_labels):
        print("style settings need a model trained with --style and --style-csv",
              file=sys.stderr)
        return 2
    spec = getattr(model, "chart_stats", None)
    stat_table = None
    if any(s["stats"] for s in settings):
        if spec is None:
            print("st / stsample need a model trained with --chart-stats", file=sys.stderr)
            return 2
        if any(s["stats"] == "sample" for s in settings):
            if not a.stats_csv.exists():
                print(f"stsample: no {a.stats_csv} (scripts/build_chart_stats.py)",
                      file=sys.stderr)
                return 2
            stat_table = chartstats.read_stats(a.stats_csv)
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
        human_share = hold_stats(z["tokens"].reshape(-1, z["tokens"].shape[-1])[:n_cells])["hold_share"]
        human_share = float(human_share) if np.isfinite(human_share) else 0.0
        mel = store.chart(row["key"], tps, offset, (len(z["tokens"]) + 1) * L)
        for s in [settings[(i - 1) % len(settings)]] if a.pairs else settings:
            buckets = None
            if s["stats"] == "human":
                buckets = chartstats.encode(spec, chartstats.chart_stats(
                    z["tokens"].reshape(-1, z["tokens"].shape[-1])[:n_cells]))
            elif s["stats"] == "sample":
                buckets = chartstats.encode(spec, chartstats.sample(
                    stat_table, sr, np.random.default_rng([a.seed, i]))[1])
            tokens = generate_song(model, mel, sr, tps, offset, n_cells, steps=s["steps"],
                                   order=s["order"], mode=s["mode"], seed=a.seed + i,
                                   temperature=s["temperature"], refine=s["refine"],
                                   lane_temperature=s["lane_temperature"],
                                   hold_bias=a.hold_bias, min_hold=a.min_hold,
                                   release_gap=a.release_gap, lanes=s["lanes"],
                                   spread=s["spread"], empty_bias=s["empty_bias"],
                                   forward_temperature=s["forward_temperature"],
                                   jack_bias=s["jack_bias"], copy_bias=s["copy_bias"],
                                   holds=s["holds"], loud_bias=s["loud_bias"],
                                   loud_side=s["loud_side"], stats=buckets,
                                   hold_share=(human_share if s["hold_share"] == "human"
                                               else s["hold_share"]),
                                   **style_args(s, vocab, style_labels.get(row["key"], {})))
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
                if not a.pairs:
                    ratings.append({"song": song, "title": meta.get("Title", ""), "label": label,
                                    "rank": "", "human?": "", "comment": ""})
        if a.pairs:
            ratings.append({"song": song, "title": meta.get("Title", ""), "human": "",
                            "better": "", "comment": ""})
        print(f"{song}  s={sr:.2f}  {row['path']}"
              + (f"  vs {charts[1][0]}" if a.pairs else ""))

    with open(pack / "ratings.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=PAIR_FIELDS if a.pairs else RATING_FIELDS)
        w.writeheader()
        w.writerows(ratings)
    readme = README_PAIRS if a.pairs else README.format(n=len(labels), labels=", ".join(labels))
    (pack / "README.txt").write_text(readme, encoding="utf-8")
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


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson interval of a proportion k / n."""
    if n == 0:
        return float("nan"), float("nan")
    p, zz = k / n, z * z
    mid = (p + zz / (2 * n)) / (1 + zz / n)
    half = z * np.sqrt(p * (1 - p) / n + zz / (4 * n * n)) / (1 + zz / n)
    return mid - half, mid + half


def score_pairs(answers: list[dict], paths: list[Path]) -> int:
    """--pairs packs: per AI setting, how often players took its chart for the human one
    (50% = could not tell them apart) and how often they found it more fun (= counts half)."""
    by_song: dict[str, dict[str, str]] = {}
    for r in answers:
        by_song.setdefault(r["song"], {})[r["label"]] = r["source"]
    taken: dict[str, list[bool]] = {}
    fun: dict[str, list[float]] = {}
    players = []
    for path in paths:
        right = total = 0
        for r in read_csv(path):
            labels = by_song.get(r.get("song", ""), {})
            ai = [lb for lb, src in labels.items() if src != "human"]
            if len(labels) != 2 or len(ai) != 1:
                continue
            source = labels[ai[0]]
            guess = r.get("human", "").strip().upper()
            if guess in labels:
                taken.setdefault(source, []).append(guess == ai[0])
                total += 1
                right += guess != ai[0]
            pick = r.get("better", "").strip().upper()
            if pick in labels or pick == "=":
                fun.setdefault(source, []).append(0.5 if pick == "=" else float(pick == ai[0]))
        players.append((path.stem, right, total))
    print(f"{'AI chart':40s} {'pairs':>5s} {'taken for human [95% CI]':>26s} {'more fun':>8s}")
    for source in sorted(set(taken) | set(fun)):
        t, f = taken.get(source, []), fun.get(source, [])
        lo, hi = wilson(sum(t), len(t))
        share = f"{np.mean(t):.0%} [{lo:.0%}, {hi:.0%}]" if t else "-"
        print(f"{source:40s} {len(t):5d} {share:>26s} {(f'{np.mean(f):.0%}' if f else '-'):>8s}")
    print("taken for human 50% = players could not tell it from the human chart")
    for who, right, total in players:
        print(f"{who}: picked the human chart in {right}/{total} pairs")
    return 0


def score(paths: list[Path]) -> int:
    """answers.csv first, then one filled ratings.csv per player."""
    answers = read_csv(paths[0])
    if paths[1:] and "human" in (read_csv(paths[1])[:1] or [{}])[0]:
        return score_pairs(answers, paths[1:])
    truth = {(r["song"], r["label"]): r["source"] for r in answers}
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
                    help="AI charts per song (--pairs: one per song, in turn): "
                         "order:steps[:mode[:extras]], extras refN[@T], fwd[@T], spread, ebX, "
                         "jbX, cpX joined by +")
    ap.add_argument("--pairs", action="store_true",
                    help="two charts per song: the human one and one AI chart, the --settings "
                         "taking turns over the songs")
    ap.add_argument("--split", default="val")
    ap.add_argument("--stats-csv", type=Path, default=Path("data/chart_stats.csv"),
                    help="stsample: the charts to draw chart stats from (build_chart_stats.py)")
    ap.add_argument("--style-csv", type=Path, default=Path("data/style.csv"),
                    help="genre and mapper per chart, for the style extra")
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
