"""Full-dataset check of the chart tokenizer (design doc §2.7).

For every 4K chart under --root it checks three invariants, which must hold
on every chart:
    grammar     grammar_violations(tokens) is empty
    balance     onsets in the tokens + dropped notes == notes in the chart, per lane
    round trip  encode(decode(tokens)) == tokens, bit for bit

and measures what the grid costs:
    onset error of the re-originated grid   |time_from_cell(cell_index(t)) - t|
    the two older measures, so the numbers in the docs can be reproduced
        local snap     §2.3 coverage table    97.21 / 97.36 / 98.00 %
        global cells   DECISIONS              90.81 % at 5 ms
    notes whose cell belongs to the next timing section (the boundary-cell rule)
    notes before the first red line          DECISIONS 16,782 notes / 3,928 charts
    one-cell gaps/overlaps of an independently rounded grid     DECISIONS 3,420
    dropped / shortened / demoted notes, token class shares, cells per chart

Usage (from the repo root):
    python scripts/validate_tokenizer.py                # data/raw, all CPU cores
    python scripts/validate_tokenizer.py --limit 300    # quick look
    python scripts/validate_tokenizer.py --selftest     # synthetic charts, no data needed

Writes docs/_stats/tokenizer_validation.csv (summary, commit it; a --limit run
writes data/tokenizer_validation_partial.csv instead) and
data/tokenizer_validation_charts.csv (one row per chart, gitignored).
Exit code 1 if any invariant fails or any chart raises.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import tempfile
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import chart_writer
from src.data.beat_grid import from_chart, grid_index, snap_error_ms
from src.data.chart_parser import Chart, Note, parse_osu
from src.data.tokenizer import (
    ChartTooLong,
    D,
    K,
    decode,
    encode,
    grammar_violations,
)

TOLS_MS = (3, 5, 10)
TOKEN_NAMES = ("empty", "tap", "hold_start", "hold_body", "hold_end", "mask", "pad")
DOC = {                                   # values currently written in the docs
    "cov_local_3": 97.21, "cov_local_5": 97.36, "cov_local_10": 98.00,
    "cov_global_5": 90.81,
    "neg_notes": 16_782, "neg_charts": 3_928,
    "indep_gap_overlap": 3_420,
}


# ---------------------------------------------------------------------------
# per chart
# ---------------------------------------------------------------------------

def load(path: Path) -> tuple[str, Chart | None]:
    """Same filters as scripts/analyze_dataset.py, so the chart set matches the docs."""
    try:
        chart = parse_osu(path)
    except ValueError:
        return "not_mania", None
    except Exception:
        return "parse_error", None
    if chart.key_count != K:
        return "wrong_keys", None
    if not chart.timing_points:
        return "no_timing", None
    if not chart.notes:
        return "no_notes", None
    return "ok", chart


def grid_measures(chart: Chart) -> dict:
    """What the grid does to note onsets. Does not touch the tokenizer."""
    grid = from_chart(chart)
    t = np.array([n.time_ms for n in chart.notes], dtype=np.float64)
    cell = grid.cell_index(t, D)
    err = {
        "reorig": np.abs(grid.time_from_cell(cell, D) - t),
        "local": snap_error_ms(grid.local_beat(t), grid.beat_length_at(t), D),
        "global": np.abs(grid.time_from_global_beat(grid_index(grid.global_beat(t), D) / D) - t),
    }
    row = {}
    for name, e in err.items():
        for tol in TOLS_MS:
            row[f"cov_{name}_{tol}"] = int(np.sum(e <= tol))

    # the boundary-cell rule: a cell equal to the next section's start belongs to that section
    owner = np.clip(np.searchsorted(grid.cell_starts(D), cell, side="right") - 1, 0, None)
    boundary = owner != grid.section_index_array(t)
    row["n_boundary"] = int(boundary.sum())
    row["n_boundary_over_5ms"] = int(np.sum(boundary & (err["reorig"] > 5)))
    row["max_boundary_err_ms"] = float(err["reorig"][boundary].max()) if boundary.any() else 0.0

    row["n_neg_cell"] = int(np.sum(cell < 0))
    row["n_neg_global"] = int(np.sum(grid.global_beat(t) < 0))

    # what independently rounded section starts, round(d * cum_i), would have done
    gap = overlap = 0
    if grid.n_sections > 1:
        indep = np.round(grid.cum * D).astype(np.int64)
        rounded_span = np.round(np.diff(grid.times) / grid.bls[:-1] * D).astype(np.int64)
        miss = np.diff(indep) - rounded_span
        gap, overlap = int(np.sum(miss > 0)), int(np.sum(miss < 0))
    row["n_indep_gap"], row["n_indep_overlap"] = gap, overlap

    row["n_sections"] = grid.n_sections
    row["n_offbeat_red_lines"] = grid.offbeat_red_lines()
    row["beat_len_min_ms"] = float(grid.bls.min())
    row["beat_len_max_ms"] = float(grid.bls.max())
    return row


def tokenizer_checks(chart: Chart) -> dict:
    tokens, metas, st = encode(chart)
    back = decode(tokens, metas)
    tokens2, metas2, _ = encode(back)
    row = {
        "n_onsets": st.n_onsets,
        "n_dropped": st.n_dropped,
        "n_shortened": st.n_shortened,
        "n_shortened_to_tap": st.n_shortened_to_tap,
        "n_demoted": st.n_demoted,
        "n_cells": st.n_cells,
        "n_chunks": len(metas),
        "cell_offset": metas[0].cell_offset if metas else 0,
        "n_violations": len(grammar_violations(tokens)),
        "balanced": bool(st.balanced),
        "round_trip": bool(np.array_equal(tokens, tokens2)
                           and [m.start_cell for m in metas] == [m.start_cell for m in metas2]),
    }
    counts = np.bincount(tokens.reshape(-1).astype(np.int64), minlength=len(TOKEN_NAMES))
    for name, c in zip(TOKEN_NAMES, counts, strict=True):
        row[f"tok_{name}"] = int(c)
    return row


def check_file(args: tuple[str, str]) -> dict:
    path, root = args
    row: dict = {"path": os.path.relpath(path, root)}
    status, chart = load(Path(path))
    row["status"] = status
    if chart is None:
        return row
    row["n_notes"] = len(chart.notes)
    with np.errstate(all="ignore"):          # extreme-BPM gimmicks overflow harmlessly here
        try:
            row.update(grid_measures(chart))
        except Exception as e:
            row["status"], row["error"] = "error", f"grid: {type(e).__name__}: {e}"
            return row
        try:
            row.update(tokenizer_checks(chart))
        except ChartTooLong as e:
            row["status"], row["error"] = "too_long", str(e)
        except Exception as e:
            row["status"], row["error"] = "error", f"tokenizer: {type(e).__name__}: {e}"
    return row


# ---------------------------------------------------------------------------
# whole dataset
# ---------------------------------------------------------------------------

def run(root: Path, workers: int, limit: int) -> list[dict]:
    paths = sorted(root.rglob("*.osu"))
    if limit:
        paths = paths[:limit]
    print(f"{len(paths):,} .osu files under {root}, {workers} worker(s)")
    jobs = [(str(p), str(root)) for p in paths]
    rows, t0 = [], time.time()
    if workers <= 1:
        results = map(check_file, jobs)
        pool = None
    else:
        pool = Pool(workers)
        results = pool.imap_unordered(check_file, jobs, chunksize=16)
    try:
        for i, row in enumerate(results, 1):
            rows.append(row)
            if i % 1000 == 0:
                print(f"  {i:,}/{len(jobs):,}  {time.time() - t0:.0f}s")
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    rows.sort(key=lambda r: r["path"])
    print(f"done in {time.time() - t0:.0f}s")
    return rows


def pct(a: float, b: float) -> float:
    return 100.0 * a / b if b else float("nan")


def summarize(rows: list[dict]) -> tuple[list[tuple[str, object]], bool]:
    """(metric, value) pairs for the CSV, and whether every invariant held."""
    status = {}
    for r in rows:
        status[r["status"]] = status.get(r["status"], 0) + 1
    measured = [r for r in rows if "cov_reorig_5" in r]      # grid measures succeeded
    tok = [r for r in rows if "round_trip" in r]             # tokenizer ran

    def total(rs: list[dict], key: str) -> int:
        return sum(r.get(key, 0) for r in rs)

    m: list[tuple[str, object]] = []
    m += [("files", len(rows))] + [(f"status_{k}", v) for k, v in sorted(status.items())]

    notes = total(measured, "n_notes")
    m.append(("notes_measured", notes))
    for name in ("reorig", "local", "global"):
        for tol in TOLS_MS:
            m.append((f"cov_{name}_{tol}ms_pct", round(pct(total(measured, f"cov_{name}_{tol}"), notes), 3)))
    m += [
        ("boundary_notes", total(measured, "n_boundary")),
        ("boundary_notes_over_5ms", total(measured, "n_boundary_over_5ms")),
        ("boundary_max_err_ms", round(max((r["max_boundary_err_ms"] for r in measured), default=0.0), 2)),
        ("neg_cell_notes", total(measured, "n_neg_cell")),
        ("neg_cell_charts", sum(r["n_neg_cell"] > 0 for r in measured)),
        ("neg_global_notes", total(measured, "n_neg_global")),
        ("neg_global_charts", sum(r["n_neg_global"] > 0 for r in measured)),
        ("indep_gaps", total(measured, "n_indep_gap")),
        ("indep_overlaps", total(measured, "n_indep_overlap")),
        ("charts_with_offbeat_red_line", sum(r["n_offbeat_red_lines"] > 0 for r in measured)),
    ]

    tok_notes = total(tok, "n_notes")
    m += [
        ("charts_tokenized", len(tok)),
        ("notes_tokenized", tok_notes),
        ("dropped_notes", total(tok, "n_dropped")),
        ("dropped_charts", sum(r["n_dropped"] > 0 for r in tok)),
        ("shortened_holds", total(tok, "n_shortened")),
        ("shortened_charts", sum(r["n_shortened"] > 0 for r in tok)),
        ("shortened_to_tap", total(tok, "n_shortened_to_tap")),
        ("demoted_holds", total(tok, "n_demoted")),
        ("dropped_pct_of_notes", round(pct(total(tok, "n_dropped"), tok_notes), 5)),
        ("shortened_pct_of_notes", round(pct(total(tok, "n_shortened"), tok_notes), 4)),
    ]
    cells = np.array([r["n_cells"] for r in tok]) if tok else np.zeros(1)
    m += [
        ("cells_median", float(np.median(cells))),
        ("cells_p95", float(np.percentile(cells, 95))),
        ("cells_max", int(cells.max())),
        ("chunks_total", total(tok, "n_chunks")),
    ]
    tok_counts = {name: total(tok, f"tok_{name}") for name in TOKEN_NAMES}
    non_pad = sum(v for k, v in tok_counts.items() if k != "pad")
    for name in TOKEN_NAMES[:5]:
        m.append((f"share_{name}_pct", round(pct(tok_counts[name], non_pad), 3)))

    fail_grammar = sum(r["n_violations"] > 0 for r in tok)
    fail_balance = sum(not r["balanced"] for r in tok)
    fail_round = sum(not r["round_trip"] for r in tok)
    errors = status.get("error", 0)
    m += [
        ("FAIL_grammar_charts", fail_grammar),
        ("FAIL_balance_charts", fail_balance),
        ("FAIL_round_trip_charts", fail_round),
        ("FAIL_error_charts", errors),
    ]
    ok = fail_grammar == fail_balance == fail_round == errors == 0
    return m, ok


def report(rows: list[dict], metrics: list[tuple[str, object]], ok: bool) -> None:
    v = dict(metrics)

    def line(label: str, value, doc=None) -> None:
        tail = f"   (문서값 {doc})" if doc is not None else ""
        print(f"  {label:<38} {value}{tail}")

    print("\n=== 불변식 (전부 0이어야 함) ===")
    line("문법 위반 채보", v["FAIL_grammar_charts"])
    line("onset 불균형 채보", v["FAIL_balance_charts"])
    line("round trip 불일치 채보", v["FAIL_round_trip_charts"])
    line("예외 발생 채보", v["FAIL_error_charts"])

    print("\n=== 처리 현황 ===")
    for k, val in metrics:
        if k.startswith("status_"):
            line(k[7:], val)

    print("\n=== 격자 커버리지 (노트 가중 %, 오차 <= 3 / 5 / 10 ms) ===")
    for name, label in (("reorig", "재원점화 셀 왕복 (지금 쓰는 격자)"),
                        ("local", "local 스냅 거리 (§2.3 표)"),
                        ("global", "누적 격자 (옛 방식)")):
        vals = " / ".join(f"{v[f'cov_{name}_{t}ms_pct']:.2f}" for t in TOLS_MS)
        doc = {"local": "97.21 / 97.36 / 98.00", "global": "- / 90.81 / -"}.get(name)
        line(label, vals, doc)
    line("경계 셀로 옮겨진 노트", v["boundary_notes"])
    line("  그중 5ms 초과", v["boundary_notes_over_5ms"])
    line("  최대 오차 ms", v["boundary_max_err_ms"])

    print("\n=== DECISIONS 숫자 재현 ===")
    line("음수 셀 노트 / 채보", f"{v['neg_cell_notes']:,} / {v['neg_cell_charts']:,}",
         f"{DOC['neg_notes']:,} / {DOC['neg_charts']:,}")
    line("음수 global beat 노트 / 채보", f"{v['neg_global_notes']:,} / {v['neg_global_charts']:,}")
    line("독립 반올림의 틈 + 겹침", f"{v['indep_gaps'] + v['indep_overlaps']:,}"
         f" ({v['indep_gaps']:,} + {v['indep_overlaps']:,})", f"{DOC['indep_gap_overlap']:,}")
    line("오프비트 레드라인 채보", f"{v['charts_with_offbeat_red_line']:,}", "4,613")

    print("\n=== 충돌 정책이 한 일 ===")
    line("버린 노트 (채보 수)", f"{v['dropped_notes']:,} ({v['dropped_charts']:,})")
    line("줄인 롱노트 (채보 수)", f"{v['shortened_holds']:,} ({v['shortened_charts']:,})")
    line("  그중 단타가 된 것", f"{v['shortened_to_tap']:,}")
    line("길이 0으로 강등된 롱노트", f"{v['demoted_holds']:,}")

    print("\n=== 토큰 ===")
    line("곡당 칸 중앙값 / p95 / 최대", f"{v['cells_median']:.0f} / {v['cells_p95']:.0f} / "
         f"{v['cells_max']:,}", "4,273 / 10,498 / -")
    line("empty 비율 % (PAD 제외)", v["share_empty_pct"], "추정 78.08")
    line("tap / start / body / end %", " / ".join(
        str(v[f"share_{n}_pct"]) for n in ("tap", "hold_start", "hold_body", "hold_end")))

    def show(title: str, rs: list[dict], key: str, n: int = 8) -> None:
        if rs:
            print(f"\n{title} (상위 {min(n, len(rs))}개)")
            for r in rs[:n]:
                print(f"  {r.get(key, '')!s:>10}  {r['path']}  {r.get('error', '')}")

    tok = [r for r in rows if "round_trip" in r]
    show("문법 위반", [r for r in tok if r["n_violations"]], "n_violations")
    show("onset 불균형", [r for r in tok if not r["balanced"]], "n_notes")
    show("round trip 불일치 (beat_len_min_ms)", [r for r in tok if not r["round_trip"]],
         "beat_len_min_ms")
    show("예외", [r for r in rows if r["status"] == "error"], "status")
    show("칸 수 초과로 제외", [r for r in rows if r["status"] == "too_long"], "beat_len_min_ms")
    show("줄인 롱노트가 많은 채보", sorted(tok, key=lambda r: -r["n_shortened"]), "n_shortened", 5)
    show("버린 노트가 있는 채보", sorted([r for r in tok if r["n_dropped"]],
                                   key=lambda r: -r["n_dropped"]), "n_dropped", 5)
    print("\n" + ("모든 불변식 통과" if ok else "불변식 실패 있음: 위 목록 확인"))


def write_outputs(rows: list[dict], metrics, summary_path: Path, charts_path: Path) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["metric", "value"])
        w.writerows(metrics)
    charts_path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(charts_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {summary_path} and {charts_path}")


# ---------------------------------------------------------------------------
# self-test on synthetic .osu files
# ---------------------------------------------------------------------------

def write_osu(path: Path, key_count: int, tps, notes, mode: int = 3, **metadata) -> None:
    chart = Chart(key_count, "audio.mp3", [tuple(tp) for tp in tps], list(notes))
    chart_writer.write_osu(path, chart, mode=mode, **metadata)


def synthetic_charts(seed: int = 0, n: int = 60):
    """Charts with the cases that matter: off-beat red lines, notes before the
    first red line, holds followed by a note 1/16 beat later, duplicates."""
    rng = np.random.default_rng(seed)
    for i in range(n):
        bl = 60000 / rng.uniform(100, 250)
        t0 = int(rng.choice([0, 0, 800]))                 # some charts start timing late
        tps = [(t0, bl)]
        if i % 3 == 0:
            tps.append((t0 + int(rng.uniform(20_000, 40_000)), bl * rng.uniform(0.7, 1.3)))
        notes = []
        for k in range(K):
            t = t0 - bl * rng.uniform(0, 2) if i % 4 == 0 else t0 + rng.uniform(0, bl)
            while t < t0 + 50_000:
                grid_step = bl / rng.choice([4, 4, 4, 8, 12, 16])
                t = round(t / grid_step) * grid_step + rng.normal(0, 1.0)
                is_ln = rng.random() < 0.35
                dur = bl * rng.choice([0.5, 1.0, 2.0])
                notes.append(Note(round(float(t)), k, round(float(t + dur)) if is_ln else None))
                t += (dur if is_ln else 0) + bl * rng.choice([1 / 16, 1 / 8, 1 / 4, 1 / 2, 1])
        if i % 7 == 0:
            notes.append(Note(notes[0].time_ms, notes[0].lane))       # duplicate object
        notes.sort(key=lambda nt: (nt.time_ms, nt.lane))
        yield f"{i:03d} Synthetic - Song {i}/lv{i % 3}.osu", 4, tps, notes


def selftest(workers: int, summary: Path | None, charts: Path | None) -> int:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "raw"
        for rel, keys, tps, notes in synthetic_charts():
            write_osu(root / rel, keys, tps, notes)
        write_osu(root / "900 Other - Seven/7k.osu", 7, [(0, 500.0)], [Note(0, 5)])
        write_osu(root / "901 Other - Std/std.osu", 4, [(0, 500.0)], [Note(0, 0)], mode=0)
        write_osu(root / "902 Other - Gimmick/gimmick.osu", 4,
                  [(0, 500.0), (1000, 0.01), (2000, 500.0)], [Note(3000, 0)])

        rows = run(root, workers, 0)
        metrics, ok = summarize(rows)
        report(rows, metrics, ok)
        write_outputs(rows, metrics, summary or Path(tmp) / "summary.csv",
                      charts or Path(tmp) / "charts.csv")
    v = dict(metrics)
    expect = {
        "invariants hold": ok,
        "7K chart skipped": v.get("status_wrong_keys") == 1,
        "non-mania file skipped": v.get("status_not_mania") == 1,
        "gimmick chart hits the cell guard": v.get("status_too_long") == 1,
        "holds were shortened": v["shortened_holds"] > 0,
        "duplicates were dropped": v["dropped_notes"] > 0,
        "notes before the first red line kept": v["neg_cell_notes"] > 0,
    }
    print("\n=== selftest ===")
    for name, passed in expect.items():
        print(f"  {'ok  ' if passed else 'FAIL'} {name}")
    return 0 if all(expect.values()) else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", type=Path, default=Path("data/raw"))
    ap.add_argument("--workers", type=int, default=None,
                    help="default: all CPU cores (1 for --selftest)")
    ap.add_argument("--limit", type=int, default=0, help="only the first N files (sorted)")
    ap.add_argument("--summary", type=Path, default=None,
                    help="default: docs/_stats/tokenizer_validation.csv")
    ap.add_argument("--charts", type=Path, default=None,
                    help="default: data/tokenizer_validation_charts.csv")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest(a.workers or 1, a.summary, a.charts)
    if not a.root.exists():
        print(f"{a.root} does not exist", file=sys.stderr)
        return 2

    rows = run(a.root, a.workers or os.cpu_count() or 1, a.limit)
    metrics, ok = summarize(rows)
    report(rows, metrics, ok)
    # a --limit run must not overwrite the committed full-dataset summary
    default_summary = (Path("docs/_stats/tokenizer_validation.csv") if not a.limit
                       else Path("data/tokenizer_validation_partial.csv"))
    write_outputs(rows, metrics, a.summary or default_summary,
                  a.charts or Path("data/tokenizer_validation_charts.csv"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
