"""How the audio files start, and whether decode_osu agrees with osu!'s own decoder.

    python scripts/check_decoder.py                     # census of every kept chart's audio
    python scripts/check_decoder.py --bass PATH/libbass.dylib --n 300   # + compare with BASS

Census (reads the first 1 MB of each file, no decoding): container, mp3 tag kind
(lame / xing / none / vbri / ...) and the trim src/data/audio.py applies, encoder
names, and the "osu file format" version of the charts (osu!lazer moves charts
older than v5 by 24 ms; none are expected in ranked mania).

--bass decodes up to N files with decode_osu and with libbass itself, with
BASS_CONFIG_MP3_OLDGAPS on as osu! sets it, and cross-correlates the first 30 s:
every file should come out at lag 0. Rare tag kinds are taken first. libbass is
not shipped here: it comes with osu! (on macOS inside osu!.app) and with
ppy/osu-framework in osu.Framework.NativeLibs/runtimes/<platform>/native/.
Writes docs/_stats/audio_census.csv (and audio_vs_bass.csv with --bass).
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import os
import random
import re
import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from scripts.preprocess_data import find_audio
from src.data.audio import decode_osu, mp3_info, read_head
from src.data.cache import read_manifest

_VERSION = re.compile(rb"osu file format v(\d+)")


def container(head: bytes) -> str:
    if head[:4] == b"OggS":
        return "ogg"
    if head[:4] == b"RIFF":
        return "wav"
    if head[:4] == b"fLaC":
        return "flac"
    if head[:3] == b"ID3" or (len(head) > 1 and head[0] == 0xFF and head[1] & 0xE0 == 0xE0):
        return "mpeg"
    return "other"


def census_one(job: tuple) -> dict:
    rel, root = job
    head = read_head(root / rel)
    row = {"audio": rel, "container": container(head), "tag": "", "trim": "", "encoder": "",
           "delay": "", "padding": "", "sample_rate": ""}
    if row["container"] in ("mpeg", "other"):
        info = mp3_info(head)
        if info.tag != "no_frame" or row["container"] == "mpeg":
            row.update(container="mpeg", tag=info.tag, trim=info.trim, encoder=info.encoder,
                       delay=info.delay, padding=info.padding, sample_rate=info.sample_rate)
    return row


def file_version(path: Path) -> int:
    with open(path, "rb") as f:
        m = _VERSION.search(f.read(200))
    return int(m.group(1)) if m else -1


# ---------- BASS, the reference ----------

class Bass:
    DECODE, FLOAT, OLDGAPS = 0x200000, 256, 68

    class Info(ctypes.Structure):
        _fields_ = [("freq", ctypes.c_uint32), ("chans", ctypes.c_uint32),
                    ("flags", ctypes.c_uint32), ("ctype", ctypes.c_uint32),
                    ("origres", ctypes.c_uint32), ("plugin", ctypes.c_uint32),
                    ("sample", ctypes.c_uint32), ("filename", ctypes.c_char_p)]

    def __init__(self, path: Path):
        lib = ctypes.CDLL(str(path))
        lib.BASS_SetConfig.argtypes = [ctypes.c_uint32, ctypes.c_uint32]
        lib.BASS_Init.argtypes = [ctypes.c_int, ctypes.c_uint32, ctypes.c_uint32,
                                  ctypes.c_void_p, ctypes.c_void_p]
        lib.BASS_StreamCreateFile.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_uint64,
                                              ctypes.c_uint64, ctypes.c_uint32]
        lib.BASS_StreamCreateFile.restype = ctypes.c_uint32
        lib.BASS_ChannelGetData.argtypes = [ctypes.c_uint32, ctypes.c_void_p, ctypes.c_uint32]
        lib.BASS_ChannelGetData.restype = ctypes.c_uint32
        lib.BASS_ChannelGetInfo.argtypes = [ctypes.c_uint32, ctypes.POINTER(Bass.Info)]
        lib.BASS_GetVersion.restype = ctypes.c_uint32
        if not lib.BASS_Init(0, 44100, 0, None, None):          # device 0: no sound output
            raise RuntimeError(f"BASS_Init failed: error {lib.BASS_ErrorGetCode()}")
        lib.BASS_SetConfig(self.OLDGAPS, 1)                    # what osu! sets
        self.lib = lib
        v = lib.BASS_GetVersion()
        self.version = ".".join(str((v >> s) & 0xFF) for s in (24, 16, 8, 0))

    def decode(self, path: Path, seconds: float) -> tuple[np.ndarray, int]:
        h = self.lib.BASS_StreamCreateFile(0, str(path).encode(), 0, 0, self.DECODE | self.FLOAT)
        if not h:
            raise RuntimeError(f"BASS_StreamCreateFile failed: error {self.lib.BASS_ErrorGetCode()}")
        info = Bass.Info()
        self.lib.BASS_ChannelGetInfo(h, ctypes.byref(info))
        want = int(seconds * info.freq * info.chans)
        buf, parts, got = (ctypes.c_float * 65536)(), [], 0
        while got < want:
            n = self.lib.BASS_ChannelGetData(h, buf, ctypes.sizeof(buf))
            if n in (0, 0xFFFFFFFF):
                break
            parts.append(np.frombuffer(buf, dtype=np.float32, count=n // 4).copy())
            got += n // 4
        self.lib.BASS_StreamFree(h)
        x = np.concatenate(parts) if parts else np.zeros(0, np.float32)
        x = x[:len(x) // info.chans * info.chans].reshape(-1, info.chans).mean(axis=1)
        return x[:int(seconds * info.freq)], info.freq


def lag(a: np.ndarray, b: np.ndarray, reach: int = 6000) -> tuple[int, float]:
    """(samples by which a sits later than b, normalized correlation at that lag)."""
    n = min(len(a), len(b))
    size = 1 << int(np.ceil(np.log2(2 * n)))
    c = np.fft.irfft(np.fft.rfft(a[:n], size) * np.conj(np.fft.rfft(b[:n], size)), size)
    c = np.concatenate([c[-reach:], c[:reach + 1]])
    k = int(np.argmax(c))
    norm = np.linalg.norm(a[:n]) * np.linalg.norm(b[:n]) + 1e-12
    return k - reach, float(c[k] / norm)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"))
    ap.add_argument("--root", type=Path, default=Path("data/raw"))
    ap.add_argument("--stats", type=Path, default=Path("docs/_stats"))
    ap.add_argument("--bass", type=Path, default=None, help="libbass to compare with")
    ap.add_argument("--n", type=int, default=300, help="--bass: files to compare")
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    a = ap.parse_args(argv)

    rows = [r for r in read_manifest(a.manifest) if r.get("drop", "") == ""]
    lookups = [(r, a.root) for r in rows]
    with Pool(a.workers) as pool:
        found = dict(pool.imap(find_audio, lookups, chunksize=64))
        audios = sorted({v for v in found.values() if v is not None})
        census = list(pool.imap(census_one, [(rel, a.root) for rel in audios], chunksize=16))
        versions = Counter(pool.imap(file_version, [a.root / r["path"] for r in rows],
                                     chunksize=64))

    a.stats.mkdir(parents=True, exist_ok=True)
    with open(a.stats / "audio_census.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(census[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(census)
    kinds = Counter((c["container"], c["tag"]) for c in census)
    print(f"{len(rows):,} kept charts, {len(audios):,} audio files "
          f"({sum(v is None for v in found.values())} charts without one)")
    for (cont, tag), n in kinds.most_common():
        trims = Counter(c["trim"] for c in census if (c["container"], c["tag"]) == (cont, tag))
        top = ", ".join(f"{t}: {k:,}" for t, k in trims.most_common(3)) if tag else ""
        print(f"  {cont:5s} {tag or '-':10s} {n:6,}  {100 * n / len(census):5.1f}%   "
              + (f"trim {top}" if top else ""))
    enc = Counter(c["encoder"][:4] or "(none)" for c in census if c["tag"] in ("lame", "xing"))
    print("  encoder names (first 4 chars): " + ", ".join(f"{k} {v:,}" for k, v in enc.most_common(8)))
    print("  osu file format versions: " + ", ".join(f"v{k} {v:,}" for k, v in sorted(versions.items())))
    early = sum(v for k, v in versions.items() if 0 <= k < 5)
    print(f"  charts older than v5 (osu!lazer adds 24 ms to those): {early}")

    if a.bass is None:
        return 0
    bass = Bass(a.bass)
    rng = random.Random(a.seed)
    rare = [c for c in census if c["container"] == "mpeg" and c["tag"] != "lame"]
    common = [c for c in census if c not in rare]
    rng.shuffle(rare)
    rng.shuffle(common)
    pick = rare[:a.n // 2]
    pick += common[:a.n - len(pick)]
    print(f"\nBASS {bass.version} (MP3_OLDGAPS on) vs decode_osu, first {a.seconds:g} s of "
          f"{len(pick)} files:")
    out = []
    for i, c in enumerate(pick, 1):
        path = a.root / c["audio"]
        try:
            ours = decode_osu(path)
            ref, rate = bass.decode(path, a.seconds)
            k, corr = lag(ours.samples[:len(ref)], ref) if rate == ours.sample_rate else (None, 0.0)
            out.append({**c, "rate_bass": rate, "rate_ours": ours.sample_rate, "lag": k,
                        "corr": round(corr, 4), "error": ""})
        except Exception as e:
            out.append({**c, "rate_bass": "", "rate_ours": "", "lag": "", "corr": "",
                        "error": f"{type(e).__name__}: {e}"})
        if i % 50 == 0:
            print(f"  [{i}/{len(pick)}]", flush=True)
    with open(a.stats / "audio_vs_bass.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(out)
    by_kind: dict[tuple, list] = {}
    for r in out:
        by_kind.setdefault((r["container"], r["tag"]), []).append(r)
    bad = [r for r in out if r["error"] or r["lag"] != 0 or r["corr"] < 0.95]
    for (cont, tag), rs in sorted(by_kind.items()):
        ok = sum(r["lag"] == 0 and not r["error"] for r in rs)
        print(f"  {cont:5s} {tag or '-':10s} {ok:4d} / {len(rs):4d} at lag 0")
    for r in bad[:15]:
        print(f"  lag {r['lag']} corr {r['corr']} {r['error']}  {r['audio']}")
    print("every file at lag 0: decode_osu starts where osu! does" if not bad else
          f"{len(bad)} files differ: see {a.stats / 'audio_vs_bass.csv'}")
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
