"""Synthetic audio for tests: clicks at known times, written as mp3 (PyAV) or wav."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def clicks(times_ms, sr: int, seconds: float, seed: int = 0) -> np.ndarray:
    """Mono float32: a short decaying noise burst starting at each time."""
    rng = np.random.default_rng(seed)
    x = np.zeros(int(sr * seconds), dtype=np.float32)
    n = int(0.005 * sr)
    env = np.exp(-np.arange(n) / (0.001 * sr)).astype(np.float32)
    for t in times_ms:
        i = round(t * sr / 1000)
        if 0 <= i < len(x) - n:
            x[i:i + n] += 0.5 * rng.normal(0, 1, n).astype(np.float32) * env
    return np.clip(x, -1, 1)


def can_write_mp3() -> bool:
    try:
        import av
        av.codec.Codec("libmp3lame", "w")
        return True
    except Exception:
        return False


def write_mp3(path: Path, x: np.ndarray, sr: int, *, xing: bool = True,
              bitrate: int = 128000) -> None:
    """Encode with LAME through PyAV. xing=True writes the Info tag with the encoder
    delay (as LAME and FFmpeg do by default); xing=False leaves it out."""
    import av
    opts = {} if xing else {"write_xing": "0"}
    with av.open(str(path), "w", format="mp3", options=opts) as out:
        stream = out.add_stream("libmp3lame", rate=sr, layout="mono")
        stream.bit_rate = bitrate
        frame = av.AudioFrame.from_ndarray(x[None, :].astype(np.float32), format="flt",
                                           layout="mono")
        frame.sample_rate = sr
        for packet in stream.encode(frame):
            out.mux(packet)
        for packet in stream.encode(None):
            out.mux(packet)


def lag(a: np.ndarray, b: np.ndarray) -> int:
    """Samples by which the content of a sits later than that of b."""
    from numpy.fft import irfft, rfft
    n = min(len(a), len(b))
    size = 1 << int(np.ceil(np.log2(2 * n)))
    c = irfft(rfft(a[:n], size) * np.conj(rfft(b[:n], size)), size)
    k = int(np.argmax(c))
    return k if k < size // 2 else k - size
