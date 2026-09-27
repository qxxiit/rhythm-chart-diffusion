"""Decode audio so that sample 0 is where osu! puts 0 ms (design doc §4.5).

Chart times are milliseconds after the first sample osu! plays. osu! decodes with
BASS and sets the BASS_CONFIG_MP3_OLDGAPS compatibility flag (osu-framework
AudioManager: "for backwards compatibility"), and decoders disagree about where an
mp3 starts by up to 1,105 samples, 25 ms at 44.1 kHz:

    an mp3 frame carries 576 samples of encoder delay (LAME) and every decoder adds
    529 samples of its own. A gapless decoder drops both when the file says how long
    the delay is (a LAME / Xing "Info" tag) and differs from player to player when
    it does not.

Samples dropped from the start, measured on synthetic files against the BASS that
osu! ships (2.4.17.48, flag on):

    first frame of the mp3                      BASS (osu!)   FFmpeg / PyAV     soundfile
    LAME tag with a name, delay or pad > 0      delay + 529   delay + 529 (*)   delay + 529
    LAME tag with a name, delay = pad = 0       0             529 (*)           529
    Xing / Info tag, name's first byte 0        0             0                 529
    no tag (an iTunSMPB comment is ignored)     0             0                 0
    VBRI tag (Fraunhofer)                       plays the tag frame, then drops max(delay, 529)
    (*) only if the name starts with LAME, Lavf or Lavc; otherwise 0

No library matches the first column everywhere, so the mp3 path decodes with the
decoder's own trimming turned off (FFmpeg flags2=+skip_manual: every sample after
the tag frame) and drops `Mp3Info.trim` samples, the first column. Every other
format (ogg, wav, ...) decodes normally: BASS and FFmpeg agree there.

tests/test_audio.py pins the rule on synthetic files; scripts/check_decoder.py
compares it with a real libbass on real files.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

SAMPLE_RATE = 22050             # what the mel stage works at
AUDIO_RULE = 2                  # bump when decoding changes: the mel cache is redone
                                # 2: ID3 tags longer than 1 MB, lenient tags, bad packets
DECODER_DELAY = 529             # samples every mp3 decoder adds

_BITRATES = {                   # kbps by (MPEG-1?, layer)
    (True, 1): (0, 32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448),
    (True, 2): (0, 32, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 384),
    (True, 3): (0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320),
    (False, 1): (0, 32, 48, 56, 64, 80, 96, 112, 128, 144, 160, 176, 192, 224, 256),
    (False, 2): (0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160),
    (False, 3): (0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160),
}
_RATES = {3: (44100, 48000, 32000), 2: (22050, 24000, 16000), 0: (11025, 12000, 8000)}


@dataclass(frozen=True)
class Mp3Info:
    """What the first frame of an mp3 says about its start.

    tag      "lame" (Xing/Info with a LAME part), "xing" (Xing/Info, no usable LAME
             part), "vbri" (Fraunhofer), "none", "not_layer3", "no_frame"
    trim     samples to drop from what FFmpeg decodes with its own trimming off, so
             that the result starts where BASS (osu!) starts; negative = prepend silence
               lame   delay + 529, when the encoder name's first byte is not 0 and
                      delay or padding is nonzero (else the tag counts as "xing")
               vbri   max(delay, 529) - (samples per frame): BASS plays the VBRI
                      frame, FFmpeg skips it
               other  0
    """
    tag: str
    trim: int = 0
    encoder: str = ""
    delay: int = 0
    padding: int = 0
    sample_rate: int = 0
    channels: int = 0
    frame_offset: int = -1


def _header(buf: bytes, p: int) -> dict | None:
    """The MPEG audio frame header at p, or None."""
    if p + 4 > len(buf) or buf[p] != 0xFF or buf[p + 1] & 0xE0 != 0xE0:
        return None
    version = (buf[p + 1] >> 3) & 3          # 3 MPEG-1, 2 MPEG-2, 0 MPEG-2.5, 1 reserved
    layer = 4 - ((buf[p + 1] >> 1) & 3)      # bits 3, 2, 1 -> layer I, II, III
    br_index, sr_index = buf[p + 2] >> 4, (buf[p + 2] >> 2) & 3
    if version == 1 or layer == 4 or br_index in (0, 15) or sr_index == 3:
        return None
    mpeg1 = version == 3
    bitrate = _BITRATES[(mpeg1, layer)][br_index] * 1000
    rate = _RATES[version][sr_index]
    pad = (buf[p + 2] >> 1) & 1
    if layer == 1:
        length = (12 * bitrate // rate + pad) * 4
    elif layer == 3 and not mpeg1:
        length = 72 * bitrate // rate + pad
    else:
        length = 144 * bitrate // rate + pad
    return {"version": version, "layer": layer, "rate": rate, "length": length,
            "mono": buf[p + 3] >> 6 == 3}


def _id3_end(buf: bytes) -> int:
    """Where the ID3v2 tags at the start end (cover art can make them megabytes long)."""
    p = 0
    while buf[p:p + 3] == b"ID3" and p + 10 <= len(buf):
        size = 0
        for byte in buf[p + 6:p + 10]:
            size = (size << 7) | (byte & 0x7F)
        p += 10 + size + (10 if buf[p + 5] & 0x10 else 0)
    return p


def read_head(path: str | Path, after: int = 1 << 20) -> bytes:
    """The ID3v2 tags plus `after` bytes: enough for mp3_info."""
    with open(path, "rb") as f:
        head = f.read(after)
        while True:
            want = _id3_end(head) + after
            if want <= len(head):
                return head
            f.seek(0)
            more = f.read(want)
            if len(more) <= len(head):          # the file ended
                return more
            head = more


def _first_frame(buf: bytes) -> tuple[int, dict] | None:
    """Skip ID3v2 tags, then the first frame header that the next frame confirms."""
    p = _id3_end(buf)
    end = min(len(buf), p + (1 << 20))
    while p < end:
        p = buf.find(b"\xff", p, end)
        if p < 0:
            return None
        h = _header(buf, p)
        if h is not None:
            q = p + h["length"]
            n = _header(buf, q)
            if q >= len(buf) or (n is not None and (n["version"], n["layer"], n["rate"])
                                  == (h["version"], h["layer"], h["rate"])):
                return p, h
        p += 1
    return None


def mp3_info(data: bytes) -> Mp3Info:
    """Parse the start of an mp3 file (read_head(path) gives enough bytes)."""
    found = _first_frame(data)
    if found is None:
        return Mp3Info("no_frame")
    p, h = found
    base = {"sample_rate": h["rate"], "channels": 1 if h["mono"] else 2, "frame_offset": p}
    if h["layer"] != 3:
        return Mp3Info("not_layer3", **base)
    side = (17 if h["mono"] else 32) if h["version"] == 3 else (9 if h["mono"] else 17)
    x = p + 4 + side                                   # as FFmpeg: the CRC is not counted
    if data[p + 36:p + 40] == b"VBRI":
        delay = int.from_bytes(data[p + 42:p + 44], "big")
        per_frame = 1152 if h["version"] == 3 else 576
        return Mp3Info("vbri", trim=max(delay, DECODER_DELAY) - per_frame, delay=delay, **base)
    if data[x:x + 4] not in (b"Xing", b"Info"):
        return Mp3Info("none", **base)
    flags = int.from_bytes(data[x + 4:x + 8], "big")
    j = x + 8 + 4 * bool(flags & 1) + 4 * bool(flags & 2) + 100 * bool(flags & 4) \
        + 4 * bool(flags & 8)
    if j + 24 > min(p + h["length"], len(data)):
        return Mp3Info("xing", **base)
    name = data[j:j + 9]
    word = int.from_bytes(data[j + 21:j + 24], "big")
    delay, padding = word >> 12, word & 0xFFF
    encoder = name.split(b"\0")[0].decode("latin-1").strip()
    if name[0] == 0 or (delay == 0 and padding == 0):
        return Mp3Info("xing", encoder=encoder, delay=delay, padding=padding, **base)
    return Mp3Info("lame", trim=delay + DECODER_DELAY, encoder=encoder, delay=delay,
                   padding=padding, **base)


@dataclass
class Decoded:
    samples: np.ndarray           # float32 [n] mono, sample 0 = 0 ms in the chart
    sample_rate: int
    codec: str
    channels: int
    mp3: Mp3Info | None
    bad_packets: int = 0

    def info(self) -> dict:
        out = {"codec": self.codec, "sample_rate": self.sample_rate, "channels": self.channels,
               "seconds": round(len(self.samples) / self.sample_rate, 4),
               "bad_packets": self.bad_packets}
        if self.mp3 is not None:
            out.update({f"mp3_{k}": v for k, v in asdict(self.mp3).items()
                        if k in ("tag", "trim", "encoder", "delay", "padding")})
        return out


_SCALE = {"s16": 32768.0, "s32": 2147483648.0, "u8": 128.0}


def _to_float(frame) -> np.ndarray:
    """PyAV audio frame -> float32 [channels, n]."""
    x = frame.to_ndarray()
    fmt = frame.format.name.rstrip("p")
    channels = len(frame.layout.channels)
    if not frame.format.is_planar:
        x = x.reshape(-1, channels).T
    if fmt in ("flt", "dbl"):
        return x.astype(np.float32, copy=False)
    if fmt == "u8":
        return (x.astype(np.float32) - 128.0) / 128.0
    if fmt in _SCALE:
        return x.astype(np.float32) / _SCALE[fmt]
    raise ValueError(f"unsupported sample format {frame.format.name}")


def decode_osu(path: str | Path) -> Decoded:
    """Mono float32 at the file's own rate, starting where osu!'s 0 ms is.

    Tags that are not valid UTF-8 (legacy Korean / Japanese encodings) are read
    leniently. A packet the decoder rejects becomes silence of the same length,
    so everything after it stays at the right time (Decoded.bad_packets counts them).
    """
    import av  # PyAV (bundles FFmpeg)

    path = Path(path)
    with av.open(str(path), metadata_errors="ignore") as f:
        stream = f.streams.audio[0]
        codec = stream.codec_context.name
        is_mp3 = codec in ("mp3", "mp3float")
        if is_mp3:                                     # every sample: trim ourselves below
            stream.codec_context.options = {"flags2": "+skip_manual"}
        parts, rate, channels, bad = [], stream.codec_context.sample_rate, 0, 0
        for packet in f.demux(stream):
            try:
                frames = packet.decode()
            except av.error.FFmpegError:
                bad += 1
                seconds = float(packet.duration * packet.time_base) \
                    if packet.duration and packet.time_base else 0.0
                n = round(seconds * rate) or (stream.codec_context.frame_size or 1152)
                parts.append(np.zeros(n, dtype=np.float32))
                continue
            for frame in frames:
                x = _to_float(frame)
                channels = max(channels, x.shape[0])
                rate = frame.sample_rate or rate
                parts.append(x.mean(axis=0))
    samples = np.concatenate(parts).astype(np.float32) if parts else np.zeros(0, np.float32)
    info = None
    if is_mp3:
        info = mp3_info(read_head(path))
        if info.trim >= 0:
            samples = samples[info.trim:]
        else:
            samples = np.concatenate([np.zeros(-info.trim, np.float32), samples])
    return Decoded(samples, int(rate), codec, channels, info, bad)


def resample(samples: np.ndarray, rate: int, target: int = SAMPLE_RATE) -> np.ndarray:
    """Band-limited resampling that keeps sample 0 at 0 ms (soxr, as librosa)."""
    if rate == target:
        return samples.astype(np.float32, copy=False)
    import soxr
    return soxr.resample(samples, rate, target, quality="HQ").astype(np.float32, copy=False)


def load_osu(path: str | Path, target: int = SAMPLE_RATE) -> tuple[np.ndarray, dict]:
    """decode_osu + resample: (mono float32 at `target` Hz, info)."""
    d = decode_osu(path)
    return resample(d.samples, d.sample_rate, target), d.info()
