"""audio.py: where osu! puts 0 ms in an mp3 (the BASS rule), and decoding.

The byte-level cases pin mp3_info to what BASS 2.4.17 with MP3_OLDGAPS (osu!) did
on the same kinds of files; scripts/check_decoder.py --bass repeats the comparison
on real files."""

import struct

import numpy as np
import pytest

from src.data.audio import (
    DECODER_DELAY,
    _first_frame,
    _to_float,
    decode_osu,
    load_osu,
    mp3_info,
    read_head,
    resample,
)
from tests.synth_audio import can_write_mp3, clicks, lag, write_mp3

MPEG1_STEREO = (0xFB, 0x90, 0x00)      # MPEG-1 layer III, 128 kbps, 44.1 kHz, stereo: 417 bytes
MPEG1_MONO = (0xFB, 0x90, 0xC0)
MPEG2_STEREO = (0xF3, 0x80, 0x00)      # MPEG-2 layer III, 64 kbps, 22.05 kHz: 208 bytes
MPEG2_MONO = (0xF3, 0x80, 0xC0)
LAYER2 = (0xFD, 0x90, 0x00)


def frame(header=MPEG1_STEREO, at: int = 0, payload: bytes = b"") -> bytes:
    lengths = {0xFB: 417, 0xF3: 208, 0xFD: 417}
    body = bytearray(lengths[header[0]])
    body[:4] = bytes([0xFF, *header])
    body[at:at + len(payload)] = payload
    return bytes(body)


def info_tag(name: bytes = b"LAME3.100", delay: int = 576, pad: int = 1000) -> bytes:
    lame = name.ljust(9, b"\0")[:9] + bytes(12) + ((delay << 12) | pad).to_bytes(3, "big")
    return b"Info" + struct.pack(">I", 0x0F) + bytes(4 + 4 + 100 + 4) + lame


def mp3(first: bytes, header=MPEG1_STEREO, n: int = 3) -> bytes:
    return first + b"".join(frame(header) for _ in range(n))


@pytest.mark.parametrize("name, delay, pad, tag, trim", [
    (b"LAME3.100", 576, 1000, "lame", 576 + DECODER_DELAY),   # the usual LAME file
    (b"Lavc60.31", 576, 936, "lame", 1105),                   # FFmpeg's own tag
    (b"GOGO3.13", 576, 0, "lame", 1105),                      # FFmpeg ignores this one, BASS not
    (b"LAME3.100", 0, 500, "lame", DECODER_DELAY),            # padding alone still counts
    (b"LAME3.100", 1000, 0, "lame", 1529),
    (b"Lavc60.31", 0, 0, "xing", 0),                          # libshine through FFmpeg
    (b"\0AME3.100", 576, 1000, "xing", 0),                    # BASS reads the first byte
    (b"", 576, 1000, "xing", 0),
])
def test_lame_tag_rule(name, delay, pad, tag, trim):
    info = mp3_info(mp3(frame(at=36, payload=info_tag(name, delay, pad))))
    assert (info.tag, info.trim) == (tag, trim)
    assert info.sample_rate == 44100 and info.channels == 2 and info.frame_offset == 0


@pytest.mark.parametrize("header, at", [(MPEG1_MONO, 21), (MPEG2_STEREO, 21), (MPEG2_MONO, 13)])
def test_tag_offset_follows_version_and_channels(header, at):
    info = mp3_info(mp3(frame(header, at=at, payload=info_tag()), header))
    assert (info.tag, info.trim) == ("lame", 1105)
    wrong = mp3_info(mp3(frame(header, at=36 if at != 36 else 21, payload=info_tag()), header))
    assert wrong.tag == "none"


def test_no_tag_vbri_and_other_layers():
    info = mp3_info(mp3(frame()))
    assert (info.tag, info.trim) == ("none", 0)
    for delay, trim in ((0, DECODER_DELAY - 1152), (576, 576 - 1152), (1500, 348)):
        vbri = b"VBRI" + struct.pack(">HHH", 1, delay, 75) + bytes(16)
        info = mp3_info(mp3(frame(at=36, payload=vbri)))
        assert (info.tag, info.trim, info.delay) == ("vbri", trim, delay)
    assert mp3_info(mp3(frame(LAYER2), LAYER2)).tag == "not_layer3"
    assert mp3_info(b"\x00" * 5000).tag == "no_frame"


def test_id3v2_and_junk_before_the_first_frame():
    body = mp3(frame(at=36, payload=info_tag()))
    id3 = b"ID3\x04\x00\x00" + bytes([0, 0, 0, 20]) + bytes(20)
    assert mp3_info(id3 + body).frame_offset == 30
    assert mp3_info(id3 + b"\x00\xff\x00junk" + body).tag == "lame"     # scanned past junk
    wrong_size = b"ID3\x04\x00\x00" + bytes([0, 0, 0, 5]) + bytes(20)  # tag says 5, is 20
    assert mp3_info(wrong_size + body).trim == 1105
    false_sync = b"\xff\xfb\x90\x00" + bytes(100)                     # no frame follows it
    assert mp3_info(false_sync + body).frame_offset == 104


requires_mp3 = pytest.mark.skipif(not can_write_mp3(), reason="PyAV without libmp3lame")


@requires_mp3
@pytest.mark.parametrize("sr", [44100, 48000])
def test_decode_puts_zero_where_osu_does(tmp_path, sr):
    x = clicks([500, 1300, 2250, 3100], sr, 4.0)
    tagged, bare = tmp_path / "tagged.mp3", tmp_path / "bare.mp3"
    write_mp3(tagged, x, sr)
    write_mp3(bare, x, sr, xing=False)
    d = decode_osu(tagged)
    assert d.sample_rate == sr and d.mp3.tag == "lame" and d.mp3.trim == 1105
    assert lag(d.samples, x) == 0                        # tag: encoder + decoder delay removed
    d = decode_osu(bare)
    assert d.mp3.tag == "none" and lag(d.samples, x) == 1105   # osu! keeps both, and so do we


@requires_mp3
def test_load_resamples_without_moving_time(tmp_path):
    sr = 44100
    x = clicks([400, 900, 1750], sr, 2.5)
    write_mp3(tmp_path / "a.mp3", x, sr)
    y, info = load_osu(tmp_path / "a.mp3")
    assert info["sample_rate"] == sr and info["mp3_trim"] == 1105
    assert lag(y, resample(x, sr)) == 0


def test_wav_and_sample_formats(tmp_path):
    sf = pytest.importorskip("soundfile")
    sr = 22050
    x = clicks([250, 700], sr, 1.0)
    sf.write(tmp_path / "a.wav", np.stack([x, x], axis=1), sr, subtype="PCM_16")
    d = decode_osu(tmp_path / "a.wav")
    assert d.mp3 is None and d.channels == 2 and d.sample_rate == sr
    assert lag(d.samples, x) == 0 and np.abs(d.samples - x).max() < 1e-3   # s16 scaled back

    class Frame:                                         # packed s16 stereo, as PyAV gives it
        class format:
            name, is_planar = "s16", False

        class layout:
            channels = (0, 1)

        @staticmethod
        def to_ndarray():
            return np.array([[16384, -16384, 32767, 0]], dtype=np.int16)

    np.testing.assert_allclose(_to_float(Frame), [[0.5, 32767 / 32768], [-0.5, 0.0]])


def id3(frames: bytes, version: int = 3) -> bytes:
    n = len(frames)
    size = bytes([(n >> 21) & 0x7F, (n >> 14) & 0x7F, (n >> 7) & 0x7F, n & 0x7F])
    return b"ID3" + bytes([version, 0, 0]) + size + frames


def test_cover_art_longer_than_a_megabyte(tmp_path):
    art = 2_500_000
    tag = id3(b"APIC" + art.to_bytes(4, "big") + b"\0\0" + bytes(art))
    path = tmp_path / "big.mp3"
    path.write_bytes(tag + mp3(frame(at=36, payload=info_tag())))
    assert mp3_info(path.read_bytes()[:1 << 20]).tag == "no_frame"   # what 1 MB used to see
    info = mp3_info(read_head(path))
    assert (info.tag, info.trim, info.frame_offset) == ("lame", 1105, len(tag))
    small = tmp_path / "small.mp3"
    small.write_bytes(mp3(frame()))
    assert read_head(small) == small.read_bytes()


def _strip_id3(data: bytes) -> bytes:
    p, _ = _first_frame(data)
    return data[p:]


@requires_mp3
def test_tags_in_legacy_encodings_are_read_leniently(tmp_path):
    sr = 44100
    x = clicks([400, 900], sr, 1.5)
    write_mp3(tmp_path / "a.mp3", x, sr)
    text = b"\x03" + "노래".encode("cp949")          # says UTF-8, is CP949
    title = b"TIT2" + len(text).to_bytes(4, "big") + b"\0\0" + text
    path = tmp_path / "b.mp3"
    path.write_bytes(id3(title, 4) + _strip_id3((tmp_path / "a.mp3").read_bytes()))
    d = decode_osu(path)
    assert d.mp3.tag == "lame" and lag(d.samples, x) == 0


@requires_mp3
def test_a_broken_packet_becomes_silence_of_its_length(tmp_path):
    sr = 44100
    x = clicks(list(range(300, 2900, 400)), sr, 3.0)
    write_mp3(tmp_path / "a.mp3", x, sr)
    data = bytearray((tmp_path / "a.mp3").read_bytes())
    p, h = _first_frame(bytes(data))
    at = p + 40 * h["length"]                         # a frame about 1 s in
    data[at:at + 4] = b"\xff\xff\xff\xff"
    data[at + 10:at + 300] = np.random.default_rng(0).integers(0, 256, 290, dtype=np.uint8).tobytes()
    (tmp_path / "b.mp3").write_bytes(bytes(data))
    d = decode_osu(tmp_path / "b.mp3")
    assert d.bad_packets >= 1 and d.info()["bad_packets"] == d.bad_packets
    assert lag(d.samples[:sr * 9 // 10], x[:sr * 9 // 10]) == 0   # before the break: exact
    # The broken header takes its own frame with it; the rejected packet is kept as
    # silence. BASS 2.4.17 gave the same samples after the break (lag 0 against it).
    tail = 2 * sr
    assert lag(d.samples[tail:], x[tail:]) in (0, -1152)
