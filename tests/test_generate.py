"""generate.estimate_timing on click tracks: beat on loud clicks, quiet off-beat
hats, an accented downbeat. Clicks in silence make flux peak ~10 ms before the
click, and the estimate subtracts the ~19 ms lag of real songs, so the offset
comes out ~29 ms early here."""

import pytest

from scripts.generate import estimate_timing
from tests.synth_audio import clicks


@pytest.mark.parametrize("bpm, offset", [(174.0, 437.0), (200.0, 1234.0), (128.5, 90.0)])
def test_estimate_timing_on_clicks(bpm, offset):
    sr, sec = 22050, 60
    bl = 60000 / bpm
    beats = [offset + i * bl for i in range(int((sec * 1000 - offset) / bl))]
    y = (clicks(beats, sr, sec, seed=1) + 0.15 * clicks([t + bl / 2 for t in beats], sr, sec, seed=2)
         + clicks(beats[::4], sr, sec, seed=3))
    est, off, sharp = estimate_timing(y)
    assert abs(est - bpm) < 0.05
    err = (off - offset + bl / 2) % bl - bl / 2                  # nearest beat
    assert -45 < err < -10
    assert round((off - offset - err) / bl) % 4 == 0             # on the downbeat
    assert sharp > 3
