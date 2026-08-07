from pathlib import Path
import pytest
from src.data.chart_parser import parse_osu

SAMPLES = sorted(Path("data/raw").glob("*/*.osu"))[:50]

@pytest.mark.parametrize("path", SAMPLES, ids=lambda p: p.parent.name)
def test_parses_without_error(path):
    c = parse_osu(path)
    assert c.key_count == 4
    assert len(c.notes) > 0
    assert all(0 <= n.lane < 4 for n in c.notes)
    assert all(n.end_ms is None or n.end_ms > n.time_ms for n in c.notes)
    assert c.timing_points, "uninherited timing point가 하나는 있어야 함"