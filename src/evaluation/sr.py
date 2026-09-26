"""Star rating computed locally with rosu-pp, a port of osu!'s own difficulty
calculation, so generated charts can be scored against their target SR (§4.11).

    star_rating(chart)       a Chart, e.g. decoded from tokens
    star_rating_file(path)   an .osu file as it is on disk
    ROSU_VERSION             SR changes whenever osu! reworks difficulty, so every
                             reported number names the version it came from

The osu! API's difficulty_rating is what training conditions on; how closely the
local number tracks it is measured by scripts/check_sr.py.
"""

from __future__ import annotations

from importlib.metadata import version
from pathlib import Path

import rosu_pp_py as rosu

from src.data.chart_parser import Chart
from src.data.chart_writer import osu_text

ROSU_VERSION = version("rosu-pp-py")


def _stars(beatmap: rosu.Beatmap) -> float:
    return float(rosu.Difficulty().calculate(beatmap).stars)


def star_rating(chart: Chart) -> float:
    return _stars(rosu.Beatmap(content=osu_text(chart)))


def star_rating_file(path: Path) -> float:
    return _stars(rosu.Beatmap(path=str(path)))
