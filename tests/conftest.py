"""Shared SWC fixtures covering both file conventions."""

from __future__ import annotations

import pytest

SPLIT_DENDRITE = """\
# a split-convention file: type column is uniformly 0
1 0 100.0 200.0 300.0 1.0 -1
2 0 110.0 200.0 300.0 1.0 1
3 0 120.0 210.0 300.0 1.0 2
"""

SPLIT_AXON = """\
1 0 100.0 200.0 300.0 1.0 -1
2 0 500.0 200.0 300.0 1.0 1
"""

COMBINED = """\
# a combined file with real SWC type codes
1 1 100.0 200.0 300.0 5.0 -1
2 2 400.0 200.0 300.0 1.0 1
3 2 800.0 200.0 300.0 1.0 2
4 3 105.0 205.0 300.0 1.0 1
5 3 110.0 210.0 300.0 1.0 4
"""


@pytest.fixture
def split_dendrite(tmp_path):
    p = tmp_path / "N009-785688-dendrite-JT.swc"
    p.write_text(SPLIT_DENDRITE)
    return p


@pytest.fixture
def split_axon(tmp_path):
    p = tmp_path / "N009-785688-axon-JT.swc"
    p.write_text(SPLIT_AXON)
    return p


@pytest.fixture
def combined(tmp_path):
    p = tmp_path / "N001-685221.swc"
    p.write_text(COMBINED)
    return p
