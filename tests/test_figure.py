"""Colour assignment: the bug here shipped a figure with duplicated hues."""

from __future__ import annotations

import pytest

from ccf_glass_render.figure import assign_colours
from ccf_glass_render.profile import RenderProfile


def test_every_cell_gets_a_distinct_colour():
    cells = [f"N{i:03d}-1" for i in range(8)]
    colours = assign_colours(cells, RenderProfile())
    assert len({tuple(v) for v in colours.values()}) == len(cells)


def test_assignment_is_deterministic():
    cells = [f"N{i:03d}-1" for i in range(8)]
    profile = RenderProfile()
    first = assign_colours(cells, profile)
    second = assign_colours(cells, profile)
    assert all((first[c] == second[c]).all() for c in cells)


def test_more_cells_than_palette_raises_rather_than_repeating():
    cells = [f"N{i:03d}-1" for i in range(9)]
    with pytest.raises(ValueError, match="palette"):
        assign_colours(cells, RenderProfile())
