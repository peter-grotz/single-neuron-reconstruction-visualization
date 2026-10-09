"""Colour assignment: the bug here shipped a figure with duplicated hues."""

from __future__ import annotations

from dataclasses import replace

import matplotlib.colors as mcolors
import numpy as np
import pytest

from ccf_glass_render import palettes
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


def test_more_cells_than_a_fixed_scheme_cycles_it():
    """Every cell is still assigned; colours repeat once the scheme runs out."""
    cells = [f"N{i:03d}-1" for i in range(46)]
    colours = assign_colours(cells, RenderProfile())
    assert len(colours) == 46
    assert len({tuple(v) for v in colours.values()}) == len(palettes.FIXED["vivid"])


def test_a_generated_scheme_sizes_itself_to_the_figure():
    cells = [f"N{i:03d}-1" for i in range(24)]
    profile = replace(RenderProfile(), skeleton=replace(
        RenderProfile().skeleton, scheme="dark"))
    colours = assign_colours(cells, profile)
    assert len({tuple(v) for v in colours.values()}) == len(cells)


def test_explicit_palette_overrides_the_scheme():
    profile = replace(RenderProfile(), skeleton=replace(
        RenderProfile().skeleton, scheme="allen", palette=("#123456",)))
    colours = assign_colours(["N001-1"], profile)
    assert np.allclose(colours["N001-1"], mcolors.to_rgb("#123456"))


def test_explicit_palette_shorter_than_the_cells_cycles_too():
    profile = replace(RenderProfile(), skeleton=replace(
        RenderProfile().skeleton, palette=("#123456",)))
    colours = assign_colours(["N001-1", "N002-1"], profile)
    assert len(colours) == 2
    assert len({tuple(v) for v in colours.values()}) == 1


@pytest.mark.parametrize("scheme", palettes.names())
def test_every_scheme_renders_eight_distinct_cells(scheme):
    cells = [f"N{i:03d}-1" for i in range(8)]
    profile = replace(RenderProfile(), skeleton=replace(
        RenderProfile().skeleton, scheme=scheme))
    colours = assign_colours(cells, profile)
    assert len({tuple(v) for v in colours.values()}) == 8
