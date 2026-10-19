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


def test_canvas_grows_to_hold_cells_outside_the_brain(monkeypatch):
    """Axons leave the atlas, and the splat drops anything off-canvas silently.

    A spinal projection runs about 3 mm past the caudal bound, which clipped
    296 rows off a sagittal figure before the canvas was sized to the cells.
    """
    import numpy as np

    from ccf_glass_render.cache import BrainView
    from ccf_glass_render.figure import render_cells
    from ccf_glass_render.skeletons import Neuron

    brain = BrainView(
        base=np.ones((40, 30, 3), np.float32), rotation=np.eye(3),
        centre_in=np.zeros(3), centre_out=np.zeros(3),
        resolution_um=10.0, view="sagittal")
    # a two-node cell sitting well past the canvas in both axes
    far = Neuron("N001-1", np.array([[0.0, 0.0, 0.0], [700.0, 500.0, 0.0]]),
                 np.array([-1, 1]), np.array([1, 2]), np.array([2, 2]))
    image = render_cells(brain, {"N001-1": [far]}, RenderProfile(),
                         colours={"N001-1": np.array([1.0, 0.0, 0.0], np.float32)})
    # transposed on return, so the long axis is width
    assert image.shape[1] > 40, "canvas should have grown past the brain"
    assert image.shape[0] > 30


def test_cells_are_not_dimmed_by_a_structure():
    """A process crossing a structure boundary must not change brightness.

    Attenuating the part behind an overlay read as the cell fading rather
    than as depth, so the same cell is rendered identically with and without
    the overlay in front of it.
    """
    import numpy as np

    from ccf_glass_render.cache import BrainView, StructureView
    from ccf_glass_render.figure import render_cells
    from ccf_glass_render.skeletons import Neuron

    brain = BrainView(
        base=np.ones((40, 40, 3), np.float32), rotation=np.eye(3),
        centre_in=np.zeros(3), centre_out=np.zeros(3),
        resolution_um=10.0, view="sagittal")
    cell = Neuron("N001-1", np.array([[100.0, 100.0, 50.0], [250.0, 250.0, 50.0]]),
                  np.array([-1, 1]), np.array([1, 2]), np.array([2, 2]))
    cells = {"N001-1": [cell]}
    colours = {"N001-1": np.array([1.0, 0.0, 0.0], np.float32)}

    # an overlay sitting entirely in front of the cell, fully opaque to it
    overlay = StructureView(
        rgb=np.zeros((40, 40, 3), np.float32), alpha=np.zeros((40, 40), np.float32),
        front=np.zeros((40, 40), np.float32),
        transmittance=np.zeros((40, 40), np.float32),
        acronyms=("X",), view="sagittal", resolution_um=10.0)

    plain = render_cells(brain, cells, RenderProfile(), colours=colours)
    over = render_cells(brain, cells, RenderProfile(), colours=colours,
                        structures=overlay)
    ink_plain = (plain.min(axis=2) < 0.9).sum()
    ink_over = (over.min(axis=2) < 0.9).sum()
    assert ink_plain == ink_over, "the structure changed how much cell was drawn"
