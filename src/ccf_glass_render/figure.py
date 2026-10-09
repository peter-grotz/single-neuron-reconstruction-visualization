"""Compositing neurons onto a cached glass view and writing the figure."""

from __future__ import annotations

import random
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from . import palettes, splat  # noqa: E402
from .cache import BrainView, StructureView  # noqa: E402
from .profile import RenderProfile, pixel_scale  # noqa: E402
from .skeletons import Neuron  # noqa: E402


def assign_colours(cell_ids: list[str], profile: RenderProfile) -> dict[str, np.ndarray]:
    """One colour per cell, drawn from the profile's scheme.

    Up to the scheme's length every cell gets a distinct colour, and that is
    asserted: sampling *with* replacement silently collided once, giving the
    same hue to three cells in an eight-cell draw. Past that length a fixed
    scheme cycles by design, so repeats are expected rather than a bug.
    """
    skeleton = profile.skeleton
    if skeleton.palette:
        pool = palettes.cycle(list(skeleton.palette), len(cell_ids), skeleton.seed)
    else:
        pool = palettes.palette(skeleton.scheme, len(cell_ids), skeleton.seed)
    chosen = random.Random(skeleton.seed).sample(pool, len(pool))
    out = {c: np.array(mcolors.to_rgb(k), np.float32)
           for c, k in zip(cell_ids, chosen, strict=True)}
    if len(cell_ids) <= len(set(pool)) and len({tuple(v) for v in out.values()}) != len(out):
        # while colours are still unique they must stay unique; beyond that the
        # scheme is deliberately cycling and repeats are expected
        raise AssertionError("colour collision")
    return out


def render_cells(brain: BrainView, cells: dict[str, list[Neuron]],
                 profile: RenderProfile,
                 colours: dict[str, np.ndarray] | None = None,
                 target_um: float | None = None,
                 structures: StructureView | None = None) -> np.ndarray:
    """Composite `cells` onto `brain` and return the RGB image.

    Parameters
    ----------
    target_um
        Output pixel size. Defaults to the brain's own resolution; a finer
        value upscales the cached glass bicubically, which adds no real detail
        and is only worth it when no finer cache exists.
    structures
        Optional CCF structure overlay, drawn between the glass and the
        neurons. Neurons deeper than its near surface are dimmed by its
        transmittance, so a cell inside a nucleus reads as inside it.
    """
    target_um = target_um or brain.resolution_um
    factor = int(round(brain.resolution_um / target_um)) or 1
    base = splat.smooth_up(brain.base.astype(np.float32), factor)
    if structures is not None:
        base = _lay_structure(base, structures, factor)
    height, width = base.shape[:2]

    sup = profile.skeleton.supersample
    scale = pixel_scale(brain.resolution_um) * factor
    radius = max(0.6 * profile.skeleton.thickness * sup * scale, 0.9)

    colours = colours or assign_colours(sorted(cells), profile)
    items = []
    for cell_id, neurons in sorted(cells.items()):
        rgb = colours[cell_id]
        for neuron in neurons:
            pixels = brain.project(neuron.xyz) * np.array([sup * factor, sup * factor, factor])
            items.append(
                (splat.densify(pixels, neuron.parent, neuron.node_id,
                               step=max(radius * 0.5, 0.45)), rgb))

    behind = None
    if structures is not None:
        behind = (splat.smooth_up_scalar(structures.front, factor) * factor,
                  splat.smooth_up_scalar(structures.transmittance, factor))
    image = splat.composite(base, items, sup, radius,
                            profile.skeleton.shade_mix, profile.skeleton.alpha_gain,
                            behind=behind)
    return np.transpose(image, (1, 0, 2))


def _lay_structure(base: np.ndarray, structures: StructureView, factor: int) -> np.ndarray:
    """Alpha-composite the structure overlay onto the glass brain."""
    rgb = splat.smooth_up(structures.rgb.astype(np.float32), factor)
    alpha = splat.smooth_up_scalar(structures.alpha, factor)[..., None]
    return np.clip(base * (1 - alpha) + rgb * alpha, 0, 1)


def save(image: np.ndarray, stem: Path, dpi: int = 300,
         formats: tuple[str, ...] = ("png", "svg"), pad: int = 24) -> list[Path]:
    """Write a borderless figure in each requested format."""
    image = splat.crop(image, pad=pad)
    height, width = image.shape[:2]
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi, facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(image, interpolation="lanczos")
    ax.axis("off")
    stem.parent.mkdir(parents=True, exist_ok=True)
    written = []
    for suffix in formats:
        path = stem.with_suffix(f".{suffix}")
        fig.savefig(path, dpi=dpi, facecolor="white")
        written.append(path)
    plt.close(fig)
    return written
