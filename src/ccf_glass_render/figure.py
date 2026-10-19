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
                 structures: StructureView | None = None,
                 extent: dict[str, list[Neuron]] | None = None) -> np.ndarray:
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

    # The cached brain is sized to the brain, but axons leave it -- a spinal
    # projection runs 3 mm past the caudal bound of the atlas -- and anything
    # outside the canvas is dropped by the splat without a word. Grow the
    # canvas to hold the cells before drawing them.
    # frame on `extent` when given: rendering one class at a time otherwise
    # frames each on its own cells, so a class with spinal projections comes
    # out far wider than one without and the two cannot be compared
    framing = extent if extent is not None else cells
    points = np.vstack([brain.project(n.xyz) for ns in framing.values() for n in ns])
    pad_top = max(int(np.ceil(-points[:, 0].min())) + 1, 0)
    pad_left = max(int(np.ceil(-points[:, 1].min())) + 1, 0)
    pad_bottom = max(int(np.ceil(points[:, 0].max())) + 2 - base.shape[0], 0)
    pad_right = max(int(np.ceil(points[:, 1].max())) + 2 - base.shape[1], 0)
    if any((pad_top, pad_left, pad_bottom, pad_right)):
        base = np.pad(base, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                      constant_values=1.0)
    offset = np.array([pad_top, pad_left, 0.0])
    height, width = base.shape[:2]

    sup = profile.skeleton.supersample
    scale = pixel_scale(brain.resolution_um) * factor
    radius = max(0.6 * profile.skeleton.thickness * sup * scale, 0.9)

    colours = colours or assign_colours(sorted(cells), profile)
    items = []
    for cell_id, neurons in sorted(cells.items()):
        rgb = colours[cell_id]
        for neuron in neurons:
            pixels = (brain.project(neuron.xyz) + offset) * np.array(
                [sup * factor, sup * factor, factor])
            items.append(
                (splat.densify(pixels, neuron.parent, neuron.node_id,
                               step=max(radius * 0.5, 0.45)), rgb))

    # Cells are drawn at full strength everywhere, including over a structure.
    # Attenuating the part of an axon that lies behind one made a single
    # process change brightness as it crossed the boundary, which reads as the
    # cell fading rather than as depth.
    behind = None
    image = splat.composite(base, items, sup, radius,
                            profile.skeleton.shade_mix, profile.skeleton.alpha_gain,
                            behind=behind)
    return np.transpose(image, (1, 0, 2))


def _lay_structure(base: np.ndarray, structures: StructureView, factor: int) -> np.ndarray:
    """Alpha-composite the structure overlay onto the glass brain."""
    rgb = splat.smooth_up(structures.rgb.astype(np.float32), factor)
    alpha = splat.smooth_up_scalar(structures.alpha, factor)[..., None]
    return np.clip(base * (1 - alpha) + rgb * alpha, 0, 1)


def crop_box(image: np.ndarray, pad: int = 24, background: float = 0.995):
    """The crop `save` would apply, so several figures can share one."""
    ink = image.min(axis=2) < background
    if not ink.any():
        return (0, image.shape[0], 0, image.shape[1])
    rows, cols = np.where(ink)
    return (max(int(rows.min()) - pad, 0),
            min(int(rows.max()) + pad + 1, image.shape[0]),
            max(int(cols.min()) - pad, 0),
            min(int(cols.max()) + pad + 1, image.shape[1]))


def save(image: np.ndarray, stem: Path, dpi: int = 300,
         formats: tuple[str, ...] = ("png", "svg"), pad: int = 24,
         box: tuple[int, int, int, int] | None = None) -> list[Path]:
    """Write a borderless figure in each requested format.

    `box` crops to a given rectangle rather than to this image's own content,
    which is what keeps a set of figures the same size.
    """
    if box is not None:
        r0, r1, c0, c1 = box
        image = image[r0:r1, c0:c1]
    else:
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
