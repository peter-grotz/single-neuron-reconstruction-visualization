"""Named colour schemes for per-cell reconstruction colouring.

Every scheme here is chosen to read on the white glass backdrop: mid-lightness
and high chroma, since pale hues disappear against it and very dark ones muddy
together once the tubes are shaded.

Schemes are either a fixed list, which caps how many cells a figure can hold,
or generated for any `n`.
"""

from __future__ import annotations

import colorsys
import random
from collections.abc import Callable

import numpy as np
from matplotlib import colormaps
from matplotlib.colors import to_hex

FIXED: dict[str, tuple[str, ...]] = {
    "vivid": (
        "#d4188c", "#8c3fe0", "#2b55e0", "#1f8fe5",
        "#11a88e", "#2fa814", "#e07b00", "#e01f33",
    ),
    "allen": (
        "#6464FF", "#FF6E00", "#00A59B", "#CD0F55",
        "#8246E1", "#DC9600", "#CDEB05", "#FF00FF",
    ),
    "okabe-ito": (
        "#E69F00", "#56B4E9", "#009E73", "#F0E442",
        "#0072B2", "#D55E00", "#CC79A7", "#000000",
    ),
    "tab10": (
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    ),
    "set1": (
        "#e41a1c", "#377eb8", "#4daf4a", "#984ea3",
        "#ff7f00", "#a65628", "#f781bf", "#999999",
    ),
    "dark2": (
        "#1b9e77", "#d95f02", "#7570b3", "#e7298a",
        "#66a61e", "#e6ab02", "#a6761d", "#666666",
    ),
}
"""Fixed schemes, keyed by name.

``vivid``
    The default. Hues spread around the circle at high chroma.
``allen``
    Allen Institute brand primaries and accents.
``okabe-ito``
    Colour-vision-deficiency safe; the usual choice for publication.
``tab10``, ``set1``, ``dark2``
    matplotlib's default and two ColorBrewer qualitative sets.
"""

DESCRIPTIONS: dict[str, str] = {
    "vivid": "high-chroma hues spread around the circle (default)",
    "allen": "Allen Institute brand primaries and accents",
    "okabe-ito": "colour-vision-deficiency safe, 8 colours",
    "tab10": "matplotlib default qualitative",
    "set1": "ColorBrewer Set1",
    "dark2": "ColorBrewer Dark2",
    "dark": "generated dark hues, alternating lightness (any n)",
    "viridis": "sampled from viridis (any n, perceptually ordered)",
    "turbo": "sampled from turbo (any n, wide hue range)",
    "husl": "evenly spaced hues at fixed lightness (any n)",
}


def _srgb_to_lab_l(rgb: tuple[float, float, float]) -> float:
    """CIELAB L* of an sRGB triple, for solving hues to equal lightness."""
    lin = np.asarray(rgb)
    lin = np.where(lin > 0.04045, ((lin + 0.055) / 1.055) ** 2.4, lin / 12.92)
    matrix = np.array([[0.4124, 0.3576, 0.1805],
                       [0.2126, 0.7152, 0.0722],
                       [0.0193, 0.1192, 0.9505]])
    xyz = (matrix @ lin) / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return float(116 * f[1] - 16)


def _at_lightness(hue: float, target: float, saturation: float) -> tuple[float, ...]:
    """Bisect HLS lightness until the hue hits a target L*.

    A plain HLS ramp makes yellow far lighter than blue at the same nominal
    lightness, so hues have to be solved individually to stay matched.
    """
    low, high = 0.02, 0.95
    for _ in range(40):
        mid = (low + high) / 2
        if _srgb_to_lab_l(colorsys.hls_to_rgb(hue, mid, saturation)) < target:
            low = mid
        else:
            high = mid
    return colorsys.hls_to_rgb(hue, (low + high) / 2, saturation)


def _dark(n: int) -> list[str]:
    """`n` dark, mutually distinct colours, alternating between two L* values.

    Alternating lightness means adjacent hues differ in two dimensions, which
    lifts the worst-case separation at n=8 from dE 18.5 to 33.8.
    """
    return [to_hex(_at_lightness(((i / n) + 0.03) % 1.0, 32.0 if i % 2 else 52.0, 0.95))
            for i in range(n)]


def _husl(n: int) -> list[str]:
    """Evenly spaced hues at a constant mid lightness."""
    return [to_hex(_at_lightness((i / n) % 1.0, 55.0, 0.9)) for i in range(n)]


def _from_colormap(name: str) -> Callable[[int], list[str]]:
    """Sample a continuous colormap at `n` evenly spaced points."""

    def sample(n: int) -> list[str]:
        cmap = colormaps[name]
        # trim the extremes: both ends of viridis and turbo are near-black or
        # near-white and vanish against the glass
        return [to_hex(cmap(x)) for x in np.linspace(0.08, 0.92, max(n, 1))]

    return sample


GENERATED: dict[str, Callable[[int], list[str]]] = {
    "dark": _dark,
    "husl": _husl,
    "viridis": _from_colormap("viridis"),
    "turbo": _from_colormap("turbo"),
}
"""Schemes that can produce any number of colours."""


def names() -> list[str]:
    """Every available scheme name, sorted."""
    return sorted(set(FIXED) | set(GENERATED))


def palette(scheme: str, n: int, seed: int | None = None) -> list[str]:
    """Return `n` hex colours for `scheme`.

    A fixed scheme shorter than `n` is cycled: each repeat is a fresh shuffle
    of the same colours, so the sequence does not simply repeat in order, and a
    repeat never lands next to its own previous use. Colours therefore recur
    once a figure holds more cells than the scheme has entries -- two cells can
    share a hue, and must be told apart by position rather than colour. Use a
    generated scheme when every cell needs a unique colour.

    Raises
    ------
    ValueError
        If the scheme is unknown.
    """
    if scheme in GENERATED:
        return GENERATED[scheme](n)
    if scheme not in FIXED:
        raise ValueError(f"unknown colour scheme {scheme!r}; choose from {names()}")
    fixed = list(FIXED[scheme])
    if n <= len(fixed):
        return fixed[:n]

    rng = random.Random(seed)
    out: list[str] = []
    while len(out) < n:
        block = fixed[:]
        rng.shuffle(block)
        if out and block[0] == out[-1] and len(block) > 1:
            # avoid the same colour straddling a block boundary
            block[0], block[1] = block[1], block[0]
        out += block
    return out[:n]


def cycle(colours: list[str], n: int, seed: int | None = None) -> list[str]:
    """Repeat `colours` to length `n`, reshuffling each pass.

    Shared by the explicit profile palette and the fixed schemes so both wrap
    the same way.
    """
    if n <= len(colours):
        return colours[:n]
    rng = random.Random(seed)
    out: list[str] = []
    while len(out) < n:
        block = colours[:]
        rng.shuffle(block)
        if out and block[0] == out[-1] and len(block) > 1:
            block[0], block[1] = block[1], block[0]
        out += block
    return out[:n]
