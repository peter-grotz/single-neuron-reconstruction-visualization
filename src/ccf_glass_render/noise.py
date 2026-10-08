"""Procedural surface irregularity for the glass material.

Sampled in object space so the ripple stays fixed to the brain rather than
swimming across it when the camera moves.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy import ndimage as ndi

FIELD_N = 96
"""Edge length of the tiling noise cubes."""


def _fractal(shape: tuple[int, ...], octaves: int, seed: int) -> np.ndarray:
    """Seamlessly tiling fractal noise; wrap-mode blur keeps it periodic."""
    rng = np.random.default_rng(seed)
    out = np.zeros(shape, np.float32)
    amp = 1.0
    for octave in range(octaves):
        n = rng.standard_normal(shape).astype(np.float32)
        n = ndi.gaussian_filter(n, 9.0 / (2**octave), mode="wrap")
        out += amp * (n / (n.std() + 1e-8))
        amp *= 0.55
    return (out / (out.std() + 1e-8)).astype(np.float32)


@lru_cache(maxsize=2)
def _field(kind: str) -> np.ndarray:
    """Cached noise cube; seeds are fixed so renders are reproducible."""
    octaves, seed = {"ripple": (3, 11), "grain": (2, 29)}[kind]
    return _fractal((FIELD_N,) * 3, octaves, seed)


def surface_noise(points_obj: np.ndarray, height: int, width: int,
                  ripple: float, grain: float,
                  f_ripple: float, f_grain: float) -> np.ndarray:
    """Sample the irregularity fields at object-space surface points.

    Parameters
    ----------
    points_obj
        ``(height * width, 3)`` surface positions in object voxel units.
    """

    def sample(field: np.ndarray, freq: float) -> np.ndarray:
        coords = (points_obj * freq) % FIELD_N
        return ndi.map_coordinates(field, coords.T, order=1, mode="wrap").reshape(height, width)

    return ripple * sample(_field("ripple"), f_ripple) + grain * sample(_field("grain"), f_grain)
