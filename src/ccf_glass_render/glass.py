"""The volumetric glass shader.

Glass reads as glass because of what it reflects and refracts, not because of
how it is lit, so the brain is given a procedural studio environment and then
shaded almost entirely by Fresnel mixing of reflection against a refracted
backdrop.

Surface normals are differenced only at the first and last hit slices. The
earlier implementation took ``np.gradient`` of the whole field and then indexed
it at exactly those two surfaces, which is the same number -- but at 10 um that
is three 4.8 GB temporaries, and it made the render impossible on a laptop.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from .noise import surface_noise
from .profile import GlassProfile, array

VIEW_DIR = np.array([0.0, 0.0, 1.0])
"""Camera ray, pointing into the screen."""


def _environment(direction: np.ndarray, glass: GlassProfile) -> np.ndarray:
    """Procedural studio: overhead softbox, back rim, graded sky, warm floor."""
    up = -direction[..., 1]
    sky = 0.26 + 0.31 * np.clip(up * 0.5 + 0.5, 0, 1) ** 0.75
    box = array(glass.softbox)
    rim = array(glass.rim)
    key = np.clip(direction @ (box / np.linalg.norm(box)), 0, 1) ** 22
    back = np.clip(direction @ (rim / np.linalg.norm(rim)), 0, 1) ** 9
    value = sky + 0.50 * key + 0.07 * back
    warm = np.clip(-up * 0.5 + 0.5, 0, 1)[..., None] * np.array([0.03, 0.01, -0.02])
    return np.clip(value[..., None] + warm, 0, 4)


def _surface_normals(field: np.ndarray, depth_index: np.ndarray,
                     rows: np.ndarray, cols: np.ndarray, sigma: float) -> np.ndarray:
    """Central differences of `field` at one hit surface, then smoothed.

    Matches ``np.gradient``: a two-sample span in the interior, one-sided at the
    array edge, without materialising a gradient volume.
    """
    height, width, depth = field.shape
    limits = (height, width, depth)

    def derivative(axis: int) -> np.ndarray:
        base = [rows, cols, depth_index]
        up = [np.clip(base[k] + (k == axis), 0, limits[k] - 1) for k in range(3)]
        down = [np.clip(base[k] - (k == axis), 0, limits[k] - 1) for k in range(3)]
        span = np.maximum(up[axis] - down[axis], 1)
        return (field[up[0], up[1], up[2]] - field[down[0], down[1], down[2]]) / span

    vec = -np.stack([derivative(0), derivative(1), derivative(2)], -1)
    vec = np.stack([ndi.gaussian_filter(vec[..., k], sigma) for k in range(3)], -1)
    return vec / np.maximum(np.linalg.norm(vec, axis=-1, keepdims=True), 1e-8)


def render_glass(field: np.ndarray, striation: np.ndarray, voxel_um: float,
                 frame: tuple, glass: GlassProfile, scale: float) -> np.ndarray:
    """Shade the glass brain to an RGB image.

    Parameters
    ----------
    field
        ``(height, width, depth)`` occupancy in the rotated frame, in [0, 1].
        May be memory-mapped; every access here is a contiguous axis-2 pass or
        a two-dimensional gather.
    striation
        ``(height, width)`` internal-striation term.
    frame
        ``(rotation, centre_in, centre_out)`` from :func:`atlas.build_affine`.
    scale
        Pixels per 20 um pixel, so pixel-denominated constants hold their
        apparent size as the grid gets finer.
    """
    height, width, depth = field.shape
    thickness_mm = field.sum(axis=2) * voxel_um / 1000.0

    inside = field > 0.5
    hit = inside.any(axis=2)
    first = np.argmax(inside, axis=2)
    last = depth - 1 - np.argmax(inside[:, :, ::-1], axis=2)
    del inside

    rows, cols = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    normal = _surface_normals(field, first, rows, cols, 1.0 * scale)
    far_normal = _surface_normals(field, last, rows, cols, 1.8 * scale)

    # object-space surface points, for noise that does not swim with the camera
    rotation, centre_in, centre_out = frame
    screen = np.stack([rows, cols, first], -1).reshape(-1, 3).astype(np.float32)
    obj = (rotation.T @ (screen - centre_out).T).T + centre_in
    # sampled in 20 um units so the pattern keeps its physical scale; the slope
    # is then rescaled because the pixels it is differenced over are smaller
    ripple = ndi.gaussian_filter(
        surface_noise(obj / scale, height, width, glass.ripple, glass.grain,
                      glass.f_ripple, glass.f_grain),
        0.6 * scale,
    )
    d_row, d_col = np.gradient(ripple)
    normal = normal + glass.bump * scale * np.stack(
        [d_row, d_col, np.zeros_like(d_row)], -1)
    normal /= np.maximum(np.linalg.norm(normal, axis=-1, keepdims=True), 1e-8)

    alpha = ndi.gaussian_filter(
        np.clip((field.max(axis=2) - 0.34) / 0.30, 0, 1) * hit, 0.7 * scale)
    backdrop = np.ones((height, width, 3), np.float32)

    facing = np.clip(np.abs(normal @ VIEW_DIR), 0, 1)
    fresnel = 0.04 + 0.96 * (1.0 - facing) ** 5
    rim_only = (1.0 - facing) ** 8

    # refraction displaces the backdrop, per channel for chromatic dispersion
    push = glass.ior_k * thickness_mm * 26.0 * scale
    channels = []
    for c, k in enumerate((1.0 - glass.dispersion, 1.0, 1.0 + glass.dispersion)):
        channels.append(ndi.map_coordinates(
            backdrop[..., c],
            [rows - push * normal[..., 1] * k, cols - push * normal[..., 0] * k],
            order=1, mode="nearest"))
    refracted = np.stack(channels, -1)

    transmit = np.exp(-glass.k_abs * thickness_mm)[..., None]
    refracted = refracted * transmit + array(glass.tint) * (1.0 - transmit)

    far_facing = (1.0 - np.clip(np.abs(far_normal @ VIEW_DIR), 0, 1)) ** 3.0
    refracted *= (1.0 - glass.back * far_facing * np.exp(-glass.k_occ * thickness_mm))[..., None]

    detail = ndi.gaussian_filter(striation, 0.7 * scale)
    detail = detail - ndi.gaussian_filter(detail, 4.0 * scale)
    refracted *= (1.0 + np.clip(detail * glass.striation, -0.22, 0.22))[..., None]

    reflected = VIEW_DIR - 2.0 * (normal @ VIEW_DIR)[..., None] * normal
    reflected /= np.maximum(np.linalg.norm(reflected, axis=-1, keepdims=True), 1e-8)
    mix = fresnel[..., None]
    colour = refracted * (1.0 - mix) + _environment(reflected, glass) * mix

    # grazing edges absorb more and refract light away from the camera; the
    # studio path brightens the rim, so darkening it back is an explicit term
    colour *= (1.0 - glass.edge * rim_only)[..., None]

    box = array(glass.softbox)
    specular = np.clip(normal @ (box / np.linalg.norm(box)), 0, 1)
    colour += (0.16 * specular**60)[..., None]
    colour += (0.018 * specular**12)[..., None] * np.array([0.97, 0.99, 1.0])

    coverage = alpha[..., None]
    return np.clip(backdrop * (1 - coverage) + np.clip(colour, 0, 1) * coverage,
                   0, 1).astype(np.float32)
