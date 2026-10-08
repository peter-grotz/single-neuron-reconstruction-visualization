"""Rendering neurons as z-buffered 3D tubes.

Each node becomes a sphere splat; two streaming passes resolve the nearest
surface and then shade it, so the fragments of a 150k-node axon never all have
to be resident at once.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

LIGHT = np.array([-0.40, -0.65, -0.65])
"""Key light, toward the viewer from upper-left."""
AMBIENT, DIFFUSE, SPECULAR, SHININESS = 0.45, 0.60, 0.30, 24.0


def densify(points: np.ndarray, parent: np.ndarray, node_id: np.ndarray,
            step: float) -> np.ndarray:
    """Resample along parent links so splatted tubes come out continuous.

    Nodes whose parent is absent (root, or trimmed by a compartment filter)
    start a new run rather than drawing a segment across the gap.
    """
    index = {int(n): i for i, n in enumerate(node_id)}
    starts, ends = [], []
    for i, p in enumerate(parent):
        j = index.get(int(p))
        if j is not None:
            starts.append(j)
            ends.append(i)
    if not starts:
        return points
    a, b = points[np.array(starts)], points[np.array(ends)]
    length = np.linalg.norm(b - a, axis=1)
    steps = np.maximum(np.ceil(length / step).astype(int), 1)
    out = [points]
    for k in range(1, int(steps.max()) + 1):
        sel = steps >= k
        t = (k / steps[sel])[:, None]
        out.append(a[sel] + (b[sel] - a[sel]) * t)
    return np.vstack(out)


def _disc(radius: float) -> list[tuple[int, int, float]]:
    """Offsets inside a sphere splat, with the height of its surface."""
    k = int(np.ceil(radius))
    return [
        (di, dj, float(np.sqrt(max(radius * radius - (di * di + dj * dj), 0.0))))
        for di in range(-k, k + 1)
        for dj in range(-k, k + 1)
        if di * di + dj * dj <= radius * radius
    ]


def splat(items: list[tuple[np.ndarray, np.ndarray]], shape: tuple[int, int],
          radius: float):
    """Z-buffered sphere splatting of coloured point sets.

    Parameters
    ----------
    items
        ``(points, rgb)`` pairs; points are ``(n, 3)`` in supersampled pixels
        with the third axis as depth.

    Returns
    -------
    depth, normal, colour, mask
    """
    height, width = shape
    r = max(radius, 0.7)
    offsets = _disc(r)
    depth = np.full((height, width), np.inf, np.float32)

    for points, _ in items:
        rows = np.rint(points[:, 0]).astype(np.int32)
        cols = np.rint(points[:, 1]).astype(np.int32)
        z = points[:, 2].astype(np.float32)
        for di, dj, dz in offsets:
            yi, yj = rows + di, cols + dj
            ok = (yi >= 0) & (yi < height) & (yj >= 0) & (yj < width)
            np.minimum.at(depth, (yi[ok], yj[ok]), z[ok] - dz)

    normal = np.zeros((height, width, 3), np.float32)
    colour = np.zeros((height, width, 3), np.float32)
    for points, rgb in items:
        rows = np.rint(points[:, 0]).astype(np.int32)
        cols = np.rint(points[:, 1]).astype(np.int32)
        z = points[:, 2].astype(np.float32)
        for di, dj, dz in offsets:
            yi, yj = rows + di, cols + dj
            ok = (yi >= 0) & (yi < height) & (yj >= 0) & (yj < width)
            yi, yj, d = yi[ok], yj[ok], z[ok] - dz
            win = d <= depth[yi, yj] + 1e-4
            if not win.any():
                continue
            wy, wx = yi[win], yj[win]
            normal[wy, wx] = np.array([di / r, dj / r, -dz / r], np.float32)
            colour[wy, wx] = rgb
    return depth, normal, colour, np.isfinite(depth)


def shade(normal: np.ndarray, colour: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Blinn-Phong shading of the splatted surface."""
    key = LIGHT / np.linalg.norm(LIGHT)
    view = np.array([0.0, 0.0, -1.0])
    half = key + view
    half /= np.linalg.norm(half)
    ndl = np.clip(np.einsum("ijk,k->ij", normal, key), 0, 1)
    ndh = np.clip(np.einsum("ijk,k->ij", normal, half), 0, 1)
    lit = (AMBIENT + DIFFUSE * ndl)[..., None] * colour
    lit = lit + (SPECULAR * ndh**SHININESS)[..., None]
    return np.clip(lit, 0, 1) * mask[..., None]


def box_down(image: np.ndarray, factor: int) -> np.ndarray:
    """Average-pool a supersampled image back to output resolution."""
    height, width = image.shape[:2]
    h2, w2 = height // factor, width // factor
    image = image[: h2 * factor, : w2 * factor]
    if image.ndim == 2:
        return image.reshape(h2, factor, w2, factor).mean((1, 3))
    return image.reshape(h2, factor, w2, factor, image.shape[2]).mean((1, 3))


def crop(image: np.ndarray, pad: int, background: float = 0.995) -> np.ndarray:
    """Trim uniform background, leaving `pad` pixels of margin."""
    ink = image.min(axis=2) < background
    if not ink.any():
        return image
    rows, cols = np.where(ink)
    # +1 because the stop index is exclusive: without it the margin is one
    # pixel short on the bottom and right, which the legacy renderer also did
    r0, r1 = max(int(rows.min()) - pad, 0), min(int(rows.max()) + pad + 1, image.shape[0])
    c0, c1 = max(int(cols.min()) - pad, 0), min(int(cols.max()) + pad + 1, image.shape[1])
    return image[r0:r1, c0:c1]


def composite(base: np.ndarray, items, supersample: int, radius: float,
              shade_mix: float, alpha_gain: float) -> np.ndarray:
    """Splat `items` over `base` and return the combined image."""
    height, width = base.shape[:2]
    _, normal, colour, mask = splat(items, (height * supersample, width * supersample), radius)
    lit = shade(normal, colour, mask)
    lit = np.clip(shade_mix * lit + (1.0 - shade_mix) * colour * mask[..., None], 0, 1)
    rgb = box_down(lit, supersample)
    alpha = np.clip(box_down(mask.astype(np.float32), supersample)[..., None] * alpha_gain, 0, 1)
    over = np.where(alpha > 0, rgb / np.maximum(alpha, 1e-6), 0)
    return np.clip(base * (1 - alpha) + over * alpha, 0, 1)


def smooth_up(image: np.ndarray, factor: int) -> np.ndarray:
    """Bicubic upscale of an RGB image, for compositing onto a coarse cache."""
    if factor == 1:
        return image
    return np.clip(
        np.stack([ndi.zoom(image[..., k], factor, order=3) for k in range(3)], -1), 0, 1)
