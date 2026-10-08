"""CCF template fetch and occupancy construction.

The template is a large external artifact, so it is downloaded to a cache
directory on first use and checksummed rather than vendored in the repo.
"""

from __future__ import annotations

import hashlib
import os
import urllib.request
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

from .profile import GlassProfile

TEMPLATE_URL = (
    "https://download.alleninstitute.org/informatics-archive/"
    "current-release/mouse_ccf/average_template/average_template_10.nrrd"
)
"""Official Allen distribution. NRRD, uint16, 1320 x 800 x 1140 at 10 um,
array axes (anterior-posterior, inferior-superior, left-right)."""

TEMPLATE_SHAPE = (1320, 800, 1140)
VOXEL_UM = 10.0
"""Native isotropic voxel size of the CCF template."""


def cache_dir() -> Path:
    """Where templates and prerendered views live; override with CCF_CACHE."""
    root = os.environ.get("CCF_CACHE") or (Path.home() / ".cache" / "ccf-glass-render")
    path = Path(root)
    path.mkdir(parents=True, exist_ok=True)
    return path


def template_array(source: str | Path | None = None) -> np.ndarray:
    """Return the template as a memory-mapped uint16 array.

    The published template is gzip-encoded NRRD, which cannot be read lazily,
    so it is decoded once into a plain ``.npy`` in the cache directory and
    memory-mapped thereafter. That keeps the downsampling pass able to walk the
    volume a slab at a time instead of holding 2.4 GB resident.

    Parameters
    ----------
    source
        Optional local NRRD or NIfTI to use instead of downloading. Must be the
        10 um average template at the documented shape; anything else raises.
    """
    raw = cache_dir() / "average_template_10.npy"
    if raw.exists():
        arr = np.load(raw, mmap_mode="r")
        _check_shape(arr)
        return arr

    src = Path(source) if source else _download(TEMPLATE_URL)
    if src.suffix in (".nrrd", ".nhdr"):
        import nrrd

        data, _ = nrrd.read(str(src))
    else:
        import nibabel as nib

        data = np.asanyarray(nib.load(str(src)).dataobj)
    data = np.ascontiguousarray(data, dtype=np.uint16)
    _check_shape(data)
    tmp = raw.with_suffix(".part.npy")
    np.save(tmp, data)
    tmp.replace(raw)
    del data
    return np.load(raw, mmap_mode="r")


def _check_shape(arr: np.ndarray) -> None:
    """Fail fast on a template that is not the 10 um CCF in the expected order."""
    if tuple(arr.shape) != TEMPLATE_SHAPE:
        raise ValueError(
            f"expected the 10 um CCF template with shape {TEMPLATE_SHAPE}, "
            f"got {tuple(arr.shape)} -- a different atlas or axis order would "
            "silently produce a wrongly oriented render"
        )


def _download(url: str, sha256: str | None = None) -> Path:
    """Fetch `url` into the cache directory once, optionally checksummed."""
    dest = cache_dir() / url.rsplit("/", 1)[-1]
    if not dest.exists():
        tmp = dest.with_suffix(dest.suffix + ".part")
        print(f"downloading {url}", flush=True)
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(dest)
    if sha256:
        got = hashlib.sha256(dest.read_bytes()).hexdigest()
        if got != sha256:
            raise ValueError(f"checksum mismatch for {dest}: expected {sha256}, got {got}")
    return dest


def _block_mean(arr: np.ndarray, down: int) -> np.ndarray:
    """Downsample the template by integer block mean, a slab at a time."""
    nx, ny, nz = arr.shape
    sx, sy, sz = nx // down, ny // down, nz // down
    acc = np.zeros((sx, sy, sz), np.float32)
    for i0 in range(0, sx * down, 200):
        i1 = min(i0 + 200, sx * down)
        sl = np.asarray(arr[i0:i1], dtype=np.float32)
        n = sl.shape[0] // down
        if n == 0:
            continue
        sl = sl[: n * down, : sy * down, : sz * down]
        acc[i0 // down : i0 // down + n] = sl.reshape(
            n, down, sy, down, sz, down
        ).mean((1, 3, 5))
    return acc


def _fill(mask: np.ndarray) -> np.ndarray:
    """Close and fill the brain mask in 3D, then slicewise along each axis.

    The slicewise pass matters: a 3D fill alone leaves ventricles and the
    interpeduncular fossa open where they connect to the outside in 3D but
    are enclosed in-plane, and those holes read as bright voids in the glass.
    """
    out = ndi.binary_fill_holes(ndi.binary_closing(mask, np.ones((3, 3, 3))))
    for axis in range(3):
        for i in range(out.shape[axis]):
            sel: list = [slice(None)] * 3
            sel[axis] = i
            out[tuple(sel)] = ndi.binary_fill_holes(out[tuple(sel)])
    return ndi.binary_closing(out, np.ones((3, 3, 3)))


def occupancy(
    down: int,
    glass: GlassProfile,
    out: np.ndarray | None = None,
    with_density: bool = True,
    template: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Build the smoothed brain occupancy field, and optionally the density.

    Parameters
    ----------
    down
        Integer downsample of the 10 um template; 1 is native.
    glass
        Supplies the intensity threshold and the smoothing sigma.
    out
        Optional destination for the occupancy -- pass a memory-mapped array at
        down=1, where the field is 4.8 GB and must not compete for RAM.
    template
        Optional local template, instead of the cached download.
    with_density
        Build the normalised intensity volume used for internal striations.
        Skipping it lets the raw accumulator be freed before the float32 copy
        the Gaussian needs, which is what keeps peak memory under control.

    Returns
    -------
    occupancy, density
        ``density`` is None when `with_density` is False.
    """
    acc = _block_mean(template_array(template), down)
    density = None
    if with_density:
        hi = max(float(np.percentile(acc[acc > glass.threshold], 99.0)), 1e-6)
        density = np.clip(acc / hi, 0.0, 1.6).astype(np.float32)

    mask = acc > glass.threshold
    if not with_density:
        del acc
    mask = _fill(mask)

    field = np.zeros(mask.shape, np.float32) if out is None else out
    np.copyto(field, mask, casting="unsafe")
    del mask
    ndi.gaussian_filter(field, glass.smooth * (2.0 / down), output=field)
    return field, density


def build_affine(shape: tuple[int, ...], yaw: float, pitch: float):
    """Rotation and the input/output centres that frame the whole volume.

    Returns ``(R, centre_in, centre_out, out_shape)``; `R` maps object-space
    voxel offsets to screen space, with the third axis pointing into the screen.
    """
    cy, sy = np.cos(np.radians(yaw)), np.sin(np.radians(yaw))
    cp, sp = np.cos(np.radians(pitch)), np.sin(np.radians(pitch))
    rot = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]]) @ np.array(
        [[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]]
    )
    corners = np.array(np.meshgrid([0, shape[0]], [0, shape[1]], [0, shape[2]])).reshape(3, -1).T
    centre_in = (np.array(shape) - 1) / 2.0
    rc = (rot @ (corners - centre_in).T).T
    out_shape = np.ceil(rc.max(0) - rc.min(0)).astype(int)
    return rot, centre_in, (out_shape - 1) / 2.0, tuple(int(v) for v in out_shape)


def rotate(vol: np.ndarray, rot, centre_in, centre_out, out_shape, out=None) -> np.ndarray:
    """Resample `vol` into the rotated frame, optionally into a given array."""
    inv = rot.T
    return ndi.affine_transform(
        vol, inv, offset=centre_in - inv @ centre_out,
        output_shape=None if out is not None else out_shape,
        output=out, order=1, cval=0.0,
    )
