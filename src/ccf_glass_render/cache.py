"""Prerendering the glass brain, once per view, to a reusable cache.

The glass pass does not depend on which neurons are drawn, so it is rendered
once and stored. Compositing cells onto a cached view takes seconds; building
the view at 10 um takes minutes and roughly 10 GB of scratch disk.

At 10 um the occupancy and its rotation are 4.8 GB each. Holding both in RAM
made ``affine_transform``'s scattered reads thrash swap -- about 1% of the
rotation in five minutes on a 19 GB machine. Both therefore live in
memory-mapped files, so the pages are file-backed and the OS evicts them to the
page cache instead of to swap.
"""

from __future__ import annotations

import gc
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

from . import atlas
from .glass import render_glass
from .profile import VIEWS, GlassProfile, RenderProfile, canonical_view, pixel_scale

PAD_FRACTION = 0.10
"""Headroom added below the object, as a fraction of its height."""


@dataclass(frozen=True)
class BrainView:
    """A prerendered glass brain and the camera that produced it."""

    base: np.ndarray
    rotation: np.ndarray
    centre_in: np.ndarray
    centre_out: np.ndarray
    resolution_um: float
    view: str

    @property
    def down(self) -> int:
        """Template downsample factor this view was built at."""
        return int(round(self.resolution_um / atlas.VOXEL_UM))

    def project(self, xyz_um: np.ndarray) -> np.ndarray:
        """Map CCF micron coordinates into this view's pixel frame."""
        voxels = xyz_um / self.resolution_um
        return (self.rotation @ (voxels - self.centre_in).T).T + self.centre_out


def cache_path(view: str, resolution_um: float, root: Path | None = None) -> Path:
    """Location of the cached view; aliases resolve to the canonical name."""
    base = Path(root) if root else atlas.cache_dir() / "views"
    base.mkdir(parents=True, exist_ok=True)
    return base / f"{canonical_view(view)}_{int(resolution_um)}um.npz"


def load_view(view: str, resolution_um: float, root: Path | None = None) -> BrainView:
    """Read a cached view, with a clear error if it has not been built."""
    path = cache_path(view, resolution_um, root)
    if not path.exists():
        raise FileNotFoundError(
            f"no cached brain for view={view} at {int(resolution_um)} um. "
            f"Build it with: ccf-render brain --views {view} "
            f"--resolution {int(resolution_um)}"
        )
    z = np.load(path)
    return BrainView(
        base=z["base"], rotation=z["rotation"], centre_in=z["centre_in"],
        centre_out=z["centre_out"], resolution_um=float(z["resolution_um"]),
        view=str(z["view"]),
    )


def _memmap(directory: Path, name: str, shape) -> np.ndarray:
    """A zeroed memory-mapped float32 array -- file-backed, never swapped."""
    directory.mkdir(parents=True, exist_ok=True)
    return np.lib.format.open_memmap(
        directory / f"{name}.npy", mode="w+", dtype=np.float32,
        shape=tuple(int(v) for v in shape))


def _striation(view: str, glass: GlassProfile, template: Path | None) -> np.ndarray:
    """Internal-striation map, always built at 20 um.

    It is blurred at sigma 0.7 and high-passed at sigma 4.0 before use, so it
    carries no detail a 10 um rebuild would add, and skipping that rebuild
    frees a second full-size volume.
    """
    yaw, pitch = VIEWS[view]
    occ, density = atlas.occupancy(2, glass, with_density=True, template=template)
    rot, centre_in, centre_out, shape = atlas.build_affine(occ.shape, yaw, pitch)
    vol = atlas.rotate(occ, rot, centre_in, centre_out, shape)
    tex = atlas.rotate(density, rot, centre_in, centre_out, shape)
    del occ, density
    gc.collect()
    out = (tex * vol).sum(axis=2) / np.maximum(vol.sum(axis=2), 1e-6)
    del vol, tex
    gc.collect()
    return out.astype(np.float32)


def build_view(view: str, resolution_um: float, profile: RenderProfile,
               root: Path | None = None, scratch: Path | None = None,
               template: Path | None = None) -> Path:
    """Render one glass view and cache it. Returns the cache path."""
    view = canonical_view(view)
    down = int(round(resolution_um / atlas.VOXEL_UM))
    if down < 1 or abs(down * atlas.VOXEL_UM - resolution_um) > 1e-6:
        raise ValueError(f"resolution must be a multiple of {atlas.VOXEL_UM} um")

    scratch = Path(scratch) if scratch else atlas.cache_dir() / "scratch"
    yaw, pitch = VIEWS[view]

    print(f"[{view}] striation map at 20 um", flush=True)
    striation = _striation(view, profile.glass, template)

    print(f"[{view}] occupancy at {int(resolution_um)} um", flush=True)
    shape = tuple(n // down for n in atlas.TEMPLATE_SHAPE)
    occ = _memmap(scratch, "occupancy", shape)
    atlas.occupancy(down, profile.glass, out=occ, with_density=False, template=template)
    occ.flush()

    rot, centre_in, centre_out, out_shape = atlas.build_affine(shape, yaw, pitch)
    pad = max(int(PAD_FRACTION * out_shape[1]), 12)
    print(f"[{view}] rotating {out_shape}", flush=True)
    vol = _memmap(scratch, "rotated", (out_shape[0], out_shape[1] + pad, out_shape[2]))
    atlas.rotate(occ, rot, centre_in, centre_out, out_shape,
                 out=vol[:, : out_shape[1], :])
    vol.flush()
    del occ
    gc.collect()

    height, width = vol.shape[:2]
    striation = ndi.zoom(
        striation, (height / striation.shape[0], width / striation.shape[1]), order=1)

    print(f"[{view}] shading {height} x {width}", flush=True)
    base = render_glass(vol, striation, resolution_um, (rot, centre_in, centre_out),
                        profile.glass, pixel_scale(resolution_um))
    del vol
    gc.collect()
    shutil.rmtree(scratch, ignore_errors=True)

    path = cache_path(view, resolution_um, root)
    np.savez_compressed(
        path, base=base, rotation=rot, centre_in=centre_in, centre_out=centre_out,
        resolution_um=np.float32(resolution_um), view=view,
        profile=profile.digest())
    print(f"  cached {path}  base {base.shape[:2]}  "
          f"{path.stat().st_size / 1e6:.0f} MB", flush=True)
    return path


@dataclass(frozen=True)
class StructureView:
    """A prerendered CCF structure in the same frame as a :class:`BrainView`.

    Attributes
    ----------
    rgb, alpha
        Premultiplied-ready colour and coverage of the structure's near surface.
    front
        Depth of the near surface in view pixels, or +inf where it is absent.
        Neurons deeper than this sit behind the structure.
    transmittance
        Fraction of a neuron's colour that survives being seen through the
        structure, from Beer-Lambert absorption along its thickness.
    """

    rgb: np.ndarray
    alpha: np.ndarray
    front: np.ndarray
    transmittance: np.ndarray
    acronyms: tuple[str, ...]
    view: str
    resolution_um: float


def structure_cache_path(view: str, resolution_um: float, acronyms: list[str],
                         root: Path | None = None) -> Path:
    """Location of a cached structure overlay; the name encodes its contents."""
    base = Path(root) if root else atlas.cache_dir() / "views"
    base.mkdir(parents=True, exist_ok=True)
    tag = "-".join(sorted(acronyms))
    return base / f"{canonical_view(view)}_{int(resolution_um)}um_struct_{tag}.npz"


def load_structures(view: str, resolution_um: float, acronyms: list[str],
                    root: Path | None = None) -> StructureView:
    """Read a cached structure overlay, or say how to build it."""
    path = structure_cache_path(view, resolution_um, acronyms, root)
    if not path.exists():
        names = ",".join(sorted(acronyms))
        raise FileNotFoundError(
            f"no cached overlay for {names} in view={view} at "
            f"{int(resolution_um)} um. Build it with: ccf-render structures "
            f"--structure {names} --views {view} --resolution {int(resolution_um)}")
    z = np.load(path)
    return StructureView(
        rgb=z["rgb"], alpha=z["alpha"], front=z["front"],
        transmittance=z["transmittance"], acronyms=tuple(z["acronyms"]),
        view=str(z["view"]), resolution_um=float(z["resolution_um"]))


def build_structures(view: str, resolution_um: float, acronyms: list[str],
                     colour: str | None = None, opacity: float = 0.55,
                     root: Path | None = None, scratch: Path | None = None) -> Path:
    """Render CCF structures in `view` and cache them as an overlay.

    The structure is shaded as a soft translucent solid rather than as glass:
    it sits inside the brain, so a second full dielectric treatment makes the
    two surfaces compete and neither reads.
    """
    from matplotlib.colors import to_rgb

    from . import structures as st

    view = canonical_view(view)
    down = int(round(resolution_um / atlas.VOXEL_UM))
    scratch = Path(scratch) if scratch else atlas.cache_dir() / "scratch"
    yaw, pitch = VIEWS[view]

    print(f"[{view}] structure mask for {sorted(acronyms)} at "
          f"{int(resolution_um)} um", flush=True)
    field = st.mask(acronyms, down)

    rot, centre_in, centre_out, out_shape = atlas.build_affine(field.shape, yaw, pitch)
    pad = max(int(PAD_FRACTION * out_shape[1]), 12)
    vol = _memmap(scratch, "structure", (out_shape[0], out_shape[1] + pad, out_shape[2]))
    atlas.rotate(field, rot, centre_in, centre_out, out_shape,
                 out=vol[:, : out_shape[1], :])
    vol.flush()
    del field
    gc.collect()

    tint = np.asarray(to_rgb(colour or st.default_colour(acronyms)), np.float32)
    overlay = _shade_structure(vol, tint, opacity, resolution_um,
                               pixel_scale(resolution_um))
    del vol
    gc.collect()
    shutil.rmtree(scratch, ignore_errors=True)

    path = structure_cache_path(view, resolution_um, acronyms, root)
    np.savez_compressed(path, acronyms=np.array(sorted(acronyms)), view=view,
                        resolution_um=np.float32(resolution_um), **overlay)
    print(f"  cached {path}  {path.stat().st_size / 1e6:.1f} MB", flush=True)
    return path


def _shade_structure(field: np.ndarray, tint: np.ndarray, opacity: float,
                     voxel_um: float, scale: float) -> dict[str, np.ndarray]:
    """Soft shading of a structure's near surface, plus its depth and opacity."""
    height, width, depth = field.shape
    thickness_mm = field.sum(axis=2) * voxel_um / 1000.0
    inside = field > 0.5
    hit = inside.any(axis=2)
    first = np.argmax(inside, axis=2).astype(np.float32)
    del inside

    rows, cols = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    peak = field.max(axis=2)
    coverage = ndi.gaussian_filter(np.clip((peak - 0.3) / 0.35, 0, 1) * hit, 0.8 * scale)

    # shape reads from a shallow relief of the near surface rather than a full
    # normal solve; the structure is a secondary cue, not the subject
    relief = ndi.gaussian_filter(np.where(hit, first, np.nan_to_num(first.max())), 1.5 * scale)
    d_row, d_col = np.gradient(relief)
    lit = np.clip(0.72 + 0.9 * (-0.5 * d_row - 0.3 * d_col) / (1.0 + scale), 0.45, 1.25)

    rgb = np.clip(tint[None, None, :] * lit[..., None], 0, 1).astype(np.float32)
    alpha = (coverage * opacity).astype(np.float32)
    front = np.where(hit, first, np.inf).astype(np.float32)
    # a neuron seen through the structure keeps this much of its colour
    transmittance = np.exp(-1.6 * opacity * thickness_mm).astype(np.float32)
    return {"rgb": rgb, "alpha": alpha, "front": front, "transmittance": transmittance}
