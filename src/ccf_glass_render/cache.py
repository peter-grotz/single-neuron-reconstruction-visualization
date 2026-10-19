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

# --- structure material -------------------------------------------------
SPECULAR = 0.29
"""Highlight strength. The shell's own is brighter; a structure sits inside it
and competes if it is as shiny."""
FRESNEL_GAIN = 0.34
RIM_DARKEN = 1.0
"""Grazing-angle darkening, which is what draws the silhouette. A synthetic
2D outline was tried instead and broke into fragments, because the coverage it
was derived from is nearly binary at this blur."""
HUE_FADE = 0.40
"""How far the CCF colour is pulled toward white before shading."""
RAMP_LOW, RAMP_SPAN = 0.34, 0.30
"""Coverage ramp across the boundary, the same mapping the shell uses."""
COVERAGE_BLUR = 0.10
ALPHA_FACE, ALPHA_EDGE = 0.28, 0.97
"""Opacity face-on and edge-on."""


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
    """Location of a cached structure overlay; the name encodes its contents.

    Structures are resolved to their acronym first, so an overlay built as `MD`
    is found when it is asked for as `362`. Naming the file after whatever
    string the caller used would cache the same overlay twice and miss it on
    the other spelling.
    """
    from . import structures as st

    base = Path(root) if root else atlas.cache_dir() / "views"
    base.mkdir(parents=True, exist_ok=True)
    tag = "-".join(sorted(st.lookup(a).acronym for a in acronyms))
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
    padded = (out_shape[0], out_shape[1] + pad, out_shape[2])
    # Only the part of the frame the structure occupies is resampled, and the
    # result stays in RAM. Rotating the whole frame into a memory-mapped file
    # -- which the brain needs, because it fills the frame -- turned a build
    # that should take seconds into tens of minutes of disk writes.
    vol = np.zeros(padded, np.float32)
    vol[:, : out_shape[1], :] = atlas.rotate_region(
        field, rot, centre_in, centre_out, out_shape)
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
    """Shade a structure as a glassy body, plus its depth and opacity.

    The material borrows from the shell -- a specular highlight, a Fresnel
    term that makes the surface clear face-on and dense edge-on, and the
    shell's own grazing-angle darkening for the silhouette -- but takes its
    hue from a direct tint rather than from absorption. Absorbing the colour
    through the body makes the hue track thickness, so a thin structure comes
    out pale at the edges and saturated through the middle, which reads as
    inconsistent rather than as depth.

    The surface is solved the way the shell's is, with sub-voxel depth and
    real normals. The earlier shortcut -- an integer argmax and a clipped
    alpha -- terraced the shading into bands on a body only tens of voxels
    thick and left 93% of MD's covered pixels at one alpha.
    """
    from .glass import _surface_normals

    height, width, depth = field.shape
    thickness_mm = field.sum(axis=2) * voxel_um / 1000.0
    inside = field > 0.5
    hit = inside.any(axis=2)
    first = np.argmax(inside, axis=2)
    del inside

    rows, cols = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")

    # where the smoothed field actually crosses 0.5, between two voxels
    below = np.clip(first - 1, 0, depth - 1)
    low = field[rows, cols, below]
    high = field[rows, cols, first]
    gap = high - low
    frac = np.where(np.abs(gap) > 1e-6,
                    (0.5 - low) / np.where(np.abs(gap) > 1e-6, gap, 1.0), 0.0)
    front = np.where(hit, first - 1 + np.clip(frac, 0, 1), np.inf).astype(np.float32)

    normal = _surface_normals(field, first, rows, cols, 1.0 * scale)
    key = np.array([-0.35, -0.86, -0.38])
    key = key / np.linalg.norm(key)
    lambert = np.clip(normal @ key, 0, 1)
    facing = np.clip(np.abs(normal @ np.array([0.0, 0.0, 1.0])), 0, 1)
    fresnel = (1.0 - facing) ** 4
    rim = (1.0 - facing) ** 8            # the shell's own silhouette term
    half = key + np.array([0.0, 0.0, -1.0])
    half = half / np.linalg.norm(half)
    specular = np.clip(normal @ half, 0, 1)

    hue = tint * (1.0 - HUE_FADE) + HUE_FADE
    lit = 0.62 + 0.42 * lambert + FRESNEL_GAIN * fresnel
    rgb = np.clip(hue[None, None, :] * lit[..., None]
                  + SPECULAR * (specular**40)[..., None]
                  + 0.10 * fresnel[..., None], 0, 1)
    rgb = (rgb * (1.0 - RIM_DARKEN * rim)[..., None]).astype(np.float32)

    # ramp coverage across the boundary, as the shell does, so the silhouette
    # stays crisp once blurred
    coverage = ndi.gaussian_filter(
        np.clip((field.max(axis=2) - RAMP_LOW) / RAMP_SPAN, 0, 1) * hit,
        COVERAGE_BLUR * scale)
    # clear face-on, dense at a grazing edge: what makes it read as glass
    alpha = (coverage * (ALPHA_FACE + (ALPHA_EDGE - ALPHA_FACE) * fresnel)
             * (opacity / 0.55)).astype(np.float32)
    transmittance = np.exp(-1.6 * opacity * thickness_mm).astype(np.float32)
    return {"rgb": rgb, "alpha": np.clip(alpha, 0, 1),
            "front": front, "transmittance": transmittance}
