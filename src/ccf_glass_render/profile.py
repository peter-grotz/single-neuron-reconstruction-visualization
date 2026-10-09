"""Render profiles: every tunable in one hashable object.

A figure is reproducible from three strings -- the input asset id, the package
version, and a profile hash -- so all shader constants, palette and geometry
live here rather than as module-level globals.
"""

from __future__ import annotations

import hashlib
import json
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

# Camera angles in (yaw, pitch) degrees, applied to the CCF array axes.
VIEWS: dict[str, tuple[float, float]] = {
    "sagittal": (0.0, 0.0),
    "coronal": (90.0, 0.0),
    "horizontal": (0.0, 90.0),
    "iso": (-38.0, 20.0),
}
"""The three anatomical planes, plus a three-quarter view."""

VIEW_ALIASES: dict[str, str] = {
    "dorsal": "horizontal", "axial": "horizontal", "transverse": "horizontal",
    "lateral": "sagittal",
    # "saggital" is the usual misspelling and reads as correct at a glance, so
    # it is accepted rather than failing a run over a doubled letter
    "saggital": "sagittal", "sagital": "sagittal", "saggittal": "sagittal",
    "horizonal": "horizontal", "coronol": "coronal",
}
"""Alternative names. `dorsal` was the earlier name for the horizontal plane,
and the common misspellings resolve rather than failing a run."""


def canonical_view(name: str) -> str:
    """Resolve a view name or alias, or say what the choices are."""
    key = name.strip().lower()
    key = VIEW_ALIASES.get(key, key)
    if key not in VIEWS:
        raise ValueError(f"unknown view {name!r}; choose from {sorted(VIEWS)}")
    return key


@dataclass(frozen=True)
class GlassProfile:
    """Shader constants for the volumetric glass pass.

    Lengths denominated in pixels (blur sigmas, refraction push, bump slope)
    are defined at 20 um and rescaled by ``2 / resolution_um * 10`` at render
    time, so a 10 um render reproduces the 20 um look at finer sampling rather
    than being a differently-tuned image.
    """

    k_abs: float = 0.012
    """Absorption per mm."""
    k_occ: float = 0.027
    """Attenuation per mm of structure seen through the glass."""
    tint: tuple[float, float, float] = (0.84, 0.95, 0.93)
    ior_k: float = 0.34
    """Refraction strength, as background displacement."""
    dispersion: float = 0.045
    back: float = 0.24
    """Far-wall silhouette contour strength."""
    edge: float = 0.55
    """Grazing-angle darkening, so the silhouette reads against white."""
    ripple: float = 0.35
    """Broad surface waviness, as in cast glass."""
    grain: float = 0.045
    f_ripple: float = 0.230
    f_grain: float = 0.900
    bump: float = 1.5
    """How strongly surface irregularity tilts the normal."""
    striation: float = 0.38
    """Internal striations from template intensity."""
    softbox: tuple[float, float, float] = (-0.35, -0.86, -0.38)
    rim: tuple[float, float, float] = (0.62, 0.10, 0.78)
    smooth: float = 1.9
    """Gaussian sigma on the occupancy mask, in 20 um voxels."""
    threshold: float = 6.0
    """Template intensity above which a voxel counts as brain."""


@dataclass(frozen=True)
class SkeletonProfile:
    """Geometry and colour for the neuron pass."""

    thickness: float = 1.15
    """Tube radius in 20 um pixels; held constant in apparent size."""
    supersample: int = 3
    shade_mix: float = 0.38
    """Weight of lit shading vs flat colour; 0 is wholly flat."""
    alpha_gain: float = 1.5
    seed: int = 3
    scheme: str = "vivid"
    """Named colour scheme; see :mod:`ccf_glass_render.palettes`."""
    palette: tuple[str, ...] = ()
    """Explicit hex colours, overriding `scheme` when non-empty."""


@dataclass(frozen=True)
class RenderProfile:
    """A complete, hashable description of one figure's appearance."""

    name: str = "vivid"
    glass: GlassProfile = field(default_factory=GlassProfile)
    skeleton: SkeletonProfile = field(default_factory=SkeletonProfile)

    @classmethod
    def from_toml(cls, path: str | Path) -> RenderProfile:
        """Load a profile, falling back to the default for any absent key."""
        raw = tomllib.loads(Path(path).read_text())
        return cls(
            name=raw.get("name", Path(path).stem),
            glass=GlassProfile(**raw.get("glass", {})),
            skeleton=SkeletonProfile(**_tuple_palette(raw.get("skeleton", {}))),
        )

    def digest(self) -> str:
        """Stable 12-character hash of every tunable, for provenance."""
        blob = json.dumps(asdict(self), sort_keys=True, default=list)
        return hashlib.sha256(blob.encode()).hexdigest()[:12]


def _tuple_palette(d: dict) -> dict:
    """TOML gives lists; the dataclass is frozen so sequences must be tuples."""
    out = dict(d)
    if "palette" in out:
        out["palette"] = tuple(out["palette"])
    return out


def pixel_scale(resolution_um: float) -> float:
    """Factor converting a 20 um pixel length to `resolution_um` pixels."""
    return 20.0 / resolution_um


def array(v: tuple[float, ...]) -> np.ndarray:
    """Profile triples are tuples so the dataclass can hash; shaders want arrays."""
    return np.asarray(v, dtype=np.float64)
