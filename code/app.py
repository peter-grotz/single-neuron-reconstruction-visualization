#!/usr/bin/env python3
"""Code Ocean App Panel entry point.

Two modes, chosen by the panel's `mode` parameter:

``render``
    The normal case. Reconstruction data assets are attached in the UI; this
    finds their SWC files, reads the prerendered glass brain from the cache
    asset, and writes figures to ``/results``.
``build_cache``
    Run once to produce the cache asset itself: it prerenders the glass views
    and any structure overlays into ``/results/ccf_cache``, from which a data
    asset is created and then attached to every later run.

Every parameter is optional and named, so adding one to the App Panel later
does not break existing runs.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from pathlib import Path

DATA = Path("/data")
RESULTS = Path("/results")
SCRATCH = Path("/scratch")
CACHE_MOUNTS = ("ccf_cache", "ccf-cache", "cache")
"""Mount names treated as the prerendered cache rather than as neuron data."""


def _log(message: str) -> None:
    print(message, flush=True)


def _csv(value: str | None) -> list[str]:
    """Split a multi-valued panel field.

    Accepts commas, plus signs or whitespace. The App Builder does not document
    whether a List parameter's values are themselves comma separated, so a
    choice like ``sagittal,iso`` might be split into two options rather than
    offered as one. ``sagittal+iso`` is unambiguous either way, and is what the
    panel should use.
    """
    if not value:
        return []
    for separator in (",", "+"):
        value = value.replace(separator, " ")
    return [v.strip() for v in value.split() if v.strip()]


def _cache_root(explicit: str | None) -> Path:
    """Locate the prerendered cache: an explicit path, a mount, or a local dir.

    Falling back to a writable local directory means a capsule with no cache
    asset still works -- it just pays to build the views itself, which the log
    says plainly so a slow run is never a mystery.
    """
    if explicit:
        return Path(explicit)
    for name in CACHE_MOUNTS:
        candidate = DATA / name
        if (candidate / "views").is_dir():
            return candidate
    for candidate in sorted(DATA.glob("*")):
        if (candidate / "views").is_dir():
            return candidate
    _log("no cache asset found; views will be built in this run, which is slow. "
         "Attach the prerendered cache asset to avoid it.")
    return SCRATCH / "ccf_cache"


def _asset_dirs(cache_root: Path) -> list[Path]:
    """Every mounted data asset that is not the cache."""
    if not DATA.is_dir():
        return []
    out = []
    for path in sorted(DATA.glob("*")):
        if not path.is_dir() or path.resolve() == cache_root.resolve():
            continue
        if (path / "views").is_dir():
            continue
        out.append(path)
    return out


def _subject_of(path: Path) -> str | None:
    """Subject id from a mount name like exaSPIM_720165_processed_..."""
    found = re.search(r"exaspim_(\d+)", path.name, re.IGNORECASE)
    return found.group(1) if found else None


def _pick(assets: list[Path], subject: str) -> list[Path]:
    """Narrow mounted assets to one subject, or explain the ambiguity.

    Every asset attached to a capsule mounts on every run, so a capsule with
    dozens of subjects attached would otherwise render all of them into one
    figure. Rendering several deliberately is still possible -- name more than
    one subject -- but it has to be asked for.
    """
    wanted = set(_csv(subject))
    if wanted:
        chosen = [a for a in assets if _subject_of(a) in wanted]
        missing = wanted - {_subject_of(a) for a in chosen}
        if missing:
            _log(f"error: no mounted asset for subject(s) {sorted(missing)}. "
                 f"Available: {sorted(filter(None, map(_subject_of, assets)))}")
            raise SystemExit(2)
        return chosen
    if len(assets) <= 1:
        return assets
    available = sorted(filter(None, map(_subject_of, assets)))
    _log(f"error: {len(assets)} reconstruction assets are mounted, which would "
         "render every subject into one figure. Set the subject parameter to "
         f"choose. Available: {available}")
    raise SystemExit(2)


def _build_cache(args, views: list[str]) -> int:
    """Prerender views and overlays into /results for capture as an asset."""
    from ccf_glass_render.cache import build_structures, build_view
    from ccf_glass_render.profile import RenderProfile

    out = RESULTS / "ccf_cache"
    (out / "views").mkdir(parents=True, exist_ok=True)
    profile = RenderProfile()
    for view in views:
        build_view(view, args.resolution, profile, root=out / "views",
                   scratch=SCRATCH / "build")
    for acronym in _csv(args.structure):
        for view in views:
            build_structures(view, args.resolution, [acronym],
                             root=out / "views", scratch=SCRATCH / "build")

    # the structure graph is small and lets `find` work offline later
    graph = Path(os.environ["CCF_CACHE"]) / "structure_graph.json"
    if graph.exists():
        shutil.copy2(graph, out / graph.name)
    _log(f"cache written to {out}")
    return 0


def _render(args, views: list[str]) -> int:
    """Render every attached reconstruction asset."""
    from ccf_glass_render import provenance
    from ccf_glass_render.cache import load_structures, load_view
    from ccf_glass_render.figure import render_cells, save
    from ccf_glass_render.skeletons import Compartment, load_cells
    from ccf_glass_render.sources import resolve

    cache_root = _cache_root(args.cache)
    # kept as strings: Path("s3://bucket/x") collapses the double slash to
    # "s3:/bucket/x", which then reads as a local path and fails to exist
    assets = [str(p) for p in _pick(_asset_dirs(cache_root), args.subject)]
    if args.asset:
        assets = _csv(args.asset)
    if not assets:
        _log("error: no reconstruction data asset is attached. Attach one in "
             "the App Panel, or give an s3:// URI in the asset field.")
        return 2

    sources = [resolve(a, subdir=args.subdir or None) for a in assets]
    for source in sources:
        _log(f"{source.asset_name}: {len(source.files)} swc files")

    profile = _profile(args)
    cells = load_cells([p for s in sources for p in s.files],
                       Compartment(args.compartment))
    cells = _select(cells, args, profile)
    if not cells:
        _log("error: no cells selected")
        return 2
    _log(f"{len(cells)} cells: {sorted(cells)}")

    wanted = _csv(args.structure)
    written = []
    for view in views:
        brain = load_view(view, args.resolution, root=cache_root / "views")
        overlay = (load_structures(view, args.resolution, wanted,
                                   root=cache_root / "views") if wanted else None)
        image = render_cells(brain, cells, profile, structures=overlay)
        written += save(image, RESULTS / f"{args.label}_{view}",
                        dpi=args.dpi, formats=tuple(_csv(args.formats) or ("png", "svg")))
        _log(f"  {view}  {image.shape[1]} x {image.shape[0]}")

    provenance.write(RESULTS / "provenance.json", sources, profile, views,
                     args.resolution, list(cells), written, structures=wanted)
    return 0


def _profile(args):
    """Build the render profile from panel fields."""
    from dataclasses import replace

    from ccf_glass_render.profile import RenderProfile

    profile = RenderProfile.from_toml(args.profile) if args.profile else RenderProfile()
    skeleton = profile.skeleton
    if args.colors:
        skeleton = replace(skeleton, scheme=args.colors, palette=())
    if args.thickness:
        skeleton = replace(skeleton, thickness=args.thickness)
    if args.seed is not None:
        skeleton = replace(skeleton, seed=args.seed)
    return replace(profile, skeleton=skeleton)


def _select(cells: dict, args, profile) -> dict:
    """Apply the named-cell and random-subset panel fields."""
    import random

    named = _csv(args.cells)
    if named:
        missing = set(named) - set(cells)
        if missing:
            _log(f"warning: not in the attached assets: {sorted(missing)}")
        cells = {k: v for k, v in cells.items() if k in set(named)}
    if args.sample and args.sample < len(cells):
        keep = random.Random(profile.skeleton.seed).sample(sorted(cells), args.sample)
        cells = {k: cells[k] for k in keep}
    return cells


ORDERED = ("views", "colors", "thickness", "sample", "cells", "seed",
           "compartment", "structure")
"""Panel order for Ordered parameters, which arrive without their names.

Named parameters are safer and are what the docs recommend: reordering the
panel then cannot change which field a value lands in. This mapping exists so
an Ordered panel runs at all, and it validates rather than trusting position --
a value that cannot be what its slot expects is reported instead of silently
becoming someone else's setting.
"""


def _as_named(argv: list[str]) -> list[str]:
    """Convert bare positional arguments into named ones, by panel order."""
    if not argv or argv[0].startswith("-"):
        return argv
    positional = []
    for value in argv:
        if value.startswith("-"):
            break
        positional.append(value)
    rest = argv[len(positional):]
    if len(positional) > len(ORDERED):
        _log(f"error: {len(positional)} ordered parameters but only "
             f"{len(ORDERED)} are mapped: {list(ORDERED)}")
        raise SystemExit(2)

    out = []
    for name, value in zip(ORDERED, positional, strict=False):
        _check_ordered(name, value)
        out += [f"--{name}", value]
    _log(f"ordered parameters mapped to {ORDERED[:len(positional)]}")
    return out + rest


def _check_ordered(name: str, value: str) -> None:
    """Reject a positional value that cannot belong to its slot.

    Position is the only thing identifying an Ordered parameter, so a panel
    reordered in the UI would otherwise feed every value to the wrong setting
    without complaint.
    """
    if not value:
        return

    def refuse(why: str) -> None:
        raise SystemExit(
            f"error: the {name!r} slot got {value!r}, which {why}. The App "
            "Panel parameters are probably not in the order this script "
            f"expects ({', '.join(ORDERED)}). Give each parameter a Parameter "
            "Name in the App Builder instead of relying on order."
        )

    if name in ("thickness", "sample", "seed", "dpi"):
        try:
            float(value)
        except ValueError:
            refuse("is not a number")
    elif name == "views":
        from ccf_glass_render.profile import canonical_view
        try:
            for one in _csv(value):
                canonical_view(one)
        except ValueError:
            refuse("is not a view name")
    elif name == "colors":
        from ccf_glass_render import palettes
        if value not in palettes.names():
            refuse("is not a colour scheme")
    elif name == "compartment" and value not in ("all", "axon", "dendrite", "soma"):
        refuse("is not a compartment")


def main(argv: list[str] | None = None) -> int:
    """Parse App Panel parameters and dispatch."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", default="render", choices=["render", "build_cache"])
    parser.add_argument("--views", default="sagittal,iso")
    parser.add_argument("--resolution", type=float, default=10.0)
    parser.add_argument("--colors", default="")
    parser.add_argument("--thickness", type=float, default=0.0)
    # the panel may name this cell_morphology_type; both spellings bind here so
    # renaming the App Panel parameter cannot silently stop reaching the script
    parser.add_argument("--compartment", "--cell-morphology-type",
                        "--cell_morphology_type", dest="compartment", default="all")
    parser.add_argument("--structure", default="")
    parser.add_argument("--cells", default="")
    parser.add_argument("--subject", default="",
                        help="subject id(s) to render when several assets are "
                             "mounted, e.g. 720165")
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--label", default="cells")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--formats", default="png,svg")
    parser.add_argument("--profile", default="")
    parser.add_argument("--asset", default="",
                        help="override the attached assets with explicit paths or s3:// URIs")
    parser.add_argument("--subdir", default="ccf_space_reconstructions/swc")
    parser.add_argument("--cache", default="", help="override the cache location")
    args = parser.parse_args(_as_named(argv if argv is not None else sys.argv[1:]))

    RESULTS.mkdir(parents=True, exist_ok=True)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    # the package caches downloads here; /scratch is the only large writable
    # path in a capsule, and a read-only asset mount cannot serve as it
    os.environ.setdefault("CCF_CACHE", str(SCRATCH / "ccf_cache"))

    views = _csv(args.views)
    if not views:
        _log("error: no views requested")
        return 2

    try:
        if args.mode == "build_cache":
            return _build_cache(args, views)
        return _render(args, views)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        _log(f"error: {exc.args[0] if exc.args else exc}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
