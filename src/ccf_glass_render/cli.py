"""Command line: ``ccf-render brain`` builds views, ``ccf-render cells`` uses them."""

from __future__ import annotations

import argparse
import random
import sys
from dataclasses import replace
from pathlib import Path

from . import palettes, provenance
from .cache import build_structures, build_view, load_structures, load_view
from .figure import render_cells, save
from .profile import VIEWS, RenderProfile
from .skeletons import Compartment, load_cells
from .sources import SWC_SUBDIR, resolve

DEFAULT_VIEWS = "sagittal,iso"


def _profile(path: str | None) -> RenderProfile:
    return RenderProfile.from_toml(path) if path else RenderProfile()


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--profile", help="TOML render profile; omit for the default")
    parser.add_argument("--resolution", type=float, default=10.0,
                        help="pixel size in microns; must be a multiple of 10")
    parser.add_argument("--views", default=DEFAULT_VIEWS,
                        help=f"comma separated, from {sorted(VIEWS)}")
    parser.add_argument("--cache", type=Path, help="directory of prerendered views")


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(prog="ccf-render", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    brain = sub.add_parser("brain", help="prerender the glass brain for each view")
    _add_common(brain)
    brain.add_argument("--template", type=Path, help="local CCF template, instead of downloading")
    brain.add_argument("--scratch", type=Path, help="scratch directory for the 10 um build")

    struct = sub.add_parser("structures",
                            help="prerender CCF structures as an overlay for a view")
    _add_common(struct)
    struct.add_argument("--structure", action="append", required=True,
                        help="CCF acronym, e.g. MD, TH, Isocortex; repeatable")
    struct.add_argument("--colour", "--color", dest="colour",
                        help="hex colour; defaults to the structure's own CCF colour")
    struct.add_argument("--opacity", type=float, default=0.55)
    struct.add_argument("--scratch", type=Path)

    find = sub.add_parser("find", help="search CCF structures by acronym or name")
    find.add_argument("text")

    cells = sub.add_parser("cells", help="render reconstructions into a cached view")
    _add_common(cells)
    cells.add_argument("--asset", action="append", required=True,
                       help="s3:// reconstruction asset or local directory; repeatable")
    cells.add_argument("--subdir", default=SWC_SUBDIR,
                       help="path to SWCs within the asset; empty if already a SWC directory")
    cells.add_argument("--out", type=Path, default=Path("out"))
    cells.add_argument("--compartment", default="all",
                       choices=[c.value for c in Compartment])
    cells.add_argument("--cell", action="append",
                       help="restrict to these cell ids; repeatable")
    cells.add_argument("--sample", type=int, help="render a random subset of this many cells")
    cells.add_argument("--label", default="cells", help="filename prefix")
    cells.add_argument("--dpi", type=int, default=300)
    cells.add_argument("--colors", "--colours", dest="colors",
                       choices=palettes.names(),
                       help="per-cell colour scheme; overrides the profile. "
                            + "; ".join(f"{k}: {v}" for k, v in palettes.DESCRIPTIONS.items()))
    cells.add_argument("--thickness", type=float,
                       help="neuron tube radius in 20 um pixels; overrides the "
                            "profile (default 1.15, try 2-3 for a thumbnail)")
    cells.add_argument("--structure", action="append",
                       help="CCF acronym to show inside the brain, e.g. MD or "
                            "Isocortex; repeatable. Needs a cached overlay")
    cells.add_argument("--pad", type=int, default=48,
                       help="margin in pixels around the rendered extent")

    args = parser.parse_args(argv)
    if args.command == "find":
        return _find(args.text)

    views = [v.strip() for v in args.views.split(",") if v.strip()]
    profile = _profile(args.profile)
    if getattr(args, "colors", None):
        # an explicit flag beats the profile, and clears any literal palette
        profile = replace(
            profile,
            skeleton=replace(profile.skeleton, scheme=args.colors, palette=()))

    if getattr(args, "thickness", None):
        profile = replace(
            profile, skeleton=replace(profile.skeleton, thickness=args.thickness))

    try:
        if args.command == "brain":
            for view in views:
                build_view(view, args.resolution, profile, root=args.cache,
                           scratch=args.scratch, template=args.template)
            return 0

        if args.command == "structures":
            for view in views:
                build_structures(view, args.resolution, _split(args.structure),
                                 colour=args.colour, opacity=args.opacity,
                                 root=args.cache, scratch=args.scratch)
            return 0
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(f"error: {exc.args[0] if exc.args else exc}", file=sys.stderr)
        return 2

    try:
        return _render_cells(args, views, profile)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        # these carry an actionable message; a traceback only buries it
        print(f"error: {exc.args[0] if exc.args else exc}", file=sys.stderr)
        return 2


def _split(values: list[str]) -> list[str]:
    """Accept both repeated flags and one comma separated value."""
    return [v.strip() for value in values for v in value.split(",") if v.strip()]


def _find(text: str) -> int:
    """Print CCF structures matching `text`."""
    from . import structures as st

    found = st.search(text)
    if not found:
        print(f"no CCF structure matches {text!r}", file=sys.stderr)
        return 1
    for s in found:
        print(f"{s.acronym:14s} {s.colour}  {s.name}")
    return 0


def _render_cells(args, views: list[str], profile: RenderProfile) -> int:
    """Load every asset, pick cells, and write one figure per view."""
    sources = [resolve(a, subdir=args.subdir or None) for a in args.asset]
    paths = [p for s in sources for p in s.files]
    cells = load_cells(paths, Compartment(args.compartment))

    if args.cell:
        want = set(args.cell)
        missing = want - set(cells)
        if missing:
            print(f"no such cells: {sorted(missing)}", file=sys.stderr)
            return 2
        cells = {k: v for k, v in cells.items() if k in want}
    if args.sample and args.sample < len(cells):
        keep = random.Random(profile.skeleton.seed).sample(sorted(cells), args.sample)
        cells = {k: cells[k] for k in keep}
    if not cells:
        print("no cells selected", file=sys.stderr)
        return 2
    print(f"{len(cells)} cells: {sorted(cells)}", flush=True)

    wanted = _split(args.structure) if args.structure else []
    written = []
    for view in views:
        brain = load_view(view, args.resolution, root=args.cache)
        overlay = (load_structures(view, args.resolution, wanted, root=args.cache)
                   if wanted else None)
        image = render_cells(brain, cells, profile, structures=overlay)
        stem = args.out / f"{args.label}_{view}"
        written += save(image, stem, dpi=args.dpi, pad=args.pad)
        print(f"  {view}  {image.shape[1]} x {image.shape[0]}", flush=True)

    provenance.write(args.out / "provenance.json", sources, profile, views,
                     args.resolution, list(cells), written, structures=wanted)
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
