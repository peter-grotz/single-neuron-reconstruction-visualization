"""Command line: ``ccf-render brain`` builds views, ``ccf-render cells`` uses them."""

from __future__ import annotations

import argparse
import random
import sys
from dataclasses import replace
from pathlib import Path

from . import palettes, provenance
from .cache import build_view, load_view
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
    cells.add_argument("--pad", type=int, default=48,
                       help="margin in pixels around the rendered extent")

    args = parser.parse_args(argv)
    views = [v.strip() for v in args.views.split(",") if v.strip()]
    profile = _profile(args.profile)
    if getattr(args, "colors", None):
        # an explicit flag beats the profile, and clears any literal palette
        profile = replace(
            profile,
            skeleton=replace(profile.skeleton, scheme=args.colors, palette=()))

    if args.command == "brain":
        for view in views:
            build_view(view, args.resolution, profile, root=args.cache,
                       scratch=args.scratch, template=args.template)
        return 0

    return _render_cells(args, views, profile)


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

    written = []
    for view in views:
        brain = load_view(view, args.resolution, root=args.cache)
        image = render_cells(brain, cells, profile)
        stem = args.out / f"{args.label}_{view}"
        written += save(image, stem, dpi=args.dpi, pad=args.pad)
        print(f"  {view}  {image.shape[1]} x {image.shape[0]}", flush=True)

    provenance.write(args.out / "provenance.json", sources, profile, views,
                     args.resolution, list(cells), written)
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
