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
        # a full mount name selects one asset outright, which is how a subject
        # attached twice -- two processing dates of the same brain -- is resolved
        by_name = [a for a in assets if a.name in wanted]
        rest = wanted - {a.name for a in by_name}
        chosen = by_name + [a for a in assets if _subject_of(a) in rest]
        missing = rest - {_subject_of(a) for a in chosen}
        if missing:
            _log(f"error: no mounted asset for subject(s) {sorted(missing)}. "
                 f"Available: {_available(assets)}")
            raise SystemExit(2)
        for one in sorted(rest):
            same = [a.name for a in chosen if _subject_of(a) == one]
            if len(same) > 1:
                _log(f"error: subject {one} is attached {len(same)} times, so it "
                     "is ambiguous. Give the full asset name instead of the "
                     f"subject id: {sorted(same)}")
                raise SystemExit(2)
        return chosen
    if len(assets) <= 1:
        return assets
    _log(f"error: {len(assets)} reconstruction assets are mounted, which would "
         "render every subject into one figure. Set the subject parameter to "
         f"choose. Available: {_available(assets)}")
    raise SystemExit(2)


def _available(assets: list[Path]) -> list[str]:
    """Subject ids on offer, marking any attached more than once."""
    counts: dict[str, int] = {}
    for a in assets:
        s = _subject_of(a)
        if s:
            counts[s] = counts.get(s, 0) + 1
    return [f"{s} (x{n}, give the full asset name)" if n > 1 else s
            for s, n in sorted(counts.items())]


def _build_cache(args, views: list[str]) -> int:
    """Prerender views and overlays into /results for capture as an asset."""
    from ccf_glass_render.cache import build_structures, build_view, cache_path
    from ccf_glass_render.profile import RenderProfile

    out = RESULTS / "ccf_cache"
    (out / "views").mkdir(parents=True, exist_ok=True)

    # Carry forward whatever any mounted cache already holds. A build writes a
    # fresh directory, so without this the new asset replaces the old views
    # instead of adding to them -- which is how a cache of sagittal and iso
    # came to be replaced by one holding only coronal and horizontal.
    carried = 0
    for mount in sorted(DATA.glob("*")) if DATA.is_dir() else []:
        if not (mount / "views").is_dir():
            continue
        for src in sorted((mount / "views").glob("*.npz")):
            dest = out / "views" / src.name
            if not dest.exists():
                shutil.copy2(src, dest)
                carried += 1
        if (mount / "structure_graph.json").exists():
            shutil.copy2(mount / "structure_graph.json", out / "structure_graph.json")
    if carried:
        _log(f"carried forward {carried} cached files")

    profile = RenderProfile()
    for view in views:
        if cache_path(view, args.resolution, out / "views").exists():
            _log(f"[{view}] already cached, kept")
            continue
        build_view(view, args.resolution, profile, root=out / "views",
                   scratch=SCRATCH / "build")
    for acronym in _csv(args.structure):
        for view in views:
            build_structures(view, args.resolution, [acronym],
                             root=out / "views", scratch=SCRATCH / "build")

    from ccf_glass_render.classify import DIVISION_FILE, build_division_volume

    _log("building the division volume")
    build_division_volume(out / DIVISION_FILE)

    # the structure graph is small and lets `find` work offline later
    graph = Path(os.environ["CCF_CACHE"]) / "structure_graph.json"
    if graph.exists():
        shutil.copy2(graph, out / graph.name)
    _log(f"cache written to {out}")
    return 0


def _render(args, views: list[str]) -> int:
    """Render every attached reconstruction asset."""
    from ccf_glass_render import provenance
    from ccf_glass_render.cache import load_view
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
        overlay = (_overlay(view, args.resolution, wanted[0],
                            cache_root / "views") if wanted else None)
        image = render_cells(brain, cells, profile, structures=overlay)
        written += save(image, RESULTS / f"{args.label}_{view}",
                        dpi=args.dpi, formats=tuple(_csv(args.formats) or ("png", "svg")))
        _log(f"  {view}  {image.shape[1]} x {image.shape[0]}")

    provenance.write(RESULTS / "provenance.json", sources, profile, views,
                     args.resolution, list(cells), written, structures=wanted)
    return 0


def _classify(args, views: list[str]) -> int:
    """Group every cell with a soma in one division by where its axon goes.

    Scans every mounted reconstruction asset rather than one subject: the
    classes are a population statement, so the cells come from wherever they
    are. Writes the per-cell measurements alongside the figures so the
    grouping can be checked without re-running the scan.
    """
    import csv
    from collections import Counter

    from ccf_glass_render.classify import (
        CLASSES,
        ambiguous,
        classify,
        load_division_volume,
        project,
    )
    from ccf_glass_render.skeletons import load_cells
    from ccf_glass_render.sources import resolve

    home = args.soma_region or "MY"
    cache_root = _cache_root(args.cache)
    assets = _asset_dirs(cache_root)
    # the panel makes a subject mandatory, but a population statement needs
    # every mounted asset, so "all" opts out of the filter
    if args.subject and args.subject.strip().lower() not in ("all", "*"):
        assets = _pick(assets, args.subject)
    _log(f"scanning {len(assets)} assets for somata in {home}")

    volume, names = load_division_volume(cache_root)
    _log(f"division volume {volume.shape} from {cache_root}")
    rows, by_class = [], {}
    for n, asset in enumerate(assets, 1):
        try:
            source = resolve(str(asset), subdir=args.subdir or None)
        except (FileNotFoundError, ValueError) as exc:
            _log(f"  [{n}/{len(assets)}] {asset.name}: skipped -- {exc.args[0][:90]}")
            continue
        cells = load_cells(source.files)
        kept = 0
        for cell_id, neurons in cells.items():
            whole = max(neurons, key=len)
            p = project(whole, volume=volume, names=names)
            if p.soma_division != home:
                continue
            kept += 1
            group = classify(p, local=home)
            rows.append({
                "cell_id": cell_id, "asset": source.asset_name,
                "soma_division": p.soma_division, "class": group or "",
                "class_name": dict(CLASSES).get(group, ""),
                "also_matches": " ".join(str(x) for x in ambiguous(p, home)),
                "axon_mm": round(p.total_mm, 2),
                "spinal_mm": round(p.spinal_mm, 2),
                "targets": " ".join(sorted(p.targets)),
                **{f"mm_{d}": round(p.length_mm.get(d, 0.0), 2)
                   for d in sorted(p.length_mm)},
            })
            if group:
                by_class.setdefault(group, []).append((cell_id, source.local))
        _log(f"  [{n}/{len(assets)}] {source.asset_name}: {kept} soma in {home}")

    if not rows:
        _log(f"error: no cells with a soma in {home} across {len(assets)} assets")
        return 2

    counts = Counter(r["class"] for r in rows)
    _log("")
    for number, name in CLASSES:
        _log(f"  class {number} {name:24s} {counts.get(number, 0):4d}")
    _log(f"  {'unclassified':31s} {counts.get('', 0):4d}")
    multi = sum(1 for r in rows if len(r["also_matches"].split()) > 1)
    _log(f"  matched more than one rule   {multi:4d}")

    fields = sorted({k for r in rows for k in r})
    with open(RESULTS / f"{home}_projection_classes.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    _log(f"wrote {RESULTS / f'{home}_projection_classes.csv'}")

    return _render_classes(args, views, by_class, home)


CLASS_COLOURS = {
    1: "#e01f33",   # cerebellum          -- red, as on the lab's slide
    2: "#2b55e0",   # medulla             -- blue
    3: "#2fa814",   # midbrain            -- green
    4: "#ff6e00",   # thalamus + midbrain -- orange; the slide's cyan sat too
                    #                        close to the medulla blue to tell
                    #                        the two apart in a dense figure
    5: "#d4188c",   # striatum / cortical -- magenta
    6: "#e0b000",   # spinal cord         -- yellow
}
"""One colour per projection class, following the published figure's scheme."""


def _render_classes(args, views: list[str], by_class: dict, home: str) -> int:
    """One figure per view, every cell coloured by its projection class."""
    import numpy as np
    from matplotlib.colors import to_rgb

    from ccf_glass_render.cache import load_view
    from ccf_glass_render.classify import CLASSES
    from ccf_glass_render.figure import render_cells, save
    from ccf_glass_render.skeletons import Compartment, load_cells

    profile = _profile(args)
    cells, colours = {}, {}
    class_of: dict[int, set[str]] = {}
    for group, members in sorted(by_class.items()):
        rgb = np.array(to_rgb(CLASS_COLOURS[group]), np.float32)
        class_of[group] = set()
        for cell_id, folder in members:
            files = [f for f in folder.glob("*.swc")
                     if f.name.startswith(cell_id)]
            loaded = load_cells(files, Compartment(args.compartment))
            for key, neurons in loaded.items():
                cells[key] = neurons
                colours[key] = rgb
                class_of[group].add(key)
    if not cells:
        _log("error: no cells to render")
        return 2
    _log(f"rendering {len(cells)} cells in {len(by_class)} classes")


    views_root = _cache_root(args.cache) / "views"
    formats = tuple(_csv(args.formats) or ("png", "svg"))
    # a plain figure always, plus one per structure asked for. The home region
    # is added by default when it happens to be cached, since the obvious
    # question of a class figure is where these cells sit.
    wanted = _csv(args.structure) or [home]

    for view in views:
        brain = load_view(view, args.resolution, root=views_root)
        image = render_cells(brain, cells, profile, colours=colours)
        written = save(image, RESULTS / f"{home}_classes_{view}",
                       dpi=args.dpi, formats=formats)
        _log(f"  {view}  {image.shape[1]} x {image.shape[0]}  -> {written[0].name}")

        # one figure per class as well: in a combined figure the dense classes
        # bury the sparse ones, and class 2 is eleven cells against ninety-one
        for group in sorted(by_class):
            only = {k: v for k, v in cells.items() if k in class_of[group]}
            if not only:
                continue
            image = render_cells(brain, only, profile, colours=colours)
            written = save(image, RESULTS / f"{home}_class{group}_{view}",
                           dpi=args.dpi, formats=formats)
            name = dict(CLASSES)[group]
            _log(f"  {view} class {group} ({name}, {len(only)} cells) "
                 f"-> {written[0].name}")

        for acronym in wanted:
            try:
                overlay = _overlay(view, args.resolution, acronym, views_root)
            except (KeyError, ValueError) as exc:
                _log(f"  {view} + {acronym}: skipped -- {exc.args[0][:70]}")
                continue
            image = render_cells(brain, cells, profile, colours=colours,
                                 structures=overlay)
            written = save(image, RESULTS / f"{home}_classes_{view}_{acronym}",
                           dpi=args.dpi, formats=formats)
            _log(f"  {view} + {acronym}  -> {written[0].name}")
    return 0


def _scratch() -> Path:
    """A writable scratch directory: /scratch in a capsule, else the cache."""
    try:
        SCRATCH.mkdir(parents=True, exist_ok=True)
        probe = SCRATCH / ".w"
        probe.touch()
        probe.unlink()
        return SCRATCH
    except OSError:
        return Path(os.environ["CCF_CACHE"]) / "scratch"


def _overlay(view: str, resolution: float, acronym: str, views_root: Path):
    """Load a structure overlay, building it if the cache does not have one.

    A cache mount is read-only and only ever holds the structures someone
    thought to pre-build, so any other region was simply unavailable. Building
    on demand costs about a minute and writes to the writable cache, which
    means any of the 1327 CCF structures can be drawn.
    """
    from ccf_glass_render.cache import build_structures, load_structures

    try:
        return load_structures(view, resolution, [acronym], root=views_root)
    except FileNotFoundError:
        pass
    local = Path(os.environ["CCF_CACHE"]) / "views"
    try:
        return load_structures(view, resolution, [acronym], root=local)
    except FileNotFoundError:
        _log(f"  building {acronym} for {view}; not in the cache")
        build_structures(view, resolution, [acronym], root=local,
                         scratch=_scratch() / "struct")
        return load_structures(view, resolution, [acronym], root=local)


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
        # the annotator initials are part of the filename but nobody remembers
        # them, so a bare id matches too -- there are no collisions once they
        # are stripped. An ambiguous prefix is reported rather than guessed.
        wanted, missing, ambiguous = set(), [], []
        for name in named:
            if name in cells:
                wanted.add(name)
                continue
            hits = [k for k in cells if k == name or k.startswith(name + "-")]
            if len(hits) == 1:
                wanted.add(hits[0])
            elif hits:
                ambiguous.append((name, sorted(hits)))
            else:
                missing.append(name)
        if ambiguous:
            # raise rather than return a code: this helper's result is used as
            # the cell map, so a sentinel would be mistaken for one
            detail = "; ".join(f"{n!r} matches {h}" for n, h in ambiguous)
            raise ValueError(f"ambiguous cell id: {detail}")
        if missing:
            _log(f"warning: not in the attached assets: {sorted(missing)}")
        cells = {k: v for k, v in cells.items() if k in wanted}
    if args.sample and args.sample < len(cells):
        keep = random.Random(profile.skeleton.seed).sample(sorted(cells), args.sample)
        cells = {k: cells[k] for k in keep}
    return cells


ORDERED = ("subject", "views", "colors", "thickness", "sample", "cells",
           "seed", "compartment", "structure", "mode", "soma_region")
"""Panel order. Code Ocean truncates an API call to the parameters the panel
defines and drops the rest silently, so anything the API needs to set -- mode
above all -- has to exist in the panel. The last two are optional: a panel
without them sends nine values and these keep their defaults."""
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
    elif name == "mode" and value not in ("render", "build_cache", "classify"):
        refuse("is not a mode")
    elif name == "subject":
        from ccf_glass_render import palettes
        from ccf_glass_render.profile import VIEWS
        if value.lower() in VIEWS or value in palettes.names():
            refuse("is a view or colour scheme, not a subject")


def main(argv: list[str] | None = None) -> int:
    """Parse App Panel parameters and dispatch."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", default="render",
                        choices=["render", "build_cache", "classify"])
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
    parser.add_argument("--soma-region", "--soma_region", dest="soma_region",
                        default="MY",
                        help="CCF division the soma must sit in, for classify mode")
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
        if args.mode == "classify":
            return _classify(args, views)
        return _render(args, views)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        _log(f"error: {exc.args[0] if exc.args else exc}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
