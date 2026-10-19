"""CCF structure masks, for rendering anatomy inside the glass brain.

A structure is named by its CCF acronym (``MD``, ``TH``, ``Isocortex``). The
annotation volume labels only leaf structures, so a parent's mask is the union
of its descendants -- asking for ``TH`` and matching label 549 alone would
return almost nothing.
"""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy import ndimage as ndi

from . import atlas

ANNOTATION_URL = (
    "https://download.alleninstitute.org/informatics-archive/"
    "current-release/mouse_ccf/annotation/ccf_2017/annotation_10.nrrd"
)
GRAPH_URL = "http://api.brain-map.org/api/v2/structure_graph_download/1.json"


@dataclass(frozen=True)
class Structure:
    """One CCF structure and the labels that make it up."""

    acronym: str
    name: str
    id: int
    colour: str
    """Official CCF colour, as ``#rrggbb``."""
    labels: tuple[int, ...]
    """This structure and every descendant, as they appear in the annotation."""


@lru_cache(maxsize=1)
def _graph() -> dict[str, Structure]:
    """Download and flatten the CCF structure graph, keyed by acronym."""
    path = atlas.cache_dir() / "structure_graph.json"
    if not path.exists():
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(GRAPH_URL, tmp)
        tmp.replace(path)
    root = json.loads(path.read_text())["msg"][0]

    nodes: list[dict] = []

    def walk(node: dict) -> None:
        nodes.append(node)
        for child in node.get("children", []):
            walk(child)

    walk(root)

    def descendants(node: dict) -> list[int]:
        out = [int(node["id"])]
        for child in node.get("children", []):
            out += descendants(child)
        return out

    table: dict[str, Structure] = {}
    for node in nodes:
        hexv = (node.get("color_hex_triplet") or "000000").zfill(6)
        table[str(node["acronym"])] = Structure(
            acronym=str(node["acronym"]), name=str(node["name"]), id=int(node["id"]),
            colour=f"#{hexv}", labels=tuple(descendants(node)))
    return table


def lookup(acronym: str) -> Structure:
    """Resolve a CCF structure by acronym or by numeric Allen id.

    Acronyms are the atlas's own spelling, but a near-miss on case is resolved
    rather than refused, and an id works too -- the App Panel asks for one, and
    an atlas id is easier to copy out of a table than an acronym is to recall.
    """
    table = _graph()
    key = acronym.strip()
    if key in table:
        return table[key]
    if key.isdigit():
        by_id = {s.id: s for s in table.values()}
        found = by_id.get(int(key))
        if found:
            return found
        raise KeyError(f"no CCF structure has id {key}")
    near = [k for k in table if k.lower() == key.lower()]
    if near:
        return table[near[0]]
    raise KeyError(f"unknown CCF structure {acronym!r}")


@lru_cache(maxsize=1)
def annotation(down: int) -> np.ndarray:
    """The CCF annotation volume at `down`, label-preserving.

    Downsampled by strided subsampling rather than averaging: labels are
    nominal, so a mean of two structure ids is a third, unrelated structure.
    """
    raw = atlas.cache_dir() / "annotation_10.npy"
    if not raw.exists():
        src = atlas._download(ANNOTATION_URL)
        import nrrd

        data, _ = nrrd.read(str(src))
        data = np.ascontiguousarray(data, dtype=np.uint32)
        if tuple(data.shape) != atlas.TEMPLATE_SHAPE:
            raise ValueError(
                f"annotation shape {tuple(data.shape)} does not match the "
                f"template {atlas.TEMPLATE_SHAPE}")
        tmp = raw.with_suffix(".part.npy")
        np.save(tmp, data)
        tmp.replace(raw)
        del data
    volume = np.load(raw, mmap_mode="r")
    return volume[::down, ::down, ::down] if down > 1 else volume


def mask(acronyms: list[str], down: int, smooth: float = 1.2) -> np.ndarray:
    """Smoothed occupancy for the union of `acronyms`, in [0, 1].

    Streams the compressed annotation one plane at a time rather than decoding
    it whole: the volume is 4.8 GB as uint32 and materialising it has killed a
    run outright. Streaming also means any of the 1327 CCF structures can be
    built on demand instead of only those baked into a cache.
    """
    labels = sorted({label for a in acronyms for label in lookup(a).labels})
    hit = _stream_mask(labels, down)
    if not hit.any():
        raise ValueError(
            f"{acronyms} selected no voxels at {down * atlas.VOXEL_UM:.0f} um; "
            "a small nucleus can vanish under heavy downsampling")
    field = hit.astype(np.float32)
    del hit
    if smooth:
        ndi.gaussian_filter(field, smooth * (2.0 / down), output=field)
    return field


def _stream_mask(labels: list[int], down: int) -> np.ndarray:
    """Boolean mask of `labels`, read plane by plane from the NRRD.

    NRRD stores raw data Fortran-ordered, so the contiguous slabs are planes of
    constant z. Reading them as if the last axis were fastest produces a volume
    with a near-identical value histogram but most voxels in the wrong place.
    """
    import gzip

    import nrrd

    wanted = np.zeros(max(labels) + 1, bool)
    wanted[list(labels)] = True

    src = atlas._download(ANNOTATION_URL)
    header = nrrd.read_header(str(src))
    nx, ny, nz = (int(v) for v in header["sizes"])
    dtype = np.dtype(np.uint32).newbyteorder(
        "<" if header.get("endian", "little") == "little" else ">")
    slab = nx * ny * dtype.itemsize

    sx, sy, sz = nx // down, ny // down, nz // down
    out = np.zeros((sx, sy, sz), bool)
    with open(src, "rb") as raw:
        while True:                      # consume the text header
            line = raw.readline()
            if not line or line.strip() == b"":
                break
        fh = gzip.GzipFile(fileobj=raw, mode="rb")
        for z in range(nz):
            buf = fh.read(slab)
            if len(buf) < slab:
                raise ValueError(f"annotation ended early at plane {z}")
            if z % down or z // down >= sz:
                continue
            plane = np.frombuffer(buf, dtype=dtype).reshape(ny, nx).T
            small = plane[::down, ::down]
            inside = small < wanted.size
            picked = np.zeros(small.shape, bool)
            picked[inside] = wanted[small[inside]]
            out[:, :, z // down] = picked
    return out


def default_colour(acronyms: list[str]) -> str:
    """CCF colour of the first structure, used when none is given."""
    return lookup(acronyms[0]).colour


def search(text: str, limit: int = 20) -> list[Structure]:
    """Structures whose acronym or name contains `text`, case-insensitively."""
    needle = text.lower()
    out = [s for s in _graph().values()
           if needle in s.acronym.lower() or needle in s.name.lower()]
    return sorted(out, key=lambda s: s.acronym)[:limit]
