"""Loading neuron reconstructions from SWC, across two file conventions.

Two conventions appear in exaSPIM CCF-space reconstruction assets:

``split``
    One file per compartment, named ``<cell>-axon-<INITIALS>.swc`` and
    ``<cell>-dendrite-<INITIALS>.swc``, with the SWC type column uniformly 0.
    The compartment is carried by the filename, not the data.
``combined``
    One file per cell with real SWC type codes (1 soma, 2 axon, 3 dendrite).

Reading the type column without checking which convention a file follows is the
bug this module exists to prevent: on a split file every node would be typed
"undefined" and a compartment filter would silently return nothing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import numpy as np

SOMA, AXON, DENDRITE = 1, 2, 3
_COMPARTMENT_IN_NAME = re.compile(r"-(axon|dendrite|soma)-", re.IGNORECASE)


class Compartment(StrEnum):
    """Which parts of a neuron to keep."""

    ALL = "all"
    AXON = "axon"
    DENDRITE = "dendrite"
    SOMA = "soma"


@dataclass(frozen=True)
class Neuron:
    """One reconstruction in CCF micron coordinates.

    Attributes
    ----------
    cell_id
        Stable identifier, e.g. ``N009-785688``.
    xyz
        ``(n, 3)`` node positions in microns.
    parent
        ``(n,)`` parent node ids; -1 marks a root.
    node_id
        ``(n,)`` node ids as written in the file.
    type_code
        ``(n,)`` SWC type codes, synthesised from the filename for split files.
    """

    cell_id: str
    xyz: np.ndarray
    parent: np.ndarray
    node_id: np.ndarray
    type_code: np.ndarray

    def __len__(self) -> int:
        return int(self.xyz.shape[0])

    @property
    def soma(self) -> np.ndarray:
        """Root node positions; the dendrite root when no node is typed soma."""
        hit = self.type_code == SOMA
        if hit.any():
            return self.xyz[hit]
        root = self.parent == -1
        return self.xyz[root] if root.any() else self.xyz[:1]

    def select(self, compartment: Compartment) -> Neuron:
        """Return a copy holding only `compartment`, renumbering nothing.

        Parent links to dropped nodes become -1, which the densifier reads as a
        break rather than drawing a segment across the gap.
        """
        if compartment is Compartment.ALL:
            return self
        want = {Compartment.AXON: AXON, Compartment.DENDRITE: DENDRITE,
                Compartment.SOMA: SOMA}[compartment]
        keep = self.type_code == want
        if compartment is Compartment.DENDRITE:
            keep |= self.type_code == SOMA
        kept_ids = set(self.node_id[keep].tolist())
        parent = np.where(
            np.isin(self.parent, list(kept_ids)), self.parent, -1
        )[keep]
        return Neuron(self.cell_id, self.xyz[keep], parent,
                      self.node_id[keep], self.type_code[keep])


def cell_id(path: str | Path) -> str:
    """Strip compartment and extension from an SWC filename."""
    stem = Path(path).name
    stem = _COMPARTMENT_IN_NAME.split(stem)[0]
    return stem.removesuffix(".swc")


def read_swc(path: str | Path) -> Neuron:
    """Read one SWC file, inferring the compartment convention.

    A file whose type column is entirely 0 is treated as `split`: its
    compartment is taken from the filename, and its root node is typed soma so
    that dendrite selections carry the soma with them.
    """
    path = Path(path)
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        f = line.split()
        rows.append((int(f[0]), int(float(f[1])), float(f[2]), float(f[3]),
                     float(f[4]), int(f[6])))
    if not rows:
        raise ValueError(f"{path} contains no SWC nodes")

    node_id = np.array([r[0] for r in rows], np.int64)
    type_code = np.array([r[1] for r in rows], np.int64)
    xyz = np.array([(r[2], r[3], r[4]) for r in rows], np.float64)
    parent = np.array([r[5] for r in rows], np.int64)

    if not type_code.any():
        type_code = _types_from_filename(path, parent)
    return Neuron(cell_id(path), xyz, parent, node_id, type_code)


def _types_from_filename(path: Path, parent: np.ndarray) -> np.ndarray:
    """Synthesise type codes for a split file, from its name."""
    found = _COMPARTMENT_IN_NAME.search(path.name)
    if not found:
        raise ValueError(
            f"{path.name} has an all-zero type column and no compartment in its "
            "name, so its compartment cannot be determined"
        )
    word = found.group(1).lower()
    code = {"axon": AXON, "dendrite": DENDRITE, "soma": SOMA}[word]
    out = np.full(parent.shape, code, np.int64)
    if code == DENDRITE:
        out[parent == -1] = SOMA
    return out


def load_cells(paths: list[str | Path], compartment: Compartment = Compartment.ALL
               ) -> dict[str, list[Neuron]]:
    """Group SWC files by cell, so a split cell's two files stay together."""
    out: dict[str, list[Neuron]] = {}
    for p in sorted(paths, key=str):
        neuron = read_swc(p).select(compartment)
        if len(neuron):
            out.setdefault(neuron.cell_id, []).append(neuron)
    return out


def find_swc(root: str | Path) -> list[Path]:
    """Every ``.swc`` under `root`, recursively, in a stable order."""
    return sorted(Path(root).rglob("*.swc"))
