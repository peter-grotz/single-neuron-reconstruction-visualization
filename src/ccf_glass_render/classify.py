"""Grouping neurons by where their axon goes.

Targets are counted at the level of the CCF's major divisions, and a division
counts as a target only when it holds a meaningful share of the axon -- a few
stray nodes clipped into a neighbouring structure should not reclassify a cell.

The CCF stops at the caudal medulla, so there is no spinal cord label. A
descending axon that leaves the atlas caudally is the only available evidence
of a spinal projection, which is why it is defined geometrically here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import atlas
from . import structures as st
from .skeletons import Compartment, Neuron

DIVISIONS: tuple[str, ...] = ("CB", "MY", "P", "MB", "TH", "HY", "STR", "PAL", "CTX")
"""Major divisions a projection is reported against."""

SPINAL_X_UM = 13200.0
"""Caudal bound of the CCF. Axon beyond this has left the atlas."""

MIN_FRACTION = 0.01
"""Share of total axon length a division needs before it counts as a target."""

MIN_LENGTH_MM = 0.2
"""...and an absolute floor, so a tiny axon cannot make a target of a few nodes."""


@dataclass(frozen=True)
class Projection:
    """Where one cell's axon goes, in millimetres per division."""

    cell_id: str
    soma_division: str | None
    length_mm: dict[str, float]
    spinal_mm: float
    total_mm: float
    targets: frozenset[str] = field(default_factory=frozenset)

    @property
    def is_spinal(self) -> bool:
        return self.spinal_mm >= MIN_LENGTH_MM


CLASSES: tuple[tuple[int, str], ...] = (
    (1, "cerebellum only"),
    (2, "medulla only"),
    (3, "midbrain, no thalamus"),
    (4, "thalamus + midbrain"),
    (5, "striatum"),
    (6, "spinal cord only"),
)


def classify(p: Projection, local: str = "MY") -> int | None:
    """Assign a projection class, or None when nothing fits.

    The six classes as specified are not mutually exclusive -- a cell reaching
    both striatum and midbrain satisfies two -- so they are tested in a fixed
    order and the first match wins. The order puts the exclusive classes first,
    so "only cerebellum" is not swallowed by a broader rule.

    `local` is the soma's own division, which is excluded when deciding whether
    a projection is confined somewhere: every medullary cell arborises locally,
    so counting MY would make "only cerebellum" impossible.
    """
    away = p.targets - {local}

    if not away and p.is_spinal:
        return 6
    if away == {"CB"}:
        return 1
    if not away and not p.is_spinal:
        return 2
    if "STR" in away:
        return 5
    if "TH" in away and "MB" in away:
        return 4
    if "MB" in away:
        return 3
    return None


def ambiguous(p: Projection, local: str = "MY") -> list[int]:
    """Every class a projection satisfies, to show where the rules overlap."""
    away = p.targets - {local}
    hits = []
    if not away and p.is_spinal:
        hits.append(6)
    if away == {"CB"}:
        hits.append(1)
    if not away and not p.is_spinal:
        hits.append(2)
    if "STR" in away:
        hits.append(5)
    if "TH" in away and "MB" in away:
        hits.append(4)
    if "MB" in away and "TH" not in away:
        hits.append(3)
    return hits


def _division_lookup() -> dict[int, str]:
    """Map every CCF label to the major division containing it."""
    out: dict[int, str] = {}
    for name in DIVISIONS:
        for label in st.lookup(name).labels:
            out.setdefault(int(label), name)
    return out


def _segment_lengths(neuron: Neuron) -> tuple[np.ndarray, np.ndarray]:
    """Midpoints and lengths of every parent-child segment, in microns."""
    index = {int(n): i for i, n in enumerate(neuron.node_id)}
    a, b = [], []
    for i, parent in enumerate(neuron.parent):
        j = index.get(int(parent))
        if j is not None:
            a.append(j)
            b.append(i)
    if not a:
        return np.empty((0, 3)), np.empty(0)
    start = neuron.xyz[np.array(a)]
    end = neuron.xyz[np.array(b)]
    return (start + end) / 2.0, np.linalg.norm(end - start, axis=1)


def project(neuron: Neuron, down: int = 1,
            lookup: dict[int, str] | None = None) -> Projection:
    """Measure one cell's axon length per division.

    Length is attributed by each segment's midpoint rather than by its nodes,
    so a long segment crossing a boundary is counted once, where it mostly lies.
    """
    lookup = _division_lookup() if lookup is None else lookup
    annotation = st.annotation(down)
    voxel = atlas.VOXEL_UM * down

    axon = neuron.select(Compartment.AXON)
    mid, length = _segment_lengths(axon)
    per: dict[str, float] = {}
    spinal = 0.0
    if len(length):
        beyond = mid[:, 0] > SPINAL_X_UM
        spinal = float(length[beyond].sum()) / 1000.0
        inside = ~beyond
        if inside.any():
            idx = np.clip((mid[inside] / voxel).astype(int), 0,
                          np.array(annotation.shape) - 1)
            labels = np.asarray(annotation[idx[:, 0], idx[:, 1], idx[:, 2]])
            lengths = length[inside]
            for label in np.unique(labels):
                name = lookup.get(int(label))
                if name:
                    per[name] = per.get(name, 0.0) + float(
                        lengths[labels == label].sum()) / 1000.0

    total = sum(per.values()) + spinal
    floor = max(MIN_LENGTH_MM, MIN_FRACTION * total)
    targets = frozenset(k for k, v in per.items() if v >= floor)

    soma = neuron.soma
    division = None
    if len(soma):
        idx = np.clip((soma[0] / voxel).astype(int), 0,
                      np.array(annotation.shape) - 1)
        division = lookup.get(int(annotation[idx[0], idx[1], idx[2]]))

    return Projection(neuron.cell_id, division, per, spinal, total, targets)
