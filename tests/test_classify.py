"""Projection classes. The six are not mutually exclusive, so the order in
which they are tested is part of the definition and is pinned here."""

from __future__ import annotations

import pytest

from ccf_glass_render.classify import (
    CLASSES,
    MIN_LENGTH_MM,
    Projection,
    ambiguous,
    classify,
)


def make(targets, spinal=0.0, cell="N001-1"):
    return Projection(cell, "MY", {}, spinal, 1.0, frozenset(targets))


@pytest.mark.parametrize(
    ("targets", "spinal", "expected"),
    [
        (["MY", "CB"], 0.0, 1),            # cerebellum only, local arbor ignored
        (["CB"], 0.0, 1),
        (["MY"], 0.0, 2),                  # nothing outside the home division
        ([], 0.0, 2),
        (["MY"], 5.0, 6),                  # local only, but leaves caudally
        (["MY", "MB"], 0.0, 3),
        (["MY", "MB", "TH"], 0.0, 4),
        (["MY", "STR", "MB"], 0.0, 5),     # striatum outranks midbrain
        (["MY", "TH"], 0.0, 4),            # thalamus alone is class 4
    ],
)
def test_class_assignment(targets, spinal, expected):
    assert classify(make(targets, spinal)) == expected


def test_local_division_is_excluded_from_only_rules():
    """Every medullary cell arborises locally; counting MY would empty class 1."""
    assert classify(make(["MY", "CB"]), local="MY") == 1
    assert classify(make(["MY", "CB"]), local="CB") is None


def test_spinal_needs_more_than_a_trace():
    assert classify(make(["MY"], spinal=MIN_LENGTH_MM / 2)) == 2
    assert classify(make(["MY"], spinal=MIN_LENGTH_MM)) == 6


def test_overlapping_rules_are_reported():
    """Striatum outranks the others when a cell reaches several."""
    assert sorted(ambiguous(make(["MY", "STR", "TH"]))) == [4, 5]
    assert classify(make(["MY", "STR", "TH"])) == 5


def test_every_class_has_a_name():
    assert {n for n, _ in CLASSES} == {1, 2, 3, 4, 5, 6}


def test_division_volume_axis_order(tmp_path):
    """NRRD is Fortran-ordered, and misreading it is near-invisible.

    Reading the stream as if the last axis were fastest produced a volume with
    almost the same value histogram -- 56.18M non-zero voxels against 56.21M --
    while 53% of voxels named the wrong division. Nothing downstream would have
    noticed, so the shape of a known structure is checked instead of its totals.
    """
    import numpy as np

    from ccf_glass_render import atlas
    from ccf_glass_render.classify import DIVISIONS, load_division_volume

    cache = atlas.cache_dir() / "divisions_20um.npy"
    if not cache.exists():
        pytest.skip("division volume not built; run a classify once")
    volume, names = load_division_volume(cache.parent)

    code = {v: k for k, v in names.items()}
    # the cerebellum sits caudal and dorsal; the olfactory-bearing cortex is
    # rostral. If the axes were transposed these would swap.
    cb = np.argwhere(volume == code["CB"])
    ctx = np.argwhere(volume == code["CTX"])
    assert cb[:, 0].mean() > ctx[:, 0].mean(), "CB should be caudal of CTX"
    assert set(names.values()) == set(DIVISIONS)


def test_spinal_does_not_displace_a_named_target():
    """Per the spec a class covers 'X', 'X + MY', 'X + MY + spinal' alike."""
    assert classify(make(["MY", "CB"], spinal=5.0)) == 1
    assert classify(make(["MY", "CB"], spinal=0.0)) == 1
    assert classify(make(["MY"], spinal=5.0)) == 6
    assert classify(make(["MY"], spinal=0.0)) == 2


def test_pons_is_local_to_a_medullary_soma():
    """Nearly every medullary cell crosses into pons; counting it as a target
    emptied the midbrain and thalamus classes outright."""
    assert classify(make(["MY", "P"])) == 2
    assert classify(make(["MY", "P", "CB"])) == 1
    assert classify(make(["MY", "P", "MB"])) == 3
    assert classify(make(["MY", "P"], spinal=5.0)) == 6


def test_spinal_does_not_block_the_inclusive_classes():
    """Classes 3-5 ask whether a target is present, not whether it is alone."""
    assert classify(make(["MY", "MB"], spinal=5.0)) == 3
    assert classify(make(["MY", "STR"], spinal=5.0)) == 5


def test_labels_do_not_claim_exclusivity():
    """'cerebellum only' was false of 60 of its 167 cells, which also reach
    the spinal cord, so no label promises a cell goes nowhere else."""
    assert not any("only" in name for _, name in CLASSES)


def test_cortex_joins_striatum_in_class_5():
    assert classify(make(["MY", "CTX"])) == 5
    assert classify(make(["MY", "STR"])) == 5
    assert classify(make(["MY", "CTX", "MB"])) == 5
