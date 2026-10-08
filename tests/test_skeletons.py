"""The SWC loader, where the convention mismatch bugs live."""

from __future__ import annotations

import numpy as np
import pytest

from ccf_glass_render.skeletons import (
    AXON,
    DENDRITE,
    SOMA,
    Compartment,
    cell_id,
    load_cells,
    read_swc,
)


def test_cell_id_strips_compartment_and_initials():
    assert cell_id("N009-785688-dendrite-JT.swc") == "N009-785688"
    assert cell_id("N009-785688-axon-JT.swc") == "N009-785688"
    assert cell_id("/a/b/N001-685221.swc") == "N001-685221"


def test_split_file_takes_compartment_from_filename(split_axon):
    """An all-zero type column must not be read as 'undefined'."""
    neuron = read_swc(split_axon)
    assert set(neuron.type_code.tolist()) == {AXON}


def test_split_dendrite_types_its_root_as_soma(split_dendrite):
    neuron = read_swc(split_dendrite)
    assert neuron.type_code[neuron.parent == -1].tolist() == [SOMA]
    assert set(neuron.type_code.tolist()) == {SOMA, DENDRITE}


def test_combined_file_keeps_its_own_types(combined):
    neuron = read_swc(combined)
    assert set(neuron.type_code.tolist()) == {SOMA, AXON, DENDRITE}


def test_split_file_without_compartment_in_name_raises(tmp_path):
    p = tmp_path / "mystery.swc"
    p.write_text("1 0 1.0 1.0 1.0 1.0 -1\n")
    with pytest.raises(ValueError, match="compartment"):
        read_swc(p)


def test_empty_file_raises(tmp_path):
    p = tmp_path / "N001-1-axon-XX.swc"
    p.write_text("# only a comment\n")
    with pytest.raises(ValueError, match="no SWC nodes"):
        read_swc(p)


@pytest.mark.parametrize(
    ("compartment", "expected"),
    [(Compartment.AXON, 2), (Compartment.DENDRITE, 3), (Compartment.SOMA, 1)],
)
def test_select_counts(combined, compartment, expected):
    """Dendrite selection carries the soma, so it is 2 dendrite + 1 soma."""
    assert len(read_swc(combined).select(compartment)) == expected


def test_select_severs_links_to_dropped_nodes(combined):
    """A kept node whose parent was dropped must become a root, not a segment."""
    axon = read_swc(combined).select(Compartment.AXON)
    orphan = axon.node_id == 2          # its parent, node 1, is the soma
    assert axon.parent[orphan].tolist() == [-1]


def test_select_all_is_identity(combined):
    neuron = read_swc(combined)
    assert neuron.select(Compartment.ALL) is neuron


def test_soma_falls_back_to_root(split_axon):
    """An axon-only file has no soma-typed node; the root stands in."""
    neuron = read_swc(split_axon)
    assert np.allclose(neuron.soma, [[100.0, 200.0, 300.0]])


def test_load_cells_groups_a_split_cell(split_axon, split_dendrite):
    cells = load_cells([split_axon, split_dendrite])
    assert set(cells) == {"N009-785688"}
    assert len(cells["N009-785688"]) == 2


def test_load_cells_drops_empty_selections(split_axon, split_dendrite):
    """The axon file contributes nothing to a dendrite-only figure."""
    cells = load_cells([split_axon, split_dendrite], Compartment.DENDRITE)
    assert len(cells["N009-785688"]) == 1
