"""Structure resolution. The graph is fetched once and cached, so these tests
exercise the parsing and selection logic against that cache rather than the
network; they skip when it is absent."""

from __future__ import annotations

import pytest

from ccf_glass_render import atlas, structures

pytestmark = pytest.mark.skipif(
    not (atlas.cache_dir() / "structure_graph.json").exists(),
    reason="CCF structure graph not cached; run `ccf-render find TH` once",
)


def test_leaf_structure_has_one_label():
    assert structures.lookup("MD").labels[0] == 362


def test_parent_expands_to_descendants():
    """TH labels almost nothing itself; its mask is the union of its children."""
    thalamus = structures.lookup("TH")
    assert len(thalamus.labels) > 50
    assert structures.lookup("MD").id in thalamus.labels


def test_structure_carries_its_ccf_colour():
    colour = structures.lookup("TH").colour
    assert colour.startswith("#") and len(colour) == 7


def test_unknown_acronym_raises():
    with pytest.raises(KeyError, match="unknown CCF structure"):
        structures.lookup("NOTASTRUCTURE")


def test_wrong_case_resolves():
    """A case near-miss is resolved rather than refused."""
    assert structures.lookup("md").acronym == "MD"
    assert structures.lookup(" th ").acronym == "TH"


def test_numeric_allen_id_resolves():
    """The App Panel asks for an allenId, so ids must work as well as acronyms."""
    assert structures.lookup("549").acronym == "TH"
    assert structures.lookup("362").acronym == "MD"


def test_unknown_numeric_id_says_so():
    with pytest.raises(KeyError, match="no CCF structure has id"):
        structures.lookup("99999999")


def test_search_finds_by_name():
    assert "DR" in {s.acronym for s in structures.search("raphe")}


def test_default_colour_follows_the_first_structure():
    assert structures.default_colour(["MD", "TH"]) == structures.lookup("MD").colour
