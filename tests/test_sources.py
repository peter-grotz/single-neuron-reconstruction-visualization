"""Asset-name parsing and local resolution."""

from __future__ import annotations

from pathlib import Path

import pytest

from ccf_glass_render.sources import Source, resolve

ASSET = "exaSPIM_685221_2024-04-12_11-46-38_reconstructions_2026-08-28_23-04-54"


def test_asset_name_and_subject():
    src = Source(uri=f"s3://aind-open-data/{ASSET}", local=Path("."), files=[])
    assert src.asset_name == ASSET
    assert src.subject == "685221"


def test_trailing_slash_is_ignored():
    src = Source(uri=f"s3://aind-open-data/{ASSET}/", local=Path("."), files=[])
    assert src.asset_name == ASSET


def test_unparseable_name_has_no_subject():
    src = Source(uri="s3://bucket/whatever", local=Path("."), files=[])
    assert src.subject is None


def test_resolve_local_directory(tmp_path):
    (tmp_path / "N001-1-axon-XX.swc").write_text("1 0 1 1 1 1 -1\n")
    src = resolve(str(tmp_path), subdir=None)
    assert len(src.files) == 1


def test_resolve_descends_into_the_asset_subdir(tmp_path):
    inner = tmp_path / "ccf_space_reconstructions" / "swc"
    inner.mkdir(parents=True)
    (inner / "N001-1-axon-XX.swc").write_text("1 0 1 1 1 1 -1\n")
    assert resolve(str(tmp_path)).local == inner


def test_missing_directory_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        resolve(str(tmp_path / "nope"), subdir=None)


def test_directory_without_swc_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="no .swc"):
        resolve(str(tmp_path), subdir=None)


def test_s3_uri_survives_a_path_round_trip():
    """Path() collapses "s3://" to "s3:/"; callers must not pre-wrap URIs.

    A capsule run failed exactly this way, reporting that
    "s3:/aind-open-data/..." did not exist.
    """
    uri = "s3://aind-open-data/exaSPIM_1_2024-01-01_00-00-00_reconstructions_x"
    assert str(Path(uri)) != uri          # the hazard is real
    src = Source(uri=uri, local=Path("."), files=[])
    assert src.uri.startswith("s3://")


def _swc(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("1 0 1 1 1 1 -1\n")


def test_finds_the_swcs_subdir_spelling(tmp_path):
    """Assets use both `swc` and `swcs`, and nest under `final/`."""
    _swc(tmp_path / "final/ccf_space_reconstructions/swcs/N001-1-axon-XX.swc")
    assert resolve(str(tmp_path)).local.name == "swcs"


def test_several_coordinate_spaces_raise_rather_than_merge(tmp_path):
    """The same cells in several spaces must not be silently combined.

    A capsule run merged seven copies of five cells and drew two of them
    outside the brain, exiting 0.
    """
    for space in ("final/ccf_space_reconstructions/swcs", "refinement/raw",
                  "alignment/aligned_swcs"):
        _swc(tmp_path / space / "N001-720165-VM.swc")
    # the known CCF spelling wins outright
    assert resolve(str(tmp_path)).local.name == "swcs"
    # but with no recognisable subdir, refuse to choose
    for space in ("space_a", "space_b"):
        _swc(tmp_path / "other" / space / "N001-720165-VM.swc")
    with pytest.raises(ValueError, match="coordinate spaces"):
        resolve(str(tmp_path / "other"), subdir=None)


def test_single_swc_directory_is_still_found_anywhere(tmp_path):
    _swc(tmp_path / "weird/nested/place/N001-1.swc")
    assert resolve(str(tmp_path), subdir=None).local.name == "place"
