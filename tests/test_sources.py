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
