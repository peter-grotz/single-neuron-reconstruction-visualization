"""The capsule app's handling of App Panel parameters."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
import app  # noqa: E402


@pytest.fixture(autouse=True)
def _sandbox(tmp_path, monkeypatch):
    for name in ("RESULTS", "SCRATCH", "DATA"):
        d = tmp_path / name.lower()
        d.mkdir()
        monkeypatch.setattr(app, name, d)


def test_named_arguments_pass_through():
    assert app._as_named(["--views", "iso"]) == ["--views", "iso"]


def test_ordered_arguments_take_the_panel_order():
    got = app._as_named(["sagittal+iso", "vivid", "1.15"])
    assert got == ["--views", "sagittal+iso", "--colors", "vivid", "--thickness", "1.15"]


def test_a_reordered_panel_is_refused_not_misassigned():
    """Position is the only identity an Ordered parameter has."""
    with pytest.raises(SystemExit, match="not a view name"):
        app._as_named(["vivid", "sagittal+iso"])


def test_a_non_numeric_thickness_is_refused():
    with pytest.raises(SystemExit, match="not a number"):
        app._as_named(["sagittal", "vivid", "all"])


def test_too_many_ordered_values_is_refused():
    with pytest.raises(SystemExit):
        app._as_named(["sagittal"] * 20)


def test_separators_accept_plus_comma_and_space():
    assert app._csv("a+b") == app._csv("a,b") == app._csv("a b") == ["a", "b"]


def _mounts(root, *subjects):
    """A /data directory holding one mount per subject, and nothing else."""
    data = root / "mounts"
    for s in subjects:
        (data / f"exaSPIM_{s}_processed_reconstructions_2026_03_10"
         / "final/ccf_space_reconstructions/swcs").mkdir(parents=True)
    return data


def test_one_mounted_asset_needs_no_subject(tmp_path, monkeypatch):
    monkeypatch.setattr(app, "DATA", _mounts(tmp_path, "720165"))
    assets = app._asset_dirs(tmp_path / "nope")
    assert len(app._pick(assets, "")) == 1


def test_many_mounted_assets_refuse_without_a_subject(tmp_path, monkeypatch):
    """Every attached asset mounts, so 55 subjects would merge into one figure."""
    monkeypatch.setattr(app, "DATA", _mounts(tmp_path, "720165", "709222", "704522"))
    assets = app._asset_dirs(tmp_path / "nope")
    with pytest.raises(SystemExit):
        app._pick(assets, "")


def test_subject_selects_one_of_many(tmp_path, monkeypatch):
    monkeypatch.setattr(app, "DATA", _mounts(tmp_path, "720165", "709222", "704522"))
    assets = app._asset_dirs(tmp_path / "nope")
    assert [a.name for a in app._pick(assets, "720165")] == [
        "exaSPIM_720165_processed_reconstructions_2026_03_10"]
    assert len(app._pick(assets, "720165+709222")) == 2


def test_unknown_subject_lists_what_is_available(tmp_path, monkeypatch):
    monkeypatch.setattr(app, "DATA", _mounts(tmp_path, "720165"))
    assets = app._asset_dirs(tmp_path / "nope")
    with pytest.raises(SystemExit):
        app._pick(assets, "999999")
