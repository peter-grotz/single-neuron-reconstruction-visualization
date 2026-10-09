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
    got = app._as_named(["685221", "sagittal+iso", "vivid", "1.15"])
    assert got == ["--subject", "685221", "--views", "sagittal+iso",
                   "--colors", "vivid", "--thickness", "1.15"]


def test_a_reordered_panel_is_refused_not_misassigned():
    """Position is the only identity an Ordered parameter has."""
    with pytest.raises(SystemExit, match="not a view name"):
        app._as_named(["685221", "vivid", "sagittal+iso"])


def test_a_view_in_the_subject_slot_is_refused():
    with pytest.raises(SystemExit, match="not a subject"):
        app._as_named(["sagittal", "sagittal+iso"])


def test_a_non_numeric_thickness_is_refused():
    with pytest.raises(SystemExit, match="not a number"):
        app._as_named(["685221", "sagittal", "vivid", "all"])


def test_too_many_ordered_values_is_refused():
    with pytest.raises(SystemExit):
        app._as_named(["685221"] * 20)


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


def test_a_subject_attached_twice_is_ambiguous(tmp_path, monkeypatch):
    """Two processing dates of one brain can both be attached."""
    data = tmp_path / "mounts"
    for d in ("2026_03_09", "2026_03_17"):
        (data / f"exaSPIM_720165_processed_reconstructions_{d}"
         / "final/ccf_space_reconstructions/swcs").mkdir(parents=True)
    monkeypatch.setattr(app, "DATA", data)
    assets = app._asset_dirs(tmp_path / "nope")
    with pytest.raises(SystemExit):
        app._pick(assets, "720165")


def test_a_full_asset_name_resolves_the_ambiguity(tmp_path, monkeypatch):
    data = tmp_path / "mounts"
    for d in ("2026_03_09", "2026_03_17"):
        (data / f"exaSPIM_720165_processed_reconstructions_{d}"
         / "final/ccf_space_reconstructions/swcs").mkdir(parents=True)
    monkeypatch.setattr(app, "DATA", data)
    assets = app._asset_dirs(tmp_path / "nope")
    picked = app._pick(assets, "exaSPIM_720165_processed_reconstructions_2026_03_17")
    assert [a.name for a in picked] == [
        "exaSPIM_720165_processed_reconstructions_2026_03_17"]


def test_available_marks_duplicated_subjects(tmp_path, monkeypatch):
    data = tmp_path / "mounts"
    for n in ("exaSPIM_720165_processed_reconstructions_2026_03_09",
              "exaSPIM_720165_processed_reconstructions_2026_03_17",
              "exaSPIM_685221_processed_reconstructions_2026_06_15"):
        (data / n / "final/ccf_space_reconstructions/swcs").mkdir(parents=True)
    monkeypatch.setattr(app, "DATA", data)
    out = app._available(app._asset_dirs(tmp_path / "nope"))
    assert "685221" in out
    assert any("720165" in x and "x2" in x for x in out)
