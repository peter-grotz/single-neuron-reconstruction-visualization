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
