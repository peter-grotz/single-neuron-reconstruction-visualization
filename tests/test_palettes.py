"""Colour schemes must stay distinguishable and keep their documented size."""

from __future__ import annotations

import matplotlib.colors as mcolors
import numpy as np
import pytest

from ccf_glass_render import palettes


def test_default_scheme_is_unchanged():
    """The vivid list is the published look; changing it rewrites old figures."""
    assert palettes.palette("vivid", 8) == [
        "#d4188c", "#8c3fe0", "#2b55e0", "#1f8fe5",
        "#11a88e", "#2fa814", "#e07b00", "#e01f33",
    ]


@pytest.mark.parametrize("scheme", palettes.names())
def test_every_scheme_returns_the_requested_count(scheme):
    assert len(palettes.palette(scheme, 6)) == 6


@pytest.mark.parametrize("scheme", palettes.names())
def test_every_scheme_is_internally_distinct(scheme):
    colours = palettes.palette(scheme, 8)
    assert len(set(colours)) == 8


@pytest.mark.parametrize("scheme", palettes.names())
def test_every_scheme_is_documented(scheme):
    assert scheme in palettes.DESCRIPTIONS


@pytest.mark.parametrize("scheme", sorted(palettes.GENERATED))
def test_generated_schemes_scale_past_the_fixed_limit(scheme):
    assert len(set(palettes.palette(scheme, 32))) == 32


def test_fixed_scheme_beyond_its_length_raises():
    with pytest.raises(ValueError, match="generated scheme"):
        palettes.palette("okabe-ito", 20)


def test_unknown_scheme_raises():
    with pytest.raises(ValueError, match="unknown colour scheme"):
        palettes.palette("nope", 4)


@pytest.mark.parametrize("scheme", sorted(palettes.FIXED))
def test_fixed_schemes_avoid_near_white(scheme):
    """Pale colours vanish against the white glass backdrop."""
    for hexv in palettes.FIXED[scheme]:
        rgb = np.array(mcolors.to_rgb(hexv))
        assert rgb.min() < 0.85, f"{scheme}:{hexv} is too pale for a white background"
