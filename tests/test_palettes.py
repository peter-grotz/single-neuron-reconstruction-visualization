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


def test_fixed_scheme_cycles_beyond_its_length():
    """Past its length a fixed scheme repeats; colours are no longer unique."""
    out = palettes.palette("okabe-ito", 20, seed=3)
    assert len(out) == 20
    assert set(out) == set(palettes.FIXED["okabe-ito"])


def test_cycled_repeats_are_reshuffled_not_tiled():
    """Each pass is shuffled, so the sequence is not the list repeated in order."""
    fixed = list(palettes.FIXED["vivid"])
    out = palettes.palette("vivid", len(fixed) * 3, seed=3)
    blocks = [out[i:i + len(fixed)] for i in range(0, len(out), len(fixed))]
    assert all(sorted(b) == sorted(fixed) for b in blocks)
    assert blocks[0] != blocks[1] or blocks[1] != blocks[2]


def test_cycling_never_puts_a_colour_next_to_itself():
    for n in (9, 17, 33, 64):
        out = palettes.palette("vivid", n, seed=7)
        assert not any(a == b for a, b in zip(out, out[1:], strict=False))


def test_cycling_is_deterministic_for_a_seed():
    assert palettes.palette("vivid", 20, seed=3) == palettes.palette("vivid", 20, seed=3)
    assert palettes.palette("vivid", 20, seed=3) != palettes.palette("vivid", 20, seed=4)


def test_unknown_scheme_raises():
    with pytest.raises(ValueError, match="unknown colour scheme"):
        palettes.palette("nope", 4)


@pytest.mark.parametrize("scheme", sorted(palettes.FIXED))
def test_fixed_schemes_avoid_near_white(scheme):
    """Pale colours vanish against the white glass backdrop."""
    for hexv in palettes.FIXED[scheme]:
        rgb = np.array(mcolors.to_rgb(hexv))
        assert rgb.min() < 0.85, f"{scheme}:{hexv} is too pale for a white background"
