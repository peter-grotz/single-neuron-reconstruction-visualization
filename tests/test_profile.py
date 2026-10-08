"""Profiles must be stable and hashable, since the digest is provenance."""

from __future__ import annotations

from dataclasses import replace

from ccf_glass_render.profile import (
    VIEWS,
    GlassProfile,
    RenderProfile,
    SkeletonProfile,
    pixel_scale,
)


def test_digest_is_stable_across_instances():
    assert RenderProfile().digest() == RenderProfile().digest()


def test_digest_changes_with_any_tunable():
    base = RenderProfile()
    tweaked = replace(base, glass=replace(base.glass, edge=0.56))
    assert base.digest() != tweaked.digest()


def test_toml_round_trip(tmp_path):
    path = tmp_path / "p.toml"
    path.write_text('name = "t"\n[skeleton]\nthickness = 2.0\npalette = ["#000000"]\n')
    profile = RenderProfile.from_toml(path)
    assert profile.name == "t"
    assert profile.skeleton.thickness == 2.0
    assert profile.skeleton.palette == ("#000000",)
    # absent keys keep their defaults
    assert profile.glass.edge == GlassProfile().edge


def test_palette_is_hashable():
    hash(SkeletonProfile())


def test_pixel_scale_is_one_at_twenty_microns():
    assert pixel_scale(20.0) == 1.0
    assert pixel_scale(10.0) == 2.0


def test_views_are_the_documented_four():
    assert set(VIEWS) == {"sagittal", "iso", "dorsal", "coronal"}
