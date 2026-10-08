"""Geometry helpers: densify, crop, box_down."""

from __future__ import annotations

import numpy as np

from ccf_glass_render.splat import box_down, crop, densify


def test_densify_fills_a_long_segment():
    points = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    out = densify(points, np.array([-1, 1]), np.array([1, 2]), step=1.0)
    assert len(out) >= 11
    spread = np.sort(out[:, 0])
    assert np.diff(spread).max() <= 1.0 + 1e-9


def test_densify_does_not_bridge_severed_links():
    """Two roots are two runs; nothing should be drawn between them."""
    points = np.array([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]])
    out = densify(points, np.array([-1, -1]), np.array([1, 2]), step=1.0)
    assert len(out) == 2


def test_densify_passes_through_a_single_node():
    points = np.array([[1.0, 2.0, 3.0]])
    out = densify(points, np.array([-1]), np.array([1]), step=1.0)
    assert np.allclose(out, points)


def test_box_down_averages():
    image = np.arange(16, dtype=np.float32).reshape(4, 4)
    assert np.allclose(box_down(image, 2), [[2.5, 4.5], [10.5, 12.5]])


def test_box_down_keeps_colour_channels():
    image = np.zeros((6, 6, 3), np.float32)
    assert box_down(image, 3).shape == (2, 2, 3)


def test_crop_trims_to_content():
    image = np.ones((40, 40, 3), np.float32)
    image[20, 20] = 0.0
    assert crop(image, pad=2).shape[:2] == (5, 5)


def test_crop_leaves_a_blank_image_alone():
    image = np.ones((8, 8, 3), np.float32)
    assert crop(image, pad=2).shape == image.shape
