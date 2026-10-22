"""Geometric invariants rather than an implementation of the tiling algorithm."""

import numpy as np
import pytest

from opensr_utils.pipeline import large_file_processing


@pytest.mark.parametrize(
    "width,height,window_size,overlap",
    [
        (w, h, window, overlap)
        for w, h in [(1, 1), (1, 19), (23, 1), (8, 8), (17, 23), (32, 32), (33, 31)]
        for window, overlap in [((8, 8), 0), ((8, 8), 4), ((12, 8), 4), ((8, 12), 6)]
    ],
)
def test_windows_are_unique_bounded_and_cover_every_pixel(
    width, height, window_size, overlap
):
    runner = object.__new__(large_file_processing)
    runner.image_meta = {"width": width, "height": height}
    runner.window_size = window_size
    runner.overlap = overlap
    windows = runner.create_image_windows()
    coordinates = [
        (int(w.col_off), int(w.row_off), int(w.width), int(w.height)) for w in windows
    ]
    assert len(coordinates) == len(set(coordinates))
    assert windows == runner.create_image_windows(), "Ordering must be reproducible"
    coverage = np.zeros((height, width), dtype=int)
    for x, y, w, h in coordinates:
        assert (w, h) == (min(width, window_size[0]), min(height, window_size[1]))
        assert 0 <= x < x + w <= width
        assert 0 <= y < y + h <= height
        coverage[y : y + h, x : x + w] += 1
    assert np.all(coverage > 0)
    # Neighboring tiles must retain at least the requested overlap on each axis.
    for axis, length, size in [(0, width, window_size[0]), (1, height, window_size[1])]:
        offsets = sorted({c[axis] for c in coordinates})
        assert offsets[0] == 0
        assert offsets[-1] + min(length, size) == length
        assert all(
            right - left <= max(1, size - overlap)
            for left, right in zip(offsets, offsets[1:])
        )


@pytest.mark.parametrize("width,height", [(0, 1), (1, 0), (-1, 3), (3, -1)])
def test_empty_or_negative_image_dimensions_are_rejected(width, height):
    runner = object.__new__(large_file_processing)
    runner.window_size = (8, 8)
    runner.overlap = 0
    runner.image_meta = {"width": width, "height": height}
    with pytest.raises(ValueError, match="dimensions must be positive"):
        runner.create_image_windows()
