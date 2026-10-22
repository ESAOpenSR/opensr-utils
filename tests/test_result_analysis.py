from __future__ import annotations

import pytest

import numpy as np
import matplotlib

matplotlib.use("Agg", force=True)

from opensr_utils.data_utils import result_analysis  # noqa: E402


def test_stretch_scales_first_three_bands_to_rgb():
    arr = np.array(
        [
            [[0, 10000], [5000, 20000]],
            [[10000, 0], [5000, 20000]],
            [[5000, 5000], [0, 10000]],
            [[9999, 9999], [9999, 9999]],
        ],
        dtype=np.uint16,
    )

    rgb = result_analysis.stretch(arr)

    assert rgb.shape == (2, 2, 3)
    assert rgb.min() >= 0
    assert rgb.max() <= 1
    assert rgb[0, 0].tolist() == pytest.approx([0.0, 1.0, 1.0])


def test_stretch_repeats_single_band_to_rgb():
    arr = np.array([[[0, 5000], [10000, 20000]]], dtype=np.uint16)

    rgb = result_analysis.stretch(arr)

    assert rgb.shape == (2, 2, 3)
    assert rgb[0, 1].tolist() == pytest.approx([1.0, 1.0, 1.0])


def test_crop_and_save_georeferenced_excerpt_returns_crop_and_writes_file(
    write_raster, dummy_runner, tmp_path
):
    data = np.arange(4 * 4 * 4, dtype=np.uint16).reshape(4, 4, 4)
    tif = write_raster(tmp_path / "sr.tif", data=data)
    out = tmp_path / "crop.tif"

    crop = result_analysis.crop_and_save_georeferenced_excerpt(
        dummy_runner,
        tif_path=str(tif),
        out_tif=str(out),
        random_crop_size=(2, 2),
    )

    assert crop.shape == (4, 2, 2)
    assert out.exists()
    assert any("Saved georeferenced" in msg for msg in dummy_runner.messages)


def test_crop_and_save_georeferenced_excerpt_clips_too_large_crop(
    write_raster, dummy_runner, tmp_path
):
    tif = write_raster(tmp_path / "sr.tif", data=np.ones((4, 3, 3), dtype=np.uint16))

    crop = result_analysis.crop_and_save_georeferenced_excerpt(
        dummy_runner,
        tif_path=str(tif),
        out_tif=str(tmp_path / "crop.tif"),
        random_crop_size=(4, 4),
    )

    assert crop.shape == (4, 3, 3)


def test_generate_side_by_side_previews_writes_png(
    write_raster, dummy_runner, tmp_path
):
    tif = write_raster(
        tmp_path / "sr.tif", data=np.ones((4, 4, 4), dtype=np.uint16) * 1000
    )
    out_dir = tmp_path / "previews"

    result_analysis.generate_side_by_side_previews(
        dummy_runner,
        tif_path=str(tif),
        out_dir=str(out_dir),
        num_examples=1,
        window_size=10,
        crop_size=2,
        target_size=(4, 4),
    )

    assert (out_dir / "example_000.png").exists()
    assert any("Image too small" in msg for msg in dummy_runner.messages)
    assert any("Saved preview PNG" in msg for msg in dummy_runner.messages)


@pytest.mark.parametrize("bands", [1, 2, 3, 5])
def test_stretch_has_explicit_rgb_mapping_and_does_not_mutate_input(bands):
    data = np.stack(
        [np.full((2, 4), value) for value in [0.1, 0.2, 0.3, 0.4, 0.5][:bands]]
    )
    before = data.copy()
    result = result_analysis.stretch(data)
    expected = {
        1: [0.35, 0.35, 0.35],
        2: [0.35, 0.7, 0.35],
        3: [0.35, 0.7, 1],
        5: [0.35, 0.7, 1],
    }[bands]
    assert result.shape == (2, 4, 3)
    np.testing.assert_allclose(result, np.broadcast_to(expected, (2, 4, 3)))
    np.testing.assert_array_equal(data, before)


def test_excerpt_pixels_and_georeferencing_match_the_source_window(
    write_raster, dummy_runner, tmp_path, monkeypatch
):
    import rasterio
    from rasterio.transform import from_origin
    from rasterio.windows import Window

    values = np.arange(2 * 6 * 8, dtype=np.uint16).reshape(2, 6, 8)
    path = write_raster(
        tmp_path / "source.tif",
        data=values,
        transform=from_origin(600000, 5100000, 10, 10),
        crs="EPSG:32632",
    )
    offsets = iter([2, 1, 0, 0])
    monkeypatch.setattr(np.random, "randint", lambda *args, **kwargs: next(offsets))
    output = tmp_path / "excerpt.tif"
    returned = result_analysis.crop_and_save_georeferenced_excerpt(
        dummy_runner, str(path), str(output), random_crop_size=(3, 2)
    )
    np.testing.assert_array_equal(returned, values[:, 1:3, 2:5])
    with rasterio.open(path) as source, rasterio.open(output) as excerpt:
        np.testing.assert_array_equal(excerpt.read(), returned)
        assert excerpt.crs == source.crs
        assert excerpt.transform == source.window_transform(Window(2, 1, 3, 2))
        assert excerpt.nodata == source.nodata
        assert excerpt.dtypes == source.dtypes
