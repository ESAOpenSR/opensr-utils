from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("rasterio")
pytest.importorskip("skimage")
matplotlib = pytest.importorskip("matplotlib")
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
