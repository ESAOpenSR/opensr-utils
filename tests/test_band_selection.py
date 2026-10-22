"""Band discovery is a public input contract, independent of model inference."""

from __future__ import annotations


import numpy as np
import pytest
from rasterio.transform import from_origin


def band_path(root, input_type, band, resolution=10):
    if input_type == "SAFE":
        return (
            root
            / "GRANULE"
            / "L2A_TEST"
            / "IMG_DATA"
            / f"R{resolution}m"
            / f"T_TEST_{band}_{resolution}m.jp2"
        )
    return root / f"{band}.tif"


@pytest.fixture(params=["SAFE", "S2GM"])
def spectral_scene(request, tmp_path, write_raster):
    input_type = request.param
    root = tmp_path / ("scene.SAFE" if input_type == "SAFE" else "scene_S2GM")
    root.mkdir()
    for band, value in {"B02": 2000, "B03": 3000, "B04": 4000, "B08": 8000}.items():
        # GeoTIFF content with SAFE naming tests discovery without a JP2 encoder.
        write_raster(
            band_path(root, input_type, band), data=np.full((1, 16, 16), value)
        )
    return root, input_type


def test_common_bands_have_rgb_nir_labels_and_values(spectral_scene, pipeline_factory):
    root, _ = spectral_scene
    runner = pipeline_factory(root)
    assert runner.image_meta["bands"] == 4
    assert set(runner.image_meta["lr_file_dict"]) == {"R", "G", "B", "NIR"}
    np.testing.assert_allclose(
        runner.datamodule.dataset[0]["image"][:, 0, 0], [0.4, 0.3, 0.2, 0.8]
    )


@pytest.mark.parametrize("auxiliary", ["AOT", "WVP", "TCI", "SCL", "MSK_CLDPRB"])
def test_auxiliary_files_are_not_counted_or_opened(
    spectral_scene, auxiliary, pipeline_factory
):
    root, input_type = spectral_scene
    path = band_path(root, input_type, auxiliary)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"Auxiliary file must never be opened as a spectral band")
    runner = pipeline_factory(root)
    assert runner.image_meta["bands"] == 4
    assert str(path) not in runner.image_meta["lr_file_dict"].values()


@pytest.mark.parametrize("band", ["B01", "B05", "B8A", "B11", "B12"])
def test_additional_spectral_bands_on_the_selected_grid_are_kept(
    spectral_scene, band, pipeline_factory, write_raster
):
    root, input_type = spectral_scene
    path = write_raster(band_path(root, input_type, band), width=16, height=16)
    runner = pipeline_factory(root)
    assert runner.image_meta["bands"] == 5
    assert runner.image_meta["lr_file_dict"][band] == str(path)
    assert runner.datamodule.dataset[0]["image"].shape == (5, 8, 8)


def test_bands_with_other_dimensions_are_excluded(
    spectral_scene, pipeline_factory, write_raster
):
    root, input_type = spectral_scene
    write_raster(band_path(root, input_type, "B11", 20), width=8, height=8)
    runner = pipeline_factory(root)
    assert runner.image_meta["bands"] == 4
    assert "B11" not in runner.image_meta["lr_file_dict"]


def test_safe_prefers_ten_meter_copy_of_common_bands(
    tmp_path, write_raster, pipeline_factory
):
    root = tmp_path / "scene.SAFE"
    for resolution, size in [(20, 8), (10, 16), (60, 3)]:
        for band in ("B02", "B03", "B04", "B08"):
            write_raster(
                band_path(root, "SAFE", band, resolution), width=size, height=size
            )
    runner = pipeline_factory(root)
    assert (runner.image_meta["width"], runner.image_meta["height"]) == (16, 16)
    assert runner.image_meta["bands"] == 4
    assert all("_10m.jp2" in p for p in runner.image_meta["lr_file_dict"].values())


def test_safe_without_usable_bands_has_a_clear_error(tmp_path, pipeline_factory):
    root = tmp_path / "empty.SAFE"
    path = band_path(root, "SAFE", "AOT")
    path.parent.mkdir(parents=True)
    path.write_bytes(b"ignored")
    with pytest.raises(FileNotFoundError, match="No usable SAFE bands"):
        pipeline_factory(root)


@pytest.mark.known_bug
@pytest.mark.xfail(
    reason="S2GM auxiliary-only input leaks StopIteration instead of a useful error",
    raises=StopIteration,
)
def test_s2gm_without_usable_bands_has_a_clear_error(
    tmp_path, write_raster, pipeline_factory
):
    root = tmp_path / "empty_S2GM"
    write_raster(root / "AOT.tif")
    with pytest.raises(FileNotFoundError, match="[Bb]ands"):
        pipeline_factory(root)


@pytest.mark.known_bug
@pytest.mark.xfail(
    reason="Band grid validation currently checks dimensions only",
    raises=pytest.fail.Exception,
)
@pytest.mark.parametrize("mismatch", ["transform", "crs"])
def test_same_size_bands_on_different_grids_are_rejected(
    spectral_scene, mismatch, write_raster, pipeline_factory
):
    root, input_type = spectral_scene
    options = (
        {"transform": from_origin(100, 200, 1, 1)}
        if mismatch == "transform"
        else {"crs": "EPSG:3857"}
    )
    write_raster(band_path(root, input_type, "B11"), width=16, height=16, **options)
    with pytest.raises(ValueError, match="(?i)grid|transform|crs|align"):
        pipeline_factory(root)
