"""Adapter-backed public workflows with real raster I/O and independent pixels."""

import zipfile
from types import SimpleNamespace

import numpy as np
import pytest
import rasterio
import torch

from opensr_utils import super_resolve
from opensr_utils.model_utils.esasrgan import ESASRGAN
from opensr_utils.model_utils.sen2sr import SEN2SRAdapter


RGBN = ("B04", "B03", "B02", "B08")


class FixedSENTile(torch.nn.Module):
    def forward(self, x):
        assert x.shape[-2:] == (128, 128)
        return x.repeat_interleave(4, -2).repeat_interleave(4, -1)


class OfficialGAN(torch.nn.Module):
    normalizer = SimpleNamespace(method="normalise_10k")

    def forward(self, x):
        pytest.fail("Official ESA inference must use predict_step")

    def predict_step(self, x):
        return x.repeat_interleave(4, -2).repeat_interleave(4, -1)


def adapter(name):
    if name == "SEN2SR":
        return SEN2SRAdapter(FixedSENTile())
    return ESASRGAN(OfficialGAN())


@pytest.mark.integration
@pytest.mark.parametrize("name", ["SEN2SR", "ESASRGAN"])
def test_declared_model_bands_reorder_files_and_preserve_validity_and_labels(
    name, write_raster, tmp_path
):
    values = np.array([8000, 2000, np.nan, 4000, 3000], dtype=np.float32)
    data = np.broadcast_to(values[:, None, None], (5, 3, 5)).copy()
    data[3, 1, 2] = np.nan
    source = write_raster(
        tmp_path / "input.tif", data=data, dtype="float32", nodata=None
    )
    with rasterio.open(source, "r+") as raster:
        raster.descriptions = ("B08", "B02", "unused", "B04", "B03")
    output = super_resolve(
        source, adapter(name), device="cpu", window_size=(8, 8), overlap=4
    )
    expected = data[[3, 4, 1, 0]].copy()
    valid = np.isfinite(expected).all(axis=0)
    expected[:, ~valid] = 0
    expected = np.repeat(np.repeat(expected, 4, 1), 4, 2)
    expected_mask = np.repeat(np.repeat(valid, 4, 0), 4, 1)
    with rasterio.open(output) as result:
        assert result.descriptions == RGBN
        assert result.shape == (12, 20)
        # Data beneath the invalid output mask is deliberately unspecified.
        np.testing.assert_allclose(
            result.read()[:, expected_mask], expected[:, expected_mask], atol=1, rtol=0
        )
        np.testing.assert_array_equal(result.read_masks(1) != 0, expected_mask)


@pytest.mark.integration
@pytest.mark.parametrize("name", ["SEN2SR", "ESASRGAN"])
def test_model_selection_ignores_unused_dictionary_spectral_bands(
    name, write_raster, tmp_path
):
    source = tmp_path / "spectral_bands"
    for band, value in (
        ("B04", 4000),
        ("B03", 3000),
        ("B02", 2000),
        ("B08", 8000),
        ("B11", 65535),
    ):
        write_raster(
            source / f"{band}.tif", data=np.full((1, 3, 5), value), nodata=65535
        )
    output = super_resolve(source, adapter(name), device="cpu")
    with rasterio.open(output) as result:
        np.testing.assert_array_equal(result.read()[:, 0, 0], [4000, 3000, 2000, 8000])
        assert np.all(result.read_masks() == 255)
        assert result.descriptions == RGBN


def test_named_public_api_loads_on_cpu_and_infers_factor(
    monkeypatch, write_raster, tmp_path
):
    from opensr_utils.model_utils import get_models

    calls = []
    source = write_raster(
        tmp_path / "input.tif", data=np.full((4, 2, 3), 2000), nodata=None
    )

    def load(name, **options):
        calls.append((name, options))
        return adapter("SEN2SR")

    monkeypatch.setattr(get_models, "get_model", load)
    output = super_resolve(
        source, "sen2sr", device="cpu", checkpoint=tmp_path / "weights", download=False
    )
    assert calls[0][0] == "SEN2SR"
    assert calls[0][1]["device"] == "cpu"
    assert calls[0][1]["download"] is False
    with rasterio.open(output) as result:
        assert result.shape == (8, 12)
        np.testing.assert_array_equal(result.read(), np.full((4, 8, 12), 2000))


def test_invalid_model_contract_keeps_source_archive(write_raster, tmp_path):
    product = tmp_path / "source" / "tile.SAFE"
    bands = product / "GRANULE" / "tile" / "IMG_DATA"
    for band in RGBN:
        write_raster(bands / f"TEST_{band}_10m.jp2", data=np.full((1, 2, 3), 2000))
    archive = tmp_path / "input.zip"
    with zipfile.ZipFile(archive, "w") as container:
        for path in product.rglob("*"):
            if path.is_file():
                container.write(path, path.relative_to(product.parent))
    with pytest.raises(ValueError, match="produces factor 4"):
        super_resolve(archive, adapter("SEN2SR"), factor=2, delete_input_zip=True)
    assert archive.is_file()
    assert not (tmp_path / "sr.tif").exists()


def test_pipeline_rejects_adapter_that_would_normalize_twice(write_raster, tmp_path):
    source = write_raster(tmp_path / "input.tif", count=4)
    model = ESASRGAN(OfficialGAN(), input_units="reflectance")
    with pytest.raises(ValueError, match="supplies normalized"):
        super_resolve(source, model)


def test_sen2sr_rejects_oversize_windows_before_inference(write_raster, tmp_path):
    source = write_raster(tmp_path / "input.tif", count=4, width=129, height=129)
    with pytest.raises(ValueError, match="accepts windows up to"):
        super_resolve(source, adapter("SEN2SR"), window_size=(256, 256))


def test_unlabeled_multiband_input_can_select_explicit_rgbn_indexes(
    write_raster, tmp_path
):
    data = np.broadcast_to(
        np.array([8000, 2000, 9999, 4000, 3000])[:, None, None], (5, 2, 3)
    )
    source = write_raster(tmp_path / "input.tif", data=data, nodata=None)
    output = super_resolve(
        source, adapter("ESASRGAN"), band_indexes=[4, 5, 2, 1], device="cpu"
    )
    with rasterio.open(output) as result:
        np.testing.assert_array_equal(result.read()[:, 0, 0], [4000, 3000, 2000, 8000])


@pytest.mark.integration
@pytest.mark.parametrize("source_kind", ["SAFE", "S2GM"])
def test_all_band_native_resolution_products_keep_order_values_and_masks(
    source_kind, write_raster, tmp_path
):
    from rasterio.transform import from_origin
    from opensr_utils.model_utils.sen2sr import ALL_BANDS

    source = tmp_path / ("tile.SAFE" if source_kind == "SAFE" else "bands")
    band_dir = (
        source / "GRANULE" / "tile" / "IMG_DATA" if source_kind == "SAFE" else source
    )
    ten_meter = {"B02", "B03", "B04", "B08"}
    for index, band in enumerate(ALL_BANDS):
        is_native_10m = band in ten_meter
        resolution = 10 if is_native_10m else 20
        shape = (4, 6) if is_native_10m else (2, 3)
        data = np.full((1, *shape), index * 500, dtype="uint16")
        if band == "B05":
            data[0, 1, 1] = 65535
        filename = (
            f"TEST_{band}_{resolution}m.jp2" if source_kind == "SAFE" else f"{band}.tif"
        )
        write_raster(
            band_dir / filename,
            data=data,
            nodata=65535,
            transform=from_origin(600000, 5100000, resolution, resolution),
            crs="EPSG:32632",
        )
    if source_kind == "SAFE":
        # This duplicate must not displace the native 20 m source.
        write_raster(
            band_dir / "TEST_B05_60m.jp2",
            data=np.full((1, 1, 1), 9999),
            nodata=65535,
            transform=from_origin(600000, 5100000, 60, 60),
            crs="EPSG:32632",
        )
    model = SEN2SRAdapter(FixedSENTile(), variant="main")
    output = super_resolve(source, model, device="cpu", resample_inputs=True)
    valid = np.ones((4, 6), dtype=bool)
    valid[2:4, 2:4] = False
    expected_mask = np.repeat(np.repeat(valid, 4, 0), 4, 1)
    with rasterio.open(output) as result:
        assert result.count == 10
        assert result.descriptions == ALL_BANDS
        assert result.shape == (16, 24)
        assert result.transform.almost_equals(from_origin(600000, 5100000, 2.5, 2.5))
        np.testing.assert_array_equal(result.dataset_mask() != 0, expected_mask)
        for index in range(10):
            np.testing.assert_array_equal(
                result.read(index + 1)[expected_mask], index * 500
            )
