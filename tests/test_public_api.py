"""Public workflows: one call, explicit destination and diagnostic-only debug runs."""

import json
import os
from pathlib import Path

import numpy as np
import pytest
import rasterio
import torch

from opensr_utils import super_resolve


class Identity(torch.nn.Module):
    def forward(self, x):
        return x


@pytest.mark.integration
def test_one_call_api_writes_requested_output_and_releases_sources(
    write_raster, tmp_path
):
    data = np.arange(20, dtype=np.uint16).reshape(1, 4, 5) * 100
    source = write_raster(tmp_path / "source.tif", data=data, nodata=None)
    destination = tmp_path / "results" / "enhanced.tif"
    result = super_resolve(source, Identity(), destination, factor=1, device="cpu")
    assert result == str(destination)
    with rasterio.open(result) as src:
        np.testing.assert_array_equal(src.read(), data)
        np.testing.assert_array_equal(src.dataset_mask(), np.full((4, 5), 255))
    assert source.exists()
    assert not list(destination.parent.glob("temp_*"))
    assert len(list(destination.parent.glob("logs_*"))) == 1
    with pytest.raises(FileExistsError):
        super_resolve(source, Identity(), destination, factor=1, device="cpu")


@pytest.mark.integration
def test_debug_retains_first_100_windows_and_never_publishes_partial_output(
    write_raster, pipeline_factory, tmp_path
):
    source = write_raster(tmp_path / "input.tif", width=22, height=22, nodata=None)
    runner = pipeline_factory(
        source, model=Identity(), factor=1, window_size=(2, 2), overlap=0, cleanup=True
    )
    output = Path(runner.final_sr_path)
    output.write_bytes(b"previous complete output")
    manifest = Path(runner.run(debug=True))
    index = json.loads(manifest.read_text())
    assert len(index["entries"]) == 100
    assert len(runner.image_meta["image_windows"]) == 121
    assert all(Path(entry["path"]).is_file() for entry in index["entries"])
    assert output.read_bytes() == b"previous complete output"
    assert not Path(runner.placeholder_filepath).exists()

    # A subsequent full run must cover the whole raster and may clean diagnostics.
    runner.run(debug=False, overwrite=True)
    with rasterio.open(output) as result, rasterio.open(source) as original:
        np.testing.assert_allclose(result.read(), original.read(), atol=1)
    assert not manifest.parent.exists()


@pytest.mark.parametrize("alias", ["same", "symlink", "hardlink"])
def test_output_cannot_replace_input_even_with_overwrite(alias, write_raster, tmp_path):
    source = write_raster(tmp_path / "input.tif")
    destination = source if alias == "same" else tmp_path / "alias.tif"
    if alias == "symlink":
        destination.symlink_to(source)
    elif alias == "hardlink":
        os.link(source, destination)
    before = source.read_bytes()
    with pytest.raises(ValueError, match="[Oo]utput.*input|[Oo]utput.*source"):
        super_resolve(source, Identity(), destination, factor=1, overwrite=True)
    assert source.read_bytes() == before


def test_default_output_cannot_replace_input_named_sr(write_raster, tmp_path):
    source = write_raster(tmp_path / "sr.tif")
    before = source.read_bytes()
    with pytest.raises(ValueError, match="[Oo]utput.*input|[Oo]utput.*source"):
        super_resolve(source, Identity(), factor=1, overwrite=True)
    assert source.read_bytes() == before


@pytest.mark.integration
def test_final_raster_preserves_source_mask_when_valid_predictions_are_zero(
    write_raster, tmp_path
):
    class Zeros(torch.nn.Module):
        def forward(self, x):
            return (
                torch.zeros_like(x[:, :1])
                .repeat_interleave(2, -2)
                .repeat_interleave(2, -1)
            )

    values = np.array([[[0, 100], [200, 300]]], dtype=np.uint16)
    source = write_raster(tmp_path / "input.tif", data=values, nodata=0)
    output = super_resolve(source, Zeros(), factor=2, device="cpu")
    with rasterio.open(output) as result:
        np.testing.assert_array_equal(
            result.read(), np.zeros((1, 4, 4), dtype=np.uint16)
        )
        expected_valid = np.repeat(np.repeat(values[0] != 0, 2, 0), 2, 1)
        np.testing.assert_array_equal(result.read_masks(1) != 0, expected_valid)
        assert result.nodata == 0
        assert not Path(output + ".msk").exists()


def test_one_call_api_rejects_auto_run_before_side_effects(tmp_path):
    with pytest.raises(TypeError, match="omit auto_run"):
        super_resolve(tmp_path / "input.tif", auto_run=True)
    assert not list(tmp_path.iterdir())


@pytest.mark.integration
def test_band_folder_discovery_does_not_require_a_special_directory_name(
    write_raster, tmp_path
):
    folder = tmp_path / "bands"
    values = {"b04": 4000, "b03": 3000, "b02": 2000, "b08": 8000}
    for band, value in values.items():
        write_raster(
            folder / f"{band}.tiff", data=np.full((1, 2, 3), value), nodata=None
        )
    output = super_resolve(folder, Identity(), factor=1, device="cpu")
    with rasterio.open(output) as result:
        np.testing.assert_array_equal(result.read()[:, 0, 0], list(values.values()))
