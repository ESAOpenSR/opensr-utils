"""Resampling must preserve source validity independently of valid zero values."""

import pickle

import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.windows import Window

from opensr_utils.data_utils.datamodule import PredictionDataset
from opensr_utils.data_utils.grids import open_input_raster


def test_resampling_preserves_internal_mask_and_valid_zeros(write_raster, tmp_path):
    path = write_raster(
        tmp_path / "source.tif",
        data=np.zeros((1, 2, 3)),
        nodata=None,
        transform=from_origin(0, 40, 20, 20),
        crs="EPSG:32632",
    )
    valid = np.ones((2, 3), dtype="uint8") * 255
    valid[0, 1] = 0
    with rasterio.open(path, "r+") as source:
        source.write_mask(valid)
    grid = dict(
        width=6,
        height=4,
        transform=from_origin(0, 40, 10, 10),
        crs=rasterio.crs.CRS.from_epsg(32632),
    )
    with open_input_raster(path, grid) as source:
        expected = valid.repeat(2, 0).repeat(2, 1)
        np.testing.assert_array_equal(source.read_masks(1), expected)
        np.testing.assert_array_equal(source.read(1)[expected != 0], 0)

    # Spawned workers reopen the same virtual grid instead of pickling GDAL handles.
    dataset = PredictionDataset(
        "S2GM",
        str(tmp_path),
        [Window(0, 0, 6, 4)],
        {"B05": str(path)},
        resample_grid=grid,
    )
    try:
        expected_values = dataset[0]["image"]
        restored = pickle.loads(pickle.dumps(dataset))
        try:
            np.testing.assert_array_equal(restored[0]["image"], expected_values)
        finally:
            restored.close()
    finally:
        dataset.close()
