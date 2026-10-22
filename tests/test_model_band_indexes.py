"""Public band index behavior for ordinary multi-band model inputs."""

import numpy as np
import rasterio
from rasterio.transform import from_origin

from opensr_utils.data_utils.datamodule import PredictionDataset
from opensr_utils.data_utils.masking import apply_input_mask


def _write_five_band(path, data):
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=2,
        height=2,
        count=5,
        dtype="float32",
        transform=from_origin(0, 2, 1, 1),
    ) as dst:
        dst.write(np.asarray(data, dtype=np.float32))


def test_file_dataset_reads_reordered_selected_band_indexes(tmp_path):
    source = tmp_path / "five-band.tif"
    values = np.stack(
        [
            np.full((2, 2), 1000),  # extra
            np.full((2, 2), 2000),  # G
            np.full((2, 2), 3000),  # B
            np.full((2, 2), 4000),  # R
            np.full((2, 2), 8000),  # NIR
        ]
    )
    _write_five_band(source, values)
    dataset = PredictionDataset(
        "file",
        str(source),
        [rasterio.windows.Window(0, 0, 2, 2)],
        {},
        band_indexes=[4, 2, 3, 5],
    )

    np.testing.assert_allclose(dataset[0]["image"][:, 0, 0], [0.4, 0.2, 0.3, 0.8])


def test_file_mask_ignores_invalid_unselected_band(tmp_path):
    source = tmp_path / "five-band-invalid-extra.tif"
    values = np.stack(
        [
            np.full((2, 2), np.nan),  # invalid, unselected extra
            np.full((2, 2), 2000),
            np.full((2, 2), 3000),
            np.full((2, 2), 4000),
            np.full((2, 2), 8000),
        ]
    )
    _write_five_band(source, values)
    output = tmp_path / "output.tif"
    with rasterio.open(
        output,
        "w",
        driver="GTiff",
        width=4,
        height=4,
        count=1,
        dtype="uint16",
        transform=from_origin(0, 2, 0.5, 0.5),
        tiled=True,
        blockxsize=16,
        blockysize=16,
    ) as dst:
        runner = type(
            "Runner",
            (),
            {
                "input_type": "file",
                "factor": 2,
                "image_meta": {
                    "lr_file_dict": {"file": str(source)},
                    "band_indexes": [4, 2, 3, 5],
                    "output_factor": 2,
                },
            },
        )()
        apply_input_mask(runner, dst)

    with rasterio.open(output) as src:
        assert np.all(src.read_masks(1) == 255)
