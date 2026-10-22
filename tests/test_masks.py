import numpy as np
import rasterio
from rasterio.transform import from_origin

from opensr_utils.data_utils.datamodule import PredictionDataset
from opensr_utils.data_utils.masking import apply_input_mask


def _write(path, values, *, nodata=None, mask=None):
    values = np.asarray(values)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=values.shape[-1],
        height=values.shape[-2],
        count=1,
        dtype=values.dtype,
        transform=from_origin(0, values.shape[-2], 1, 1),
        nodata=nodata,
    ) as dst:
        dst.write(values[None])
        if mask is not None:
            dst.write_mask(np.asarray(mask, dtype=np.uint8) * 255)


def test_dataset_masks_nonfinite_and_intersects_dictionary_inputs(tmp_path):
    first = tmp_path / "first.tif"
    second = tmp_path / "second.tif"
    _write(first, np.array([[0, 10000], [20000, 30000]], dtype=np.float32))
    _write(
        second, np.array([[1, np.nan], [1, 1]], dtype=np.float32), mask=[[1, 1], [0, 1]]
    )

    dataset = PredictionDataset(
        "S2GM",
        str(tmp_path),
        [rasterio.windows.Window(0, 0, 2, 2)],
        {"R": str(first), "G": str(second)},
    )
    image = dataset[0]["image"].numpy()
    assert image[0, 0, 0] == 0  # valid zero remains data
    assert image[0, 0, 1] == 0  # invalid through the other source
    assert image[1, 1, 0] == 0  # source mask is intersected
    assert image[1, 0, 1] == 0  # NaN is filled before normalization


def test_apply_input_mask_upscales_nearest_and_keeps_valid_zero(tmp_path):
    source = tmp_path / "source.tif"
    _write(
        source, np.ones((2, 2), dtype=np.uint16), nodata=65535, mask=[[1, 0], [1, 1]]
    )
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
        nodata=0,
        tiled=True,
        blockxsize=16,
        blockysize=16,
    ) as dst:
        runner = type(
            "Runner",
            (),
            {"image_meta": {"lr_file_dict": {"R": str(source)}, "output_factor": 2}},
        )()
        apply_input_mask(runner, dst)
        dst.write(np.zeros((1, 4, 4), dtype=np.uint16))

    with rasterio.open(output) as src:
        mask = src.read_masks(1)
        assert np.all(mask[:2, :2] == 255)
        assert np.all(mask[:2, 2:] == 0)
        assert src.nodata == 65535


def test_apply_input_mask_is_exact_across_partial_blocks_and_factor_three(tmp_path):
    """The last 1-pixel output rows/columns must map to the right LR pixels."""
    height, width, factor = 173, 171, 3
    transform = rasterio.Affine(1.2, 0.17, 42.0, -0.11, -1.3, 9.0)
    first = tmp_path / "first-large.tif"
    second = tmp_path / "second-large.tif"
    first_values = np.ones((height, width), dtype=np.float32)
    second_values = np.full((height, width), 2.0, dtype=np.float32)
    # A finite source value is still invalid when its source mask is zero.
    second_values[169, 170] = np.nan
    first_mask = np.ones((height, width), dtype=np.uint8)
    second_mask = np.ones((height, width), dtype=np.uint8)
    first_mask[2, 4] = 0
    second_mask[100, 80] = 0
    _write(first, first_values, mask=first_mask)
    _write(second, second_values, mask=second_mask)

    output = tmp_path / "large-output.tif"
    out_height, out_width = height * factor, width * factor
    with rasterio.open(
        output,
        "w",
        driver="GTiff",
        width=out_width,
        height=out_height,
        count=5,  # output channels need not match input channels
        dtype="uint16",
        transform=transform * rasterio.Affine.scale(1 / factor),
        tiled=True,
        blockxsize=256,
        blockysize=256,
    ) as dst:
        runner = type(
            "Runner",
            (),
            {
                "image_meta": {
                    "lr_file_dict": {"R": str(first), "G": str(second)},
                    "output_factor": factor,
                }
            },
        )()
        apply_input_mask(runner, dst)

    with rasterio.open(output) as src:
        actual = src.read_masks(1) == 255
    expected_lr = first_mask.astype(bool) & second_mask.astype(bool)
    expected_lr[169, 170] = False  # nonfinite source value
    expected = expected_lr[
        np.floor_divide(np.arange(out_height), factor)[:, None],
        np.floor_divide(np.arange(out_width), factor)[None, :],
    ]
    np.testing.assert_array_equal(actual, expected)
