"""Tiny real CPU runs check pixels as well as metadata and lifecycle behavior."""

from pathlib import Path

import numpy as np
import pytest
import rasterio
import torch
from affine import Affine

pytestmark = pytest.mark.integration


class Nearest(torch.nn.Module):
    def __init__(self, factor, channels):
        super().__init__()
        self.factor = factor
        self.channels = channels

    def forward(self, x):
        return torch.nn.functional.interpolate(
            x[:, : self.channels], scale_factor=self.factor, mode="nearest"
        )


@pytest.mark.parametrize(
    "height,width,factor,channels,window,overlap,discard,compressed",
    [
        (3, 5, 1, 1, (8, 8), 0, 0, False),
        (11, 17, 1, 2, (8, 8), 4, 0, False),
        (11, 17, 2, 4, (8, 8), 4, 0, False),
        (13, 19, 3, 2, (12, 10), 6, 2, True),
        (1, 19, 2, 1, (8, 8), 4, 0, True),
        (19, 1, 2, 2, (8, 8), 4, 0, False),
    ],
)
def test_tiled_inference_matches_whole_image_and_preserves_georeferencing(
    height,
    width,
    factor,
    channels,
    window,
    overlap,
    discard,
    compressed,
    write_raster,
    pipeline_factory,
    tmp_path,
):
    data = (np.arange(4 * height * width).reshape(4, height, width) * 7 % 10001).astype(
        np.uint16
    )
    # Include valid zero-valued predictions at internal and external boundaries.
    data[:, ::3, ::2] = 0
    transform = Affine(10, 0, 600000, 0, -10, 5100000)
    path = write_raster(
        tmp_path / "input.tif", data=data, crs="EPSG:32632", transform=transform
    )
    runner = pipeline_factory(
        path,
        model=Nearest(factor, channels),
        factor=factor,
        window_size=window,
        overlap=overlap,
        eliminate_border_px=discard,
        compressed_patches=compressed,
    )
    output = runner.run()
    expected = np.repeat(np.repeat(data[:channels], factor, axis=1), factor, axis=2)
    with rasterio.open(path) as original, rasterio.open(output) as result:
        assert result.shape == (height * factor, width * factor)
        assert result.count == channels
        assert result.dtypes == ("uint16",) * channels
        assert result.crs == original.crs
        assert result.transform.almost_equals(transform * Affine.scale(1 / factor))
        np.testing.assert_allclose(tuple(result.bounds), tuple(original.bounds))
        # Float32 accumulation followed by integer casting may lose one DN.
        np.testing.assert_allclose(result.read(), expected, atol=1, rtol=0)
    with rasterio.open(path) as original:
        np.testing.assert_array_equal(original.read(), data)
    assert output == runner.final_sr_path
    assert not Path(runner.placeholder_filepath).exists()


@pytest.mark.known_bug
@pytest.mark.xfail(
    reason="Upscaling divides affine a/e but leaves rotation/shear b/d unchanged",
    raises=AssertionError,
)
def test_upscaling_rotated_raster_preserves_its_spatial_footprint(
    write_raster, pipeline_factory, tmp_path
):
    transform = Affine(10, 2, 600000, 1, -10, 5100000)
    path = write_raster(tmp_path / "rotated.tif", transform=transform)
    runner = pipeline_factory(path)
    with rasterio.open(runner.run()) as result:
        assert result.transform.almost_equals(transform * Affine.scale(0.5))


def test_construction_only_prepares_and_does_not_invoke_model_or_touch_output(
    write_raster,
    pipeline_factory,
    tmp_path,
):
    class MustNotRun(torch.nn.Module):
        def forward(self, x):
            pytest.fail("Construction must not start inference")

    path = write_raster(tmp_path / "input.tif")
    final = tmp_path / "sr.tif"
    final.write_bytes(b"existing output")
    runner = pipeline_factory(path, model=MustNotRun())
    assert final.read_bytes() == b"existing output"
    assert not Path(runner.placeholder_filepath).exists()
    assert not list(Path(runner.temp_folder).iterdir())
    with pytest.raises(FileExistsError):
        runner.run()
    assert final.read_bytes() == b"existing output"


def test_successful_cleanup_preserves_input_logs_and_unrelated_files(
    write_raster, pipeline_factory, tmp_path
):
    path = write_raster(tmp_path / "input.tif")
    unrelated = tmp_path / "temp_other_run"
    unrelated.mkdir()
    sentinel = unrelated / "important.txt"
    sentinel.write_text("keep")
    runner = pipeline_factory(path, cleanup=True)
    assert Path(runner.run()).is_file()
    assert not Path(runner.temp_folder).exists()
    assert Path(runner.log_dir).is_dir()
    assert path.exists()
    assert sentinel.read_text() == "keep"


def test_model_failure_preserves_input_and_diagnostics(
    write_raster, pipeline_factory, tmp_path
):
    class Fails(torch.nn.Module):
        def forward(self, x):
            raise RuntimeError("inference failed deliberately")

    path = write_raster(tmp_path / "input.tif")
    before = path.read_bytes()
    runner = pipeline_factory(path, model=Fails(), cleanup=True)
    with pytest.raises(RuntimeError, match="inference failed deliberately"):
        runner.run()
    assert path.read_bytes() == before
    assert Path(runner.temp_folder).exists()
    assert Path(runner.log_dir).exists()
    assert not Path(runner.final_sr_path).exists()


def test_real_inference_disables_gradients_and_training_layers(
    write_raster, pipeline_factory, tmp_path
):
    class ChecksInference(torch.nn.Module):
        def forward(self, x):
            assert not self.training
            assert not torch.is_grad_enabled()
            return x

    path = write_raster(tmp_path / "input.tif")
    runner = pipeline_factory(path, model=ChecksInference(), factor=1)
    assert Path(runner.run()).is_file()


def test_explicit_overwrite_replaces_old_output_with_readable_result(
    write_raster, pipeline_factory, tmp_path
):
    path = write_raster(tmp_path / "input.tif", data=np.full((1, 8, 8), 2000))
    final = tmp_path / "sr.tif"
    final.write_bytes(b"previous output")
    runner = pipeline_factory(path)
    assert runner.run(overwrite=True) == str(final)
    with rasterio.open(final) as result:
        np.testing.assert_array_equal(result.read(), np.full((1, 16, 16), 2000))


@pytest.mark.known_bug
@pytest.mark.xfail(
    reason="Stitcher deletes patches every 50 tiles even with cleanup=False",
    raises=AssertionError,
)
def test_keep_temp_preserves_every_prediction_for_reproducibility(
    write_raster, pipeline_factory, tmp_path
):
    path = write_raster(tmp_path / "input.tif", width=16, height=16)
    runner = pipeline_factory(
        path,
        model=Nearest(1, 1),
        factor=1,
        window_size=(2, 2),
        overlap=0,
        cleanup=False,
    )
    runner.run()
    patches = list(Path(runner.temp_folder).glob("*.npy"))
    assert len(patches) == len(runner.image_meta["image_windows"])
