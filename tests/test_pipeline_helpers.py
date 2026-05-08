from __future__ import annotations

from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("rasterio")
pytest.importorskip("torch")
pytest.importorskip("pytorch_lightning")

from rasterio.transform import from_origin  # noqa: E402

from opensr_utils import pipeline  # noqa: E402


def blank_runner():
    return object.__new__(pipeline.large_file_processing)


@pytest.mark.parametrize(
    ("kwargs", "exc", "match"),
    [
        ({"model": object()}, TypeError, "Model must"),
        ({"window_size": (0, 8)}, ValueError, "positive"),
        ({"window_size": (8,)}, TypeError, "two positive"),
        ({"overlap": 3}, ValueError, "even"),
        ({"eliminate_border_px": 5}, ValueError, "even"),
        ({"overlap": 0, "eliminate_border_px": 2}, ValueError, "must be 0"),
        ({"overlap": 4, "eliminate_border_px": 4}, ValueError, "smaller"),
        ({"overlap": 6, "eliminate_border_px": 4}, ValueError, "at least 4"),
        ({"window_size": (8, 8), "overlap": 8}, ValueError, "smaller than both"),
        ({"factor": 0}, ValueError, "positive integer"),
        ({"device": "tpu"}, ValueError, "device"),
        ({"batch_size": 0}, ValueError, "greater than zero"),
        ({"num_workers": -1}, ValueError, "non-negative"),
        ({"gpus": "0"}, TypeError, "gpus"),
        ({"device": "cuda", "gpus": [-1]}, ValueError, "GPU ids"),
    ],
)
def test_large_file_processing_validates_constructor_args(tmp_path, kwargs, exc, match):
    args = {
        "root": str(tmp_path / "missing.tif"),
        "model": None,
        "window_size": (8, 8),
        "factor": 2,
        "overlap": 4,
        "eliminate_border_px": 0,
        "device": "cpu",
        "batch_size": 1,
        "num_workers": 0,
        "prefetch_factor": 0,
    }
    args.update(kwargs)

    with pytest.raises(exc, match=match):
        pipeline.large_file_processing(**args)


def test_large_file_processing_can_prepare_without_running(write_raster, tmp_path):
    tif = write_raster(
        tmp_path / "input.tif", data=np.ones((1, 16, 16), dtype=np.uint16)
    )

    runner = pipeline.large_file_processing(
        root=str(tif),
        model=None,
        window_size=(8, 8),
        factor=2,
        overlap=4,
        eliminate_border_px=0,
        device="cpu",
        batch_size=1,
        num_workers=0,
        prefetch_factor=0,
        auto_run=False,
    )

    assert runner.input_type == "file"
    assert runner.gpus is None
    assert runner.datamodule.dataset.root == str(tif)
    assert runner.model.factor == 2
    assert Path(runner.log_dir).is_dir()
    assert Path(runner.temp_folder).is_dir()


def test_large_file_processing_runs_small_image_factor_three_two_band_model(
    write_raster, tmp_path
):
    torch = pytest.importorskip("torch")
    rasterio = pytest.importorskip("rasterio")

    class TwoBandUpsampler(torch.nn.Module):
        def forward(self, x):
            return torch.nn.functional.interpolate(
                x[:, :2], scale_factor=3, mode="nearest"
            )

    tif = write_raster(
        tmp_path / "input.tif",
        data=np.ones((4, 5, 3), dtype=np.uint16) * 1000,
    )

    runner = pipeline.large_file_processing(
        root=str(tif),
        model=TwoBandUpsampler(),
        window_size=(8, 8),
        factor=3,
        overlap=4,
        eliminate_border_px=0,
        device="cpu",
        batch_size=1,
        num_workers=0,
        prefetch_factor=0,
        auto_run=False,
        cleanup=False,
        overwrite=True,
    )

    out = runner.run()

    with rasterio.open(out) as src:
        assert (src.width, src.height, src.count) == (9, 15, 2)
        assert src.dtypes == ("uint16", "uint16")


def test_large_file_processing_runs_factor_one_tiled_identity_model(
    write_raster, tmp_path
):
    torch = pytest.importorskip("torch")
    rasterio = pytest.importorskip("rasterio")

    class FirstBandIdentity(torch.nn.Module):
        def forward(self, x):
            return x[:, :1]

    data = np.stack(
        [
            np.arange(15, dtype=np.uint16).reshape(5, 3),
            np.full((5, 3), 2000, dtype=np.uint16),
        ],
        axis=0,
    )
    tif = write_raster(tmp_path / "input.tif", data=data)

    runner = pipeline.large_file_processing(
        root=str(tif),
        model=FirstBandIdentity(),
        window_size=(2, 2),
        factor=1,
        overlap=0,
        eliminate_border_px=0,
        device="cpu",
        batch_size=2,
        num_workers=0,
        prefetch_factor=0,
        auto_run=False,
        cleanup=False,
        overwrite=True,
    )

    out = runner.run()

    with rasterio.open(out) as src:
        assert (src.width, src.height, src.count) == (3, 5, 1)
        np.testing.assert_array_equal(src.read(1), data[0])


def test_create_image_windows_covers_right_and_bottom_edges():
    runner = blank_runner()
    runner.window_size = (8, 8)
    runner.overlap = 2
    runner.image_meta = {"width": 21, "height": 17}

    windows = runner.create_image_windows()
    coords = {(int(w.col_off), int(w.row_off)) for w in windows}

    assert (0, 0) in coords
    assert (13, 0) in coords
    assert (0, 9) in coords
    assert (13, 9) in coords
    assert all(w.width == 8 and w.height == 8 for w in windows)


def test_create_image_windows_uses_single_clipped_window_for_small_images():
    runner = blank_runner()
    runner.window_size = (8, 8)
    runner.overlap = 2
    runner.image_meta = {"width": 7, "height": 5}

    windows = runner.create_image_windows()

    assert len(windows) == 1
    assert windows[0].col_off == 0
    assert windows[0].row_off == 0
    assert windows[0].width == 7
    assert windows[0].height == 5


@pytest.mark.parametrize(
    ("width", "height", "window_size", "overlap"),
    [
        (21, 17, (8, 8), 2),
        (9, 9, (8, 8), 4),
        (257, 129, (128, 128), 8),
        (7, 5, (8, 8), 2),
        (16, 16, (8, 8), 0),
    ],
)
def test_create_image_windows_cover_pixels_without_underlapping(
    width, height, window_size, overlap
):
    runner = blank_runner()
    runner.window_size = window_size
    runner.overlap = overlap
    runner.image_meta = {"width": width, "height": height}

    windows = runner.create_image_windows()
    coverage = np.zeros((height, width), dtype=np.uint8)
    for win in windows:
        x0 = int(win.col_off)
        y0 = int(win.row_off)
        x1 = x0 + int(win.width)
        y1 = y0 + int(win.height)
        assert 0 <= x0 < x1 <= width
        assert 0 <= y0 < y1 <= height
        coverage[y0:y1, x0:x1] += 1

    assert coverage.min() > 0

    for axis_length, win_length, offsets in [
        (width, min(window_size[0], width), sorted({int(w.col_off) for w in windows})),
        (
            height,
            min(window_size[1], height),
            sorted({int(w.row_off) for w in windows}),
        ),
    ]:
        assert offsets[0] == 0
        assert offsets[-1] + win_length == axis_length
        for left, right in zip(offsets, offsets[1:]):
            assert right - left <= win_length - overlap


def test_create_placeholder_file_writes_scaled_geotiff(tmp_path):
    rasterio = pytest.importorskip("rasterio")
    runner = blank_runner()
    runner.temp_folder = str(tmp_path / "temp_unit")
    runner.placeholder_filepath = str(tmp_path / "sr_placeholder.tif")
    runner.output_file_path = runner.placeholder_filepath
    runner.factor = 2
    runner.image_meta = {
        "bands": 3,
        "width": 5,
        "height": 4,
        "dtype": "uint16",
        "crs": "EPSG:4326",
        "transform": from_origin(10, 20, 10, 10),
    }
    runner._log = lambda message: None

    runner.create_placeholder_file(force=True)

    with rasterio.open(runner.placeholder_filepath) as src:
        assert (src.width, src.height, src.count) == (10, 8, 3)
        assert src.transform.a == 5
        assert src.transform.e == -5
        assert src.nodata == 0


def test_create_placeholder_file_skips_existing_readable_file(tmp_path):
    runner = blank_runner()
    runner.temp_folder = str(tmp_path / "temp_unit")
    runner.placeholder_filepath = str(tmp_path / "sr_placeholder.tif")
    runner.output_file_path = runner.placeholder_filepath
    runner.factor = 2
    runner.image_meta = {
        "bands": 1,
        "width": 2,
        "height": 2,
        "dtype": "uint16",
        "crs": "EPSG:4326",
        "transform": from_origin(0, 2, 1, 1),
    }
    messages = []
    runner._log = messages.append

    runner.create_placeholder_file(force=True)
    runner.create_placeholder_file(force=False)

    assert any("skipping creation" in msg for msg in messages)


def test_create_placeholder_file_can_use_prediction_output_metadata(tmp_path):
    rasterio = pytest.importorskip("rasterio")
    runner = blank_runner()
    runner.temp_folder = str(tmp_path / "temp_unit")
    runner.placeholder_filepath = str(tmp_path / "sr_placeholder.tif")
    runner.output_file_path = runner.placeholder_filepath
    runner.factor = 2
    runner.image_meta = {
        "bands": 4,
        "output_bands": 2,
        "output_dtype": "uint16",
        "width": 3,
        "height": 2,
        "dtype": "uint8",
        "crs": "EPSG:4326",
        "transform": from_origin(0, 2, 9, 9),
    }
    runner._log = lambda message: None

    runner.create_placeholder_file(force=True, factor=3)

    with rasterio.open(runner.placeholder_filepath) as src:
        assert (src.width, src.height, src.count) == (9, 6, 2)
        assert src.dtypes == ("uint16", "uint16")
        assert src.transform.a == 3


def test_delete_lr_temp_removes_only_run_scoped_temp_folder(tmp_path):
    runner = blank_runner()
    runner.temp_folder = str(tmp_path / "temp_unit")
    Path(runner.temp_folder).mkdir()
    (Path(runner.temp_folder) / "patch.npy").write_bytes(b"data")
    runner._log = lambda message: None

    runner.delete_LR_temp()

    assert not Path(runner.temp_folder).exists()


def test_delete_lr_temp_refuses_unexpected_folder(tmp_path):
    runner = blank_runner()
    runner.temp_folder = str(tmp_path / "not_temp")
    Path(runner.temp_folder).mkdir()

    with pytest.raises(RuntimeError, match="Refusing to delete"):
        runner.delete_LR_temp()


def test_is_rank0_uses_trainer_torch_and_environment(monkeypatch):
    runner = blank_runner()
    assert (
        runner._is_rank0(trainer=type("Trainer", (), {"is_global_zero": False})())
        is False
    )
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("LOCAL_RANK", "1")

    assert runner._is_rank0() is False


def test_log_buffers_until_log_file_exists(tmp_path):
    runner = blank_runner()
    runner._is_rank0 = lambda trainer=None: False

    runner._log("before")
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    runner.log_file = log_dir / "log.txt"
    runner._log("after")

    text = runner.log_file.read_text()
    assert "before" in text
    assert "after" in text
    assert runner._log_backlog == []


def test_run_refuses_to_overwrite_existing_output(tmp_path):
    runner = blank_runner()
    runner.debug = False
    runner.save_preview = False
    runner.cleanup = True
    runner.overwrite = False
    runner.final_sr_path = str(tmp_path / "sr.tif")
    Path(runner.final_sr_path).write_bytes(b"existing")

    with pytest.raises(FileExistsError, match="Output already exists"):
        runner.run()


def test_write_to_file_delegates_to_ddp_safe_stitch(monkeypatch, tmp_path):
    from opensr_utils.data_utils import writing_utils

    runner = blank_runner()
    runner.overwrite = False
    calls = []

    def fake_ddp_safe_stitch(self, **kwargs):
        calls.append((self, kwargs))
        return ("out.tif", 0)

    monkeypatch.setattr(writing_utils, "ddp_safe_stitch", fake_ddp_safe_stitch)

    assert runner.write_to_file(
        index_path=str(tmp_path / "index.json"), limit=2, overwrite=True
    ) == ("out.tif", 0)
    assert runner.overwrite is True
    assert calls[0][0] is runner
    assert calls[0][1]["index_path"].endswith("index.json")
    assert calls[0][1]["limit"] == 2
    assert calls[0][1]["profile"] == "sigmoid"


def test_write_to_file_creates_placeholder_from_prediction_index(
    monkeypatch, tmp_path
):
    from opensr_utils.data_utils import writing_utils

    rasterio = pytest.importorskip("rasterio")
    runner = blank_runner()
    runner.overwrite = True
    runner.trainer = type("Trainer", (), {"is_global_zero": True})()
    runner.temp_folder = str(tmp_path)
    runner.placeholder_filepath = str(tmp_path / "sr_placeholder.tif")
    runner.output_file_path = runner.placeholder_filepath
    runner.final_sr_path = str(tmp_path / "sr.tif")
    runner.factor = 2
    runner.image_meta = {
        "bands": 4,
        "width": 3,
        "height": 2,
        "dtype": "uint8",
        "crs": "EPSG:4326",
        "transform": from_origin(0, 2, 9, 9),
    }
    runner._log = lambda message: None
    index_path = tmp_path / "index.json"
    index_path.write_text(
        '{"factor":3,"output_bands":2,"saved_dtype":"uint16","entries":[]}'
    )

    monkeypatch.setattr(writing_utils, "ddp_safe_stitch", lambda self, **kwargs: "ok")

    assert runner.write_to_file(index_path=str(index_path)) == "ok"
    with rasterio.open(runner.placeholder_filepath) as src:
        assert (src.width, src.height, src.count) == (9, 6, 2)
        assert src.dtypes == ("uint16", "uint16")
