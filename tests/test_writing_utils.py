from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import numpy as np
import rasterio

from rasterio.transform import from_origin  # noqa: E402
from rasterio.windows import Window  # noqa: E402

from opensr_utils.data_utils import writing_utils  # noqa: E402


def make_placeholder(path, *, width=12, height=4, count=1, dtype="uint16"):
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=count,
        dtype=dtype,
        transform=from_origin(0, height, 1, 1),
        crs="EPSG:4326",
        nodata=0,
    ):
        pass
    return path


def test_write_to_placeholder_writes_patch_and_accepts_hwc_input(
    tmp_path, dummy_runner
):
    path = make_placeholder(tmp_path / "placeholder.tif", width=3, height=4, count=3)
    image_meta = {
        "placeholder_path": str(path),
        "dtype": "uint16",
        "window_coordinates": [Window(0, 0, 3, 4)],
    }
    sr = np.zeros((4, 3, 3), dtype=np.uint16)
    sr[..., 0] = 100
    sr[..., 1] = 200
    sr[..., 2] = 300

    writing_utils.write_to_placeholder(dummy_runner, sr, 0, image_meta, factor=1)

    with rasterio.open(path) as src:
        written = src.read()
    assert written.shape == (3, 4, 3)
    assert written[:, 0, 0].tolist() == [100, 200, 300]


def test_write_to_placeholder_blends_overlapping_patch_edges(tmp_path, dummy_runner):
    path = make_placeholder(tmp_path / "placeholder.tif", width=12, height=4, count=1)
    image_meta = {
        "placeholder_path": str(path),
        "dtype": "uint16",
        "window_coordinates": [Window(0, 0, 2, 1), Window(1, 0, 2, 1)],
    }

    writing_utils.write_to_placeholder(
        dummy_runner,
        np.full((1, 4, 8), 100, dtype=np.uint16),
        0,
        image_meta,
        factor=4,
        overlap=4,
    )
    writing_utils.write_to_placeholder(
        dummy_runner,
        np.full((1, 4, 8), 200, dtype=np.uint16),
        1,
        image_meta,
        factor=4,
        overlap=4,
        profile="linear",
    )

    with rasterio.open(path) as src:
        row = src.read(1)[0]
    assert row[:4].tolist() == [100, 100, 100, 100]
    assert row[4:8].tolist() == [100, 133, 166, 200]
    assert row[8:].tolist() == [200, 200, 200, 200]


def test_write_to_placeholder_blends_valid_zero_predictions(tmp_path, dummy_runner):
    path = make_placeholder(tmp_path / "placeholder.tif", width=12, height=4, count=1)
    image_meta = {
        "placeholder_path": str(path),
        "dtype": "uint16",
        "window_coordinates": [Window(0, 0, 2, 1), Window(1, 0, 2, 1)],
    }

    writing_utils.write_to_placeholder(
        dummy_runner,
        np.zeros((1, 4, 8), dtype=np.uint16),
        0,
        image_meta,
        factor=4,
        overlap=4,
    )
    writing_utils.write_to_placeholder(
        dummy_runner,
        np.full((1, 4, 8), 100, dtype=np.uint16),
        1,
        image_meta,
        factor=4,
        overlap=4,
        profile="linear",
    )

    with rasterio.open(path) as src:
        row = src.read(1)[0]
    assert row[4:8].tolist() == [0, 33, 66, 100]


def test_write_to_placeholder_discards_both_internal_patch_edges(
    tmp_path, dummy_runner
):
    path = make_placeholder(tmp_path / "placeholder.tif", width=12, height=4, count=1)
    image_meta = {
        "placeholder_path": str(path),
        "dtype": "uint16",
        "window_coordinates": [Window(0, 0, 2, 1), Window(1, 0, 2, 1)],
    }
    left = np.full((1, 4, 8), 100, dtype=np.uint16)
    right = np.full((1, 4, 8), 200, dtype=np.uint16)
    left[:, :, -1] = 9000
    right[:, :, 0] = 8000

    writing_utils.write_to_placeholder(
        dummy_runner,
        left,
        0,
        image_meta,
        factor=4,
        overlap=4,
        eliminate_border_px=1,
        profile="linear",
    )
    writing_utils.write_to_placeholder(
        dummy_runner,
        right,
        1,
        image_meta,
        factor=4,
        overlap=4,
        eliminate_border_px=1,
        profile="linear",
    )

    with rasterio.open(path) as src:
        row = src.read(1)[0]
    assert row[4:8].tolist() == [100, 100, 200, 200]
    assert 8000 not in row
    assert 9000 not in row


def test_write_to_placeholder_discards_vertical_internal_patch_edges(
    tmp_path, dummy_runner
):
    path = make_placeholder(tmp_path / "placeholder.tif", width=4, height=12, count=1)
    image_meta = {
        "placeholder_path": str(path),
        "dtype": "uint16",
        "window_coordinates": [Window(0, 0, 1, 2), Window(0, 1, 1, 2)],
    }
    top = np.full((1, 8, 4), 100, dtype=np.uint16)
    bottom = np.full((1, 8, 4), 200, dtype=np.uint16)
    top[:, -1, :] = 9000
    bottom[:, 0, :] = 8000

    writing_utils.write_to_placeholder(
        dummy_runner,
        top,
        0,
        image_meta,
        factor=4,
        overlap=4,
        eliminate_border_px=1,
        profile="linear",
    )
    writing_utils.write_to_placeholder(
        dummy_runner,
        bottom,
        1,
        image_meta,
        factor=4,
        overlap=4,
        eliminate_border_px=1,
        profile="linear",
    )

    with rasterio.open(path) as src:
        col = src.read(1)[:, 0]
    assert col[4:8].tolist() == [100, 100, 200, 200]
    assert 8000 not in col
    assert 9000 not in col


def test_write_to_placeholder_blending_is_order_independent(tmp_path, dummy_runner):
    arrays = [
        np.full((1, 4, 8), 100, dtype=np.uint16),
        np.full((1, 4, 8), 200, dtype=np.uint16),
    ]

    def run_case(name, order):
        path = make_placeholder(tmp_path / name, width=12, height=4, count=1)
        image_meta = {
            "placeholder_path": str(path),
            "dtype": "uint16",
            "window_coordinates": [Window(0, 0, 2, 1), Window(1, 0, 2, 1)],
        }
        for idx in order:
            writing_utils.write_to_placeholder(
                dummy_runner,
                arrays[idx],
                idx,
                image_meta,
                factor=4,
                overlap=4,
                profile="linear",
            )
        with rasterio.open(path) as src:
            return src.read(1)

    np.testing.assert_array_equal(
        run_case("forward.tif", [0, 1]),
        run_case("reverse.tif", [1, 0]),
    )


def test_write_to_placeholder_clips_integer_outputs(tmp_path, dummy_runner):
    path = make_placeholder(
        tmp_path / "placeholder.tif", width=2, height=2, count=1, dtype="uint8"
    )
    image_meta = {
        "placeholder_path": str(path),
        "dtype": "uint8",
        "window_coordinates": [Window(0, 0, 2, 2)],
    }

    writing_utils.write_to_placeholder(
        dummy_runner, np.full((1, 2, 2), 999, dtype=np.float32), 0, image_meta, factor=1
    )

    with rasterio.open(path) as src:
        assert src.read(1).max() == 255


def test_write_to_placeholder_rejects_unknown_profile_when_blending(
    tmp_path, dummy_runner
):
    path = make_placeholder(tmp_path / "placeholder.tif", width=6, height=4, count=1)
    with rasterio.open(path, "r+") as dst:
        dst.write(np.ones((1, 4, 6), dtype=np.uint16), indexes=[1])
    image_meta = {
        "placeholder_path": str(path),
        "dtype": "uint16",
        "window_coordinates": [Window(1, 0, 4, 4)],
    }

    with pytest.raises(ValueError, match="Unknown profile"):
        writing_utils.write_to_placeholder(
            dummy_runner,
            np.full((1, 4, 4), 5, dtype=np.uint16),
            0,
            image_meta,
            factor=1,
            overlap=2,
            profile="bad",
        )


def test_write_to_placeholder_rejects_band_mismatch(tmp_path, dummy_runner):
    path = make_placeholder(tmp_path / "placeholder.tif", width=2, height=2, count=2)
    image_meta = {
        "placeholder_path": str(path),
        "dtype": "uint16",
        "window_coordinates": [Window(0, 0, 2, 2)],
    }

    with pytest.raises(ValueError, match="Band mismatch"):
        writing_utils.write_to_placeholder(
            dummy_runner, np.ones((3, 3, 3), dtype=np.uint16), 0, image_meta, factor=1
        )


def test_stitch_sr_patches_loads_index_writes_final_and_cleans_patches(tmp_path):
    placeholder = make_placeholder(
        tmp_path / "sr_placeholder_unit.tif", width=4, height=2, count=1
    )
    patch_a = tmp_path / "a.npy"
    patch_b = tmp_path / "b.npy"
    np.save(patch_a, np.full((1, 2, 2), 10, dtype=np.uint16))
    np.save(patch_b, np.full((1, 2, 2), 20, dtype=np.uint16))
    index = {
        "saved_container": "npy",
        "saved_key": "arr",
        "output_bands": 1,
        "entries": [
            {
                "path": str(patch_b),
                "row_off_lr": 0,
                "col_off_lr": 2,
                "width_lr": 2,
                "height_lr": 2,
            },
            {
                "path": str(patch_a),
                "row_off_lr": 0,
                "col_off_lr": 0,
                "width_lr": 2,
                "height_lr": 2,
            },
        ],
    }
    index_path = tmp_path / "index.json"
    index_path.write_text(json.dumps(index))
    runner = SimpleNamespace(
        temp_folder=str(tmp_path),
        placeholder_filepath=str(placeholder),
        final_sr_path=str(tmp_path / "sr.tif"),
        factor=1,
        overlap=0,
        eliminate_border_px=0,
        image_meta={"dtype": "uint16"},
        _log=lambda message: None,
    )

    out, missing = writing_utils.stitch_sr_patches(
        runner, index_path=str(index_path), cleanup_every=1
    )

    assert out == str(tmp_path / "sr.tif")
    assert missing == 0
    assert not placeholder.exists()
    assert not (tmp_path / "sr_placeholder_unit.tif.sum.tif").exists()
    assert not (tmp_path / "sr_placeholder_unit.tif.weights.tif").exists()
    assert not patch_a.exists()
    assert not patch_b.exists()
    with rasterio.open(out) as src:
        assert src.read(1).tolist() == [[10, 10, 20, 20], [10, 10, 20, 20]]


def test_stitch_sr_patches_reports_missing_patch_files(tmp_path):
    placeholder = make_placeholder(
        tmp_path / "sr_placeholder_unit.tif", width=2, height=2, count=1
    )
    index_path = tmp_path / "index.json"
    index_path.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "path": str(tmp_path / "missing.npy"),
                        "row_off_lr": 0,
                        "col_off_lr": 0,
                        "width_lr": 2,
                        "height_lr": 2,
                    }
                ]
            }
        )
    )
    runner = SimpleNamespace(
        temp_folder=str(tmp_path),
        placeholder_filepath=str(placeholder),
        final_sr_path=str(tmp_path / "sr.tif"),
        factor=1,
        overlap=0,
        eliminate_border_px=0,
        image_meta={"dtype": "uint16"},
        _log=lambda message: None,
    )

    with pytest.raises(RuntimeError, match="patch files were missing"):
        writing_utils.stitch_sr_patches(runner, index_path=str(index_path))


def test_stitch_sr_patches_rejects_placeholder_band_mismatch(tmp_path):
    placeholder = make_placeholder(
        tmp_path / "sr_placeholder_unit.tif", width=2, height=2, count=1
    )
    patch = tmp_path / "patch.npy"
    np.save(patch, np.ones((2, 2, 2), dtype=np.uint16))
    index_path = tmp_path / "index.json"
    index_path.write_text(
        json.dumps(
            {
                "saved_container": "npy",
                "output_bands": 2,
                "entries": [
                    {
                        "path": str(patch),
                        "row_off_lr": 0,
                        "col_off_lr": 0,
                        "width_lr": 2,
                        "height_lr": 2,
                        "bands": 2,
                    }
                ],
            }
        )
    )
    runner = SimpleNamespace(
        temp_folder=str(tmp_path),
        placeholder_filepath=str(placeholder),
        final_sr_path=str(tmp_path / "sr.tif"),
        factor=1,
        overlap=0,
        eliminate_border_px=0,
        image_meta={"dtype": "uint16"},
        _log=lambda message: None,
    )

    with pytest.raises(ValueError, match="Placeholder has 1 bands"):
        writing_utils.stitch_sr_patches(runner, index_path=str(index_path))


def test_ddp_safe_stitch_skips_non_rank_zero(tmp_path):
    runner = SimpleNamespace(trainer=SimpleNamespace(is_global_zero=False))

    assert writing_utils.ddp_safe_stitch(runner) is None


def test_ddp_safe_stitch_refuses_existing_output_without_overwrite(tmp_path):
    placeholder = tmp_path / "sr_placeholder.tif"
    final = tmp_path / "sr.tif"
    final.write_bytes(b"existing")
    runner = SimpleNamespace(
        trainer=SimpleNamespace(is_global_zero=True),
        placeholder_filepath=str(placeholder),
        final_sr_path=str(final),
        overwrite=False,
        _log=lambda message: None,
    )

    with pytest.raises(FileExistsError, match="Output already exists"):
        writing_utils.ddp_safe_stitch(runner)


def test_ddp_safe_stitch_overwrites_existing_output_and_calls_stitcher(
    monkeypatch, tmp_path
):
    placeholder = tmp_path / "sr_placeholder.tif"
    final = tmp_path / "sr.tif"
    final.write_bytes(b"existing")
    runner = SimpleNamespace(
        trainer=SimpleNamespace(is_global_zero=True),
        placeholder_filepath=str(placeholder),
        final_sr_path=str(final),
        overwrite=True,
        _log=lambda message: None,
    )
    calls = []

    def fake_stitch(self, **kwargs):
        calls.append(kwargs)
        return ("done.tif", 0)

    monkeypatch.setattr(writing_utils, "stitch_sr_patches", fake_stitch)

    assert writing_utils.ddp_safe_stitch(runner, limit=3) == ("done.tif", 0)
    assert final.read_bytes() == b"existing"
    assert calls[0]["limit"] == 3


@pytest.mark.parametrize("profile", ["linear", "sigmoid", "cosine"])
def test_four_way_overlap_has_an_analytic_weighted_mean_and_no_black_frame(
    tmp_path, dummy_runner, profile
):
    path = make_placeholder(tmp_path / "four_way.tif", width=7, height=7, count=2)
    windows = [
        Window(0, 0, 5, 5),
        Window(2, 0, 5, 5),
        Window(0, 2, 5, 5),
        Window(2, 2, 5, 5),
    ]
    meta = {"placeholder_path": str(path), "window_coordinates": windows}
    # At (3,3), all four tiles have equal weights for all symmetric curves.
    for index, value in enumerate([0, 100, 200, 300]):
        data = np.stack([np.full((5, 5), value), np.full((5, 5), value + 1000)])
        writing_utils.write_to_placeholder(
            dummy_runner, data, index, meta, factor=1, overlap=3, profile=profile
        )
    with rasterio.open(path) as src:
        result = src.read()
    np.testing.assert_allclose(result[:, 3, 3], [150, 1150], atol=1, rtol=0)
    np.testing.assert_array_equal(result[:, 0, 0], [0, 1000])
    np.testing.assert_array_equal(result[:, -1, -1], [300, 1300])
    assert np.all(result[1] >= 1000)


@pytest.mark.parametrize("profile", ["linear", "sigmoid", "cosine"])
@pytest.mark.parametrize("reverse", [False, True])
def test_vertical_and_horizontal_blending_agree_under_transposition(
    tmp_path, dummy_runner, profile, reverse
):
    def run(transposed):
        path = make_placeholder(
            tmp_path / f"{transposed}.tif",
            width=4 if transposed else 12,
            height=12 if transposed else 4,
        )
        windows = (
            [Window(0, 0, 4, 8), Window(0, 4, 4, 8)]
            if transposed
            else [Window(0, 0, 8, 4), Window(4, 0, 8, 4)]
        )
        meta = {"placeholder_path": str(path), "window_coordinates": windows}
        for i in [1, 0] if reverse else [0, 1]:
            patch = np.full(
                (1, 8, 4) if transposed else (1, 4, 8), 100 * i, dtype=np.uint16
            )
            writing_utils.write_to_placeholder(
                dummy_runner,
                patch,
                i,
                meta,
                factor=1,
                overlap=4,
                eliminate_border_px=1,
                profile=profile,
            )
        with rasterio.open(path) as src:
            return src.read(1)

    np.testing.assert_array_equal(run(False), run(True).T)


@pytest.mark.parametrize(
    "layout", ["2d", "chw", "hwc", "list", "tensor", "pil", "xarray"]
)
def test_writer_accepts_documented_patch_representations(
    tmp_path, dummy_runner, layout
):
    data = np.arange(20, dtype=np.uint16).reshape(4, 5)
    if layout == "2d":
        patch = data
    elif layout == "chw":
        patch = data[None]
    elif layout == "hwc":
        patch = data[..., None]
    elif layout == "list":
        patch = [data]
    elif layout == "tensor":
        import torch

        patch = torch.tensor(data[None].astype(np.float32), requires_grad=True)
    elif layout == "pil":
        from PIL import Image

        patch = Image.fromarray(data)
    else:
        import xarray as xr

        patch = xr.DataArray(data[None], dims=("band", "y", "x"))
    path = make_placeholder(tmp_path / "output.tif", width=5, height=4)
    meta = {"placeholder_path": str(path), "window_coordinates": [Window(0, 0, 5, 4)]}
    writing_utils.write_to_placeholder(
        dummy_runner, patch, 0, meta, factor=1, overlap=0
    )
    with rasterio.open(path) as src:
        np.testing.assert_array_equal(src.read(1), data)


@pytest.mark.parametrize(
    "window",
    [Window(-1, 0, 2, 2), Window(0, -1, 2, 2), Window(1, 0, 2, 2), Window(0, 1, 2, 2)],
)
def test_out_of_bounds_patches_fail_without_modifying_output(
    tmp_path, dummy_runner, window
):
    path = make_placeholder(tmp_path / "output.tif", width=2, height=2)
    meta = {"placeholder_path": str(path), "window_coordinates": [window]}
    with pytest.raises(ValueError, match="exceeds placeholder bounds"):
        writing_utils.write_to_placeholder(
            dummy_runner, np.ones((1, 2, 2)), 0, meta, factor=1, overlap=0
        )
    with rasterio.open(path) as src:
        assert not src.read().any()
    assert not list(tmp_path.glob("*.sum.tif"))


@pytest.mark.parametrize("patch", [np.zeros((1, 1, 2, 2)), np.zeros(3), "invalid"])
def test_writer_rejects_unsupported_shapes_and_types(tmp_path, dummy_runner, patch):
    path = make_placeholder(tmp_path / "output.tif", width=2, height=2)
    meta = {"placeholder_path": str(path), "window_coordinates": [Window(0, 0, 2, 2)]}
    with pytest.raises(TypeError):
        writing_utils.write_to_placeholder(
            dummy_runner, patch, 0, meta, factor=1, overlap=0
        )


def saved_patch_run(tmp_path, *, container="npy", coverage="complete", factor=1):
    """A hand-written prediction manifest: no model or manifest writer involved."""
    placeholder = make_placeholder(
        tmp_path / "placeholder.tif", width=4 * factor, height=2 * factor
    )
    entries = []
    for x, value in [(0, 100), (2, 200)]:
        path = tmp_path / f"patch_{x}.{container}"
        data = np.full((1, 2 * factor, 2 * factor), value, dtype=np.uint16)
        if container == "npz":
            np.savez_compressed(path, prediction=data)
        else:
            np.save(path, data)
        entries.append(
            dict(
                path=str(path),
                row_off_lr=0,
                col_off_lr=x,
                width_lr=2,
                height_lr=2,
                bands=1,
            )
        )
    if coverage == "gap":
        entries.pop()
    elif coverage == "missing":
        Path(entries[-1]["path"]).unlink()
    elif coverage == "corrupt":
        Path(entries[-1]["path"]).write_bytes(b"invalid numpy payload")
    (tmp_path / "index.json").write_text(
        json.dumps(
            dict(
                entries=entries[::-1],
                factor=factor,
                saved_container=container,
                saved_key="prediction",
                output_bands=1,
            )
        )
    )
    return SimpleNamespace(
        temp_folder=str(tmp_path),
        placeholder_filepath=str(placeholder),
        final_sr_path=str(tmp_path / "sr.tif"),
        factor=1,
        overlap=0,
        eliminate_border_px=0,
        image_meta={"dtype": "uint16"},
        _log=lambda message: None,
    )


@pytest.mark.parametrize("container", ["npy", "npz"])
@pytest.mark.parametrize("factor", [1, 3])
def test_stitcher_obeys_manifest_scale_container_and_custom_key(
    tmp_path, container, factor
):
    runner = saved_patch_run(tmp_path, container=container, factor=factor)
    output, missing = writing_utils.stitch_sr_patches(runner)
    assert missing == 0
    with rasterio.open(output) as src:
        expected = np.repeat(np.array([[100, 100, 200, 200]]), 2 * factor, axis=0)
        np.testing.assert_array_equal(src.read(1), np.repeat(expected, factor, axis=1))


@pytest.mark.parametrize(
    "coverage,exception",
    [("gap", RuntimeError), ("missing", RuntimeError), ("corrupt", ValueError)],
)
def test_failed_stitch_does_not_publish_incomplete_output_and_keeps_diagnostics(
    tmp_path, coverage, exception
):
    runner = saved_patch_run(tmp_path, coverage=coverage)
    with pytest.raises(exception):
        writing_utils.stitch_sr_patches(runner)
    assert not Path(runner.final_sr_path).exists()
    assert Path(runner.placeholder_filepath).exists()
    # Failed stitches retain accumulators as diagnostics for inspection.
    assert list(tmp_path.glob("*.sum.tif"))
    assert list(tmp_path.glob("*.weights.tif"))


def test_failed_overwrite_keeps_last_successful_output(tmp_path):
    runner = saved_patch_run(tmp_path, coverage="missing")
    runner.overwrite = True
    final = Path(runner.final_sr_path)
    final.write_bytes(b"last successful output")
    with pytest.raises(RuntimeError):
        writing_utils.ddp_safe_stitch(runner)
    assert final.exists()
    assert final.read_bytes() == b"last successful output"


def test_partial_limit_does_not_publish_or_replace_existing_output(tmp_path):
    runner = saved_patch_run(tmp_path)
    runner.overwrite = True
    final = Path(runner.final_sr_path)
    final.write_bytes(b"last successful output")

    with pytest.raises(RuntimeError, match="output pixels were not covered"):
        writing_utils.ddp_safe_stitch(runner, limit=1)

    assert final.read_bytes() == b"last successful output"
    assert Path(runner.placeholder_filepath).exists()


def test_cleanup_false_retains_manifest_and_every_prediction(tmp_path):
    runner = saved_patch_run(tmp_path)
    patch_paths = [
        Path(entry["path"])
        for entry in json.loads((tmp_path / "index.json").read_text())["entries"]
    ]

    writing_utils.stitch_sr_patches(runner, cleanup=False)

    assert (tmp_path / "index.json").exists()
    assert all(path.exists() for path in patch_paths)


def test_concurrent_destination_creation_cannot_bypass_no_overwrite(
    monkeypatch, tmp_path
):
    runner = saved_patch_run(tmp_path)
    final = Path(runner.final_sr_path)
    count_uncovered = writing_utils._count_uncovered_pixels

    def concurrent_publication(weights):
        uncovered = count_uncovered(weights)
        final.write_bytes(b"another successful run")
        return uncovered

    monkeypatch.setattr(
        writing_utils, "_count_uncovered_pixels", concurrent_publication
    )
    with pytest.raises(FileExistsError, match="Output already exists"):
        writing_utils.ddp_safe_stitch(runner)
    assert final.read_bytes() == b"another successful run"
    assert Path(runner.placeholder_filepath).exists()
    assert len(list(tmp_path.glob("patch_*.npy"))) == 2


def test_successful_overwrite_removes_stale_external_raster_sidecars(tmp_path):
    runner = saved_patch_run(tmp_path)
    runner.overwrite = True
    final = Path(runner.final_sr_path)
    make_placeholder(final, width=4, height=2)
    with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=False):
        with rasterio.open(final, "r+") as old:
            old.write_mask(np.zeros((2, 4), dtype=np.uint8))
    assert Path(str(final) + ".msk").exists()
    for suffix in (".ovr", ".aux.xml"):
        Path(str(final) + suffix).write_bytes(b"old raster sidecar")

    writing_utils.ddp_safe_stitch(runner)
    for suffix in (".msk", ".ovr", ".aux.xml"):
        assert not Path(str(final) + suffix).exists()
    with rasterio.open(final) as result:
        np.testing.assert_array_equal(result.read(1), [[100, 100, 200, 200]] * 2)
        assert np.all(result.read_masks(1) == 255)


def test_failed_overwrite_preserves_previous_raster_sidecars(tmp_path):
    runner = saved_patch_run(tmp_path, coverage="missing")
    runner.overwrite = True
    final = Path(runner.final_sr_path)
    final.write_bytes(b"previous raster")
    sidecar = Path(str(final) + ".msk")
    sidecar.write_bytes(b"previous mask")
    with pytest.raises(RuntimeError):
        writing_utils.ddp_safe_stitch(runner)
    assert final.read_bytes() == b"previous raster"
    assert sidecar.read_bytes() == b"previous mask"
