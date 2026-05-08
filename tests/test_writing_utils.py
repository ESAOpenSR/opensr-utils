from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")
rasterio = pytest.importorskip("rasterio")

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
    assert not final.exists()
    assert calls[0]["limit"] == 3
