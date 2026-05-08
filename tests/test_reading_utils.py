from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

pytest.importorskip("rasterio")

from opensr_utils.data_utils import reading_utils


def test_can_read_directly_with_rasterio_accepts_valid_raster(
    write_raster, dummy_runner, tmp_path
):
    tif = write_raster(tmp_path / "scene.tif", data=[[[1, 2], [3, 4]]])

    assert reading_utils.can_read_directly_with_rasterio(dummy_runner, str(tif)) is True
    assert dummy_runner.messages == []


def test_can_read_directly_with_rasterio_rejects_unknown_extension(
    dummy_runner, tmp_path
):
    path = tmp_path / "scene.txt"
    path.write_text("not a raster")

    assert (
        reading_utils.can_read_directly_with_rasterio(dummy_runner, str(path)) is False
    )
    assert "Unsupported file extension" in dummy_runner.messages[-1]


def test_can_read_directly_with_rasterio_logs_rasterio_error(dummy_runner, tmp_path):
    path = tmp_path / "broken.tif"
    path.write_text("not a geotiff")

    assert (
        reading_utils.can_read_directly_with_rasterio(dummy_runner, str(path)) is False
    )
    assert "Rasterio can't open" in dummy_runner.messages[-1]


def test_verify_input_file_type_detects_file(write_raster, dummy_runner, tmp_path):
    tif = write_raster(tmp_path / "scene.tif")

    reading_utils.verify_input_file_type(dummy_runner, str(tif))

    assert dummy_runner.root == str(tif)
    assert dummy_runner.input_type == "file"
    assert any("Raster file OK" in msg for msg in dummy_runner.messages)


def test_verify_input_file_type_detects_safe_folder(dummy_runner, safe_folder):
    reading_utils.verify_input_file_type(dummy_runner, str(safe_folder))

    assert dummy_runner.root == str(safe_folder)
    assert dummy_runner.input_type == "SAFE"


def test_verify_input_file_type_detects_s2gm_folder(dummy_runner, s2gm_folder):
    reading_utils.verify_input_file_type(dummy_runner, str(s2gm_folder))

    assert dummy_runner.root == str(s2gm_folder)
    assert dummy_runner.input_type == "S2GM"


def test_verify_input_file_type_extracts_safe_zip(dummy_runner, tmp_path):
    archive = tmp_path / "download.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("nested/S2A_TEST.SAFE/manifest.safe", "")

    reading_utils.verify_input_file_type(dummy_runner, str(archive))

    assert dummy_runner.input_type == "SAFE"
    assert dummy_runner.root.endswith("S2A_TEST.SAFE")
    assert Path(dummy_runner.root).is_dir()
    assert archive.exists()


def test_verify_input_file_type_can_delete_zip_after_extract(dummy_runner, tmp_path):
    archive = tmp_path / "download.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("S2A_TEST.SAFE/manifest.safe", "")
    dummy_runner.delete_input_zip = True

    reading_utils.verify_input_file_type(dummy_runner, str(archive))

    assert dummy_runner.input_type == "SAFE"
    assert not archive.exists()


def test_verify_input_file_type_rejects_zip_slip(dummy_runner, tmp_path):
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("../outside.SAFE/manifest.safe", "")

    with pytest.raises(RuntimeError, match="Unsafe path"):
        reading_utils.verify_input_file_type(dummy_runner, str(archive))


def test_verify_input_file_type_rejects_unsupported_paths(dummy_runner, tmp_path):
    with pytest.raises(NotImplementedError, match="not valid"):
        reading_utils.verify_input_file_type(dummy_runner, str(tmp_path / "missing"))


def test_create_dirs_uses_file_parent_and_run_id(dummy_runner, write_raster, tmp_path):
    tif = write_raster(tmp_path / "input.tif")
    dummy_runner.root = str(tif)
    dummy_runner.input_type = "file"
    dummy_runner.image_meta = {}
    dummy_runner.run_id = "unit"

    reading_utils.create_dirs(dummy_runner)

    assert dummy_runner.placeholder_path == str(tmp_path)
    assert dummy_runner.output_dir == str(tmp_path)
    assert Path(dummy_runner.log_dir).name == "logs_unit"
    assert Path(dummy_runner.temp_folder).name == "temp_unit"
    assert Path(dummy_runner.log_file).exists()
    assert dummy_runner.placeholder_filepath == str(
        tmp_path / "sr_placeholder_unit.tif"
    )
    assert dummy_runner.image_meta["final_sr_path"] == str(tmp_path / "sr.tif")


def test_create_dirs_uses_s2gm_folder_as_base(dummy_runner, s2gm_folder):
    dummy_runner.root = str(s2gm_folder)
    dummy_runner.input_type = "S2GM"
    dummy_runner.image_meta = {}
    dummy_runner.run_id = "unit"

    reading_utils.create_dirs(dummy_runner)

    assert dummy_runner.placeholder_path == str(s2gm_folder)
    assert Path(dummy_runner.log_dir).parent == s2gm_folder
