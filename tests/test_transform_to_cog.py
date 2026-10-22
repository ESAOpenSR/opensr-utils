from __future__ import annotations

import sys
import types

import pytest

from opensr_utils.data_utils import transform_to_cog


def install_fake_cog_validate(monkeypatch, result=(True, [], [])):
    package = types.ModuleType("rio_cogeo")
    cogeo = types.ModuleType("rio_cogeo.cogeo")
    cogeo.cog_validate = lambda path: result
    tqdm_module = types.ModuleType("tqdm")

    class FakeTqdm:
        def __init__(self, iterable, *args, **kwargs):
            self.iterable = iterable

        def __iter__(self):
            return iter(self.iterable)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def set_postfix(self, **kwargs):
            self.postfix = kwargs

    tqdm_module.tqdm = FakeTqdm
    monkeypatch.setitem(sys.modules, "rio_cogeo", package)
    monkeypatch.setitem(sys.modules, "rio_cogeo.cogeo", cogeo)
    monkeypatch.setitem(sys.modules, "tqdm", tqdm_module)


def test_cog_profile_uses_deflate_profile(monkeypatch):
    package = types.ModuleType("rio_cogeo")
    profiles = types.ModuleType("rio_cogeo.profiles")
    profiles.cog_profiles = {
        "deflate": {"compress": "zstd", "blocksize": 128, "bigtiff": "NO"}
    }
    monkeypatch.setitem(sys.modules, "rio_cogeo", package)
    monkeypatch.setitem(sys.modules, "rio_cogeo.profiles", profiles)

    profile = transform_to_cog._cog_profile()

    assert profile["compress"] == "deflate"
    assert profile["blocksize"] == 512
    assert profile["bigtiff"] == "IF_SAFER"


def test_make_cog_falls_back_to_rio_cogeo(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        transform_to_cog,
        "make_cog_rio",
        lambda *args: (_ for _ in ()).throw(RuntimeError("rio failed")),
    )
    monkeypatch.setattr(
        transform_to_cog, "make_cog_rcg", lambda *args: calls.append(args)
    )

    transform_to_cog.make_cog(tmp_path / "in.tif", tmp_path / "out.tif")

    assert calls == [(tmp_path / "in.tif", tmp_path / "out.tif")]


def test_readable_with_checksums_opens_valid_raster(write_raster, tmp_path):
    __import__("rasterio")
    tif = write_raster(tmp_path / "sr.tif", data=[[[1, 2], [3, 4]]])

    assert transform_to_cog.readable_with_checksums(tif) is True


def test_process_tree_dry_run_prints_planned_outputs(monkeypatch, tmp_path, capsys):
    install_fake_cog_validate(monkeypatch)
    sr = tmp_path / "a" / "sr.tif"
    sr.parent.mkdir()
    sr.write_bytes(b"not read in dry run")

    assert transform_to_cog.process_tree(tmp_path, dry_run=True) == (0, 0)

    output = capsys.readouterr().out
    assert f"Found: {sr}" in output
    assert f"Would create: {sr.with_name('sr_cog.tif')}" in output


def test_process_tree_counts_success_and_can_remove_original(monkeypatch, tmp_path):
    install_fake_cog_validate(monkeypatch)
    sr = tmp_path / "nested" / "sr.tif"
    sr.parent.mkdir()
    sr.write_bytes(b"input")

    def fake_make_cog(in_tif, out_tif):
        out_tif.write_bytes(b"output")

    monkeypatch.setattr(transform_to_cog, "make_cog", fake_make_cog)
    monkeypatch.setattr(transform_to_cog, "readable_with_checksums", lambda path: True)

    assert transform_to_cog.process_tree(tmp_path, remove_original=True) == (1, 0)
    assert not sr.exists()
    assert sr.with_name("sr_cog.tif").exists()


def test_process_tree_counts_validation_failures(monkeypatch, tmp_path):
    install_fake_cog_validate(monkeypatch, result=(False, ["bad layout"], ["warn"]))
    sr = tmp_path / "sr.tif"
    sr.write_bytes(b"input")
    monkeypatch.setattr(
        transform_to_cog,
        "make_cog",
        lambda in_tif, out_tif: out_tif.write_bytes(b"output"),
    )
    monkeypatch.setattr(transform_to_cog, "readable_with_checksums", lambda path: True)

    assert transform_to_cog.process_tree(tmp_path, remove_original=True) == (0, 1)
    assert sr.exists()


@pytest.mark.integration
def test_real_cog_conversion_preserves_pixels_and_spatial_metadata(
    write_raster, tmp_path
):
    import numpy as np
    import rasterio
    from rio_cogeo.cogeo import cog_validate

    data = np.arange(3 * 16 * 20, dtype=np.uint16).reshape(3, 16, 20)
    source = write_raster(tmp_path / "source.tif", data=data, crs="EPSG:32632")
    output = tmp_path / "cog.tif"
    transform_to_cog.make_cog(source, output)
    assert cog_validate(output)[0]
    assert transform_to_cog.readable_with_checksums(output)
    with rasterio.open(source) as original, rasterio.open(output) as cog:
        np.testing.assert_array_equal(cog.read(), data)
        assert cog.crs == original.crs
        assert cog.transform == original.transform
        assert cog.nodata == original.nodata
        assert cog.dtypes == original.dtypes
    assert source.exists()


@pytest.mark.parametrize("failure", ["conversion", "validation", "readability"])
def test_cog_failure_preserves_original_and_continues_to_next_file(
    monkeypatch, tmp_path, failure
):
    install_fake_cog_validate(monkeypatch)
    sources = [tmp_path / name / "sr.tif" for name in ["bad", "good"]]
    for source in sources:
        source.parent.mkdir()
        source.write_bytes(b"original")

    def convert(source, output):
        if source.parent.name == "bad" and failure == "conversion":
            raise OSError("conversion failed")
        output.write_bytes(b"cog")

    def validate(path):
        return (not (path.parent.name == "bad" and failure == "validation"), [], [])

    monkeypatch.setattr(transform_to_cog, "make_cog", convert)
    monkeypatch.setattr(sys.modules["rio_cogeo.cogeo"], "cog_validate", validate)
    monkeypatch.setattr(
        transform_to_cog,
        "readable_with_checksums",
        lambda path: not (path.parent.name == "bad" and failure == "readability"),
    )
    assert transform_to_cog.process_tree(tmp_path, remove_original=True) == (1, 1)
    assert sources[0].read_bytes() == b"original"
    assert not sources[1].exists()


def test_cog_dry_run_does_not_modify_existing_files(monkeypatch, tmp_path):
    install_fake_cog_validate(monkeypatch)
    source, output = tmp_path / "sr.tif", tmp_path / "sr_cog.tif"
    source.write_bytes(b"source")
    output.write_bytes(b"existing cog")
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    assert transform_to_cog.process_tree(
        tmp_path, dry_run=True, remove_original=True
    ) == (0, 0)
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before
