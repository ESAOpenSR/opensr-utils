from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

os.environ.setdefault("MPLBACKEND", "Agg")


class DummyRunner(SimpleNamespace):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.messages = []

    def _log(self, message):
        self.messages.append(message)


@pytest.fixture(autouse=True)
def clean_rank_env(monkeypatch):
    for name in ("LOCAL_RANK", "RANK", "WORLD_SIZE", "OPENSR_RUN_ID"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(autouse=True)
def block_http(monkeypatch):
    """Network/model downloads belong behind a mocked boundary in this suite."""
    import requests

    def unexpected_request(*args, **kwargs):
        pytest.fail("Unexpected HTTP request: mock model downloads in tests")

    monkeypatch.setattr(requests.sessions.Session, "request", unexpected_request)


@pytest.fixture
def dummy_runner():
    return DummyRunner()


@pytest.fixture
def write_raster():
    def _write_raster(
        path: Path,
        data=None,
        *,
        width: int = 8,
        height: int = 8,
        count: int = 1,
        dtype: str = "uint16",
        transform=None,
        crs: str = "EPSG:4326",
        nodata=0,
    ):
        import numpy as np
        import rasterio
        from rasterio.transform import from_origin

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if data is None:
            base = np.arange(count * height * width, dtype=np.float32).reshape(
                count, height, width
            )
            data = base.astype(dtype)
        else:
            data = np.asarray(data).astype(dtype)
            if data.ndim == 2:
                data = data[None, ...]
            count, height, width = data.shape

        profile = {
            "driver": "GTiff",
            "height": height,
            "width": width,
            "count": count,
            "dtype": str(data.dtype),
            "crs": crs,
            "transform": transform or from_origin(0, height, 1, 1),
            "nodata": nodata,
        }
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(data)
        return path

    return _write_raster


@pytest.fixture
def pipeline_factory():
    """Build CPU runners without inference; close raster handles after each test."""
    from opensr_utils.pipeline import large_file_processing

    runners = []

    def make(root, **overrides):
        options = dict(
            model=None,
            window_size=(8, 8),
            factor=2,
            overlap=4,
            eliminate_border_px=0,
            device="cpu",
            batch_size=2,
            num_workers=0,
            prefetch_factor=0,
            auto_run=False,
            cleanup=False,
        )
        options.update(overrides)
        runner = large_file_processing(root=str(root), **options)
        runners.append(runner)
        return runner

    yield make
    for runner in runners:
        runner.datamodule.dataset.close()


@pytest.fixture
def dataset_factory():
    from opensr_utils.data_utils.datamodule import PredictionDataset

    datasets = []

    def make(**kwargs):
        dataset = PredictionDataset(**kwargs)
        datasets.append(dataset)
        return dataset

    yield make
    for dataset in datasets:
        dataset.close()


@pytest.fixture
def safe_folder(tmp_path):
    root = tmp_path / "S2A_TEST.SAFE" / "GRANULE" / "L1C_TEST" / "IMG_DATA"
    root.mkdir(parents=True)
    for band in ("B02", "B03", "B04", "B08"):
        (root / f"T_TEST_{band}_10m.jp2").write_bytes(b"placeholder")
    return root.parents[2]


@pytest.fixture
def s2gm_folder(tmp_path, write_raster):
    root = tmp_path / "example_S2GM"
    root.mkdir()
    for band, value in {"B02": 2000, "B03": 3000, "B04": 4000, "B08": 8000}.items():
        write_raster(root / f"{band}.tif", data=[[[value] * 4] * 4], width=4, height=4)
    return root


@pytest.fixture
def chdir_tmp(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path


def assert_no_unexpected_delete(path: Path):
    assert os.path.exists(path)
