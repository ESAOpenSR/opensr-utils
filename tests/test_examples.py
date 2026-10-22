"""Smoke tests for the import-light instructional scripts."""

from __future__ import annotations

import subprocess
import sys
import types
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_demo_help_is_available_without_model_imports():
    result = subprocess.run(
        [sys.executable, str(ROOT / "examples" / "run_model.py"), "--help"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--checkpoint" in result.stdout
    assert "--model" in result.stdout


def test_model_examples_expose_help_without_downloading_weights():
    for script in ("run_sen2sr.py", "run_esasrgan.py", "run_ldsrs2.py"):
        result = subprocess.run(
            [sys.executable, str(ROOT / "examples" / script), "--help"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert "usage:" in result.stdout


def test_demo_runs_tiny_model_none_workflow(tmp_path):
    rasterio = pytest.importorskip("rasterio")
    import numpy as np

    source = tmp_path / "input.tif"
    output = tmp_path / "output.tif"
    with rasterio.open(
        source,
        "w",
        driver="GTiff",
        width=4,
        height=4,
        count=1,
        dtype="uint16",
        crs="EPSG:4326",
        transform=rasterio.transform.from_origin(0, 4, 1, 1),
    ) as dst:
        dst.write(np.arange(16, dtype="uint16").reshape(1, 4, 4))
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "examples" / "run_model.py"),
            str(source),
            "--model",
            "none",
            "--factor",
            "1",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert output.exists()


def test_demo_passes_checkpoint_options_to_model_registry(monkeypatch, tmp_path):
    from examples import run_model as demo
    import opensr_utils

    calls = []
    fake_loader = types.ModuleType("opensr_utils.model_utils.get_models")

    def get_model(name, **kwargs):
        calls.append((name, kwargs))
        return "fake-model"

    fake_loader.get_model = get_model
    monkeypatch.setitem(sys.modules, "opensr_utils.model_utils.get_models", fake_loader)
    forwarded = {}
    monkeypatch.setattr(
        opensr_utils, "super_resolve", lambda *a, **kw: forwarded.update(kw) or "result"
    )
    checkpoint = tmp_path / "model.ckpt"
    config = tmp_path / "model.yaml"
    assert (
        demo.main(
            [
                "input.tif",
                "--model",
                "ESASRGAN",
                "--checkpoint",
                str(checkpoint),
                "--config",
                str(config),
                "--no-download",
            ]
        )
        == 0
    )
    assert calls == [
        (
            "esasrgan",
            {
                "device": "cpu",
                "checkpoint": checkpoint,
                "cache_dir": None,
                "download": False,
                "config": config,
            },
        )
    ]


def test_demo_translates_cuda_device_selector_to_pipeline_gpu(monkeypatch):
    from examples import run_model as demo
    import opensr_utils

    forwarded = {}
    monkeypatch.setattr(
        opensr_utils, "super_resolve", lambda *a, **kw: forwarded.update(kw) or "result"
    )
    assert demo.main(["input.tif", "--model", "none", "--device", "cuda:3"]) == 0
    assert forwarded["device"] == "cuda"
    assert forwarded["gpus"] == [3]


def test_esasrgan_example_forwards_auto_download_and_cache_options(
    monkeypatch, tmp_path
):
    import opensr_utils
    from examples import run_esasrgan

    calls = []
    fake_loader = types.ModuleType("opensr_utils.model_utils.get_models")

    def get_model(name, **kwargs):
        calls.append((name, kwargs))
        return "fake-model"

    fake_loader.get_model = get_model
    monkeypatch.setitem(sys.modules, "opensr_utils.model_utils.get_models", fake_loader)
    monkeypatch.setattr(opensr_utils, "super_resolve", lambda *a, **kw: "result")
    cache_dir = tmp_path / "cache"
    assert run_esasrgan.main(["input.tif", "--cache-dir", str(cache_dir)]) == 0
    assert calls == [
        (
            "esasrgan",
            {
                "device": "cpu",
                "checkpoint": None,
                "config": None,
                "cache_dir": cache_dir,
                "download": True,
            },
        )
    ]


def test_sen2sr_example_selects_all_bands_and_ldsr_backbone(monkeypatch, tmp_path):
    import opensr_utils
    from examples import run_sen2sr
    from opensr_utils.model_utils import get_models

    calls = []
    monkeypatch.setattr(
        get_models,
        "get_model",
        lambda name, **kw: calls.append((name, kw)) or "composite",
    )
    forwarded = {}
    monkeypatch.setattr(
        opensr_utils, "super_resolve", lambda *a, **kw: forwarded.update(kw) or "result"
    )
    assert run_sen2sr.main(["scene.SAFE", "--sampling-steps", "20"]) == 0
    assert calls[0][1]["variant"] == "main"
    assert calls[0][1]["backbone"] == "ldsrs2"
    assert calls[0][1]["sampling_steps"] == 20
    assert forwarded["resample_inputs"] is True
    assert forwarded["batch_size"] == 1


def test_ldsrs2_example_selects_rgbn_and_forwards_sampling(monkeypatch):
    import opensr_utils
    from examples import run_ldsrs2
    from opensr_utils.model_utils import ldsrs2

    calls = []
    monkeypatch.setattr(
        ldsrs2, "load_ldsrs2", lambda **kw: calls.append(kw) or "rgbn-model"
    )
    forwarded = {}
    monkeypatch.setattr(
        opensr_utils, "super_resolve", lambda *a, **kw: forwarded.update(kw) or "result"
    )
    assert (
        run_ldsrs2.main(
            ["input.tif", "--bands", "3", "2", "1", "4", "--sampling-steps", "10"]
        )
        == 0
    )
    assert calls[0]["sampling_steps"] == 10
    assert forwarded["band_indexes"] == [3, 2, 1, 4]


@pytest.mark.parametrize(
    "script,options",
    [
        ("run_sen2sr.py", ["scene.SAFE", "--no-download"]),
        ("run_sen2sr.py", ["scene.SAFE", "--backbone-config", "config.yaml"]),
        ("run_ldsrs2.py", ["scene.SAFE", "--no-download"]),
        ("run_ldsrs2.py", ["scene.SAFE", "--config", "config.yaml"]),
    ],
)
def test_invalid_offline_examples_fail_before_model_loading(script, options):
    result = subprocess.run(
        [sys.executable, str(ROOT / "examples" / script), *options],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "requires" in result.stderr or "together" in result.stderr
