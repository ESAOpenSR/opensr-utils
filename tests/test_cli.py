from __future__ import annotations

import sys
import types

from opensr_utils import cli


def install_fake_pipeline(monkeypatch, calls):
    module = types.ModuleType("opensr_utils.pipeline")

    class FakeRunner:
        def __init__(self, **kwargs):
            calls.append(("init", kwargs))

        def run(self):
            calls.append(("run", None))
            return "sr.tif"

    module.large_file_processing = FakeRunner
    monkeypatch.setitem(sys.modules, "opensr_utils.pipeline", module)


def test_cli_runs_placeholder_model_with_cpu_defaults(monkeypatch, capsys):
    calls = []
    install_fake_pipeline(monkeypatch, calls)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "opensr-run",
            "/data/input.tif",
            "None",
            "--window_size",
            "32",
            "64",
            "--factor",
            "3",
            "--overlap",
            "4",
            "--batch_size",
            "3",
            "--num_workers",
            "0",
            "--keep_temp",
            "--compressed_patches",
        ],
    )

    cli.main()

    kwargs = calls[0][1]
    assert kwargs["root"] == "/data/input.tif"
    assert kwargs["model"] is None
    assert kwargs["window_size"] == (32, 64)
    assert kwargs["factor"] == 3
    assert kwargs["gpus"] is None
    assert kwargs["cleanup"] is False
    assert kwargs["compressed_patches"] is True
    assert calls[-1] == ("run", None)
    assert "Using interpolation placeholder" in capsys.readouterr().out


def test_cli_loads_ldsrs2_model_and_passes_cuda_gpus(monkeypatch):
    calls = []
    install_fake_pipeline(monkeypatch, calls)
    get_models = types.ModuleType("opensr_utils.model_utils.get_models")
    get_models.get_ldsrs2 = lambda device="cpu": f"model-on-{device}"
    monkeypatch.setitem(sys.modules, "opensr_utils.model_utils.get_models", get_models)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "opensr-run",
            "/data/input.tif",
            "LDSRS2",
            "--device",
            "cuda",
            "--gpus",
            "1",
            "2",
            "--overwrite",
            "--delete_input_zip",
        ],
    )

    cli.main()

    kwargs = calls[0][1]
    assert kwargs["model"] == "model-on-cpu"
    assert kwargs["device"] == "cuda"
    assert kwargs["gpus"] == [1, 2]
    assert kwargs["overwrite"] is True
    assert kwargs["delete_input_zip"] is True


def test_cli_returns_when_ldsrs2_model_load_fails(monkeypatch, capsys):
    calls = []
    install_fake_pipeline(monkeypatch, calls)
    get_models = types.ModuleType("opensr_utils.model_utils.get_models")

    def fail(device="cpu"):
        raise RuntimeError("download failed")

    get_models.get_ldsrs2 = fail
    monkeypatch.setitem(sys.modules, "opensr_utils.model_utils.get_models", get_models)
    monkeypatch.setattr(sys, "argv", ["opensr-run", "/data/input.tif", "LDSRS2"])

    cli.main()

    assert calls == []
    output = capsys.readouterr().out
    assert "Could not load LDSR-S2 model" in output
    assert "download failed" in output
