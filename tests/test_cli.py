from __future__ import annotations

import sys
import types

import pytest

from opensr_utils import cli


def install_fake_pipeline(monkeypatch, calls):
    module = types.ModuleType("opensr_utils.pipeline")

    class FakeRunner:
        def __init__(self, **kwargs):
            calls.append(("init", kwargs))

        def close(self):
            pass

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
    assert "sr.tif" in capsys.readouterr().out


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


@pytest.mark.parametrize(
    "name,canonical", [("sen2sr", "SEN2SR"), ("SRGAN", "ESASRGAN")]
)
def test_cli_named_models_use_local_artifacts_and_declared_contract(
    monkeypatch, name, canonical
):
    calls = []
    install_fake_pipeline(monkeypatch, calls)
    registry = types.ModuleType("opensr_utils.model_utils.get_models")
    model = types.SimpleNamespace(factor=4, output_units="normalized")
    loads = []
    registry.get_model = lambda *args, **kwargs: loads.append((args, kwargs)) or model
    monkeypatch.setitem(sys.modules, "opensr_utils.model_utils.get_models", registry)
    cli.main(
        [
            "input.tif",
            name,
            "--checkpoint",
            "local-weights",
            "--no-download",
            "--bands",
            "3",
            "2",
            "1",
            "4",
            "--device",
            "cpu",
        ]
    )
    assert loads == [
        (
            (canonical,),
            dict(
                device="cpu",
                checkpoint="local-weights",
                config=None,
                cache_dir=None,
                variant=None,
                download=False,
            ),
        )
    ]
    options = calls[0][1]
    assert options["model"] is model
    assert options["factor"] == 4
    assert options["output_units"] == "normalized"
    assert options["band_indexes"] == [3, 2, 1, 4]
    assert calls[-1] == ("run", None)


def test_cli_returns_when_ldsrs2_model_load_fails(monkeypatch, capsys):
    calls = []
    install_fake_pipeline(monkeypatch, calls)
    get_models = types.ModuleType("opensr_utils.model_utils.get_models")

    def fail(device="cpu"):
        raise RuntimeError("download failed")

    get_models.get_ldsrs2 = fail
    monkeypatch.setitem(sys.modules, "opensr_utils.model_utils.get_models", get_models)
    monkeypatch.setattr(sys, "argv", ["opensr-run", "/data/input.tif", "LDSRS2"])

    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 1
    assert calls == []
    output = capsys.readouterr().err
    assert "Could not load LDSR-S2 model" in output
    assert "download failed" in output


@pytest.mark.parametrize(
    "argv",
    [
        [],
        ["input.tif", "unknown"],
        ["input.tif", "None", "--device", "tpu"],
        ["input.tif", "None", "--window_size", "8"],
        ["input.tif", "None", "--factor", "1.5"],
        ["input.tif", "None", "--gpus"],
        ["input.tif", "None", "--not_an_option"],
    ],
)
def test_invalid_cli_arguments_fail_before_starting_pipeline(monkeypatch, capsys, argv):
    calls = []
    install_fake_pipeline(monkeypatch, calls)
    monkeypatch.setattr(sys, "argv", ["opensr-run", *argv])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2
    assert calls == []
    assert "usage:" in capsys.readouterr().err


def test_cli_help_succeeds_without_running_inference(monkeypatch, capsys):
    calls = []
    install_fake_pipeline(monkeypatch, calls)
    monkeypatch.setattr(sys, "argv", ["opensr-run", "--help"])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 0
    assert calls == []
    help_text = capsys.readouterr().out
    assert "--overwrite" in help_text
    assert "--keep_temp" in help_text


def test_cli_passes_all_explicit_processing_options_and_runs_once(monkeypatch):
    calls = []
    install_fake_pipeline(monkeypatch, calls)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "opensr-run",
            "path with spaces/input.tif",
            "None",
            "--factor",
            "1",
            "--window_size",
            "16",
            "24",
            "--overlap",
            "6",
            "--eliminate_border_px",
            "2",
            "--batch_size",
            "7",
            "--num_workers",
            "3",
            "--prefetch_factor",
            "4",
            "--save_preview",
            "--debug",
            "--overwrite",
            "--delete_input_zip",
            "--compressed_patches",
            "--keep_temp",
        ],
    )
    cli.main()
    assert [name for name, _ in calls] == ["init", "run"]
    assert calls[0][1] == dict(
        root="path with spaces/input.tif",
        model=None,
        factor=1,
        window_size=(16, 24),
        overlap=6,
        eliminate_border_px=2,
        device="auto",
        output_path=None,
        output_units="normalized",
        gpus=None,
        batch_size=7,
        num_workers=3,
        prefetch_factor=4,
        save_preview=True,
        debug=True,
        cleanup=False,
        overwrite=True,
        delete_input_zip=True,
        compressed_patches=True,
    )


def test_cli_propagates_pipeline_errors(monkeypatch):
    calls = []
    install_fake_pipeline(monkeypatch, calls)
    module = sys.modules["opensr_utils.pipeline"]

    def fail(**kwargs):
        raise FileExistsError("output exists")

    monkeypatch.setattr(module, "large_file_processing", fail)
    monkeypatch.setattr(sys, "argv", ["opensr-run", "input.tif", "None"])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 1


def test_model_loading_failure_returns_nonzero_exit_status(monkeypatch):
    install_fake_pipeline(monkeypatch, [])
    get_models = types.ModuleType("opensr_utils.model_utils.get_models")

    def fail(device):
        raise RuntimeError("download failed")

    get_models.get_ldsrs2 = fail
    monkeypatch.setitem(sys.modules, "opensr_utils.model_utils.get_models", get_models)
    monkeypatch.setattr(sys, "argv", ["opensr-run", "input.tif", "LDSRS2"])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code != 0


def test_cli_minimal_call_and_output_destination(monkeypatch, capsys):
    calls = []
    install_fake_pipeline(monkeypatch, calls)
    monkeypatch.setattr(
        sys, "argv", ["opensr-run", "input.tif", "-o", "results/output.tif"]
    )
    cli.main()
    options = calls[0][1]
    assert options["model"] is None
    assert options["device"] == "auto"
    assert options["output_path"] == "results/output.tif"
    assert "sr.tif" in capsys.readouterr().out


@pytest.mark.integration
def test_cli_real_inference_retains_all_patches_and_writes_explicit_destination(
    write_raster, tmp_path
):
    import json
    import os
    from pathlib import Path
    import subprocess

    import numpy as np
    import rasterio

    data = np.arange(256, dtype=np.uint16).reshape(1, 16, 16)
    source = write_raster(tmp_path / "input.tif", data=data, nodata=None)
    output = tmp_path / "results" / "output.tif"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "opensr_utils",
            str(source),
            "-o",
            str(output),
            "--device",
            "cpu",
            "--factor",
            "1",
            "--window-size",
            "2",
            "2",
            "--overlap",
            "0",
            "--batch-size",
            "7",
            "--keep-temp",
        ],
        capture_output=True,
        text=True,
        timeout=45,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip().splitlines()[-1] == str(output)
    with rasterio.open(output) as result:
        np.testing.assert_allclose(result.read(), data, atol=1, rtol=0)
    (manifest,) = output.parent.glob("temp_*/index.json")
    entries = json.loads(manifest.read_text())["entries"]
    assert len(entries) == 64
    assert all(Path(entry["path"]).is_file() for entry in entries)
