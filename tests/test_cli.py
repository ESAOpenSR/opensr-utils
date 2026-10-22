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


@pytest.mark.parametrize(
    "argv",
    [
        [],
        ["input.tif"],
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
        device="cpu",
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
    with pytest.raises(FileExistsError):
        cli.main()


@pytest.mark.known_bug
@pytest.mark.xfail(
    reason="CLI returns success when model loading fails", raises=pytest.fail.Exception
)
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
