"""Model loading mocks the HTTP/checkpoint boundary and never downloads weights."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.fixture
def loader(monkeypatch):
    response = Mock(text="ckpt_version: fake-v1")
    get = Mock(return_value=response)
    config = SimpleNamespace(ckpt_version="fake-v1")
    load_config = Mock(return_value=config)
    model = Mock()
    construct = Mock(return_value=model)
    for name, attrs in {
        "requests": {"get": get},
        "omegaconf": {"OmegaConf": SimpleNamespace(load=load_config)},
        "opensr_model": {"SRLatentDiffusion": construct},
    }.items():
        module = types.ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
    # Keep mocked imports out of the real module cache and package attributes.
    path = (
        Path(__file__).resolve().parents[1] / "opensr_utils/model_utils/get_models.py"
    )
    spec = importlib.util.spec_from_file_location("isolated_model_loader", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return SimpleNamespace(
        module=module,
        get=get,
        response=response,
        config=config,
        load_config=load_config,
        model=model,
        construct=construct,
    )


@pytest.mark.parametrize("device", ["cpu", "cuda:0", "cuda:2"])
def test_model_loader_validates_response_and_loads_requested_checkpoint(loader, device):
    assert loader.module.get_ldsrs2(device=device) is loader.model
    args, kwargs = loader.get.call_args
    assert args[0].endswith("/opensr_model/configs/config_10m.yaml")
    assert kwargs["timeout"] > 0
    loader.response.raise_for_status.assert_called_once_with()
    assert loader.load_config.call_args.args[0].getvalue() == "ckpt_version: fake-v1"
    loader.construct.assert_called_once_with(loader.config, device=device)
    loader.model.load_pretrained.assert_called_once_with("fake-v1")


@pytest.mark.parametrize(
    "stage", ["request", "http_status", "config", "constructor", "checkpoint"]
)
def test_model_loading_errors_propagate_and_stop_later_stages(loader, stage):
    operations = [
        loader.get,
        loader.response.raise_for_status,
        loader.load_config,
        loader.construct,
        loader.model.load_pretrained,
    ]
    index = ["request", "http_status", "config", "constructor", "checkpoint"].index(
        stage
    )
    error = RuntimeError(f"{stage} failed")
    operations[index].side_effect = error
    with pytest.raises(RuntimeError) as exc:
        loader.module.get_ldsrs2()
    assert exc.value is error
    for operation in operations[index + 1 :]:
        operation.assert_not_called()
