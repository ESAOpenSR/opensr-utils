from __future__ import annotations

import importlib
import sys
import types
from types import SimpleNamespace


def test_get_ldsrs2_downloads_config_and_loads_checkpoint(monkeypatch):
    calls = {}

    class FakeResponse:
        text = "ckpt_version: fake-v1"

        def raise_for_status(self):
            calls["raised"] = True

    fake_requests = types.ModuleType("requests")

    def get(url, timeout):
        calls["url"] = url
        calls["timeout"] = timeout
        return FakeResponse()

    fake_requests.get = get

    class FakeOmegaConf:
        @staticmethod
        def load(stream):
            calls["config_text"] = stream.read()
            return SimpleNamespace(ckpt_version="fake-v1")

    fake_omegaconf = types.ModuleType("omegaconf")
    fake_omegaconf.OmegaConf = FakeOmegaConf

    fake_opensr_model = types.ModuleType("opensr_model")

    class FakeSRLatentDiffusion:
        def __init__(self, config, device):
            self.config = config
            self.device = device
            self.loaded = None

        def load_pretrained(self, version):
            self.loaded = version

    fake_opensr_model.SRLatentDiffusion = FakeSRLatentDiffusion

    monkeypatch.setitem(sys.modules, "requests", fake_requests)
    monkeypatch.setitem(sys.modules, "omegaconf", fake_omegaconf)
    monkeypatch.setitem(sys.modules, "opensr_model", fake_opensr_model)
    sys.modules.pop("opensr_utils.model_utils.get_models", None)

    module = importlib.import_module("opensr_utils.model_utils.get_models")
    model = module.get_ldsrs2(device="cuda:0")

    assert calls["url"].endswith("/opensr_model/configs/config_10m.yaml")
    assert calls["timeout"] == 30
    assert calls["raised"] is True
    assert calls["config_text"] == "ckpt_version: fake-v1"
    assert model.device == "cuda:0"
    assert model.loaded == "fake-v1"
    sys.modules.pop("opensr_utils.model_utils.get_models", None)
