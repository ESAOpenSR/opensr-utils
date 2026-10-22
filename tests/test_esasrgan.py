from pathlib import Path
import sys
import types

import pytest
import torch

from opensr_utils.model_utils.esasrgan import ESASRGAN, load_esasrgan


class _TinySR(torch.nn.Module):
    def forward(self, x):
        return torch.nn.functional.interpolate(x, scale_factor=2, mode="nearest")


class _PredictSR(_TinySR):
    def forward(self, x):
        return super().forward(x) * 0.25

    def predict_step(self, x):
        return super().forward(x) * 0.5


class _BrokenPredict(_TinySR):
    def predict_step(self, x):
        raise TypeError("upstream failure")


class _ConfiguredSR(_TinySR):
    config = types.SimpleNamespace(
        Model=types.SimpleNamespace(in_bands=4),
        Generator=types.SimpleNamespace(scaling_factor=2),
    )


def test_adapter_normalizes_dn_and_returns_expected_shape():
    model = ESASRGAN(_TinySR(), scale_factor=2, input_units="reflectance")
    out = model(torch.full((2, 4, 3, 5), 5000.0))
    assert out.shape == (2, 4, 6, 10)
    assert torch.allclose(out, torch.full_like(out, 0.5))


def test_adapter_can_return_reflectance_units():
    model = ESASRGAN(_TinySR(), scale_factor=2, output_units="reflectance")
    out = model(torch.full((1, 4, 2, 2), 0.25))
    assert torch.allclose(out, torch.full_like(out, 2500.0))


def test_adapter_prefers_predict_step_without_brightness_heuristic():
    out = ESASRGAN(_PredictSR(), scale_factor=2)(torch.ones(1, 4, 2, 2))
    assert torch.allclose(out, torch.full_like(out, 0.5))


def test_predict_step_errors_are_not_silently_replaced_by_forward():
    with pytest.raises(TypeError, match="upstream failure"):
        ESASRGAN(_BrokenPredict(), scale_factor=2)(torch.ones(1, 4, 2, 2))


def test_predict_step_rejects_non_10k_normalizer():
    model = _PredictSR()
    model.normalizer = types.SimpleNamespace(method="zero_one")
    with pytest.raises(ValueError, match="normalise_10k"):
        ESASRGAN(model, scale_factor=2)(torch.ones(1, 4, 2, 2))


@pytest.mark.parametrize("bad", [torch.zeros(1, 3, 2, 2), torch.zeros(1, 4, 2, 2, 1)])
def test_adapter_rejects_wrong_input_shape(bad):
    with pytest.raises(ValueError, match="expects|4-D"):
        ESASRGAN(_TinySR(), scale_factor=2)(bad)


def test_adapter_rejects_wrong_model_scale():
    with pytest.raises(ValueError, match="does not match"):
        ESASRGAN(_TinySR(), scale_factor=4)(torch.zeros(1, 4, 2, 2))


def test_loader_uses_official_optional_api_on_cpu(monkeypatch):
    seen = {}

    def load(preset, **kwargs):
        seen.update(preset=preset, **kwargs)
        return _TinySR()

    monkeypatch.setitem(
        sys.modules, "opensr_srgan", types.SimpleNamespace(load_inference_model=load)
    )
    adapter = load_esasrgan("RGB-NIR", device="cpu", scale_factor=2)
    assert isinstance(adapter, ESASRGAN)
    assert seen == {"preset": "RGB-NIR", "map_location": "cpu", "cache_dir": None}


def test_loader_local_module_checkpoint(tmp_path):
    path = Path(tmp_path) / "model.pt"
    torch.save(_TinySR(), path)
    adapter = load_esasrgan(checkpoint=path, scale_factor=2)
    assert torch.allclose(adapter(torch.ones(1, 4, 1, 1)), torch.ones(1, 4, 2, 2))


def test_loader_explains_missing_optional_package(monkeypatch):
    monkeypatch.setitem(sys.modules, "opensr_srgan", None)
    with pytest.raises(ImportError, match="opensr-srgan"):
        load_esasrgan()


def test_loader_infers_config_scale_and_rejects_explicit_mismatch(monkeypatch):
    def load_from_config(*args, **kwargs):
        return _ConfiguredSR()

    fake = types.SimpleNamespace(load_from_config=load_from_config)
    monkeypatch.setitem(sys.modules, "opensr_srgan", fake)
    model = load_esasrgan(config_path="config.yaml", checkpoint="weights.ckpt")
    assert model.factor == 2
    with pytest.raises(ValueError, match="conflicts"):
        load_esasrgan(
            config_path="config.yaml", checkpoint="weights.ckpt", scale_factor=4
        )


def test_loader_rejects_config_without_checkpoint():
    with pytest.raises(ValueError, match="requires checkpoint"):
        load_esasrgan(config_path="config.yaml")


def test_loader_rejects_swir_preset_until_six_band_pipeline_is_explicit():
    with pytest.raises(ValueError, match="SWIR"):
        load_esasrgan("SWIR")
