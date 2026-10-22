"""Named dispatch stays lazy and makes local/offline artifact choices explicit."""

from pathlib import Path

import pytest

from opensr_utils.model_utils import get_models


@pytest.mark.parametrize(
    "name,expected",
    [
        (None, "None"),
        ("bilinear", "None"),
        ("LDSR-S2", "LDSRS2"),
        ("sen2sr", "SEN2SR"),
        ("SEN2SRLite", "SEN2SR"),
        ("srgan", "ESASRGAN"),
        ("ESASRGAN", "ESASRGAN"),
    ],
)
def test_model_aliases_have_one_canonical_dispatch(name, expected):
    assert get_models.normalize_model_name(name) == expected


def test_unknown_model_reports_available_choices():
    with pytest.raises(ValueError, match="Unknown model.*SEN2SR"):
        get_models.get_model("unavailable")


def test_sen2sr_local_manifest_uses_offline_directory(monkeypatch, tmp_path):
    manifest = tmp_path / "mlm.json"
    manifest.write_text("{}")
    calls = []
    sentinel = object()
    monkeypatch.setattr(
        get_models, "get_sen2sr", lambda **kwargs: calls.append(kwargs) or sentinel
    )
    assert (
        get_models.get_model(
            "SEN2SR", checkpoint=manifest, download=False, variant="lite"
        )
        is sentinel
    )
    assert calls == [
        dict(
            device="cpu",
            cache_dir=None,
            download=False,
            variant="lite",
            model_dir=tmp_path,
        )
    ]


def test_srgan_local_artifacts_are_forwarded_to_official_loader(monkeypatch, tmp_path):
    checkpoint = tmp_path / "generator.ckpt"
    checkpoint.write_bytes(b"mock checkpoint boundary")
    config = tmp_path / "config.yaml"
    config.write_text("Model: {}")
    calls = []
    monkeypatch.setattr(
        get_models, "get_esasrgan", lambda **kwargs: calls.append(kwargs) or "adapter"
    )
    assert (
        get_models.get_model(
            "SRGAN", config=config, checkpoint=checkpoint, download=False
        )
        == "adapter"
    )
    assert calls == [
        dict(device="cpu", checkpoint=checkpoint, config_path=config, cache_dir=None)
    ]


@pytest.mark.parametrize("model", ["SEN2SR", "SRGAN"])
def test_offline_loading_requires_explicit_local_artifacts(model):
    with pytest.raises(ValueError, match="requires"):
        get_models.get_model(model, download=False)


def test_interpolation_does_not_accept_a_checkpoint(tmp_path):
    with pytest.raises(ValueError, match="Interpolation"):
        get_models.get_model(None, checkpoint=Path(tmp_path / "model.pt"))
