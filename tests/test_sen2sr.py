from __future__ import annotations

import sys
import types

import pytest
import torch
from torch import nn

from opensr_utils.model_utils.sen2sr import SEN2SRAdapter, load_sen2sr


class UpsampleRGBN(nn.Module):
    def forward(self, x):
        return torch.nn.functional.interpolate(x, scale_factor=4, mode="nearest")


def test_adapter_registers_upstream_plain_hard_constraint_mask():
    class HardConstraintLike(nn.Module):
        def __init__(self):
            super().__init__()
            self.low_pass_mask = torch.ones(1)

        def forward(self, x):
            return x

    model = HardConstraintLike()
    adapter = SEN2SRAdapter(model)
    assert "low_pass_mask" in dict(model.named_buffers())
    assert "low_pass_mask" not in model.state_dict()
    adapter.to("meta")
    assert model.low_pass_mask.device.type == "meta"


def test_adapter_preserves_batch_and_rgbn_shape_on_cpu():
    model = SEN2SRAdapter(UpsampleRGBN())
    output = model(torch.rand(2, 4, 128, 128))
    assert output.shape == (2, 4, 512, 512)
    assert output.device.type == "cpu"
    assert not model.training


@pytest.mark.parametrize("height,width", [(1, 1), (1, 127), (63, 91), (128, 64)])
def test_adapter_pads_edge_tiles_and_crops_output(height, width):
    output = SEN2SRAdapter(UpsampleRGBN())(torch.rand(2, 4, height, width))
    assert output.shape == (2, 4, height * 4, width * 4)


@pytest.mark.parametrize(
    "shape", [(4, 128, 128), (2, 3, 128, 128), (2, 5, 128, 128), (2, 4, 129, 128)]
)
def test_adapter_rejects_wrong_band_or_rank(shape):
    with pytest.raises(
        (ValueError, TypeError), match="(4 channels|shape|128x128|no larger)"
    ):
        SEN2SRAdapter(UpsampleRGBN())(torch.ones(shape))


def test_adapter_rejects_bad_output_contract():
    class Wrong(nn.Module):
        def forward(self, x):
            return x

    with pytest.raises(ValueError, match="unexpected shape"):
        SEN2SRAdapter(Wrong())(torch.ones(1, 4, 128, 128))


def test_loader_uses_official_manifest_and_compiles_on_cpu(monkeypatch, tmp_path):
    calls = {}

    class Loaded:
        def compiled_model(self, *, device):
            calls["device"] = device
            return UpsampleRGBN()

    def download(*, file, output_dir):
        calls["file"] = file
        calls["output_dir"] = output_dir
        (tmp_path / "NonReference_RGBN_x4").mkdir()

    def load(path):
        calls["path"] = path
        return Loaded()

    monkeypatch.setitem(
        sys.modules, "mlstac", types.SimpleNamespace(download=download, load=load)
    )
    model = load_sen2sr(cache_dir=tmp_path, device="cpu")
    result = model(torch.ones(1, 4, 128, 128))
    assert result.shape == (1, 4, 512, 512)
    assert calls["file"].endswith("SEN2SRLite/NonReference_RGBN_x4/mlm.json")
    assert calls["device"] == torch.device("cpu")


def test_loader_accepts_local_serialized_module(tmp_path):
    path = tmp_path / "model.pt"
    torch.save(UpsampleRGBN(), path)
    model = load_sen2sr(checkpoint=path)
    assert model(torch.ones(1, 4, 128, 128)).shape == (1, 4, 512, 512)


def test_loader_rejects_state_dict_checkpoint(tmp_path):
    path = tmp_path / "state.pt"
    torch.save(UpsampleRGBN().state_dict(), path)
    with pytest.raises(ValueError, match="state_dict"):
        load_sen2sr(checkpoint=path)


def test_all_band_model_uses_ldsrs2_backbone_and_keeps_fusion(monkeypatch, tmp_path):
    from opensr_utils.model_utils import get_models
    from opensr_utils.model_utils.sen2sr import ALL_BANDS

    class Diffusion(nn.Module):
        device = "initial"

        def __init__(self):
            super().__init__()
            self.device = "initial"
            self.received = None

        def forward(self, x, *, sampling_steps):
            assert self.device == x.device
            self.received = sampling_steps
            return (x + 0.1).repeat_interleave(4, -2).repeat_interleave(4, -1)

    class Composite(nn.Module):
        def __init__(self):
            super().__init__()
            self.sr_model = nn.Identity()

        def forward(self, x):
            result = x.repeat_interleave(4, -2).repeat_interleave(4, -1)
            result[:, [2, 1, 0, 6]] = self.sr_model(x[:, [2, 1, 0, 6]])
            return result

    diffusion = Diffusion()
    composite = Composite()
    monkeypatch.setattr(get_models, "get_ldsrs2", lambda **kw: diffusion)
    monkeypatch.setitem(
        sys.modules,
        "mlstac",
        types.SimpleNamespace(
            load=lambda path: types.SimpleNamespace(
                compiled_model=lambda **kw: composite
            )
        ),
    )
    model = load_sen2sr(tmp_path, variant="main", backbone="ldsrs2", sampling_steps=7)
    values = torch.arange(10, dtype=torch.float32)[None, :, None, None] / 100
    result = model(values)
    expected = values.clone()
    expected[:, [2, 1, 0, 6]] += 0.1
    torch.testing.assert_close(
        result, expected.repeat_interleave(4, -2).repeat_interleave(4, -1)
    )
    assert model.input_bands == model.output_band_names == ALL_BANDS
    assert diffusion.received == 7


def test_ldsr_backbone_requires_all_band_variant():
    with pytest.raises(ValueError, match="ten-band"):
        load_sen2sr(backbone="ldsrs2")
