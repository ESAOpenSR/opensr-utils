"""The four-band diffusion boundary is explicit and device placement is current."""

from types import SimpleNamespace

import pytest
import torch

from opensr_utils.model_utils.ldsrs2 import LDSRS2Adapter, load_ldsrs2


class Diffusion(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.device = "stale"
        self.config = SimpleNamespace(
            denoiser_settings=SimpleNamespace(sampling_steps=None)
        )
        self.steps = None

    def forward(self, x, *, sampling_steps):
        assert self.device == x.device
        self.steps = sampling_steps
        return x.repeat_interleave(4, -2).repeat_interleave(4, -1)


def test_rgbn_diffusion_pads_crops_and_supports_legacy_auto_steps():
    diffusion = Diffusion()
    model = LDSRS2Adapter(diffusion)
    values = torch.arange(4, dtype=torch.float32)[None, :, None, None] / 10
    result = model(values.expand(2, 4, 3, 5))
    assert model.output_band_names == ("B04", "B03", "B02", "B08")
    assert result.shape == (2, 4, 12, 20)
    torch.testing.assert_close(result, values.expand(2, 4, 12, 20))
    assert diffusion.steps == 100


@pytest.mark.parametrize("steps", [0, -1, True, 1.5])
def test_invalid_sampling_steps_are_rejected(steps):
    with pytest.raises(ValueError, match="sampling_steps"):
        LDSRS2Adapter(Diffusion(), sampling_steps=steps)


@pytest.mark.parametrize(
    "options",
    [
        {"download": False},
        {"checkpoint": "missing.ckpt"},
        {"config_path": "config.yaml"},
    ],
)
def test_offline_loading_requires_matching_local_artifacts(options):
    with pytest.raises(ValueError, match="requires"):
        load_ldsrs2(**options)
