"""RGB-NIR LDSR-S2 inference, also usable as SEN2SR's four-band backbone."""

from __future__ import annotations

from pathlib import Path
import inspect

from torch import nn

from .sen2sr import RGBN_BANDS, SEN2SRAdapter


class _DiffusionInference(nn.Module):
    def __init__(self, model, sampling_steps=None):
        super().__init__()
        self.model = model
        # Older opensr-model releases interpret a null step count literally,
        # whereas recent releases choose a count for their configured sampler.
        settings = getattr(getattr(model, "config", None), "denoiser_settings", None)
        if (
            sampling_steps is None
            and settings is not None
            and getattr(settings, "sampling_steps", None) is None
            and "sampler" not in inspect.signature(model.forward).parameters
        ):
            sampling_steps = 100
        self.sampling_steps = sampling_steps

    def forward(self, x):
        # Upstream caches ordinary device attributes, which Module.to() does
        # not update. Keep them consistent with Lightning's actual placement.
        for module in self.model.modules():
            if "device" in vars(module):
                module.device = x.device
        options = (
            {}
            if self.sampling_steps is None
            else {"sampling_steps": self.sampling_steps}
        )
        return self.model(x, **options)


class LDSRS2Adapter(SEN2SRAdapter):
    """Apply the same fixed-tile, RGB-NIR contract as the published model.

    Padding to 128 square pixels also supports older opensr-model releases
    whose encoder assumes square inputs. Output is cropped to the true extent.
    """

    input_bands = RGBN_BANDS

    def __init__(self, model, *, sampling_steps=None, device="cpu"):
        if sampling_steps is not None and (
            isinstance(sampling_steps, bool)
            or not isinstance(sampling_steps, int)
            or sampling_steps <= 0
        ):
            raise ValueError("sampling_steps must be a positive integer.")
        super().__init__(_DiffusionInference(model, sampling_steps), device=device)


def load_ldsrs2(
    *,
    checkpoint=None,
    config_path=None,
    sampling_steps=None,
    device="cpu",
    download=True,
):
    """Load the official preset or an explicit local config/checkpoint pair.

    Local artifacts make offline inference reproducible. A missing local
    checkpoint never falls back to a downloaded preset.
    """
    if (checkpoint is None) != (config_path is None):
        raise ValueError(
            "Local LDSR-S2 loading requires both checkpoint and config_path."
        )
    if checkpoint is None:
        if not download:
            raise ValueError(
                "Offline LDSR-S2 requires a local checkpoint and config_path."
            )
        from .get_models import get_ldsrs2

        model = get_ldsrs2(device="cpu")
    else:
        checkpoint = Path(checkpoint).expanduser()
        config_path = Path(config_path).expanduser()
        for path in (checkpoint, config_path):
            if not path.is_file():
                raise FileNotFoundError(f"LDSR-S2 artifact does not exist: {path}")
        try:
            import opensr_model
            from omegaconf import OmegaConf
        except ImportError as exc:
            raise ImportError(
                "LDSR-S2 requires: python -m pip install 'opensr-utils[model]'"
            ) from exc
        model = opensr_model.SRLatentDiffusion(
            OmegaConf.load(config_path), device="cpu"
        )
        model.load_pretrained(str(checkpoint))
    return LDSRS2Adapter(model, sampling_steps=sampling_steps, device=device)
