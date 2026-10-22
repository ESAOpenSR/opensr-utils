"""Adapters for the ESA OpenSR SRGAN/ESRGAN inference package.

The ESA implementation is an optional dependency.  Keeping its import inside
the loader means that installing ``opensr-utils`` does not pull in the GAN
training stack, while ``ESASRGAN`` remains an ordinary ``torch.nn.Module``
that can be handed to :class:`opensr_utils.large_file_processing`.

The official package currently exposes ``load_inference_model`` presets.  The
adapter uses its ``predict_step`` when available, preserving the published
histogram matching and EMA inference behavior.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


class ESASRGAN(torch.nn.Module):
    """Wrap an ESA OpenSR-SRGAN model for tiled OpenSR inference.

    Parameters
    ----------
    model:
        An already constructed ESA SRGAN/ESRGAN model.  It must implement
        ``forward`` or expose a callable ``generator``.
    scale_factor:
        Spatial scale produced by the checkpoint.  ESA's published RGB-NIR
        preset is 4x; use 2 for a 20 m preset.
    input_units:
        ``"normalized"`` (the opensr-utils default) or ``"reflectance"`` for
        Sentinel-2 DN values in [0, 10000].
    output_units:
        The units returned by this adapter.  The pipeline's normalized mode
        expects [0, 1]; reflectance mode returns [0, 10000].
    """

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        scale_factor: int = 4,
        input_units: str = "normalized",
        output_units: str = "normalized",
        bands: int = 4,
        input_bands: tuple[str, ...] | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(model, torch.nn.Module) and not callable(model):
            raise TypeError("model must be a torch.nn.Module or callable")
        if (
            isinstance(scale_factor, bool)
            or not isinstance(scale_factor, int)
            or scale_factor <= 0
        ):
            raise ValueError("scale_factor must be a positive integer")
        if bands <= 0 or isinstance(bands, bool) or not isinstance(bands, int):
            raise ValueError("bands must be a positive integer")
        if input_units not in {"normalized", "reflectance"}:
            raise ValueError("input_units must be normalized or reflectance")
        if output_units not in {"normalized", "reflectance"}:
            raise ValueError("output_units must be normalized or reflectance")
        self.model = model
        self.scale_factor = scale_factor
        # ``factor`` is the name used by opensr-utils' pipeline and by other
        # model adapters; retain scale_factor as the descriptive alias.
        self.factor = scale_factor
        self.input_units = input_units
        self.output_units = output_units
        self.bands = bands
        self.input_bands = tuple(input_bands or ("B04", "B03", "B02", "B08"))
        if len(self.input_bands) != bands:
            raise ValueError("input_bands length must equal bands")
        # Consumers can use this stable metadata when writing named SR bands.
        self.output_bands = self.input_bands
        self.output_band_names = self.input_bands

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not isinstance(x, torch.Tensor) or x.ndim != 4:
            raise ValueError("ESASRGAN input must be a 4-D BCHW tensor")
        if x.shape[1] != self.bands:
            raise ValueError(
                f"ESASRGAN expects {self.bands} channels, got {x.shape[1]}"
            )
        if not torch.isfinite(x).all().item():
            raise ValueError("ESASRGAN input contains NaN or infinite values")
        if self.input_units == "reflectance":
            x = x / 10000.0
        with torch.inference_mode():
            # ESA's predict_step performs histogram matching and EMA-aware
            # inference.  Because x is explicitly normalized first, its
            # normalise_10k implementation deterministically returns [0, 1].
            predict_step = getattr(self.model, "predict_step", None)
            if callable(predict_step):
                normalizer = getattr(self.model, "normalizer", None)
                method = getattr(normalizer, "method", None)
                if normalizer is not None and method != "normalise_10k":
                    raise ValueError(
                        "ESA SRGAN predict_step requires normalizer.method="
                        "'normalise_10k'; use a plain generator for other units."
                    )
                result = predict_step(x)
            else:
                try:
                    result = self.model(x)
                except (AttributeError, NotImplementedError):
                    generator = getattr(self.model, "generator", None)
                    if generator is None or not callable(generator):
                        raise
                    result = generator(x)
        result = _unwrap_prediction(result)
        if result.ndim != 4 or result.shape[0] != x.shape[0]:
            raise ValueError("ESASRGAN model must return a BCHW tensor")
        expected = (x.shape[-2] * self.scale_factor, x.shape[-1] * self.scale_factor)
        if tuple(result.shape[-2:]) != expected:
            raise ValueError(
                f"ESASRGAN output shape {tuple(result.shape[-2:])} does not match "
                f"scale_factor={self.scale_factor} (expected {expected})"
            )
        if result.shape[1] != self.bands:
            raise ValueError(
                f"ESASRGAN returned {result.shape[1]} channels, expected {self.bands}"
            )
        if not torch.isfinite(result).all().item():
            raise ValueError("ESASRGAN output contains NaN or infinite values")
        result = result.clamp(0, 1)
        return result * 10000.0 if self.output_units == "reflectance" else result


def _unwrap_prediction(value: Any) -> torch.Tensor:
    """Extract a tensor from common Lightning/inference return containers."""
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, dict):
        for key in ("sr", "output", "prediction", "pred", "image"):
            if key in value:
                return _unwrap_prediction(value[key])
    if isinstance(value, (tuple, list)) and value:
        return _unwrap_prediction(value[0])
    raise TypeError("ESA SRGAN model returned no tensor prediction")


def _load_checkpoint(path: str | Path, *, map_location: str = "cpu") -> Any:
    """Load a trusted local Torch module export or serialized module checkpoint."""
    checkpoint = torch.load(Path(path), map_location=map_location, weights_only=False)
    if isinstance(checkpoint, torch.nn.Module):
        return checkpoint
    if isinstance(checkpoint, dict) and isinstance(
        checkpoint.get("model"), torch.nn.Module
    ):
        return checkpoint["model"]
    raise ValueError(
        "A local ESA SRGAN checkpoint must contain a serialized torch.nn.Module; "
        "use load_inference_model for official presets."
    )


def load_esasrgan(
    preset: str = "RGB-NIR",
    *,
    checkpoint: str | Path | None = None,
    config_path: str | Path | None = None,
    cache_dir: str | Path | None = None,
    device: str | torch.device = "cpu",
    scale_factor: int | None = None,
    input_units: str = "normalized",
    output_units: str = "normalized",
    bands: int = 4,
    input_bands: tuple[str, ...] | None = None,
) -> ESASRGAN:
    """Load an ESA SRGAN preset or local Torch module checkpoint.

    The official ``opensr-srgan`` package is imported only when this function
    is called.  Loading always starts on CPU, then moves the adapter to the
    requested device, avoiding accidental GPU allocation during construction.
    """
    if config_path is not None:
        if checkpoint is None:
            raise ValueError(
                "config_path requires checkpoint; refusing to construct an "
                "untrained ESA SRGAN model for inference."
            )
        try:
            from opensr_srgan import load_from_config
        except ImportError as exc:
            raise ImportError(
                "ESA SRGAN support requires the optional 'opensr-srgan' package; "
                "install it with `python -m pip install opensr-srgan`."
            ) from exc
        model = load_from_config(
            config_path,
            checkpoint,
            map_location="cpu",
            mode="eval",
        )
    elif checkpoint is not None:
        model = _load_checkpoint(checkpoint, map_location="cpu")
    else:
        if preset.strip().replace("_", "-").upper() == "SWIR":
            raise ValueError(
                "The SWIR ESA SRGAN preset has six bands and 8x scaling; "
                "use an explicit six-band adapter configuration."
            )
        try:
            from opensr_srgan import load_inference_model
        except ImportError as exc:
            raise ImportError(
                "ESA SRGAN support requires the optional 'opensr-srgan' package; "
                "install it with `python -m pip install opensr-srgan`."
            ) from exc
        model = load_inference_model(
            preset,
            cache_dir=cache_dir,
            map_location="cpu",
        )
    # Prefer architecture metadata from official configs.  The public
    # pipeline currently defaults to RGB-NIR; requiring explicit band labels
    # for other configurations prevents silently feeding SWIR channels in the
    # wrong order.
    model_config = getattr(model, "config", None)
    configured_bands = getattr(getattr(model_config, "Model", None), "in_bands", None)
    configured_factor = getattr(
        getattr(model_config, "Generator", None), "scaling_factor", None
    )
    if configured_bands is not None:
        configured_bands = int(configured_bands)
        if configured_bands != bands and input_bands is None:
            raise ValueError(
                f"Loaded ESA SRGAN config expects {configured_bands} bands; "
                "provide matching input_bands explicitly."
            )
        bands = configured_bands
    if configured_factor is not None:
        configured_factor = int(configured_factor)
        if scale_factor is not None and int(scale_factor) != configured_factor:
            raise ValueError(
                f"scale_factor={scale_factor} conflicts with loaded ESA config "
                f"scaling_factor={configured_factor}."
            )
        scale_factor = configured_factor
    if scale_factor is None:
        scale_factor = 4
    adapter = ESASRGAN(
        model,
        scale_factor=scale_factor,
        input_units=input_units,
        output_units=output_units,
        bands=bands,
        input_bands=input_bands,
    )
    adapter.eval()
    return adapter.to(torch.device(device))


# Friendly aliases used by model registries and older examples.
load_srgan = load_esasrgan
load_esar = load_esasrgan
get_esasrgan = load_esasrgan
SRGAN = ESASRGAN

__all__ = [
    "ESASRGAN",
    "SRGAN",
    "load_esasrgan",
    "get_esasrgan",
    "load_srgan",
    "load_esar",
]
