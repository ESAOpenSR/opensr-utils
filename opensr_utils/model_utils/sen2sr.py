"""Adapters and loaders for the ESAOpenSR SEN2SR models.

The upstream ``sen2sr`` package uses MLSTAC manifests rather than exposing a
stable PyTorch constructor.  This module keeps that detail at the boundary so
the rest of OpenSR Utils can consume an ordinary ``torch.nn.Module``.  The
default is the small, CPU-capable ``SEN2SRLite`` RGBN model.
"""

from __future__ import annotations

from pathlib import Path
import tempfile

import torch
from torch import Tensor, nn


DEFAULT_VARIANT = "NonReference_RGBN_x4"
MODEL_REPOSITORY = "https://huggingface.co/tacofoundation/sen2sr/resolve/main"
VARIANT_MANIFESTS = {
    "nonreference_rgbn_x4": "SEN2SRLite/NonReference_RGBN_x4/mlm.json",
    "sen2srlite_nonreference_rgbn_x4": "SEN2SRLite/NonReference_RGBN_x4/mlm.json",
    "lite": "SEN2SRLite/NonReference_RGBN_x4/mlm.json",
}


def _device(value: str | torch.device) -> torch.device:
    target = torch.device(value)
    if target.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("SEN2SR requested CUDA, but CUDA is not available")
    return target


class SEN2SRAdapter(nn.Module):
    """Validate and adapt a loaded SEN2SR model for OpenSR Utils.

    The RGBN model expects channels in ``B04, B03, B02, B08`` order and input
    values normalized to reflectance ``[0, 1]``.  A batch of shape
    ``(batch, 4, height, width)`` is returned as ``(batch, 4, height*4,
    width*4)``.  Validation here gives the pipeline an actionable error before
    a less clear convolution or manifest error occurs.
    """

    input_bands = ("B04", "B03", "B02", "B08")
    output_bands = input_bands
    output_band_names = input_bands
    input_units = "normalized"
    input_size = (128, 128)
    max_window_size = input_size
    factor = 4
    output_units = "normalized"

    def __init__(self, model: nn.Module, *, device: str | torch.device = "cpu"):
        super().__init__()
        if not isinstance(model, nn.Module) and not callable(model):
            raise TypeError("model must be a torch.nn.Module or callable")
        self.model = model
        # ``.to(...)`` may be called later by Lightning/DDP.  Do not cache the
        # initial device: forward() discovers the module's current placement.
        self.requested_device = _device(device)
        self._register_hard_constraint_masks()
        self.to(self.requested_device)
        self.eval()

    def _register_hard_constraint_masks(self) -> None:
        """Make SEN2SR's plain Fourier mask tensor follow device placement.

        The upstream ``HardConstraint`` stores ``low_pass_mask`` as a regular
        tensor rather than a registered buffer.  Registering the known tensor
        as nonpersistent preserves the upstream checkpoint format while making
        ``.to(cuda)`` and Lightning/DDP placement correct.
        """
        for module in self.modules():
            mask = getattr(module, "low_pass_mask", None)
            if isinstance(mask, Tensor) and "low_pass_mask" not in module._buffers:
                delattr(module, "low_pass_mask")
                module.register_buffer("low_pass_mask", mask, persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        if not isinstance(x, Tensor):
            raise TypeError("SEN2SR input must be a torch.Tensor")
        if x.ndim != 4:
            raise ValueError("SEN2SR input must have shape (batch, 4, height, width)")
        if x.shape[1] != 4:
            raise ValueError(
                "SEN2SR RGBN expects four channels in B04,B03,B02,B08 order; "
                f"received {x.shape[1]}"
            )
        height, width = (int(x.shape[2]), int(x.shape[3]))
        if height > self.input_size[0] or width > self.input_size[1]:
            raise ValueError(
                "SEN2SR RGBN accepts input tiles no larger than 128x128 "
                f"(the upstream hard constraint is fixed at 512x512); received {(height, width)}"
            )
        if min(x.shape[0], x.shape[2], x.shape[3]) <= 0:
            raise ValueError("SEN2SR input dimensions must be positive")
        if not torch.is_floating_point(x):
            x = x.float()
        if not torch.isfinite(x).all().item():
            raise ValueError("SEN2SR input contains NaN or infinite values")
        model_device = x.device
        if isinstance(self.model, nn.Module):
            try:
                model_device = next(self.model.parameters()).device
            except StopIteration:
                model_device = x.device
        x = x.to(model_device)
        padded = torch.nn.functional.pad(
            x,
            (0, self.input_size[1] - width, 0, self.input_size[0] - height),
            mode="replicate",
        )
        with torch.inference_mode():
            output = self.model(padded)
        if isinstance(output, (tuple, list)):
            output = output[0]
        if not isinstance(output, Tensor) or output.ndim != 4:
            raise ValueError("SEN2SR model must return a 4-D tensor")
        expected = (
            padded.shape[0],
            4,
            self.input_size[0] * self.factor,
            self.input_size[1] * self.factor,
        )
        if tuple(output.shape) != expected:
            raise ValueError(
                "SEN2SR RGBN output has unexpected shape: "
                f"received {tuple(output.shape)}, expected {expected}"
            )
        if not torch.isfinite(output).all().item():
            raise ValueError("SEN2SR output contains NaN or infinite values")
        return output[..., : height * self.factor, : width * self.factor]


def _manifest_url(variant: str) -> str:
    key = variant.strip().lower().replace("-", "_").replace(" ", "_")
    try:
        path = VARIANT_MANIFESTS[key]
    except KeyError as exc:
        choices = ", ".join(sorted(VARIANT_MANIFESTS))
        raise ValueError(
            f"Unknown SEN2SR variant {variant!r}; choose one of {choices}"
        ) from exc
    return f"{MODEL_REPOSITORY}/{path}"


def _load_checkpoint(path: Path, target: torch.device) -> nn.Module:
    """Load a local TorchScript or serialized module checkpoint."""
    try:
        model = torch.jit.load(str(path), map_location=target)
    except RuntimeError:
        payload = torch.load(str(path), map_location="cpu", weights_only=False)
        if not isinstance(payload, nn.Module):
            raise ValueError(
                "A local SEN2SR checkpoint must be TorchScript or a serialized "
                "torch.nn.Module; a state_dict needs the upstream MLSTAC manifest."
            )
        model = payload
    return model.to(target).eval()


def load_sen2sr(
    model_dir: str | Path | None = None,
    *,
    variant: str = DEFAULT_VARIANT,
    device: str | torch.device = "cpu",
    download: bool = True,
    cache_dir: str | Path | None = None,
    checkpoint: str | Path | None = None,
) -> SEN2SRAdapter:
    """Load a SEN2SR RGBN model from MLSTAC or a local checkpoint.

    ``model_dir`` points to a directory containing the upstream ``mlm.json``
    manifest.  ``checkpoint`` may point to a TorchScript file or a serialized
    ``torch.nn.Module``.  When neither is supplied, the official manifest is
    downloaded lazily through ``mlstac``.  Weights are constructed on CPU and
    moved only after loading, avoiding accidental GPU allocation during setup.
    """
    target = _device(device)
    if model_dir is not None and checkpoint is not None:
        raise ValueError("Provide model_dir or checkpoint, not both")

    if checkpoint is not None:
        path = Path(checkpoint).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"SEN2SR checkpoint does not exist: {path}")
        return SEN2SRAdapter(_load_checkpoint(path, target), device=target)

    try:
        import mlstac
    except ImportError as exc:
        raise ImportError(
            "Loading SEN2SR requires the optional 'mlstac' package. "
            "Install it with: pip install sen2sr mlstac"
        ) from exc

    if model_dir is None:
        if not download:
            raise ValueError("download=False requires model_dir or checkpoint")
        root = (
            Path(cache_dir).expanduser()
            if cache_dir
            else Path(tempfile.gettempdir()) / "opensr-utils" / "sen2sr"
        )
        root.mkdir(parents=True, exist_ok=True)
        model_dir = root / variant
        manifest = _manifest_url(variant)
        mlstac.download(file=manifest, output_dir=str(model_dir))
    path = Path(model_dir).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"SEN2SR model directory does not exist: {path}")
    loaded = mlstac.load(str(path))
    # MLSTAC's documented compiled_model API accepts a torch device.
    model = loaded.compiled_model(device=torch.device("cpu"))
    if not isinstance(model, nn.Module) and not callable(model):
        raise TypeError("MLSTAC compiled_model() did not return a callable model")
    return SEN2SRAdapter(model, device=target)


get_sen2sr = load_sen2sr

__all__ = ["DEFAULT_VARIANT", "SEN2SRAdapter", "get_sen2sr", "load_sen2sr"]
