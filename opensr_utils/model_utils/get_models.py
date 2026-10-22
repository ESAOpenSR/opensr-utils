"""Lazy pretrained model loaders with a shared, explicit inference interface.

Importing this module never imports Torch or any optional model package. Models
are loaded on CPU by default; the pipeline lets Lightning place them later.
"""

from __future__ import annotations

from pathlib import Path


MODEL_NAMES = ("None", "LDSRS2", "SEN2SR", "ESASRGAN", "SRGAN")


def normalize_model_name(name) -> str:
    """Resolve documented aliases without importing model dependencies."""
    key = (
        "NONE"
        if name is None
        else str(name).strip().upper().replace("-", "").replace("_", "")
    )
    aliases = {
        "NONE": "None",
        "BILINEAR": "None",
        "INTERPOLATION": "None",
        "LDSRS2": "LDSRS2",
        "SEN2SR": "SEN2SR",
        "SEN2SRLITE": "SEN2SR",
        "ESASRGAN": "ESASRGAN",
        "SRGAN": "ESASRGAN",
    }
    try:
        return aliases[key]
    except KeyError as exc:
        raise ValueError(
            f"Unknown model {name!r}. Choose one of {', '.join(MODEL_NAMES)}."
        ) from exc


def get_ldsrs2(device: str = "cpu"):
    """Load ESA's pretrained four-band LDSR-S2 diffusion model.

    Install ``opensr-utils[model]`` first. The model's own loader manages its
    pretrained checkpoint; model construction starts on the requested device.
    """
    from io import StringIO

    import requests
    from omegaconf import OmegaConf

    try:
        import opensr_model
    except ImportError as exc:
        raise ImportError(
            "LDSR-S2 requires: python -m pip install 'opensr-utils[model]'"
        ) from exc

    config_url = "https://raw.githubusercontent.com/ESAOpenSR/opensr-model/refs/heads/main/opensr_model/configs/config_10m.yaml"
    response = requests.get(config_url, timeout=30)
    response.raise_for_status()
    config = OmegaConf.load(StringIO(response.text))
    model = opensr_model.SRLatentDiffusion(config, device=device)
    model.load_pretrained(config.ckpt_version)
    return model


def get_sen2sr(*args, **kwargs):
    """Load a four- or ten-band SEN2SR adapter; see ``sen2sr.load_sen2sr``."""
    from .sen2sr import load_sen2sr

    return load_sen2sr(*args, **kwargs)


def get_esasrgan(*args, **kwargs):
    """Load ESA's RGB-NIR SRGAN adapter; see ``esasrgan.load_esasrgan``."""
    from .esasrgan import load_esasrgan

    return load_esasrgan(*args, **kwargs)


def get_model(
    name,
    *,
    device="cpu",
    checkpoint=None,
    config=None,
    cache_dir=None,
    variant=None,
    download=True,
    **model_options,
):
    """Load a named model, with downloads and local artifacts kept explicit.

    ``SEN2SR`` defaults to four-band SEN2SRLite RGBN. ``variant='main'`` selects
    all ten bands; ``backbone='ldsrs2'`` replaces its RGB-NIR generator with
    LDSR-S2, keeping SEN2SR's learned fusion stages. A local ``checkpoint``
    can name its MLSTAC directory/manifest, or a supported exported model file.
    ``ESASRGAN`` and ``SRGAN`` are aliases for ESA's RGB-NIR preset; official
    local state dictionaries require both ``config`` and ``checkpoint``.
    ``None`` selects interpolation and returns ``None`` for the pipeline.

    ``download=False`` requires local artifacts and never falls back to a
    remote preset. Advanced adapter options may be passed as keyword arguments.
    """
    name = normalize_model_name(name)
    if name == "None":
        if (
            checkpoint is not None
            or config is not None
            or variant is not None
            or model_options
        ):
            raise ValueError(
                "Interpolation does not use model checkpoints, configuration or variants."
            )
        return None
    if name == "LDSRS2":
        if (
            checkpoint is not None
            or config is not None
            or not download
            or model_options
        ):
            from .ldsrs2 import load_ldsrs2

            if cache_dir is not None or variant is not None:
                raise ValueError("LDSR-S2 does not use cache_dir or variant overrides.")
            return load_ldsrs2(
                device=device,
                checkpoint=checkpoint,
                config_path=config,
                download=download,
                **model_options,
            )
        if (
            checkpoint is not None
            or config is not None
            or cache_dir is not None
            or variant is not None
            or not download
            or model_options
        ):
            raise ValueError(
                "LDSRS2 uses its upstream pretrained loader; checkpoint/config/cache/variant overrides and offline loading are not supported here."
            )
        return get_ldsrs2(device=device)
    if name == "SEN2SR":
        if config is not None:
            raise ValueError(
                "SEN2SR uses an MLSTAC manifest; pass its directory with checkpoint= instead of config=."
            )
        if not download and checkpoint is None:
            raise ValueError(
                "Offline SEN2SR loading requires local checkpoint artifacts."
            )
        options = dict(
            device=device, cache_dir=cache_dir, download=download, **model_options
        )
        if variant is not None:
            options["variant"] = variant
        if checkpoint is not None:
            path = Path(checkpoint).expanduser()
            if path.is_dir() or path.name == "mlm.json":
                options["model_dir"] = path if path.is_dir() else path.parent
            else:
                options["checkpoint"] = path
        return get_sen2sr(**options)
    if variant is not None:
        raise ValueError(
            "variant is a SEN2SR option; ESA SRGAN uses the RGB-NIR preset."
        )
    if not download:
        if checkpoint is None:
            raise ValueError(
                "Offline ESA SRGAN loading requires a local checkpoint (and config for official state dictionaries)."
            )
        if not Path(checkpoint).expanduser().is_file():
            raise FileNotFoundError(
                f"Local ESA SRGAN checkpoint does not exist: {checkpoint}"
            )
    return get_esasrgan(
        device=device,
        checkpoint=checkpoint,
        config_path=config,
        cache_dir=cache_dir,
        **model_options,
    )


__all__ = [
    "MODEL_NAMES",
    "normalize_model_name",
    "get_model",
    "get_ldsrs2",
    "get_sen2sr",
    "get_esasrgan",
]
