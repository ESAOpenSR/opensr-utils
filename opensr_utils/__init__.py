# -*- coding: utf-8 -*-
"""OpenSR-Utils: Utilities for Super-Resolution of Sentinel-2 imagery."""

from importlib.metadata import version

try:
    __version__ = version("opensr-utils")
except Exception:
    __version__ = "unknown"

__all__ = ["super_resolve", "large_file_processing", "__version__"]


def super_resolve(input_path, model=None, output=None, **options):
    """Run tiled inference and return the output GeoTIFF path.

    ``input_path`` accepts a raster, SAFE directory/archive, or S2GM directory.
    ``model`` accepts a PyTorch/Lightning module, ``"LDSRS2"``, ``"SEN2SR"``,
    ``"ESASRGAN"``/``"SRGAN"``, or ``None`` for bilinear interpolation.
    Named models need their optional extra. Device selection,
    tiling, batching, blending, temporary storage and cleanup are automatic.

    Pass advanced options supported by :class:`large_file_processing`, such as
    ``factor``, ``batch_size``, ``cleanup=False`` or ``overwrite=True``. Named
    model downloads accept ``checkpoint``, ``model_config`` and ``cache_dir``.
    Declared model scale and units are used automatically. Model
    outputs default to normalized [0, 1]; choose ``output_units="reflectance"``
    for values already in [0, 10000]. Debug runs return their manifest path and
    preserve predictions without creating a final raster.
    """
    if "auto_run" in options:
        raise TypeError("super_resolve always runs inference; omit auto_run.")
    from .pipeline import large_file_processing

    with large_file_processing(
        root=input_path, model=model, output_path=output, **options
    ) as runner:
        return runner.run()


def __getattr__(name):
    if name == "large_file_processing":
        from .pipeline import large_file_processing

        return large_file_processing
    raise AttributeError(f"module 'opensr_utils' has no attribute {name!r}")
