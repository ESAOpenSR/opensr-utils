# -*- coding: utf-8 -*-
"""OpenSR-Utils: Utilities for Super-Resolution of Sentinel-2 imagery."""

from importlib.metadata import version

try:
    __version__ = version("opensr-utils")
except Exception:
    __version__ = "unknown"

__all__ = ["large_file_processing", "__version__"]


def __getattr__(name):
    if name == "large_file_processing":
        from .pipeline import large_file_processing

        return large_file_processing
    raise AttributeError(f"module 'opensr_utils' has no attribute {name!r}")
