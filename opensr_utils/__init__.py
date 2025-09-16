# -*- coding: utf-8 -*-
"""OpenSR-Utils: Utilities for Super-Resolution of Sentinel-2 imagery."""

from importlib.metadata import version

try:
    __version__ = version("opensr-utils")
except Exception:
    __version__ = "unknown"

# expose high-level class for convenience
from .pipeline import large_file_processing

__all__ = ["large_file_processing", "__version__"]