# -*- coding: utf-8 -*-
"""Model utilities for OpenSR-Utils."""

__all__ = [
    "preprocess_model",
    "get_model",
    "get_ldsrs2",
    "get_sen2sr",
    "get_esasrgan",
]


def __getattr__(name):
    if name in {"get_model", "get_ldsrs2", "get_sen2sr", "get_esasrgan"}:
        from . import get_models

        return getattr(get_models, name)
    if name == "preprocess_model":
        from .prepare_model import preprocess_model

        return preprocess_model
    raise AttributeError(f"module 'opensr_utils.model_utils' has no attribute {name!r}")
