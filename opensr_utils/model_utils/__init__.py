# -*- coding: utf-8 -*-
"""Model utilities for OpenSR-Utils."""

__all__ = [
    "preprocess_model",
]


def __getattr__(name):
    if name == "preprocess_model":
        from .prepare_model import preprocess_model

        return preprocess_model
    raise AttributeError(f"module 'opensr_utils.model_utils' has no attribute {name!r}")
