# -*- coding: utf-8 -*-
"""Data utilities for OpenSR-Utils."""

__all__ = [
    "write_to_placeholder",
    "PredictionDataModule",
    "can_read_directly_with_rasterio",
]


def __getattr__(name):
    if name == "write_to_placeholder":
        from .writing_utils import write_to_placeholder

        return write_to_placeholder
    if name == "PredictionDataModule":
        from .datamodule import PredictionDataModule

        return PredictionDataModule
    if name == "can_read_directly_with_rasterio":
        from .reading_utils import can_read_directly_with_rasterio

        return can_read_directly_with_rasterio
    raise AttributeError(f"module 'opensr_utils.data_utils' has no attribute {name!r}")
