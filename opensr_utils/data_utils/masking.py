"""Blockwise input validity and output mask handling.

Rasterio's per-band masks are intersected, so a pixel is valid only when all
input bands consider it valid and all of their values are finite.  This is
kept independent of pixel values: zero is a valid value whenever the source
mask says it is valid.
"""

from __future__ import annotations

import math
from contextlib import ExitStack
import numpy as np
from rasterio.windows import Window
from .grids import open_input_raster


def _source_paths(runner) -> list[str]:
    meta = getattr(runner, "image_meta", {})
    paths = meta.get("lr_file_dict") or {}
    if isinstance(paths, dict):
        # Preserve the dataset's established band order for deterministic
        # intersections, while avoiding duplicate paths.
        values = paths.values()
    else:
        values = paths
    result = []
    for path in values:
        path = str(path)
        if path not in result:
            result.append(path)
    if not result:
        raise ValueError("Cannot build an input mask without source rasters")
    return result


def _common_nodata(sources, indexes=None) -> float | int | None:
    """Return a common representable nodata value, if one exists."""
    values = []
    for src in sources:
        # A file may have distinct nodata values per band.  They all have to
        # agree for a single output nodata value to be meaningful.
        nodatavals = src.nodatavals
        if indexes is not None:
            nodatavals = [nodatavals[index - 1] for index in indexes]
        values.extend(nodatavals)
    if not values or any(value is None for value in values):
        return None
    first = values[0]
    if any(value != first for value in values[1:]):
        return None
    try:
        value = float(first)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value) or value < 0 or value > np.iinfo(np.uint16).max:
        return None
    if not value.is_integer():
        return None
    return int(value)


def _valid_window(sources, window: Window, indexes=None) -> np.ndarray:
    valid = None
    for src in sources:
        masks = src.read_masks(indexes=indexes, window=window)
        values = src.read(indexes=indexes, window=window)
        band_valid = np.all(masks != 0, axis=0)
        band_valid &= np.all(np.isfinite(values), axis=0)
        valid = band_valid if valid is None else (valid & band_valid)
    return valid


def apply_input_mask(runner, dst):
    """Apply the source validity mask to an upscaled output dataset.

    ``dst`` must be an open rasterio dataset in update mode.  The source is
    read a small LR window for each destination block and enlarged with
    nearest-neighbour indexing.  Rasterio writes the resulting validity as an
    internal TIFF mask, so no full-resolution boolean array is retained.

    The common source nodata value is copied when it is representable by
    ``uint16``; otherwise the output nodata value is cleared.  The internal
    mask remains authoritative, which means valid zero predictions stay
    valid even when the nodata value is zero.
    """
    meta = getattr(runner, "image_meta", {})
    factor = int(
        meta.get("output_factor") or meta.get("factor") or getattr(runner, "factor", 1)
    )
    if factor <= 0:
        raise ValueError(f"Upscaling factor must be positive, got {factor}")

    paths = _source_paths(runner)
    with ExitStack() as stack:
        sources = [
            stack.enter_context(open_input_raster(path, meta.get("resample_grid")))
            for path in paths
        ]
        indexes = None
        if getattr(runner, "input_type", None) == "file":
            indexes = meta.get("band_indexes")
            if indexes is not None:
                indexes = tuple(int(index) for index in indexes)
                if (
                    not indexes
                    or any(index < 1 for index in indexes)
                    or len(set(indexes)) != len(indexes)
                    or any(index > src.count for src in sources for index in indexes)
                ):
                    raise ValueError(
                        "band_indexes must refer to unique positive source bands."
                    )
        source_width = sources[0].width
        source_height = sources[0].height
        if any(
            src.width != source_width or src.height != source_height for src in sources
        ):
            raise ValueError(
                "Input rasters must have matching dimensions to build a mask"
            )

        nodata = _common_nodata(sources, indexes=indexes)
        # Rasterio exposes nodata as a profile value.  Updating the writer's
        # nodata values also ensures the value survives the atomic rename.
        dst.nodata = nodata

        for _, out_win in dst.block_windows(1):
            row0 = int(out_win.row_off)
            col0 = int(out_win.col_off)
            row1 = row0 + int(out_win.height)
            col1 = col0 + int(out_win.width)
            lr_row0 = row0 // factor
            lr_col0 = col0 // factor
            lr_row1 = min(source_height, (row1 + factor - 1) // factor)
            lr_col1 = min(source_width, (col1 + factor - 1) // factor)
            lr_window = Window(lr_col0, lr_row0, lr_col1 - lr_col0, lr_row1 - lr_row0)
            lr_valid = _valid_window(sources, lr_window, indexes=indexes)
            # Indexing avoids allocating a potentially large repeated source
            # window and handles edge blocks whose dimensions are partial.
            rows = (np.arange(row0, row1) // factor) - lr_row0
            cols = (np.arange(col0, col1) // factor) - lr_col0
            out_valid = lr_valid[np.ix_(rows, cols)]
            dst.write_mask(np.where(out_valid, 255, 0).astype(np.uint8), window=out_win)
    return dst


__all__ = ["apply_input_mask"]
