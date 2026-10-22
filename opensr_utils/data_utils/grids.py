"""Open explicitly resampled inputs without materializing a full raster stack."""

from contextlib import contextmanager

import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT


@contextmanager
def open_input_raster(path, grid=None):
    """Read a source on the requested reference grid, preserving source masks.

    Nearest-neighbor mapping preserves native 20 m samples for SEN2SR's learned
    20 m -> 10 m stage. Both data loading and final validity use this same view.
    """
    with rasterio.open(path) as source:
        if grid is None or all(
            getattr(source, key) == value for key, value in grid.items()
        ):
            yield source
        else:
            if source.crs is None:
                raise ValueError(f"Cannot resample a source without a CRS: {path}")
            with WarpedVRT(
                source,
                **grid,
                resampling=Resampling.nearest,
                dtype="float32",
                nodata=float("nan"),
            ) as aligned:
                yield aligned
