import os
import rasterio

def can_read_directly_with_rasterio(filename):
    """
    Check whether a given raster file can be opened directly with rasterio.

    This function first verifies that the file extension is among a set of
    common geospatial and raster formats (GeoTIFF, JPEG2000, PNG, NetCDF,
    HDF4/5, ENVI, DIMAP, VRT, etc.). If the extension is supported, it then
    attempts to actually open the file with rasterio. This ensures that the
    file is not only of a supported type by name but also valid and readable.

    Parameters
    ----------
    filename : str
        Path to the raster file to check.

    Returns
    -------
    bool
        True if rasterio successfully opens the file, False otherwise.

    Notes
    -----
    - Prints a message if the extension is unsupported or if rasterio fails
      to open the file.
    - Useful for input validation in preprocessing pipelines before attempting
      large-scale reading or tiling.

    Examples
    --------
    >>> can_read_directly_with_rasterio("scene.tif")
    True
    >>> can_read_directly_with_rasterio("scene.xyz")
    Unsupported file extension for scene.xyz.
    False
    """
    # Define a set of supported file extensions
    supported_extensions = {
        '.tif', '.tiff',  # GeoTIFF
        '.jpg', '.jpeg',  # JPEG
        '.jp2', '.j2k', '.jpf', '.jpx', '.jpm', '.mj2',  # JPEG2000
        '.png',  # PNG
        '.nc',  # NetCDF
        '.hdf', '.h4', '.hd4',  # HDF4
        '.h5', '.hdf5', '.he5',  # HDF5
        '.img',  # ERDAS IMG and possibly ENVI
        '.hdr',  # ENVI header (data file extension needs checking)
        '.dim',  # DIMAP
        '.vrt',  # VRT
    }

    # Extract the file extension and check if it's in the set of supported extensions
    _, ext = os.path.splitext(filename)
    if ext.lower() in supported_extensions:
        try:
            # Attempt to open the file with Rasterio to ensure it's not just a matching extension
            with rasterio.open(filename) as src:
                return True
        except rasterio.errors.RasterioIOError:
            print(f"Rasterio can't open {filename}.")
            return False
    else:
        print(f"Unsupported file extension for {filename}.")
        return False