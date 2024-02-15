import os
import rasterio

def can_read_directly_with_rasterio(filename):
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
