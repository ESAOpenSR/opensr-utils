import rasterio
import os, shutil, zipfile
from pathlib import Path
from datetime import datetime



def can_read_directly_with_rasterio(self,filename):
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
            self._log(f"Rasterio can't open {filename}.")
            return False
    else:
        self._log(f"Unsupported file extension for {filename}.")
        return False


def create_dirs(self):
    """
    Create the working directory structure for large-file super-resolution.

    Based on the resolved input type (`file`, `SAFE`, or `S2GM`), this function 
    determines the appropriate *base directory* and sets up the following:

    - `logs/` folder: stores log files and preview outputs.
    - `temp/` folder: stores intermediate SR patches during inference.
    - Final SR outputs (e.g., `sr.tif`) are written directly into the base directory.
    - `sr_placeholder.tif`: an empty GeoTIFF placeholder created in the base 
      directory to receive stitched SR results.

    Rules for determining the base directory
    ----------------------------------------
    - If input is a `.SAFE` folder → use its parent folder.
    - If input is a single raster file → use the file's parent folder.
    - If input is an `S2GM` folder → use the folder itself.

    Side Effects
    ------------
    - Creates directories if they do not exist.
    - Sets the following attributes on `self`:
        * `self.placeholder_path` : str → path to the base directory
        * `self.log_dir`          : str → path to `logs/` folder
        * `self.temp_folder`      : str → path to `temp/` folder
        * `self.output_dir`       : str → path to base directory (for `sr.tif`)
        * `self.placeholder_filepath` : str → path to placeholder GeoTIFF
        * `self.output_file_path`     : str → alias for placeholder filepath
        * `self.image_meta["placeholder_dir"]`
        * `self.image_meta["placeholder_filepath"]`

    Notes
    -----
    - This function does not create the placeholder file itself; it only sets
      paths and directories. Call `create_placeholder_file()` afterwards to
      initialize `sr_placeholder.tif`.
    - Ensures idempotency: repeated calls will not overwrite directories.
    - Keeps both `self.output_file_path` and `self.placeholder_filepath` for
      compatibility with downstream code.
    """
    p = Path(self.root)
    if self.input_type == "SAFE":
        base = p.parent
    elif self.input_type == "file":
        base = p.parent
    elif self.input_type == "S2GM":
        base = p
    else:
        base = p

    self.placeholder_path = str(base)

    # --- Handle logs folder with increment ---
    log_base = base / "logs"
    log_dir = log_base
    counter = 1
    while log_dir.exists():
        counter += 1
        log_dir = base / f"logs_{counter}"
    os.makedirs(log_dir, exist_ok=True)
    
    # Always use "temp" without increment
    temp_dir = base / "temp"
    os.makedirs(temp_dir, exist_ok=True)

    # Assign attributes
    self.log_dir = str(log_dir)
    self.temp_folder = str(temp_dir)

    # Create log file
    self.log_file = log_dir / "log.txt"
    open(self.log_file, 'a').close()
    self._log(f"🗂️  Logs will be saved to: {self.log_file}")

    # Final SR output goes directly into base
    self.output_dir = str(base)

    # Placeholder paths
    self.placeholder_filepath = str(base / "sr_placeholder.tif")
    self.output_file_path = self.placeholder_filepath

    # Metadata
    self.image_meta["placeholder_dir"] = self.placeholder_path
    self.image_meta["placeholder_filepath"] = self.placeholder_filepath

def verify_input_file_type(self, root):
    """
    Resolve the input type before any dirs/logging are created:
    - Regular raster file (rasterio-readable)
    - Sentinel-2 .SAFE folder
    - Sentinel-2 S2GM folder
    - .zip archive containing a .SAFE folder (from Copernicus Hub)

    Sets:
        self.root        → final usable path (file or folder)
        self.input_type  → "file" | "SAFE" | "S2GM"

    Notes:
        Logs here are printed to stdout.
    """
    self.root = str(root)
    p = Path(self.root)

    # --- ZIP case ---
    if p.is_file() and p.suffix.lower() == ".zip":
        self._log(f"📦 Unzipping: {p.name}")
        extract_dir = p.parent / p.stem
        extract_dir.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(p, "r") as zf:
            zf.extractall(extract_dir)

        if self.debug==False: # only delete if not debugging
            p.unlink()  # delete the zip
            self._log(f"🗑️ Deleted archive: {p.name}")

        safe_candidates = list(extract_dir.rglob("*.SAFE"))
        if not safe_candidates:
            raise NotImplementedError("🚫 Archive does not contain a .SAFE folder.")
        self.root = str(safe_candidates[0])
        self.input_type = "SAFE"
        self._log(f"📁 Found .SAFE folder: {self.root}")

    # --- File case ---
    elif p.is_file():
        self.input_type = "file"
        if not can_read_directly_with_rasterio(self, self.root):
            raise NotImplementedError("🚫 File type not supported by rasterio ❌")
        self._log("📄 Raster file OK — processing possible! 🚀")

    # --- SAFE folder ---
    elif p.is_dir() and p.name.endswith(".SAFE"):
        self.input_type = "SAFE"
        self._log("📁 Input is Sentinel-2 .SAFE folder — processing possible! 🚀")

    # --- S2GM folder ---
    elif p.is_dir() and "S2GM" in str(p):
        self.input_type = "S2GM"
        self._log("📁 Input is Sentinel-2 S2GM folder — processing possible! 🚀")

    else:
        raise NotImplementedError("🚫 Input path is not valid (file, .SAFE, or S2GM).")

    # ✅ At this point, self.root + self.input_type are FINAL
