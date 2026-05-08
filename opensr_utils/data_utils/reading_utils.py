import rasterio
import os, shutil, zipfile
from pathlib import Path
from datetime import datetime
import time


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

    def _is_rank0_env() -> bool:
        # Works before torch.distributed init: rely on launcher env vars if present
        lr = os.environ.get("LOCAL_RANK", "")
        r  = os.environ.get("RANK", "")
        ws = os.environ.get("WORLD_SIZE", "1")
        if ws not in ("", "1"):   # multi-process likely
            # treat rank0 when both LOCAL_RANK and RANK are 0 or empty
            return (lr in ("", "0")) and (r in ("", "0"))
        return True  # single process → rank0

    def _wait_until(predicate, timeout_s=600.0, sleep_s=0.1, what="dirs"):
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            try:
                if predicate():
                    return
            except Exception:
                pass
            time.sleep(sleep_s)
        raise TimeoutError(f"Timed out waiting for {what} prepared by rank0.")

    p = Path(self.root)
    if self.input_type == "SAFE":
        base = p.parent
    elif self.input_type == "file":
        base = p.parent
    elif self.input_type == "S2GM":
        base = p
    else:
        base = p

    run_id = getattr(self, "run_id", os.environ.get("OPENSR_RUN_ID", "default"))
    # Where we drop tiny markers so other ranks can read the chosen paths.
    logs_marker = base / f".opensr_{run_id}_logs_path"
    temp_marker = base / f".opensr_{run_id}_temp_path"
    ready_marker = base / f".opensr_{run_id}_dirs_ready"

    is_r0 = _is_rank0_env()

    if is_r0:
        # --- Pick run-scoped log/temp dirs and create them ---
        log_dir = base / f"logs_{run_id}"
        os.makedirs(log_dir, exist_ok=True)

        temp_dir = base / f"temp_{run_id}"
        os.makedirs(temp_dir, exist_ok=True)

        # --- Create log file once ---
        self.log_file = log_dir / "log.txt"
        open(self.log_file, "a").close()

        # --- Write markers so non-rank0 can discover paths deterministically ---
        try:
            logs_marker.write_text(str(log_dir))
        except Exception:
            pass
        try:
            temp_marker.write_text(str(temp_dir))
        except Exception:
            pass
        try:
            ready_marker.write_text("ok")
        except Exception:
            pass

        # Assign attributes
        self.log_dir = str(log_dir)
        self.temp_folder = str(temp_dir)
        self._log(f"🗂️  Logs will be saved to: {self.log_file}")

    else:
        # --- Non-rank0: wait until rank0 finished mkdirs ---
        def _dirs_ready():
            if ready_marker.exists() and logs_marker.exists() and temp_marker.exists():
                return True
            return False

        _wait_until(_dirs_ready, what="directory creation")

        # Resolve exact logs dir deterministically
        log_dir_path = None
        if logs_marker.exists():
            try:
                text = logs_marker.read_text().strip()
                if text:
                    maybe = Path(text)
                    if maybe.exists():
                        log_dir_path = maybe
            except Exception:
                log_dir_path = None

        if log_dir_path is None:
            raise RuntimeError("Could not locate logs directory created by rank0.")

        temp_dir = None
        if temp_marker.exists():
            text = temp_marker.read_text().strip()
            if text:
                maybe = Path(text)
                if maybe.exists():
                    temp_dir = maybe
        if temp_dir is None:
            raise RuntimeError("Could not locate temp directory created by rank0.")
        if not temp_dir.exists():
            raise RuntimeError("Temp directory does not exist yet; rank0 mkdir likely failed.")

        # Assign attributes (NO file writes here)
        self.log_dir = str(log_dir_path)
        self.temp_folder = str(temp_dir)
        # Do NOT set self.log_file on non-rank0 to avoid concurrent writes

    # Common paths for all ranks
    self.placeholder_path = str(base)
    self.output_dir = str(base)
    self.placeholder_filepath = str(base / f"sr_placeholder_{run_id}.tif")
    self.output_file_path = self.placeholder_filepath
    self.final_sr_path = str(base / "sr.tif")

    # Metadata
    self.image_meta["placeholder_dir"] = self.placeholder_path
    self.image_meta["placeholder_filepath"] = self.placeholder_filepath
    self.image_meta["final_sr_path"] = self.final_sr_path

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
    """
    import os, time, zipfile
    from pathlib import Path

    def _is_rank0_env() -> bool:
        lr = os.environ.get("LOCAL_RANK", "")
        r  = os.environ.get("RANK", "")
        ws = os.environ.get("WORLD_SIZE", "1")
        if ws not in ("", "1"):
            return (lr in ("", "0")) and (r in ("", "0"))
        return True  # CPU/1-GPU → rank0 behavior

    def _can_read_with_rasterio(self, path: str) -> bool:
        try:
            import rasterio
            with rasterio.open(path):
                return True
        except Exception:
            return False

    def _safe_extract(zf, destination: Path):
        dest = destination.resolve()
        for member in zf.infolist():
            member_path = (destination / member.filename).resolve()
            try:
                member_path.relative_to(dest)
            except ValueError as exc:
                raise RuntimeError(f"Unsafe path in zip archive: {member.filename}") from exc
        zf.extractall(destination)

    self.root = str(root)
    p = Path(self.root)

    # ---------------- ZIP-like case (works even if .zip already deleted) ----------------
    if str(p).lower().endswith(".zip"):
        extract_dir = p.parent / p.stem         # .../<zip_stem>/
        sentinel    = extract_dir / ".unzipped_ok"

        def _find_safe():
            # search nested (zip may contain a top-level folder before the .SAFE)
            return sorted(extract_dir.rglob("*.SAFE"))

        if _is_rank0_env():
            # Ensure target dir
            extract_dir.mkdir(parents=True, exist_ok=True)
            need_unzip = not _find_safe()

            # Only unzip if we actually still have the zip and no .SAFE yet
            if need_unzip and p.exists():
                self._log(f"📦 Unzipping: {p.name}")
                with zipfile.ZipFile(p, "r") as zf:
                    _safe_extract(zf, extract_dir)

            # Mark ready for other ranks
            try:
                sentinel.write_text("ok")
            except Exception:
                pass

            # Optionally delete zip (rank0 only). Keep the source archive by default.
            if getattr(self, "delete_input_zip", False) and p.exists():
                p.unlink()
                self._log(f"🗑️ Deleted archive: {p.name}")
        else:
            # Non-rank0: wait until .SAFE appears (or sentinel says ok)
            timeout_s = 600.0
            poll = 0.5
            waited = 0.0
            while waited < timeout_s:
                if sentinel.exists() or _find_safe():
                    break
                time.sleep(poll)
                waited += poll
            if not _find_safe():
                raise RuntimeError(
                    f"Timeout waiting for rank0 to unzip {p.name}; expected .SAFE in {extract_dir}"
                )

        safe_candidates = _find_safe()
        if not safe_candidates:
            raise NotImplementedError("🚫 Archive does not contain a .SAFE folder.")
        self.root = str(safe_candidates[0])
        self.input_type = "SAFE"
        self._log(f"📁 Found .SAFE folder: {self.root}")
        return  # done

    # ---------------- File case ----------------
    if p.is_file():
        if not _can_read_with_rasterio(self, self.root):
            raise NotImplementedError("🚫 File type not supported by rasterio ❌")
        self.input_type = "file"
        self._log("📄 Raster file OK — processing possible! 🚀")
        return

    # ---------------- SAFE folder ----------------
    if p.is_dir() and p.name.endswith(".SAFE"):
        self.input_type = "SAFE"
        self._log("📁 Input is Sentinel-2 .SAFE folder — processing possible! 🚀")
        return

    # ---------------- S2GM folder ----------------
    if p.is_dir() and "S2GM" in str(p):
        self.input_type = "S2GM"
        self._log("📁 Input is Sentinel-2 S2GM folder — processing possible! 🚀")
        return

    # ---------------- Not recognized ----------------
    raise NotImplementedError("🚫 Input path is not valid (file, .SAFE, or S2GM).")
