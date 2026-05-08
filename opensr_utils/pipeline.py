# -*- coding: utf-8 -*-


# torch stuff
import torch
from pytorch_lightning import LightningModule
torch.set_float32_matmul_precision('medium')

# general imports
import os
import shutil
from datetime import datetime

# Geo
import rasterio
from rasterio.transform import Affine


# local imports
from opensr_utils.data_utils.datamodule import PredictionDataModule
from opensr_utils.model_utils.prepare_model import preprocess_model


class large_file_processing():
    """
    Large-scale super-resolution pipeline for Sentinel-2 and other geospatial inputs.

    This class orchestrates the full workflow for processing large raster datasets
    (e.g. Sentinel-2 SAFE archives, S2GM folders, or standalone GeoTIFFs) with a 
    super-resolution (SR) model. It handles input validation, directory setup, 
    patch-based inference, distributed (DDP) support, and final stitched outputs.

    Workflow
    --------
    1. **Validate input and arguments**
        - Checks existence of `root`, model type, overlap/border settings, SR factor.
    2. **Detect input type**
        - Supports raster file, `.SAFE` folder, `S2GM` folder, or zipped SAFE archive.
        - If zipped, unzips and updates `self.root`.
    3. **Create directories**
        - Sets up `logs/`, `temp/` next to the input for intermediate files.
    4. **Extract image metadata**
        - Reads size, CRS, transforms, and computes sliding-window coordinates.
    5. **Create placeholder file**
        - Writes an empty GeoTIFF (`sr_placeholder.tif`) to receive SR results.
    6. **Initialize datamodule**
        - Builds a PyTorch Lightning DataModule for batched inference windows.
    7. **Preprocess model**
        - Places model on correct device, switches to eval mode, prepares for inference.
    8. **Run super-resolution**
        - If not in debug mode, runs patch-based inference and saves results in `temp/`.
    9. **Stitch patches into final output**
        - On rank 0 only: stitches patches into the placeholder file → `final_sr_path`.
        10. **Cleanup and previews**
        - Deletes temporary LR patches.
        - Optionally saves example crops and preview images in `logs/`.

    Parameters
    ----------
    root : str
        Path to input (SAFE folder, S2GM folder, single raster, or Copernicus zip).
    model : torch.nn.Module | LightningModule | None, default=None
        Super-resolution model to use. If `None`, runs in a placeholder/no-model mode.
    window_size : tuple[int, int], default=(128, 128)
        Spatial size of low-resolution windows (before SR).
    factor : int, default=4
        Super-resolution upscaling factor. Must be one of {2, 4, 6, 8}.
    overlap : int, default=8
        Overlap (in LR pixels) between adjacent windows. Must be even.
    eliminate_border_px : int, default=0
        Number of border pixels to discard per window in the SR output.
        Must be even and smaller than `overlap`.
    device : {"cpu","cuda"}, default="cpu"
        Device on which to run the model.
    gpus : int | list[int] | None, default=1
        GPU index or list of GPU indices. If `device="cpu"`, ignored.
    save_preview : bool, default=False
        If True, saves 1 cropped SR excerpt and 10 side-by-side LR/SR previews.
    debug : bool, default=False
        If True, processes only 100 windows globally (DDP-safe).

    Attributes (after init)
    -----------------------
    root : str
        Final resolved input path (after unzip if necessary).
    input_type : str
        One of {"file","SAFE","S2GM"}.
    placeholder_path : str
        Base folder containing logs/temp/sr.
    log_dir : str
        Path to the logs directory.
    temp_folder : str
        Path to temporary patch storage.
    output_dir : str
        Path to SR output directory.
    image_meta : dict
        Metadata such as CRS, transforms, dimensions, and window layout.
    model : torch.nn.Module | LightningModule
        Preprocessed model in eval mode.
    final_sr_path : str
        Path to the stitched super-resolved GeoTIFF (set after stitching).

    Notes
    -----
    - Designed for multi-GPU environments with PyTorch Lightning (DDP).
    - Uses weighted overlap blending to avoid patch seams.
    - On rank 0, the stitched result is written to disk and previews are saved.
    - Non-rank0 processes perform inference only, without logging or stitching.
    """
    def __init__(self,
                 root: str,
                 model=None,
                 window_size: tuple = (128, 128),
                 factor: int = 4,
                 overlap: int = 8,
                 eliminate_border_px=0,
                 device: str = "cpu",
                 gpus=None,
                 save_preview: bool = False,
                 debug=False,
                 auto_run: bool = False,
                 cleanup: bool = True,
                 overwrite: bool = False,
                 delete_input_zip: bool = False,
                 batch_size: int = 16,
                 num_workers: int = 4,
                 prefetch_factor: int = 2,
                 compressed_patches: bool = False,
                 ):
        # 1) Validate public inputs. Do not use assert: optimized Python removes it.
        if model is not None and not isinstance(model, (LightningModule, torch.nn.Module)):
            raise TypeError("Model must be a PyTorch module, LightningModule, or None.")
        if not isinstance(window_size, (tuple, list)) or len(window_size) != 2:
            raise TypeError("window_size must be a tuple/list of two positive integers.")
        if not all(isinstance(v, int) and v > 0 for v in window_size):
            raise ValueError("window_size must contain two positive integers.")
        if not isinstance(overlap, int) or overlap < 0 or overlap % 2 != 0:
            raise ValueError("overlap must be a non-negative even integer.")
        if not isinstance(eliminate_border_px, int) or eliminate_border_px < 0 or eliminate_border_px % 2 != 0:
            raise ValueError("eliminate_border_px must be a non-negative even integer.")
        if eliminate_border_px >= overlap:
            raise ValueError("eliminate_border_px must be smaller than overlap.")
        if (overlap - eliminate_border_px) < 4:
            raise ValueError("overlap must be at least 4 px bigger than eliminate_border_px.")
        if overlap >= min(window_size):
            raise ValueError("overlap must be smaller than both window dimensions.")
        if factor not in [2, 4, 6, 8]:
            raise ValueError("Factor must be one of [2,4,6,8].")
        if device not in ("cpu", "cuda"):
            raise ValueError("device must be 'cpu' or 'cuda'.")
        for name, value in {
            "batch_size": batch_size,
            "num_workers": num_workers,
            "prefetch_factor": prefetch_factor,
        }.items():
            if not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer.")
        if batch_size == 0:
            raise ValueError("batch_size must be greater than zero.")


        # General Settings
        self.debug = bool(debug) # if True, only process 100 windows globally (DDP-safe)
        self.root = root # path to folder containing S2 SAFE data format
        self.window_size = tuple(window_size) # window size of the LR image
        self.factor = factor # sr factor of the model
        self.overlap = overlap # overlap in px of the LR image windows
        self.eliminate_border_px = eliminate_border_px # border pixels to eliminate in px on each side of the SR image
        self.device = device # device to run model on
        self.save_preview = bool(save_preview)
        self.cleanup = bool(cleanup)
        self.overwrite = bool(overwrite)
        self.delete_input_zip = bool(delete_input_zip)
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.prefetch_factor = int(prefetch_factor)
        self.compressed_patches = bool(compressed_patches)
        existing_run_id = os.environ.get("OPENSR_RUN_ID")
        launcher_env = (
            os.environ.get("WORLD_SIZE", "1") not in ("", "1")
            or os.environ.get("LOCAL_RANK") is not None
            or os.environ.get("RANK") is not None
        )
        if existing_run_id and launcher_env:
            self.run_id = existing_run_id
        else:
            self.run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{os.getpid()}"
            os.environ["OPENSR_RUN_ID"] = self.run_id
        if isinstance(gpus, int): # pass GPU/s to Trainer as list
            self.gpus = [gpus] # number of gpus or list of gpu ids to use
        elif gpus is None:
            self.gpus = None
        elif isinstance(gpus, (list, tuple)):
            self.gpus = list(gpus)
        else:
            raise TypeError("gpus must be None, an int, or a list/tuple of ints.")
        if self.gpus is not None and not all(isinstance(g, int) and g >= 0 for g in self.gpus):
            raise ValueError("gpus must contain non-negative integer GPU ids.")
        if self.device == "cpu":
            self.gpus = None
        elif self.gpus is None:
            self.gpus = [0]
            
        # 3) Locals/init-only (no paths derived yet!) - leave empty
        self.image_meta = {}
        self.input_type = None
        self.placeholder_filepath = None
        self.placeholder_path = None
        self.log_dir = None
        self.temp_folder = None
        self.final_sr_path = None

        # 4) Verify input type FIRST (will also unzip and update self.root if needed)
        from opensr_utils.data_utils.reading_utils import verify_input_file_type
        verify_input_file_type(self, self.root) # multi-GPU safe

        # 5) Create directories AFTER type detection
        from opensr_utils.data_utils.reading_utils import create_dirs
        create_dirs(self)  # multi-GPU safe
        
        # Get Image information based on input, including image coordinate windows
        self.get_image_meta(self.root) # multi-GPU safe
        
        # Create Datamodule based on input files and Windows
        self.create_datamodule() # Runs on all Ranks

        # Make sure model is useable and in eval mode
        self.model = preprocess_model(self,model,
                                      temp_folder=self.temp_folder,
                                      windows=self.image_meta["image_windows"],
                                      factor=self.factor)

        # Print Status
        self._log("📊 Status: Model and Data ready for inference ✅")

        if auto_run:
            self.run()

    def create_datamodule(self, windows=None):
        """
        Create a PredictionDataModule for patch-based inference.

        The datamodule wraps the input type, root path, and sliding windows,
        and handles batching for PyTorch Lightning’s predict loop.
        """
        dm = PredictionDataModule(input_type=self.input_type,
                              root=self.root,
                              windows=windows or self.image_meta["image_windows"],
                              lr_file_dict=self.image_meta["lr_file_dict"],
                              prefetch_factor=self.prefetch_factor,
                              batch_size=self.batch_size,
                              num_workers=self.num_workers)
        dm.setup()
        self._log(f"📦 Created PredictionDataModule with {len(dm.dataset)} patches.")
        self.datamodule = dm

    def get_image_meta(self, root_dir):
        """
        Extract raster metadata (width, height, dtype, CRS, transform, band count).

        - For files: opens directly with rasterio.
        - For SAFE: locates JP2 bands B02, B03, B04, B08.
        - For S2GM: locates GeoTIFF bands B02, B03, B04, B08.

        Adds a list of sliding image windows to `self.image_meta`.
        """
        if self.input_type == "file":
            self.image_meta["lr_file_dict"] = {"file": root_dir}
            with rasterio.open(root_dir) as src:
                self.image_meta["width"], self.image_meta["height"], self.image_meta["dtype"] = src.width, src.height, src.dtypes[0]
                self.image_meta["transform"] = src.transform
                self.image_meta["crs"] = src.crs
                self.image_meta["bands"] = src.count

        elif self.input_type == "SAFE":
            # Get LR image file paths
            jp2s = []
            for root, _, files in os.walk(self.root):
                if "IMG_DATA" in root:
                    for f in files:
                        if f.endswith(".jp2") and any(f"_{b}_" in f or f"_{b}." in f for b in ("B02", "B03", "B04", "B08")):
                            jp2s.append(os.path.join(root, f))
            jp2s = sorted(jp2s)  # Sort to maintain consistent band order

            def pick_safe_band(band):
                matches = [p for p in jp2s if f"_{band}_" in os.path.basename(p) or f"_{band}." in os.path.basename(p)]
                if not matches:
                    raise FileNotFoundError(f"Could not locate SAFE band {band}.")
                ten_meter = [p for p in matches if "_10m" in os.path.basename(p)]
                return sorted(ten_meter or matches)[0]

            self.image_meta["lr_file_dict"] = {
                "R": pick_safe_band("B04"),
                "G": pick_safe_band("B03"),
                "B": pick_safe_band("B02"),
                "NIR": pick_safe_band("B08"),
            }

            # Get image properties
            with rasterio.open(self.image_meta["lr_file_dict"]["R"]) as src:
                self.image_meta["width"], self.image_meta["height"], self.image_meta["dtype"] = src.width, src.height, src.dtypes[0]
                self.image_meta["transform"] = src.transform
                self.image_meta["crs"] = src.crs
                self.image_meta["bands"] = len(self.image_meta["lr_file_dict"])

        elif self.input_type == "S2GM":
            # Get LR image file paths
            tifs = []
            for root, _, files in os.walk(self.root):
                for f in files:
                    if f.lower().endswith(".tif"):
                        tifs.append(os.path.join(root, f))

            # Hard-map to RGB + NIR
            self.image_meta["lr_file_dict"] = {
                "R":  [p for p in tifs if os.path.basename(p) == "B04.tif"][0],
                "G":  [p for p in tifs if os.path.basename(p) == "B03.tif"][0],
                "B":  [p for p in tifs if os.path.basename(p) == "B02.tif"][0],
                "NIR":[p for p in tifs if os.path.basename(p) == "B08.tif"][0],
            }

            # Get image properties
            with rasterio.open(self.image_meta["lr_file_dict"]["R"]) as src:
                self.image_meta["width"] = src.width
                self.image_meta["height"] = src.height
                self.image_meta["dtype"] = src.dtypes[0]
                self.image_meta["transform"] = src.transform
                self.image_meta["crs"] = src.crs
                self.image_meta["bands"] = len(self.image_meta["lr_file_dict"])
                
        # finally, add image windows to metadata
        self.image_meta["image_windows"] = self.create_image_windows()
        if not self.image_meta["image_windows"]:
            raise ValueError("No image windows were created; check image size, window_size, and overlap.")

    def create_placeholder_file(self, force: bool = False, wait_timeout_s: float = 600.0, wait_sleep_s: float = 0.2):
        """
        Create an empty HR GeoTIFF placeholder for stitched SR results.

        - Output path: `<input_dir>/sr_placeholder.tif`.
        - Only rank 0 creates/writes the file and ensures `temp/` exists.
        - Other ranks wait until the placeholder is readable, then return.
        - Safe on CPU and single-GPU (treated as rank 0).
        """
        import os, time
        import rasterio
        from rasterio.transform import Affine

        def _is_rank0_env() -> bool:
            # Works before torch.distributed is initialized
            lr = os.environ.get("LOCAL_RANK", "")
            r  = os.environ.get("RANK", "")
            ws = os.environ.get("WORLD_SIZE", "1")
            if ws not in ("", "1"):
                return (lr in ("", "0")) and (r in ("", "0"))
            return True  # single process → rank0

        def _placeholder_readable() -> bool:
            """True if placeholder exists and can be opened by rasterio (not truncated)."""
            if not os.path.exists(self.placeholder_filepath):
                return False
            try:
                with rasterio.open(self.placeholder_filepath, "r"):
                    return True
            except Exception:
                return False

        is_r0 = _is_rank0_env()

        if is_r0:
            # Ensure temp dir exists (rank 0 only)
            os.makedirs(self.temp_folder, exist_ok=True)
            self._log(f"📂 Ensured temporary folder: {self.temp_folder}")

            # If exists and not forcing, bail early
            if _placeholder_readable() and not force:
                self._log("⚠️ Placeholder file already exists — skipping creation.")
                return

            # Create (or overwrite) placeholder
            nb = int(self.image_meta["bands"])
            W  = int(self.image_meta["width"]  * self.factor)
            H  = int(self.image_meta["height"] * self.factor)
            tr = self.image_meta["transform"]
            save_transform = Affine(tr.a / self.factor, tr.b, tr.c, tr.d, tr.e / self.factor, tr.f)

            profile = {
                "driver": "GTiff",
                "dtype": self.image_meta["dtype"],
                "count": nb,
                "width": W,
                "height": H,
                "crs": self.image_meta["crs"],
                "transform": save_transform,
                "compress": "none",   # keep uncompressed while many writes happen
                "tiled": True,
                "blockxsize": 512,
                "blockysize": 512,
                "bigtiff": "YES",
                "sparse_ok": True,
                "nodata": 0,
            }

            # Open with "w" to (re)create atomically from the POV of other ranks
            with rasterio.open(self.output_file_path, "w", **profile):
                pass

            self._log(f"💾 Saved empty placeholder SR image at: {self.output_file_path}")

        else:
            # Non-rank0: wait until rank0 created a usable file and temp/
            t0 = time.time()
            while True:
                if os.path.isdir(self.temp_folder) and _placeholder_readable():
                    break
                if time.time() - t0 > wait_timeout_s:
                    raise TimeoutError(
                        f"Timed out waiting for placeholder at {self.placeholder_filepath} "
                        f"and temp dir {self.temp_folder} created by rank 0."
                    )
                time.sleep(wait_sleep_s)
            # No logging here (avoid concurrent writes); rank 0 logs already.

    def create_image_windows(self): # Works
        """
        Generate overlapping rasterio Windows covering the input image.

        - Windows are defined at LR scale with specified size and overlap.
        - Extra windows are added at image borders to ensure full coverage.
        - Returns a list of rasterio.windows.Window objects.

        Returns
        -------
        list of rasterio.windows.Window
            Coordinates of all LR patches to process.
        """
        # get amount of overlap
        overlap = self.overlap
        if self.image_meta["width"] < self.window_size[0] or self.image_meta["height"] < self.window_size[1]:
            raise ValueError(
                f"Input image ({self.image_meta['width']}x{self.image_meta['height']}) "
                f"is smaller than window_size {self.window_size}."
            )
        # Calculate the number of windows in each dimension
        n_windows_x = (self.image_meta["width"] - overlap) // (self.window_size[0] - overlap)
        n_windows_y = (self.image_meta["height"] - overlap) // (self.window_size[1] - overlap)
        # Create list of batch windows coordinates
        window_coordinates = []
        for win_y in range(n_windows_y):
            for win_x in range(n_windows_x):
                # Define window to read with overlap
                window = rasterio.windows.Window(
                    win_x * (self.window_size[0] - overlap),
                    win_y * (self.window_size[1] - overlap),
                    self.window_size[0],
                    self.window_size[1]
                )
                window_coordinates.append(window)

        # Check for any remaining space after the sliding window approach
        final_x = self.image_meta["width"] - self.window_size[0]
        final_y = self.image_meta["height"] - self.window_size[1]

        # Add extra windows for the edges if there's remaining space
        # Adjust the check to handle the overlap correctly
        if final_x % (self.window_size[0] - overlap) > 0:
            for win_y in range(n_windows_y):
                window = rasterio.windows.Window(
                    self.image_meta["width"] - self.window_size[0],
                    win_y * (self.window_size[1] - overlap),
                    self.window_size[0],
                    self.window_size[1]
                )
                window_coordinates.append(window)

        if final_y % (self.window_size[1] - overlap) > 0:
            for win_x in range(n_windows_x):
                window = rasterio.windows.Window(
                    win_x * (self.window_size[0] - overlap),
                    self.image_meta["height"] - self.window_size[1],
                    self.window_size[0],
                    self.window_size[1]
                )
                window_coordinates.append(window)

        # Handle corner case if both x and y have remaining space
        if (final_x % (self.window_size[0] - overlap) > 0 and
                final_y % (self.window_size[1] - overlap) > 0):
            window = rasterio.windows.Window(
                self.image_meta["width"] - self.window_size[0],
                self.image_meta["height"] - self.window_size[1],
                self.window_size[0],
                self.window_size[1]
            )
            window_coordinates.append(window)

        # Return filled list of coordinates
        return window_coordinates
    
    def delete_LR_temp(self):
        """
        Delete the temporary folder created for LR patches and intermediate files.

        Cleans up after prediction and stitching.
        """
        temp_folder = os.path.abspath(self.temp_folder)
        if not os.path.basename(temp_folder).startswith("temp_"):
            raise RuntimeError(f"Refusing to delete unexpected temp folder: {temp_folder}")
        if os.path.isdir(temp_folder):
            shutil.rmtree(temp_folder)
            self._log(f"🗑️📂 Deleted temporary folder at: {temp_folder}")

    def run(self, debug=None, save_preview=None, cleanup=None, overwrite=None):
        """
        Execute inference, stitch the patches, and optionally write preview outputs.

        Construction prepares the object and validates inputs; this method performs
        the expensive and destructive work.
        """
        debug = self.debug if debug is None else bool(debug)
        save_preview = self.save_preview if save_preview is None else bool(save_preview)
        cleanup = self.cleanup if cleanup is None else bool(cleanup)
        overwrite = self.overwrite if overwrite is None else bool(overwrite)

        if os.path.exists(self.final_sr_path) and not overwrite:
            raise FileExistsError(
                f"Output already exists at {self.final_sr_path}. "
                "Pass overwrite=True to replace it."
            )

        # Create a fresh placeholder for this run. This avoids stale pixels from
        # failed or repeated runs contaminating the final mosaic.
        self.create_placeholder_file(force=True)
        self.start_super_resolution(debug=debug, save_preview=save_preview)

        trainer = getattr(self, "trainer", None)
        if self._is_rank0(trainer):
            self._log("🪡 Stitching results into final sr.tif...")
            self.write_to_file(overwrite=overwrite)

            if cleanup:
                self.delete_LR_temp()

            if save_preview:
                from opensr_utils.data_utils.result_analysis import (
                    crop_and_save_georeferenced_excerpt,
                    generate_side_by_side_previews,
                )
                tif_path = self.final_sr_path
                crop_and_save_georeferenced_excerpt(
                    self,
                    tif_path=tif_path,
                    out_tif=os.path.join(self.log_dir, "cropped_excerpt.tif"),
                    random_crop_size=(512, 512),
                )
                self._log("✅ Saved an example SR patch to cropped_excerpt.tif in logs folder.")
                num_examples = 10
                generate_side_by_side_previews(
                    self,
                    tif_path=tif_path,
                    out_dir=self.log_dir,
                    num_examples=num_examples,
                )
                self._log(f"✅ Saved {num_examples} side-by-side preview images to logs folder.")

        self._log("🎉 Processing done! SR process exited.")
        return self.final_sr_path

    def start_super_resolution(self, debug: bool = False, save_preview: bool = False):
        """
        Run super-resolution inference across all LR windows.

        - Creates a fresh PredictionDataModule for the selected windows.
        - Passes temp folder and window metadata to the model.
        - Uses a PyTorch Lightning Trainer to run `predict_step` in inference mode.
        - Saves SR patches into `temp/` as `.npy`/`.npz`.
        """
        import inspect
        import torch
        from pytorch_lightning import Trainer

        # Pick windows (debug trims to first 100 globally; DDP sampler will shard those across ranks)
        self._log("🚀 Running SR...")
        windows_all = self.image_meta["image_windows"]
        windows_run = windows_all[:100] if debug else windows_all
        self.create_datamodule(windows=windows_run)
        self.model.model_dm_length = len(self.datamodule.dataset)

        # Hand the hook context to the model (consumed inside predict_step)
        self.model._save_temp_folder = self.temp_folder
        self.model._save_windows     = windows_run
        self.model._save_factor      = int(self.factor)
        self.model._save_compressed  = bool(self.compressed_patches)
        self.model._save_progress_preview = bool(save_preview)

        # Devices: list[int] or 1 for CPU/single-GPU
        devices = self.gpus if self.gpus is not None else 1

        # Build Trainer kwargs robustly across PL versions
        trainer_kwargs = dict(
            accelerator=self.device,        # "cpu" or "cuda"
            devices=devices,                # int or list of GPU ids
            logger=False,
            enable_checkpointing=False,
            enable_model_summary=False,
            inference_mode=True,            # no_grad + eval
            enable_progress_bar=True,      # optional: cleaner logs
        )

        # Use DDP only when we truly have >1 CUDA devices
        if isinstance(devices, (list, tuple)) and len(devices) > 1 and self.device == "cuda":
            trainer_kwargs["strategy"] = "ddp"

        # Make sure PL doesn't override our custom sampler
        sig = inspect.signature(Trainer)
        if "replace_sampler_ddp" in sig.parameters:          # PL 1.x
            trainer_kwargs["replace_sampler_ddp"] = False
        if "use_distributed_sampler" in sig.parameters:      # PL 2.x
            trainer_kwargs["use_distributed_sampler"] = False

        trainer = Trainer(**trainer_kwargs)
        self.trainer = trainer  # let _log() and later code see rank via trainer.is_global_zero

        # Stream predictions to disk inside predict_step (we return None there)
        try:
            trainer.predict(self.model, datamodule=self.datamodule, return_predictions=False)
        except RuntimeError as e:
            msg = str(e)
            if "Lightning can't create new processes if CUDA is already initialized" in msg:
                # Stop the workflow cleanly (no noisy traceback for users)
                self._log(
                    "🚨🔥 STOPPING WORKFLOW 🔥🚨\n"
                    "⚠️  CUDA was already initialized before launching DDP!\n"
                    "🖥️  This is a multi-GPU processing limitation of PyTorch Lightning.\n"
                    "🤔  This can happen if you pass 'None' as model with gpus>1.\n"
                    "🔄  Please restart the Python kernel or switch to single-GPU (devices=1).\n"
                    "✅  After restarting, re-run your workflow."
                )
                raise SystemExit(3)
            raise  # unrelated error: re-raise

        # Rank-aware status message
        is_ddp = torch.distributed.is_available() and torch.distributed.is_initialized()
        rank = torch.distributed.get_rank() if is_ddp else 0
        if rank == 0:
            self._log(f"✅ Prediction complete! SR patches saved in 📂: {self.temp_folder}")
            if debug:
                self._log("Debug mode was ON → processed only 100 windows.")

    def write_to_file(self, index_path=None, limit=None, overwrite=None):
        """
        Multi-GPU safe entry point for stitching super-resolved patches.

        This method wraps around ``writing_utils.ddp_safe_stitch`` to ensure that
        the expensive GeoTIFF write phase is performed only once (on global rank 0),
        while all other ranks in a distributed run wait at a synchronization barrier.
        This prevents file corruption and crashes when using PyTorch DDP.

        Behavior by environment:
        ------------------------
        - **CPU / single process**: stitches normally.
        - **Single GPU**: stitches normally (rank 0 is the only rank).
        - **Multi-GPU (DDP)**: only rank 0 writes to the GeoTIFF, all other ranks
        return immediately after the barrier.

        Parameters
        ----------
        index_path : str, optional
            Path to ``index.json`` describing all patches. Defaults to
            ``<temp_folder>/index.json``.
        limit : int, optional
            If given, only the first ``limit`` patches are stitched
            (useful for debugging).

        Notes
        -----
        - Actual stitching and file I/O are handled by
        :func:`opensr_utils.data_utils.writing_utils.ddp_safe_stitch`.
        - Ensure that the placeholder GeoTIFF has already been created before
        calling this function.
        - Environment variables such as ``GDAL_CACHEMAX=256`` and
        ``GDAL_NUM_THREADS=1`` are recommended for stability when running with
        multiple processes.
        """
        from opensr_utils.data_utils.writing_utils import ddp_safe_stitch
        if overwrite is not None:
            self.overwrite = bool(overwrite)
        return ddp_safe_stitch(
            self,
            index_path=index_path,
            limit=limit,
            cleanup_every=50,     # tune if RAM pressure
            profile="sigmoid",     # "linear"|"sigmoid"|"cosine"
            profile_alpha=6.0,
            gdal_cache_mb=256,
        )
    
    def _is_rank0(self, trainer=None):
        """Return True only for the process that should print (rank 0)."""
        import os, torch

        # When Trainer exists, trust it
        if trainer is not None and hasattr(trainer, "is_global_zero"):
            return bool(trainer.is_global_zero)

        # If torch.distributed already initialized, trust it
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            try:
                return torch.distributed.get_rank() == 0
            except Exception:
                pass

        # Pre-init: rely on env set by the launcher
        r  = os.environ.get("RANK", "")
        lr = os.environ.get("LOCAL_RANK", "")
        ws = os.environ.get("WORLD_SIZE", "")
        if ws not in ("", "1"):  # multi-process expected
            return (r in ("", "0")) and (lr in ("", "0"))

        # Single process
        return True

    def _log(self, message: str):
        """
        Rank-0 logging with print + file write and backlog support.

        Behavior
        --------
        - Always prints the message to stdout if called on rank 0.
        - Buffers messages in-memory (backlog) until `self.log_file` is set.
        - Once `self.log_file` exists (created in `create_dirs()`), all backlog 
        messages are flushed and new entries are appended to the log file.
        - If writing fails (e.g. permissions), the message is preserved in backlog.

        Notes
        -----
        - Ensures no logs are lost, even if logging starts before the log file exists.
        - Timestamp is automatically prepended to every message.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] {message}"

        # always print on rank 0
        if self._is_rank0(getattr(self, "trainer", None)):
            print(line)

        # ensure backlog buffer exists
        if not hasattr(self, "_log_backlog"):
            self._log_backlog = []

        # if log_file not ready yet → backlog
        if not hasattr(self, "log_file") or not self.log_file or not os.path.exists(os.path.dirname(self.log_file)):
            self._log_backlog.append(line)
            return

        # if log_file ready → flush backlog first
        try:
            with open(self.log_file, "a") as f:
                if self._log_backlog:
                    for bl_line in self._log_backlog:
                        f.write(bl_line + "\n")
                    self._log_backlog.clear()
                f.write(line + "\n")
        except Exception as e:
            # in case file is not writable, keep line in backlog
            self._log_backlog.append(line)
