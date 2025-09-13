# -*- coding: utf-8 -*-


# torch stuff
import torch
from pytorch_lightning import LightningModule,Trainer
torch.set_float32_matmul_precision('medium')

# general imports
from einops import rearrange
from tqdm import tqdm
import numpy as np
import random
import json
import os
import glob
import shutil

# Geo
import rasterio
from rasterio.transform import Affine
from rasterio.windows import Window
from rasterio.shutil import copy as rio_copy


# local imports
from opensr_utils.data_utils.writing_utils import write_to_placeholder as blend_write
from opensr_utils.data_utils.datamodule import PredictionDataModule
from opensr_utils.model_utils.prepare_model import preprocess_model
from opensr_utils.data_utils.reading_utils import can_read_directly_with_rasterio


class large_file_processing():
    
    def __init__(self,
                 root: str,
                 model=None,
                 window_size:tuple=(128, 128),
                 factor:int=4,
                 overlap:int=8,
                 eliminate_border_px=0,
                 device:str="cpu",
                 gpus: int = 1 or list,
                 debug=False,
                 ):

        """
        High-level pipeline for patch-based super-resolution (SR) on large geospatial
        raster files.

        This class handles the entire flow of:
        1. Verifying input type (single file, Sentinel-2 SAFE, or S2GM folder).
        2. Extracting metadata (dimensions, CRS, transform, band count).
        3. Generating a list of sliding LR windows with overlap.
        4. Creating an empty placeholder GeoTIFF at HR resolution.
        5. Building a PyTorch Lightning datamodule for inference.
        6. Running SR model predictions on windows, streamed to disk as `.npy`/`.npz`.
        7. Stitching predictions back into the placeholder with overlap-aware blending.
        8. Cleaning up temporary files and writing final `sr.tif`.

        The design enables processing of arbitrarily large rasters without exhausting
        system memory, by relying on tiling, temporary storage, and rasterio’s
        windowed I/O.

        Parameters
        ----------
        root : str
            Path to the input data. Can be:
            - A single raster file (readable by rasterio).
            - A Sentinel-2 SAFE folder (expects JP2 bands B02, B03, B04, B08).
            - A Sentinel-2 Global Mosaic (S2GM) folder with TIFF bands.
        model : torch.nn.Module or LightningModule, optional
            Super-resolution model to run. If None, only preprocessing/placeholder
            setup is performed.
        window_size : tuple of int, default=(128, 128)
            Size of each LR patch to read from disk.
        factor : int, default=4
            Upscaling factor for SR model. Must be one of {2,4,6,8}.
        overlap : int, default=8
            Amount of overlap (in LR pixels) between adjacent windows.
        eliminate_border_px : int, default=0
            Number of pixels at each SR patch border to suppress before feathering.
            Must be smaller than overlap.
        device : {"cpu","cuda"}, default="cpu"
            Device for model inference.
        gpus : int or list of int, default=1
            GPU configuration for PyTorch Lightning Trainer. Accepts integer count
            or explicit device IDs.

        Attributes
        ----------
        root : str
            Path to input file/folder.
        window_size : tuple
            LR patch dimensions.
        factor : int
            Upscaling factor.
        overlap : int
            Overlap size in pixels.
        eliminate_border_px : int
            Pixels to suppress at SR patch edges.
        device : str
            Inference device.
        gpus : list of int
            List of GPU IDs used for Trainer.
        image_meta : dict
            Metadata dictionary with keys:
            - width, height, dtype, crs, transform, bands
            - image_windows : list of rasterio.windows.Window
            - placeholder_filepath : str
            - placeholder_dir : str
        input_type : {"file","SAFE","S2GM"}
            Type of input data detected.
        placeholder_filepath : str
            Path to the temporary placeholder GeoTIFF for SR results.
        temp_folder : str
            Temporary folder for per-patch `.npy`/`.npz` predictions.
        datamodule : PredictionDataModule
            Lightning datamodule managing data loading and batching.
        model : torch.nn.Module
            Preprocessed SR model set to evaluation mode.

        Methods
        -------
        create_datamodule()
            Build the PredictionDataModule for input windows.
        verify_input_file_type(root)
            Identify input as raster file, SAFE folder, or S2GM folder.
        get_image_meta(root_dir)
            Extract metadata (dimensions, dtype, CRS, bands) from input.
        create_placeholder_file()
            Initialize empty HR GeoTIFF placeholder for stitched results.
        create_image_windows()
            Generate overlapping rasterio Windows covering the input image.
        delete_LR_temp()
            Delete temporary LR patch folder.
        start_super_resolution(debug=False)
            Run inference on all LR windows and save SR patches to temp folder.
        write_to_file(index_path=None, limit=None)
            Stitch saved SR patches from temp folder into placeholder GeoTIFF,
            blend overlaps, clean up, and finalize `sr.tif`.

        Notes
        -----
        - Overlap-aware blending uses `opensr_utils.data_utils.writing_utils.write_to_placeholder`.
        - The pipeline is designed for very large Sentinel-2 mosaics and similar
        datasets where in-memory inference is not feasible.
        - Temporary predictions are streamed to disk, avoiding large GPU <-> CPU transfers.
        - Placeholder is written with ZSTD compression and BigTIFF enabled for large files.

        Example
        -------
        >>> o = large_file_processing(
        ...     root="/data/mosaic/S2_tile.tif",
        ...     model=my_model,
        ...     window_size=(128, 128),
        ...     factor=4,
        ...     overlap=12,
        ...     eliminate_border_px=2,
        ...     device="cuda",
        ...     gpus=[0]
        ... )
        >>> o.start_super_resolution()
        >>> o.write_to_file()
        Saved stitched SR image at: /data/mosaic/sr.tif
        """        
        
        # First, do some asserts to make sure input is valid
        assert os.path.exists(root), "Input folder/file path does not exist"
        assert model==None or isinstance(model, LightningModule) or isinstance(model, torch.nn.Module), "Model must be a PyTorch, Lightning, or 'None'."
        assert eliminate_border_px < overlap, "eliminate_border_px must be smaller than overlap, since it is a subset of the overlap area."
        assert overlap % 2 == 0, "Overlap must be an even integer."
        assert eliminate_border_px % 2 == 0, "eliminate_border_px must be an even integer."
        assert overlap-eliminate_border_px >= 4, "overlap must be at least 4 pixels bigger than eliminate_border_px, otherwise there is no point in doing it."
        assert factor in [2,4,6,8], "Factor must be one of [2,4,6,8]."

        # General Settings
        self.debug = debug # if True, only process 100 windows globally (DDP-safe)
        self.root = root # path to folder containing S2 SAFE data format
        self.window_size = window_size # window size of the LR image
        self.factor = factor # sr factor of the model
        self.overlap = overlap # overlap in px of the LR image windows
        self.eliminate_border_px = eliminate_border_px # border pixels to eliminate in px on each side of the SR image
        self.device = device # device to run model on
        if type(gpus) == int: # pass GPU/s to Trainer as list
            self.gpus = [gpus] # number of gpus or list of gpu ids to use
        else:
            self.gpus = gpus
        if self.device == "cpu":
            self.gpus = None
            
        # ---
        # Local variable definitions
        self.image_meta = {} # dict to hold image metadata - gets filled by get_image_meta
        self.input_type = None # file, SAFE or S2GM - gets filled by verify_input_file_type
        self.placeholder_filepath = None # filepath of empty SR placeholder - gets filled by create_placeholder_file
        self.placeholder_dir = None # directory of empty SR placeholder - gets filled by create_placeholder_file
        self.temp_folder = None # folder of temporary files - gets filled by create_placeholder_file

        # Verifying type of input: file, SAFE or S2GM folder
        self.verify_input_file_type(self.root)
        
        # Get Image information based on input, including image coordinate windows
        self.get_image_meta(self.root)

        # Create LR placeholder file
        self.create_placeholder_file()
        
        # Create Datamodule based on input files and Windows
        self.create_datamodule()
        
        # Make sure model is useable and in eval mode
        self.model = preprocess_model(model,
                                      temp_folder=self.temp_folder,
                                      windows=self.image_meta["image_windows"],
                                      factor=self.factor)

        # Print Status
        print("\n")
        print("📊 Status: Model and Data ready for inference ✅")
        
        # If in standard mode, run everything right away
        if self.debug!=True:
            print("🚀 Runing SR...")
            self.start_super_resolution(debug=self.debug)    
            
        trainer = getattr(self, "trainer", None)
        if self._is_rank0(trainer):
            print(f"✅ Prediction complete! SR patches saved in 📂: {self.temp_folder}")
            self.write_to_file()     # calls ddp_safe_stitch()
            self.delete_LR_temp()
        else:
            # Non-rank0 processes: don't print, don't stitch, don't delete
            pass


    def _is_rank0(self,trainer=None):
        """Return True only on the global rank 0 process."""
        # Prefer Lightning's flag (correct across DDP/TPU/etc.)
        if trainer is not None and hasattr(trainer, "is_global_zero"):
            return bool(trainer.is_global_zero)
        # Fallback to torch.distributed if initialized
        try:
            import torch
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                return torch.distributed.get_rank() == 0
        except Exception:
            pass
        # Single process
        return True
    
    def create_datamodule(self):
        """
        Create a PredictionDataModule for patch-based inference.

        The datamodule wraps the input type, root path, and sliding windows,
        and handles batching for PyTorch Lightning’s predict loop.
        """
        dm = PredictionDataModule(input_type=self.input_type,
                              root=self.root,
                              windows = self.image_meta["image_windows"],
                              prefetch_factor=2, batch_size=16, num_workers=4)
        dm.setup()
        self.datamodule = dm

    def verify_input_file_type(self, root):
        """
        Determine whether input is a single raster file, a SAFE folder, or an S2GM folder.

        - For files: ensures rasterio can open it.
        - For folders: checks SAFE or S2GM conventions.
        - Sets `self.input_type` accordingly.
        - Updates placeholder path in metadata.

        Raises
        ------
        NotImplementedError
            If the input path is not a valid file or recognized folder type.
        """
        # Check if root is a file or a folder
        if os.path.isfile(root):
            input_type = "file"
        elif os.path.isdir(root):
            input_type = "folder"
        else:
            raise NotImplementedError(
                "🚫 Input path is neither a 📄 file nor a 📁 folder. 👉 Please provide a valid input path."
            )
        
        # Verifying type of input: file, SAFE or S2GM folder
        if input_type=="file":
            self.input_type = "file"
            self.placeholder_path = os.path.dirname(self.root)
            self.image_meta["placeholder_path"] = self.placeholder_path
            if can_read_directly_with_rasterio(self.root) == False:
                raise NotImplementedError(
                    "🚫 Input is a file, but this file type cannot be opened by 'rasterio' ❌📂"
                )
            else:
                print("📄 Input is a file, can be opened with rasterio — processing possible! 🚀")
        elif input_type=="folder":
            self.placeholder_path = self.root
            if self.root.replace("/","")[-5:] == ".SAFE":
                print("📁 Input is Sentinel-2 .SAFE folder, processing possible! 🚀")
                self.input_type = "SAFE"
            elif "S2GM" in self.root:
                print("📁 Input is Sentinel-2 S2GM folder, processing possible! 🚀")
                self.input_type = "S2GM"
            else:
                raise NotImplementedError("🚫 Input folder is not in .SAFE format or S2GM format. 👉 Please provide a valid input folder.")
        
    def get_image_meta(self, root_dir):
        """
        Extract raster metadata (width, height, dtype, CRS, transform, band count).

        - For files: opens directly with rasterio.
        - For SAFE: locates JP2 bands B02, B03, B04, B08.
        - For S2GM: locates GeoTIFF bands B02, B03, B04, B08.

        Adds a list of sliding image windows to `self.image_meta`.
        """
        if self.input_type == "file":
            with rasterio.open(root_dir) as src:
                self.image_meta["width"], self.image_meta["height"], self.image_meta["dtype"] = src.width, src.height, src.dtypes[0]
                self.image_meta["transform"] = src.transform
                self.image_meta["crs"] = src.crs
                self.image_meta["bands"] = src.count

        elif self.input_type == "SAFE":
                files_ls = []   
                for root, dirs, files in os.walk(root_dir):
                    for file in files:
                        if file.endswith('.jp2'):
                            if any(band in file for band in ["B04","B03","B02","B08",]):
                                full_path = os.path.join(root, file)
                                if "IMG_DATA" in full_path:
                                    files_ls.append(full_path)
                image_files = {"R":[file for file in files_ls if "B04" in file][0],
                            "G":[file for file in files_ls if "B03" in file][0],
                            "B":[file for file in files_ls if "B02" in file][0],
                            "NIR":[file for file in files_ls if "B08" in file][0]}
                with rasterio.open(image_files["R"]) as src:
                    self.image_meta["width"], self.image_meta["height"], self.image_meta["dtype"] = src.width, src.height, src.dtypes[0]
                    self.image_meta["transform"] = src.transform
                    self.image_meta["crs"] = src.crs
                    self.image_meta["bands"] = len(image_files)
        
        elif self.input_type == "S2GM":
            band_names = ["B04.tif","B03.tif","B02.tif","B08.tif",]
            tif_files = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.lower().endswith(".tif")]
            tif_files = [f for f in tif_files if os.path.basename(f) in band_names]
            with rasterio.open(tif_files[0]) as src:
                self.image_meta["width"] = src.width
                self.image_meta["height"] = src.height
                self.image_meta["dtype"] = src.dtypes[0]
                self.image_meta["transform"] = src.transform
                self.image_meta["crs"] = src.crs
                self.image_meta["bands"] = len(tif_files)
                
        # finally, add image windows to metadata
        self.image_meta["image_windows"] = self.create_image_windows()
        
    def create_placeholder_file(self,force=False):
        """
        Create an empty HR GeoTIFF placeholder for stitched SR results.

        - Output path: `<input_dir>/sr_placeholder.tif`.
        - Uses ZSTD compression, tiling, and BigTIFF for large file safety.
        - Applies appropriate transform scaling (factor).
        - Creates a `temp/` folder alongside placeholder for intermediate patches.

        Skips creation if placeholder already exists.
        """
        # 1. Create placeholder file path variable
        out_name = "sr_placeholder.tif"
        output_file_path = os.path.join(self.placeholder_path, out_name)
        # also set placeholder path in metadata
        self.image_meta["placeholder_filepath"] = output_file_path
        self.image_meta["placeholder_dir"] = self.placeholder_dir
        self.placeholder_filepath = output_file_path
        self.placeholder_dir = self.placeholder_dir
        
        # 2. Creating temporary folder for storing batches during prediction
        temp_folder_path = os.path.join(os.path.dirname(output_file_path),"temp")
        self.temp_folder = temp_folder_path
        
        # 3. Create placeholder file if it does not exist yet, otherwise skip
        if os.path.exists(output_file_path):
            print("⚠️ Placeholder already exists: ",temp_folder_path)
        
        # 4. If it does not exist, create it
        else:
            os.makedirs(temp_folder_path, exist_ok=True)
            print(f"📂 Created temporary folder at: {output_file_path}")
            
        # If sr_placeholder file exists and force is False, skip creation
        if os.path.exists(self.placeholder_filepath) and force==False:
            print(f"⚠️ Placeholder file already exists — skipping creation.")
        else:  
            nb = int(self.image_meta["bands"])
            W  = int(self.image_meta["width"]  * self.factor)
            H  = int(self.image_meta["height"] * self.factor)
            tr = self.image_meta["transform"]
            save_transform = Affine(tr.a / self.factor, tr.b, tr.c, tr.d, tr.e / self.factor, tr.f)

            profile = {  # robust placeholder for stitching (no compression!)
                "driver": "GTiff",
                "dtype": self.image_meta["dtype"],
                "count": nb,
                "width": W,
                "height": H,
                "crs": self.image_meta["crs"],
                "transform": save_transform,

                # 🔒 Important: keep uncompressed while many writes happen
                "compress": "none",        # <-- no ZSTD/DEFLATE during stitching
                # remove "zlevel" and "predictor" when uncompressed

                "tiled": True,
                "blockxsize": 512,
                "blockysize": 512,

                # Huge rasters: don't risk "IF_SAFER" surprises—just use BIGTIFF
                "bigtiff": "YES",

                # Safe to keep; helps GDAL skip empty tiles
                "sparse_ok": True,
            }
            # Create the dataset and close without writing any pixels
            with rasterio.open(output_file_path, "w", **profile):
                pass
            print(f"💾 Saved empty placeholder SR image at: {output_file_path}")


        
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
    
    def delete_LR_temp(self): # Works for now, ToDo: Check
        """
        Delete the temporary folder created for LR patches and intermediate files.

        Cleans up after prediction and stitching.
        """
        shutil.rmtree(self.temp_folder)
        print("🗑️📂 Deleted temporary folder at:",self.temp_folder)

    def start_super_resolution(self, debug: bool = False):
        """
        Run super-resolution inference across all LR windows.

        - Creates a fresh PredictionDataModule for the selected windows.
        - Passes temp folder and window metadata to the model.
        - Uses a PyTorch Lightning Trainer to run `predict_step` in inference mode.
        - Saves SR patches into `temp/` as `.npy`/`.npz`.

        Parameters
        ----------
        debug : bool, default=False
            If True, process only the first 100 windows globally (DDP-safe).

        Notes
        -----
        - Multi-GPU support is handled via Lightning DDP.
        - No checkpoints or logs are created during prediction.
        """
        # Pick windows (debug trims to first 100 globally; DDP sampler will shard those across ranks)
        windows_all = self.image_meta["image_windows"]
        windows_run = windows_all[:100] if debug else windows_all

        # Hand the hook context to the model (these are consumed inside predict_step)
        self.model._save_temp_folder = self.temp_folder
        self.model._save_windows     = windows_run
        self.model._save_factor      = int(self.factor)

        devices = self.gpus if self.gpus is not None else 1
        # Trainer config: no logging/checkpointing; DDP handled automatically if multiple GPUs
        trainer = Trainer(
            accelerator=self.device,
            devices=devices,
            logger=False,
            enable_checkpointing=False,
            enable_model_summary=False,
            inference_mode=True,           # no_grad + eval
            # precision="16-mixed",        # uncomment if your model supports AMP
        )

        # Stream predictions to disk inside predict_step (we return None there to avoid gathers)
        try:
            trainer.predict(self.model, datamodule=self.datamodule, return_predictions=False)    
        except RuntimeError as e:
            msg = str(e)
            if "Lightning can't create new processes if CUDA is already initialized" in msg:
                # Stop the workflow cleanly (no noisy traceback for users)
                print(
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
            print(f"✅ Prediction complete! SR patches saved in 📂: {self.temp_folder}")
            if debug:
                print("Debug mode was ON → processed only 100 windows.")

    def write_to_file(self, index_path=None, limit=None):
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

        # Optional: set conservative GDAL threading via env in launcher
        # (export GDAL_CACHEMAX=256; export GDAL_NUM_THREADS=1; etc.)

        ddp_safe_stitch(
            self,
            index_path=index_path,
            limit=limit,
            cleanup_every=50,     # tune if RAM pressure
            profile="sigmoid",     # "linear"|"sigmoid"|"cosine"
            profile_alpha=6.0,
            gdal_cache_mb=256,
        )

    


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="🚀 Patch-based super-resolution for large geospatial rasters"
    )
    parser.add_argument("root", type=str, help="📂 Path to input file/folder")
    
    parser.add_argument("model", type=str, choices=["LDSRS2", "None"],
                        help="🤖 Model to run: 'LDSRS2' or 'None'")
    
    parser.add_argument("--window_size", type=int, nargs=2, default=(128, 128),
                        help="🔲 LR window size (default: 128 128)")
    parser.add_argument("--factor", type=int, default=4, choices=[2, 4, 6, 8],
                        help="⬆️ Upscaling factor (default: 4)")
    parser.add_argument("--overlap", type=int, default=8,
                        help="🤝 Overlap in LR pixels (default: 8)")
    parser.add_argument("--eliminate_border_px", type=int, default=0,
                        help="✂️ Pixels to eliminate at patch borders (default: 0)")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"],
                        help="⚡ Device for inference (default: cpu)")
    parser.add_argument("--gpus", type=int, nargs="+", default=[0],
                        help="💻 GPU IDs to use (default: 0)")
    parser.add_argument("--debug", action="store_true",
                        help="🐞 Debug mode: process only ~100 windows")

    args = parser.parse_args()
    

    # Resolve model argument
    if args.model == "LDSRS2":
        print("Using LDSR-S2 model.")
        from io import StringIO
        from omegaconf import OmegaConf
        import requests, opensr_model
        config_url = "https://raw.githubusercontent.com/ESAOpenSR/opensr-model/a5474a07258d632e09236a58db34fa8640678c22/opensr_model/configs/config_10m.yaml"
        response = requests.get(config_url)
        config = OmegaConf.load(StringIO(response.text))
        model = opensr_model.SRLatentDiffusion(config, device=args.device) # create model
        model.load_pretrained(config.ckpt_version)
    elif args.model == "None":
        print("Using placeholder (interpolation) mode.")
        model = None
    else:
        print("⚠️ From CLI, you can only run the LDSR-S2 model. Using placeholder (interpolation) mode.")
        model = None

    # Create processing object
    processor = large_file_processing(
        root=args.root,
        window_size=tuple(args.window_size),
        factor=args.factor,
        overlap=args.overlap,
        eliminate_border_px=args.eliminate_border_px,
        device=args.device,
        gpus=args.gpus,
        debug=args.debug,
    )
    
if __name__ == "__main__":
    main()
    
    
    
    # Other stuff
    """
    # open file and look at all gdal settings
    import rasterio
    with rasterio.open("/data2/simon/mosaic/Q10_20240701_20240930_Global-S2GM-m36p8_STD_v2.0.5/S2GM_Q10_20240701_20240930_Global-S2GM-m36p8_STD_v2.0.5/tile_0/sr.tif") as src:
        print(src.profile)
        print(src.tags())
        print(src.tags(ns="IMAGE_STRUCTURE"))
        print(src.tags(ns="TIFF"))
        print(src.tags(ns="ZSTD"))
        print(src.compression)
        print(src.compression.name) 
        print(src.compression.value)
        
    import rasterio
    with rasterio.open("/data2/simon/mosaic/individual_tile/sr_placeholder.tif") as src:
        print(src.profile)
        print(src.tags())
        print(src.tags(ns="IMAGE_STRUCTURE"))
        print(src.tags(ns="TIFF"))
        print(src.tags(ns="ZSTD"))
        print(src.compression)
        print(src.compression.name) 
        print(src.compression.value)
        
    import rasterio
    with rasterio.open("/data2/simon/mosaic/individual_tile/S2_BA_smaller.tif") as src:
        print(src.profile)
        print(src.tags())
        print(src.tags(ns="IMAGE_STRUCTURE"))
        print(src.tags(ns="TIFF"))
        print(src.tags(ns="ZSTD"))
        print(src.compression)
        
    """