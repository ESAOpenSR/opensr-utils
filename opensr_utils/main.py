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


# local imports
from opensr_utils.data_utils.writing_utils import write_to_placeholder as blend_write
from opensr_utils.data_utils.datamodule import PredictionDataModule
from opensr_utils.model_utils.prepare_model import preprocess_model
from opensr_utils.data_utils.reading_utils import can_read_directly_with_rasterio
#from opensr_utils.denormalize_image_per_band_batch import denormalize_image_per_band_batch as denorm
#from opensr_utils.stretching import hq_histogram_matching
#from opensr_utils.bands10m_stacked_from_S2_folder import extract_10mbands_from_S2_folder
#from opensr_utils.bands20m_stacked_from_S2_folder import extract_20mbands_from_S2_folder
#from opensr_utils.weighted_overlap import weighted_overlap
#from opensr_utils.utils import SuppressPrint


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
                 ):
        
        # First, do some asserts to make sure input is valid
        assert os.path.exists(root), "Input folder/file path does not exist"
        assert model==None or isinstance(model, LightningModule), "Model must be a PyTorch Lightning Module or None"

        # General Settings
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
        # ---
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

        print("\n")
        print("Status: Model and Data ready for inference.")
        print("Run SR with .start_super_resolution() method.")

    def create_datamodule(self): # Works
        dm = PredictionDataModule(input_type=self.input_type,
                              root=self.root,
                              windows = self.image_meta["image_windows"])
        dm.setup()
        self.datamodule = dm

    def verify_input_file_type(self, root): # Works
        # Check if root is a file or a folder
        if os.path.isfile(root):
            input_type = "file"
        elif os.path.isdir(root):
            input_type = "folder"
        else:
            raise NotImplementedError("Input path is neither a file nor a folder. Please provide a valid input path.")
        # Verifying type of input: file, SAFE or S2GM folder
        if input_type=="file":
            self.placeholder_path = os.path.dirname(self.root)
            self.image_meta["placeholder_path"] = self.placeholder_path
            if can_read_directly_with_rasterio(self.root) == False:
                raise NotImplementedError("Input is a file. File type can not be opened by 'rasterio'.")
            else:
                print("Input is a file, can be opened with rasterio, processing possible.")
        elif input_type=="folder":
            self.placeholder_path = self.root
            if self.root.replace("/","")[-5:] == ".SAFE":
                print("Input is Sentinel-2 .SAFE folder, processing possible.")
                self.input_type = "SAFE"
            elif "S2GM" in self.root:
                print("Input is Sentinel-2 S2GM folder, processing possible.")
                self.input_type = "S2GM"
            else:
                raise NotImplementedError("Input folder is not in .SAFE format or S2GM format. Please provide a valid input folder.")
        self.input_type = input_type
        
    def get_image_meta(self, root_dir): # Works
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
        
    def create_placeholder_file(self): # Works - tmp file unconfirmed
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
        os.makedirs(temp_folder_path, exist_ok=True)
        self.temp_folder = temp_folder_path
        print("Created temporary folder at:",temp_folder_path)
        
        # 3. Create placeholder file if it does not exist yet, otherwise skip
        if os.path.exists(output_file_path):
            print(f"Placeholder already exists: {output_file_path}")
            self.placeholder_path = output_file_path
        
        # 4. If it does not exist, create it
        else:
            print(f"Creating placeholder: {output_file_path}")
            nb = int(self.image_meta["bands"])
            W  = int(self.image_meta["width"]  * self.factor)
            H  = int(self.image_meta["height"] * self.factor)
            tr = self.image_meta["transform"]
            save_transform = Affine(tr.a / self.factor, tr.b, tr.c, tr.d, tr.e / self.factor, tr.f)

            profile = { # define profile for output file
                "driver": "GTiff",
                "dtype": self.image_meta["dtype"],
                "count": nb,
                "width": W,
                "height": H,
                "crs": self.image_meta["crs"],
                "transform": save_transform,
                "compress": "deflate",
                "tiled": True,
                "blockxsize": 512,
                "blockysize": 512,
                "bigtiff": "IF_SAFER",
                "sparse_ok": True,  # rasterio passes SPARSE_OK when present in profile >=1.3
            }
            # Create the dataset and close without writing any pixels
            with rasterio.open(output_file_path, "w", **profile):
                pass
            """
            with rasterio.open(output_file_path, "w", **profile) as dst:
                block_h, block_w = dst.block_shapes[0]      # (rows, cols)
                zeros_block = np.zeros((nb, block_h, block_w), dtype=dst.dtypes[0])

                for row in tqdm(range(0, H, block_h),desc="Writing placeholder file ..."):
                    h = min(block_h, H - row)
                    for col in range(0, W, block_w):
                        w = min(block_w, W - col)
                        window = rasterio.windows.Window(col_off=col, row_off=row, width=w, height=h)
                        dst.write(zeros_block[:, :h, :w], window=window)
            """

            print(f"Saved empty placeholder SR image at: {output_file_path}")
        
    def create_image_windows(self): # Works
        """
        Creates a list of window coordinates for the input image. The windows overlap by a specified amount.
        Output type is a list of rasterio.windows.Window objects.
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
        # delete LR stack folder
        shutil.rmtree(self.temp_folder)
        print("Deleted temp folder at",self.temp_folder)

    def start_super_resolution(self, debug: bool = False):
        """
        Run SR inference and stream predictions to disk (temp/*.npy) via model hooks.
        If debug=True, process only the first 100 LR windows total (DDP-safe).
        """
        # Pick windows (debug trims to first 100 globally; DDP sampler will shard those across ranks)
        windows_all = self.image_meta["image_windows"]
        windows_run = windows_all[:100] if debug else windows_all

        # Rebuild a datamodule for this run (debug-safe) and update model hook context
        dm = PredictionDataModule(input_type=self.input_type, root=self.root, windows=windows_run)
        dm.setup()

        # Hand the hook context to the model (these are consumed inside predict_step)
        self.model._save_temp_folder = self.temp_folder
        self.model._save_windows     = windows_run
        self.model._save_factor      = int(self.factor)

        # Trainer config: no logging/checkpointing; DDP handled automatically if multiple GPUs
        trainer = Trainer(
            accelerator=self.device,
            devices=self.gpus,
            logger=False,
            enable_checkpointing=False,
            enable_model_summary=False,
            inference_mode=True,           # no_grad + eval
            # precision="16-mixed",        # uncomment if your model supports AMP
        )

        # Stream predictions to disk inside predict_step (we return None there to avoid gathers)
        trainer.predict(self.model, datamodule=dm, return_predictions=False)

        # Rank-aware status message
        is_ddp = torch.distributed.is_available() and torch.distributed.is_initialized()
        rank = torch.distributed.get_rank() if is_ddp else 0
        if rank == 0:
            print(f"Prediction complete. Tiles saved in: {self.temp_folder}")
            if debug:
                print("Debug mode was ON → processed only 100 windows.")

    def write_to_file(self, index_path=None, limit=None):
        """
        Stream SR patches from temp/ to the placeholder GeoTIFF using your blending writer.
        Uses self.image_meta / self.factor / self.overlap / self.eliminate_border_px for config.
        Only the per-tile LR windows & paths come from index.json.
        """
        # 1) load merged index
        if index_path is None:
            index_path = os.path.join(self.temp_folder, "index.json")
        with open(index_path) as f:
            idx = json.load(f)

        entries = idx["entries"]
        entries.sort(key=lambda e: (e["row_off_lr"], e["col_off_lr"]))
        if limit is not None:
            entries = entries[:int(limit)]

        # 2) build once-off info dict FROM SELF (no derived junk)
        info_dict = {
            "placeholder_path": os.path.join(self.placeholder_filepath),
            "dtype": str(self.image_meta["dtype"]),
            "factor": int(self.factor),
            "overlap": int(self.overlap),
            "eliminate_border_px": int(self.eliminate_border_px),
            # window list matches loop order; index i maps directly
            "window_coordinates": [
                Window(e["col_off_lr"], e["row_off_lr"], e["width_lr"], e["height_lr"])
                for e in entries
            ],
        }

        # 3) stream tiles and write
        container = idx.get("saved_container", "npz")
        key = idx.get("saved_key", "arr")

        to_delete = []  # paths staged for deletion
        cleanup_every = 50  # delete staged files every N tiles (0 = never)
        for i, e in enumerate(tqdm(entries, desc="Stitching → placeholder", unit="tile"), start=1):
                p = e["path"]

                # load patch
                if container == "npz" and p.endswith(".npz"):
                    with np.load(p) as z:
                        sr = z[key]  # (C,H,W), uint16 ×10000
                else:
                    sr = np.load(p)

                # blend-write into placeholder
                blend_write(self, sr, i-1, info_dict)

                # stage for deletion
                to_delete.append(p)

                # every cleanup_every tiles, delete staged files
                if cleanup_every and (i % cleanup_every == 0):
                    for fp in to_delete:
                        try:
                            os.remove(fp)
                        except FileNotFoundError:
                            pass
                        # also nuke common temp leftovers next to it
                        for leftover in glob.glob(fp + ".*"):
                            try:
                                os.remove(leftover)
                            except FileNotFoundError:
                                pass
                    to_delete.clear()

        self.delete_LR_temp()
        # rename placeholder file to sr.tif
        sr_path = os.path.join(self.placeholder_filepath.replace("sr_placeholder.tif","sr.tif"))
        os.rename(self.placeholder_filepath, sr_path)
        print(f"Stitched {len(entries)} tiles into: {sr_path}")

if __name__ == "__main__":
    path = "/data2/simon/mosaic/individual_tile/S2_BA_small.tif"
    o = large_file_processing( 
                 root=path,
                 model=None,
                 window_size=(128, 128),
                 factor=4,
                 overlap=8,
                 eliminate_border_px=0,
                 device="cuda",
                 gpus=[0]
                 )
    o.start_super_resolution(debug=False)
    o.write_to_file()
    
    
