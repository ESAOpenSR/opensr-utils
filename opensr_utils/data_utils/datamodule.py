import torch
# import dataloader and dataset functions
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
import os
import rasterio
import numpy as np
from rasterio.windows import Window

# Dataset class for PyTorch Lightning
class PredictionDataset(Dataset):
    """
    PyTorch Dataset for patch-based inference on large Sentinel-2 imagery.

    Handles three input types:
      - **file** : a single GeoTIFF readable by rasterio.
      - **SAFE** : a Sentinel-2 SAFE folder, where RGB+NIR JP2 bands are located
        under `IMG_DATA`.
      - **S2GM** : a Sentinel-2 Global Mosaic (S2GM) folder containing per-band TIFFs.

    For each window, reads the corresponding patch, stacks RGB+NIR, and returns
    a normalized FloatTensor.

    Parameters
    ----------
    input_type : {"file", "SAFE", "S2GM"}
        Type of input data.
    root : str
        Path to the raster file or folder.
    windows : list of rasterio.windows.Window
        Sliding windows defining LR patches to read.

    Returns
    -------
    dict
        {
          "image" : torch.FloatTensor (C,H,W), scaled to [0,1],
          "meta"  : dict with patch metadata (row_off, col_off, height, width, index).
        }

    Notes
    -----
    - Input values are divided by 10000.0 and clamped to [0,1].
    - Band order is always RGB+NIR (4 channels).
    """

    def __init__(self,input_type,root,windows):
        # Set properties
        self.input_type = input_type
        self.root = root
        self.windows = windows
        
        # 1. If file, nothing necessary
        if self.input_type == "file":
            pass
        # 2. If SAFE, parse SAFE structure to get file paths
        elif self.input_type == "SAFE":
            # Find JP2s for 10 m bands (B02,B03,B04,B08) under IMG_DATA
            jp2s = []
            for root, _, files in os.walk(self.root):
                if "IMG_DATA" in root:
                    for f in files:
                        if f.endswith(".jp2") and any(b in f for b in ("B02", "B03", "B04", "B08")):
                            jp2s.append(os.path.join(root, f))
            jp2s = sorted(jp2s)  # Sort to maintain consistent band order
            self.jp2_dict = {
                "R": [p for p in jp2s if "B04" in p][0],
                "G": [p for p in jp2s if "B03" in p][0],
                "B": [p for p in jp2s if "B02" in p][0],
                "NIR": [p for p in jp2s if "B08" in p][0],
            }
        elif self.input_type == "S2GM":
            # Collect per-band GeoTIFFs in the given tile folder
            tifs = []
            for root, _, files in os.walk(self.root):
                for f in files:
                    if f.lower().endswith(".tif"):
                        tifs.append(os.path.join(root, f))

            # Hard-map to RGB + NIR
            self.s2gm_files = {
                "R":  [p for p in tifs if os.path.basename(p) == "B04.tif"][0],
                "G":  [p for p in tifs if os.path.basename(p) == "B03.tif"][0],
                "B":  [p for p in tifs if os.path.basename(p) == "B02.tif"][0],
                "NIR":[p for p in tifs if os.path.basename(p) == "B08.tif"][0],
            }

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        # Get metadata of this datapoint and carry over
        win = self.windows[idx]
        meta = {
        "row_off": int(win.row_off),
        "col_off": int(win.col_off),
        "height":  int(win.height),
        "width":   int(win.width),
        "index":   int(idx),
}
        
        # Get image window from file depending on input type
        if self.input_type == "file":
            img = self.get_from_file(idx)
        elif self.input_type == "SAFE":
            img = self.get_from_SAFE(idx)
        elif self.input_type == "S2GM":
            img = self.get_from_S2GM(idx)
        else:
            raise NotImplementedError(f"Input type {self.input_type} not supported.")    
        # perform normalizations here
        img = img / 10000.0  # Scale to [0,1] assuming input is in [0,10000]
        img = torch.clamp(img, 0.0, 1.0)
        # dtype to float32
        img = img.type(torch.float32)
        # return batch image
        return {"image": img, "meta": meta}
    
    def get_from_file(self, idx):
        """
        Read a patch from a single GeoTIFF file.

        Parameters
        ----------
        idx : int
            Index of the window to read.

        Returns
        -------
        torch.FloatTensor
            (C,H,W) patch in native band order, float32.
        """
        window = self.windows[idx]
        with rasterio.open(self.root) as src:
            data = src.read(window=window)  # np.ndarray
        data = data.astype(np.float32)
        return torch.from_numpy(data)
    
    def get_from_SAFE(self, idx):
        """
        Read RGB+NIR bands from a Sentinel-2 SAFE folder for a given window.

        Parameters
        ----------
        idx : int
            Index of the window to read.

        Returns
        -------
        torch.FloatTensor
            (4,H,W) patch with bands in [R,G,B,NIR] order, float32.
        """
        window = self.windows[idx]
        bands_order = ["R", "G", "B", "NIR"]
        tiles = []
        for b in bands_order:
            with rasterio.open(self.jp2_dict[b]) as src:
                tile = src.read(1, window=window)
                tiles.append(tile)
        arr = np.stack(tiles, axis=0).astype(np.float32)
        return torch.from_numpy(arr)
    
    def get_from_S2GM(self, idx):
        """
        Read RGB+NIR bands from a Sentinel-2 Global Mosaic (S2GM) tile folder.

        Parameters
        ----------
        idx : int
            Index of the window to read.

        Returns
        -------
        torch.FloatTensor
            (4,H,W) patch with bands in [R,G,B,NIR] order, float32.
        """
        window = self.windows[idx]
        order = ["R", "G", "B", "NIR"]

        tiles = []
        for k in order:
            with rasterio.open(self.s2gm_files[k]) as src:
                tile = src.read(1, window=window)
                tiles.append(tile)
        arr = np.stack(tiles, axis=0).astype(np.float32)
        return torch.from_numpy(arr)
    


class PredictionDataModule(LightningDataModule):
    """
    PyTorch Lightning DataModule for patch-based SR inference.

    Wraps `PredictionDataset` and provides a DataLoader suitable for
    Lightning's `.predict()` loop.

    Parameters
    ----------
    input_type : {"file", "SAFE", "S2GM"}
        Type of input data.
    root : str
        Path to raster file or folder.
    windows : list of rasterio.windows.Window
        List of windows defining patches to read.
    prefetch_factor : int, default=2
        Number of batches to prefetch per worker.
    batch_size : int, default=16
        Number of patches per batch.
    num_workers : int, default=4
        Number of worker processes for data loading.

    Attributes
    ----------
    dataset : PredictionDataset
        Dataset instance created during `setup()`.
    """
    def __init__(self, input_type,root,windows,
                 prefetch_factor=2, batch_size=16, num_workers=4):
        super().__init__()
        self.input_type = input_type
        self.root = root
        self.windows = windows
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.prefetch_factor = prefetch_factor

    def setup(self, stage=None):
        """
        Instantiate `PredictionDataset` with given input type, root, and windows.

        Parameters
        ----------
        stage : str, optional
            Lightning stage flag (ignored in this implementation).
        """
        self.dataset = PredictionDataset(input_type=self.input_type,
                                     root=self.root,
                                     windows=self.windows)
        print(f"📦 Created PredictionDataModule with {len(self.dataset)} patches.")

    def predict_dataloader(self):
        return DataLoader(self.dataset, num_workers=self.num_workers,
                          batch_size=self.batch_size, prefetch_factor=self.prefetch_factor)
