import torch
# import dataloader and dataset functions
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
import os
import rasterio
import numpy as np
from rasterio.windows import Window
from torch.utils.data import Sampler



class ShardedInferenceSampler(Sampler[int]):
    """
    No-duplicate sampler for DDP inference.
    Splits [0, N) into `world_size` contiguous shards; last shards may be shorter.
    """
    def __init__(self, dataset_len: int, *, rank: int, world_size: int):
        self.dataset_len = int(dataset_len)
        self.rank = int(rank)
        self.world_size = int(world_size)
        if not (0 <= self.rank < self.world_size):
            raise ValueError(f"Bad rank/world_size: {self.rank}/{self.world_size}")

        base = self.dataset_len // self.world_size
        rem  = self.dataset_len %  self.world_size
        # First `rem` ranks get one extra item
        self.length = base + (1 if self.rank < rem else 0)
        self.start  = self.rank * base + min(self.rank, rem)
        self.end    = self.start + self.length

    def __iter__(self):
        # Range is empty if dataset_len < rank start (rare but valid)
        return iter(range(self.start, self.end))

    def __len__(self):
        return self.length


def _infer_rank_world():
    """Robust rank/world detection (works even if user queries before init)."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank(), torch.distributed.get_world_size()
    # Fallback to env if called a bit early; OK because we only use it in predict_dataloader
    rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    return rank, world



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

    def __init__(self,input_type,root,windows,lr_file_dict):
        # Set properties
        self.input_type = input_type
        self.root = root
        self.windows = windows
        
        # 1. If file
        if self.input_type == "file":
            self.lr_file_dict = lr_file_dict
        # 2. If SAFE
        elif self.input_type == "SAFE":
            self.lr_file_dict = lr_file_dict
        # 3. If S2GM
        elif self.input_type == "S2GM":
            self.lr_file_dict = lr_file_dict


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
            img = self.get_from_dict(idx)
        elif self.input_type == "S2GM":
            img = self.get_from_dict(idx)
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
    
    def get_from_dict(self, idx):
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
            with rasterio.open(self.lr_file_dict[b]) as src:
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
    def __init__(self, input_type,root,windows,lr_file_dict,
                 prefetch_factor=2, batch_size=16, num_workers=4):
        super().__init__()
        self.input_type = input_type
        self.root = root
        self.windows = windows
        self.lr_file_dict = lr_file_dict
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
                                     windows=self.windows,
                                     lr_file_dict=self.lr_file_dict)

    def predict_dataloader(self):
        # Determine if we’re in DDP
        is_ddp = torch.distributed.is_available() and torch.distributed.is_initialized()
        rank, world = _infer_rank_world()

        # Build a shard-aware sampler ONLY when DDP is active
        sampler = None
        if is_ddp:
            sampler = ShardedInferenceSampler(
                dataset_len=len(self.dataset),
                rank=rank,
                world_size=world,
            )

        # Pin memory only if running on CUDA
        pin_memory = torch.cuda.is_available()

        # prefetch_factor must be None when num_workers == 0
        pf = self.prefetch_factor if self.num_workers and self.num_workers > 0 else None

        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,                 # IMPORTANT: never shuffle for predict with a sampler
            sampler=sampler,               # None on CPU/1-GPU; sharded on DDP
            num_workers=self.num_workers,
            pin_memory=pin_memory,
            persistent_workers=False,      # safer for repeated runs
            prefetch_factor=pf,
        )
