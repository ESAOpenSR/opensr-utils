import os
import json
import torch
import numpy as np
from pytorch_lightning import LightningModule
import time
import glob
import math
from PIL import Image

"""
Multi-GPU SR Inference Workflow
================================

   ┌─────────────┐
   │ Trainer.run │
   └──────┬──────┘
          │
          ▼
   ┌─────────────────────────┐
   │ Each GPU rank executes: │
   └──────────┬──────────────┘
              │
              ▼
   [on_predict_start]
        │  Initialize:
        │   - rank/world_size
        │   - temp folder
        │   - local entries[]
        ▼
   [predict_step]
        │  For each batch:
        │   - Run model (SR patch)
        │   - Save patch → temp/sr_rXXXX_cXXXX.npz
        │   - Append offsets + path to entries[]
        ▼
   [on_predict_end]
        │  Write shard index_rank{r}.json
        │
        │───► DDP barrier ───(all ranks wait)───┐
        │                                      │
        ▼                                      │
   Rank >0:                                    │ Rank 0:
   - Finished                                  │  - Collect all shard JSONs
   - Optionally delete shards                  │  - Merge entries[]
                                               │  - Deduplicate + sort
                                               │  - Write global index.json
                                               │  - Clean up shard files
                                               ▼
                                        ┌───────────────┐
                                        │ index.json    │
                                        │ (global list) │
                                        └───────────────┘
"""


"""
Module: preprocess_model

Provides utilities to wrap a PyTorch or Lightning model for distributed
super-resolution (SR) inference. The core workflow integrates with PyTorch
Lightning’s `.predict()` loop:

1. **Model wrapping**
   - Accepts a plain `torch.nn.Module`, `LightningModule`, or `None`.
   - If `None`, a placeholder bilinear upsampling model is used.
   - If plain torch model, it is wrapped into a minimal LightningModule.

2. **Hook injection**
   - Adds `on_predict_start`, `predict_step`, and `on_predict_end` hooks.
   - These hooks handle patch inference, saving results to disk, and writing
     per-rank index files.

3. **Distributed execution (DDP)**
   - Each GPU rank saves its own shard (`index_rank{r}.json`).
   - Synchronization barrier ensures all ranks finish writing.
   - Rank 0 merges shards into a global `index.json` in deterministic order.

4. **Temporary storage**
   - SR patches are saved to `.npz` files in `temp/`.
   - Index JSON stores file paths, LR/HR offsets, and dimensions.

This design allows inference on arbitrarily large raster datasets across
multiple GPUs without memory bottlenecks. Patches are streamed to disk and
stitched later.
"""

# -------------------------------------------------------------------------
# Placeholder interpolation model
class SRModelPL(LightningModule): 
    """
    Placeholder LightningModule for SR inference.

    Uses bilinear interpolation to resize input patches to 512×512. Useful when
    no trained model is provided, for testing the end-to-end pipeline.

    Methods
    -------
    forward(x)
        Perform bilinear upsampling on input tensor.
    predict(x)
        Alias for forward; called inside predict_step.
    """
    def __init__(self):
        super(SRModelPL, self).__init__()
    def forward(self, x):
        sr = torch.nn.functional.interpolate(x, size=(512, 512), mode="bilinear")
        return sr
    def predict(self,x):
        return self.forward(x)
    
    
def get_world_info():
    """
    Return (world_size, rank) in a robust way:
    - CPU or single-GPU: world_size = 1
    - Multi-GPU DDP: world_size = number of GPUs (world size)
    """
    ddp = torch.distributed.is_available() and torch.distributed.is_initialized()

    if ddp:
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
    else:
        # if CUDA is available, check how many GPUs are visible
        gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        world_size = gpu_count if gpu_count > 1 else 1
        rank = 0

    return world_size, rank


# -------------------------------------------------------------------------
# Preprocessor that also attaches predict_step + hooks that save predictions to disk
def preprocess_model(self,model,temp_folder,windows,factor):
    """
    Prepare a model for distributed patch-based super-resolution inference.

    This function ensures the model is a LightningModule with custom hooks
    attached for saving predictions to disk. It configures the model with
    temporary output paths, sliding windows, and SR scale factor.

    Parameters
    ----------
    model : LightningModule, torch.nn.Module, or None
        The SR model to wrap. If None, uses a placeholder bilinear model.
    temp_folder : str
        Directory for saving per-patch `.npz` predictions and index files.
    windows : list of rasterio.windows.Window
        List of LR windows corresponding to input patches.
    factor : int
        Upscaling factor (e.g., 4 for 10m → 2.5m).

    Returns
    -------
    LightningModule
        Model ready for `Trainer.predict()` with hooks for disk streaming.

    Workflow
    --------
    1. **Model normalization**:
       - LightningModule → returned directly.
       - torch.nn.Module → wrapped into minimal LightningModule.
       - None → replaced with SRModelPL (bilinear).

    2. **Hook injection**:
       Attaches:
         - `on_predict_start`: initializes rank info, temp folder, and index.
         - `predict_step`: runs inference, saves patches as `.npz`, logs offsets.
         - `on_predict_end`: writes per-rank index JSON, synchronizes DDP,
           merges shards on rank 0.

    3. **Multi-GPU / DDP details**:
       - Each rank runs `predict_step` on its shard of the dataset.
       - Saved files include row/col offsets in both LR and HR grids.
       - `on_predict_end`:
         * Writes `index_rank{r}.json` on each rank.
         * Calls `torch.distributed.barrier()` to sync.
         * Rank 0 merges all shards into `index.json` in deterministic order.
         * Optionally deletes shard JSONs and partial files.

    Notes
    -----
    - Saved arrays are clamped/scaled to [0,10000] and stored as uint16.
    - File naming encodes HR offsets: `sr_r{row_off_hr}_c{col_off_hr}.npz`.
    - Final `index.json` is consumed by the stitching stage.
    """
    
    # --- define hooks here so they stay out of the global namespace ---
    def _rank_world(self):
        """
        Utility to detect current DDP rank and world size.

        Returns
        -------
        (rank, world_size) : tuple of int
            (0,1) if not running under torch.distributed.
        """
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank(), torch.distributed.get_world_size()
        return 0, 1

    def on_predict_start(self):
        """
        Hook executed at the start of prediction.

        - Ensures temp folder exists.
        - Records DDP rank and world size.
        - Initializes local index list (`self._entries`).
        - Sets counters for per-rank patch saving.
        """
        assert hasattr(self, "_save_temp_folder"), "set model._save_temp_folder"
        assert hasattr(self, "_save_windows"), "set model._save_windows"
        assert hasattr(self, "_save_factor"), "set model._save_factor"
        os.makedirs(self._save_temp_folder, exist_ok=True)
        self._rank, self._world = _rank_world(self)
        self._local_idx = 0
        self._entries = []

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """
        Hook executed for each batch during prediction.

        Workflow
        --------
        1. Run model forward/predict to obtain SR patches (B,C,Hh,Ww).
        2. Normalize metadata dicts from batch.
        3. For each sample:
        - Compute HR offsets from LR offsets × scale.
        - Clamp and scale values to [0,10000], convert to uint16.
        - Save patch as compressed `.npz` in temp folder.
        - Append entry with file path, offsets, and dimensions to local index.
        4. Return None → prevents Lightning from gathering predictions.

        Notes
        -----
        - Handles both dict-of-lists (PL default) and list-of-dict meta formats.
        - Saves filenames as `sr_r{row_off_hr}_c{col_off_hr}.npz`.
        """
        x    = batch["image"]                        # (B,C,H,W) tensor
        meta = batch["meta"]                         # list[dict] or dict of lists
        sr = self.predict(x) if hasattr(self, "predict") else self.forward(x)
        sr = sr.detach().cpu()                       # (B,C,Hh,Ww)

        scale = int(self._save_factor)
        B, C, Hh, Ww = sr.shape

        # meta may be a dict of lists (PL default); normalize to per-sample dicts
        if isinstance(meta, dict) and isinstance(meta["row_off"], torch.Tensor):
            meta = [{k: int(v[i].item()) for k, v in meta.items()} for i in range(B)]
        elif isinstance(meta, dict):
            meta = [{k: int(v[i]) for k, v in meta.items()} for i in range(B)]

        for i in range(B):
            mi = meta[i]
            row_off_lr = mi["row_off"]; col_off_lr = mi["col_off"]
            row_off_hr = row_off_lr * scale; col_off_hr = col_off_lr * scale

            s = sr[i]                                # (C,Hh,Ww) float
            mean_val = float(torch.mean(s))
            # simple range heuristic: [0,1] vs already ~[0,10000]
            if mean_val < 10:
                s = torch.clamp(s, 0.0, 1.0) * 10000.0
            else:
                s = torch.clamp(s, 0.0, 10000.0)

            s = torch.round(s)
            arr = s.numpy().astype(np.uint16, copy=False)

            fn = f"sr_r{row_off_hr:09d}_c{col_off_hr:09d}.npz"
            out_path = os.path.join(self._save_temp_folder, fn)
            np.savez_compressed(out_path, arr=arr)   # direct compressed save

            self._entries.append({
                "path": out_path,
                "row_off_lr": row_off_lr, "col_off_lr": col_off_lr,
                "height_lr":  mi["height"], "width_lr": mi["width"],
                "row_off_hr": row_off_hr,   "col_off_hr": col_off_hr,
                "height_hr":  int(Hh),      "width_hr":  int(Ww),
            })

            # logic to calculate logging steps - every 10% of total
            ddp  = torch.distributed.is_available() and torch.distributed.is_initialized()
            rank = torch.distributed.get_rank() if ddp else 0
            world= torch.distributed.get_world_size() if ddp else 1
            is_rank0 = (rank == 0)
            global_total = int(model.model_dm_length)
            local_interval = max(1, math.ceil((global_total / max(world,1)) / 10))
            if is_rank0:
                # avoid firing at 0
                if global_total > 0 and self._local_idx > 0 and (self._local_idx % local_interval == 0):
                    # CHW -> take RGB (or gray→RGB)
                    if arr.shape[0] >= 3:
                        rgb = arr[:3, :, :]
                    else:
                        rgb = np.repeat(arr[:1, :, :], 3, axis=0)
                    
                    x_i = x[i].detach().cpu()             # CHW float
                    if x_i.shape[0] >= 3:
                        lr_rgb = x_i[:3, :, :].numpy()
                    else:
                        lr_rgb = np.repeat(x_i[:1, :, :].numpy(), 3, axis=0)

                    # scale [0..10000] -> [0..255] with optional gain
                    gain = 3.5
                    rgb = (((rgb/10_000)*gain)*255.0).astype(np.uint8).clip(0,255)
                    lr_rgb = (((lr_rgb)*gain)*255.0).astype(np.uint8).clip(0,255)

                    # CHW -> HWC for PIL
                    rgb,lr_rgb = np.transpose(rgb, (1, 2, 0)), np.transpose(lr_rgb, (1, 2, 0))

                    # save
                    Image.fromarray(rgb).save(os.path.join(self.log_dir, f"preview_sr_{self._local_idx:06d}.png"))
                    Image.fromarray(lr_rgb).save(os.path.join(self.log_dir, f"preview_lr_{self._local_idx:06d}.png"))


        return None

    def on_predict_end(self):
        """
        Hook executed once prediction finishes.

        Workflow
        --------
        1. Write per-rank shard JSON:
        - Contains rank, world_size, factor, saved dtype/scale, and patch entries.
        2. Synchronize ranks via `torch.distributed.barrier()` (if DDP).
        3. Rank 0 collects shard JSONs:
        - Waits for all shards to exist.
        - Reads entries, merges, deduplicates, sorts by LR offsets.
        - Writes merged global `index.json`.
        4. Cleanup:
        - Deletes shard JSONs (optional).
        - Deletes leftover partial files (optional).

        Multi-GPU Details
        -----------------
        - Each GPU saves only the patches it predicted.
        - Index merging ensures a single consistent record of all patches.
        - Sorting by (row_off_lr, col_off_lr) guarantees deterministic order.

        Prints
        ------
        - Per-rank: number of entries written.
        - Rank 0: total merged entries and output path of `index.json`.
        """
        shard = {
            "rank": self._rank,
            "world_size": self._world,
            "factor": int(self._save_factor),
            "entries": self._entries,
            "saved_dtype": "uint16",
            "saved_scale": 10000,
            "saved_container": "npz",
            "saved_key": "arr",
        }
        shard_path = os.path.join(self._save_temp_folder, f"index_rank{self._rank}.json")
        with open(shard_path, "w") as f:
            json.dump(shard, f, separators=(",", ":"))  # compact
            f.flush()
            os.fsync(f.fileno())
        #print(f"[rank {self._rank}] wrote {len(self._entries)} entries → {shard_path}")

        # ---- DDP sync ----
        ddp = torch.distributed.is_available() and torch.distributed.is_initialized()
        if ddp:
            torch.distributed.barrier()

        # ---- Merge on rank 0 (or single-process) ----
        if (not ddp) or (torch.distributed.get_rank() == 0):
            world = self._world if ddp else 1
            shard_paths = [os.path.join(self._save_temp_folder, f"index_rank{r}.json") for r in range(world)]

            # (Optional) wait briefly if FS is slow
            for _ in range(50):
                if all(os.path.exists(p) for p in shard_paths): break
                time.sleep(0.05)

            merged_entries = []
            meta = {
                "factor": int(self._save_factor),
                "saved_dtype": "uint16",
                "saved_scale": 10000,
                "saved_container": "npz",
                "saved_key": "arr",
            }
            for p in shard_paths:
                if not os.path.exists(p):
                    continue  # tolerate empty shard or failed worker
                with open(p) as f:
                    data = json.load(f)
                merged_entries.extend(data.get("entries", []))

            # dedupe & deterministic order
            # key by LR offsets + path (handles accidental dupes)
            seen = set()
            uniq = []
            for e in merged_entries:
                k = (e["row_off_lr"], e["col_off_lr"], e["path"])
                if k not in seen:
                    seen.add(k)
                    uniq.append(e)
            uniq.sort(key=lambda e: (e["row_off_lr"], e["col_off_lr"]))

            merged = {**meta, "entries": uniq}
            merged_path = os.path.join(self._save_temp_folder, "index.json")
            with open(merged_path, "w") as f:
                json.dump(merged, f, separators=(",", ":"))
                f.flush()
                os.fsync(f.fileno())
            #print(f"[rank 0] merged {len(uniq)} entries → {merged_path}")
        
        # CleanUp
        try:
            KEEP_SHARDS = False  # toggle for debugging
            KEEP_PARTS = False   # toggle for debugging
            if not KEEP_SHARDS:
                for p in shard_paths:
                    try: os.remove(p)
                    except FileNotFoundError: pass
                    
            # optional: nuke partials
            if KEEP_PARTS==False:
                for p in glob.glob(os.path.join(self._save_temp_folder, "*.part")):
                    try: os.remove(p)
                    except FileNotFoundError: pass
        except UnboundLocalError:
            pass # If we're in single GPU, or in CPU, these variables dont exist


    # ---------------------------------------------------------------------
    # Attach hooks to model
    def _attach_hooks(m):
        """
        Bind custom hook functions to the model instance.

        - Replaces Lightning’s default hooks with custom implementations.
        - Ensures `Trainer.predict()` will call our save/merge workflow.

        Parameters
        ----------
        m : LightningModule
            Model to modify.

        Returns
        -------
        LightningModule
            Same model with hooks attached.
        """
        m.on_predict_start = on_predict_start.__get__(m, m.__class__)
        m.predict_step = predict_step.__get__(m, m.__class__)
        m.on_predict_end = on_predict_end.__get__(m, m.__class__)
        return m

    # --- wrap model if needed ---
    if isinstance(model, LightningModule):
        self._log("⚡ Model is a LightningModule ✅")
        model.eval()
    elif isinstance(model, torch.nn.Module):
        self._log("Model is a torch.nn.Module. Wrapping in LightningModule ✅.")
        class WrappedModel(LightningModule):
            def __init__(self, mdl):
                super().__init__()
                self.model = mdl
            def forward(self, x): return self.model(x)
            def predict(self, x): return self.forward(x)
        wrapped_model = WrappedModel(model)
        wrapped_model.eval()
        model = wrapped_model
    elif model is None:
        self._log("⚠️ No model provided - using placeholder interpolation model.")
        placeholder_model = SRModelPL()
        placeholder_model.eval()
        model = placeholder_model
    else:
        raise NotImplementedError("Model must be a LightningModule, torch.nn.Module, or None.")


    # set model variables
    model.log_dir = self.log_dir # set log dir for previews
    model.model_dm_length = len(self.datamodule.dataset) # set datamodule length for preview interval
    model = _attach_hooks(model)
    model._save_temp_folder = temp_folder
    model._save_windows = windows
    model._save_factor = factor
    return model