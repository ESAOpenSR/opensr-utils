"""Wrap SR models for streamed inference and collective manifest finalization.

Each rank persists patches and a shard index. All ranks exchange write status;
rank zero validates and atomically publishes the merged index. Hooks are bound
on preparation so plain PyTorch and Lightning models share the same workflow.
"""

import os
import json
import torch
import numpy as np
from pytorch_lightning import LightningModule
import time
import math
from numbers import Integral
from datetime import timedelta
from PIL import Image


# -------------------------------------------------------------------------
# Placeholder interpolation model
class SRModelPL(LightningModule):
    """
    Placeholder LightningModule for SR inference.

    Uses bilinear interpolation to resize input patches by the requested factor. Useful when
    no trained model is provided, for testing the end-to-end pipeline.

    Methods
    -------
    forward(x)
        Perform bilinear upsampling on input tensor.
    predict(x)
        Alias for forward; called inside predict_step.
    """

    def __init__(self, factor=4):
        super(SRModelPL, self).__init__()
        if not isinstance(factor, int) or isinstance(factor, bool) or factor <= 0:
            raise ValueError("factor must be a positive integer.")
        self.factor = int(factor)

    def forward(self, x):
        if self.factor == 1:
            return x
        sr = torch.nn.functional.interpolate(
            x,
            scale_factor=self.factor,
            mode="bilinear",
            align_corners=False,
        )
        return sr

    def predict(self, x):
        return self.forward(x)


def get_world_info():
    """Return process world size and global rank, regardless of visible GPUs."""
    ddp = torch.distributed.is_available() and torch.distributed.is_initialized()

    if ddp:
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
    else:
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))

    return world_size, rank


def _distributed_barrier(timeout_seconds=30):
    """Synchronize ranks while turning gloo failures into regular exceptions."""
    dist = torch.distributed
    backend = None
    try:
        backend = dist.get_backend()
    except (AttributeError, RuntimeError, ValueError):
        pass
    if backend == "gloo" and hasattr(dist, "monitored_barrier"):
        try:
            return dist.monitored_barrier(timeout=timedelta(seconds=timeout_seconds))
        except TypeError:  # older torch versions have no timeout parameter
            return dist.monitored_barrier()
    return dist.barrier()


# -------------------------------------------------------------------------
# Preprocessor that also attaches predict_step + hooks that save predictions to disk
def preprocess_model(
    self, model, temp_folder, windows, factor, output_units="normalized"
):
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
        Positive integer output scale. Use 1 for models that preserve resolution.
    output_units : {"normalized", "reflectance"}, default="normalized"
        Interpretation of model predictions before persistence. Normalized values
        are clipped to [0, 1] and scaled to uint16 [0, 10000]; reflectance/DN
        values are clipped directly to [0, 10000].

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

    if not isinstance(factor, int) or isinstance(factor, bool) or factor <= 0:
        raise ValueError("factor must be a positive integer.")
    factor = int(factor)
    unit_aliases = {
        "normalized": "normalized",
        "normalised": "normalized",
        "reflectance": "reflectance",
        "dn": "reflectance",
    }
    units = unit_aliases.get(str(output_units).strip().lower())
    if units is None:
        raise ValueError("output_units must be 'normalized' or 'reflectance'.")

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
        self._output_bands = None
        self._save_compressed = bool(getattr(self, "_save_compressed", False))
        self._save_progress_preview = bool(
            getattr(self, "_save_progress_preview", False)
        )

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
        # An empty rank needs hooks to run, but has no image to infer.
        if "_opensr_empty_rank" in batch:
            return None
        x = batch["image"]  # (B,C,H,W) tensor
        meta = batch["meta"]  # list[dict] or dict of lists
        if not isinstance(x, torch.Tensor) or x.ndim != 4:
            raise ValueError("Input batch image must be a 4-D tensor (B,C,H,W).")
        input_batch = int(x.shape[0])
        sr = self.predict(x) if hasattr(self, "predict") else self.forward(x)
        if not isinstance(sr, torch.Tensor) or sr.ndim != 4:
            raise ValueError("Model output must be a 4-D tensor (B,C,H,W).")
        if int(sr.shape[0]) != input_batch:
            raise ValueError(
                "Model output batch size does not match input batch size: "
                f"got {int(sr.shape[0])}, expected {input_batch}."
            )
        if not torch.isfinite(sr).all().item():
            raise ValueError(
                "Model output contains NaN or infinite values; predictions must be finite."
            )
        sr = sr.detach().cpu()  # (B,C,Hh,Ww)

        scale = int(self._save_factor)
        B, C, Hh, Ww = sr.shape
        if C <= 0 or Hh <= 0 or Ww <= 0:
            raise ValueError(
                "Model output must have positive channel and spatial dimensions."
            )
        if self._output_bands is None:
            self._output_bands = int(C)
        elif int(C) != int(self._output_bands):
            raise ValueError(
                f"Model output band count changed from {self._output_bands} to {C}."
            )

        # meta may be a dict of lists (PL default); normalize to per-sample dicts
        if isinstance(meta, dict):
            required = ("row_off", "col_off", "height", "width")
            missing = [key for key in required if key not in meta]
            if missing:
                raise ValueError(
                    f"Prediction metadata is missing required fields: {', '.join(missing)}."
                )
            try:
                lengths = {len(value) for value in meta.values()}
            except TypeError as exc:
                raise ValueError(
                    "Prediction metadata values must be batched sequences."
                ) from exc
            if lengths != {B}:
                raise ValueError(
                    f"Prediction metadata batch size does not match input batch size ({B})."
                )
            meta = [
                {
                    k: v[i].item() if isinstance(v, torch.Tensor) else v[i]
                    for k, v in meta.items()
                }
                for i in range(B)
            ]
        elif isinstance(meta, (list, tuple)):
            if len(meta) != B:
                raise ValueError(
                    f"Prediction metadata batch size does not match input batch size ({B})."
                )
            meta = list(meta)
        else:
            raise ValueError(
                "Prediction metadata must be a dict of batches or a list of records."
            )
        for i, record in enumerate(meta):
            missing = [
                key
                for key in ("row_off", "col_off", "height", "width")
                if key not in record
            ]
            if missing:
                raise ValueError(
                    f"Prediction metadata record {i} is missing: {', '.join(missing)}."
                )
            for key in ("row_off", "col_off", "height", "width"):
                value = record[key]
                if isinstance(value, torch.Tensor) and value.numel() == 1:
                    value = value.item()
                if isinstance(value, bool) or not isinstance(value, Integral):
                    raise ValueError(
                        f"Prediction metadata record {i} requires an integer {key}."
                    )
                record[key] = int(value)
                if record[key] < 0 and key in ("row_off", "col_off"):
                    raise ValueError(
                        f"Prediction metadata record {i} has a negative {key}."
                    )
                if record[key] <= 0 and key in ("height", "width"):
                    raise ValueError(
                        f"Prediction metadata record {i} has non-positive {key}."
                    )
            expected_h = record["height"] * scale
            expected_w = record["width"] * scale
            if Hh != expected_h or Ww != expected_w:
                raise ValueError(
                    f"Model output shape {Hh}x{Ww} does not match expected "
                    f"{expected_h}x{expected_w} for factor {scale} (record {i})."
                )

        for i in range(B):
            mi = meta[i]
            row_off_lr = mi["row_off"]
            col_off_lr = mi["col_off"]
            row_off_hr = row_off_lr * scale
            col_off_hr = col_off_lr * scale
            s = sr[i]  # (C,Hh,Ww) float
            if units == "normalized":
                s = torch.clamp(s, 0.0, 1.0) * 10000.0
            else:
                s = torch.clamp(s, 0.0, 10000.0)

            s = torch.round(s)
            arr = s.numpy().astype(np.uint16, copy=False)

            ext = "npz" if self._save_compressed else "npy"
            fn = f"sr_r{row_off_hr:09d}_c{col_off_hr:09d}.{ext}"
            out_path = os.path.join(self._save_temp_folder, fn)
            if self._save_compressed:
                np.savez_compressed(out_path, arr=arr)
            else:
                np.save(out_path, arr)

            self._entries.append(
                {
                    "path": out_path,
                    "row_off_lr": row_off_lr,
                    "col_off_lr": col_off_lr,
                    "height_lr": mi["height"],
                    "width_lr": mi["width"],
                    "row_off_hr": row_off_hr,
                    "col_off_hr": col_off_hr,
                    "height_hr": int(Hh),
                    "width_hr": int(Ww),
                    "bands": int(C),
                }
            )

            # logic to calculate logging steps - every 10% of total
            ddp = (
                torch.distributed.is_available() and torch.distributed.is_initialized()
            )
            rank = torch.distributed.get_rank() if ddp else 0
            world = torch.distributed.get_world_size() if ddp else 1
            is_rank0 = rank == 0
            global_total = int(self.model_dm_length)
            local_interval = max(1, math.ceil((global_total / max(world, 1)) / 10))
            if is_rank0 and self._save_progress_preview:
                # avoid firing at 0
                if (
                    global_total > 0
                    and self._local_idx > 0
                    and (self._local_idx % local_interval == 0)
                ):
                    # CHW -> take RGB (or gray→RGB)
                    if arr.shape[0] >= 3:
                        rgb = arr[:3, :, :]
                    else:
                        rgb = np.repeat(arr[:1, :, :], 3, axis=0)

                    x_i = x[i].detach().cpu()  # CHW float
                    if x_i.shape[0] >= 3:
                        lr_rgb = x_i[:3, :, :].numpy()
                    else:
                        lr_rgb = np.repeat(x_i[:1, :, :].numpy(), 3, axis=0)

                    # scale [0..10000] -> [0..255] with optional gain
                    gain = 3.5
                    rgb = (
                        (((rgb / 10_000) * gain) * 255.0).astype(np.uint8).clip(0, 255)
                    )
                    lr_rgb = (((lr_rgb) * gain) * 255.0).astype(np.uint8).clip(0, 255)

                    # CHW -> HWC for PIL
                    rgb, lr_rgb = (
                        np.transpose(rgb, (1, 2, 0)),
                        np.transpose(lr_rgb, (1, 2, 0)),
                    )

                    # save
                    Image.fromarray(rgb).save(
                        os.path.join(
                            self.log_dir, f"preview_sr_{self._local_idx:06d}.png"
                        )
                    )
                    Image.fromarray(lr_rgb).save(
                        os.path.join(
                            self.log_dir, f"preview_lr_{self._local_idx:06d}.png"
                        )
                    )

            self._local_idx += 1

        return None

    def manifest_metadata(self):
        return {
            "factor": int(self._save_factor),
            "output_bands": int(self._output_bands or 0),
            "saved_dtype": "uint16",
            "saved_scale": 10000,
            "output_units": units,
            "saved_container": "npz" if self._save_compressed else "npy",
            "saved_key": "arr",
        }

    def write_manifest(path, data):
        # Readers see complete JSON, even on a slow shared filesystem.
        part = path + ".part"
        with open(part, "w", encoding="utf-8") as stream:
            json.dump(data, stream, separators=(",", ":"))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(part, path)

    def merge_shards(self, world):
        """Validate and publish a deterministic index; retain shards on failure."""
        paths = [
            os.path.join(self._save_temp_folder, f"index_rank{rank}.json")
            for rank in range(world)
        ]
        # Permit a short metadata-visibility delay on shared filesystems.
        for _ in range(50):
            if all(os.path.exists(path) for path in paths):
                break
            time.sleep(0.05)

        metadata = manifest_metadata(self)
        output_bands = None
        entries = []
        for path in paths:
            if not os.path.exists(path):
                raise RuntimeError(f"Missing prediction shard index: {path}")
            with open(path, encoding="utf-8") as stream:
                shard = json.load(stream)
            for field in (
                "factor",
                "saved_dtype",
                "saved_scale",
                "output_units",
                "saved_container",
            ):
                if field in shard and shard[field] != metadata[field]:
                    raise ValueError(f"Prediction shards disagree on {field}.")
            bands = shard.get("output_bands")
            if bands:
                if output_bands is not None and int(bands) != output_bands:
                    raise ValueError(
                        "Prediction shards disagree on output band count: "
                        f"{output_bands} vs {bands}."
                    )
                output_bands = int(bands)
            entries.extend(shard.get("entries", []))

        # Duplicate records from the same patch are harmless. Distinct windows
        # must remain in deterministic row/column order for reproducibility.
        unique = {}
        for entry in entries:
            key = (entry["row_off_lr"], entry["col_off_lr"], entry["path"])
            unique.setdefault(key, entry)
        entries = sorted(
            unique.values(),
            key=lambda entry: (entry["row_off_lr"], entry["col_off_lr"]),
        )
        entry_bands = {int(entry["bands"]) for entry in entries if entry.get("bands")}
        if len(entry_bands) > 1 or (
            entry_bands and output_bands is not None and entry_bands != {output_bands}
        ):
            raise ValueError("Prediction entries disagree on output band count.")
        metadata["output_bands"] = output_bands or next(iter(entry_bands), 0)
        write_manifest(
            os.path.join(self._save_temp_folder, "index.json"),
            {**metadata, "entries": entries},
        )
        for path in paths:
            os.remove(path)

    def on_predict_end(self):
        """Persist every shard and exchange errors at matching collective phases.

        Every live rank reaches the barrier even if its local write failed.
        After synchronization, all ranks exchange write status before rank zero
        merges. Merge errors are broadcast so workers cannot report success.
        A failed barrier exits directly: another collective on a broken group
        could hang. The launcher's process-group timeout bounds other collectives.
        """
        dist = torch.distributed
        ddp = dist.is_available() and dist.is_initialized()
        backend = None
        if ddp:
            try:
                backend = dist.get_backend()
            except (AttributeError, RuntimeError, ValueError):
                pass  # Lightweight tests may supply only rank/barrier metadata.

        original_error = None
        error = None
        try:
            write_manifest(
                os.path.join(self._save_temp_folder, f"index_rank{self._rank}.json"),
                {
                    **manifest_metadata(self),
                    "rank": self._rank,
                    "world_size": self._world,
                    "entries": self._entries,
                },
            )
        except Exception as exc:
            original_error = exc
            error = f"{type(exc).__name__}: {exc}"

        if ddp:
            try:
                _distributed_barrier(getattr(self, "_save_sync_timeout", 30))
            except Exception as exc:
                raise RuntimeError(
                    f"Distributed prediction finalization failed: synchronization barrier: {exc}"
                ) from exc

        if backend is not None:
            statuses = [None] * self._world
            try:
                dist.all_gather_object(statuses, error)
            except Exception as exc:
                raise RuntimeError(
                    f"Distributed prediction finalization failed: write-status exchange: {exc}"
                ) from exc
            error = next((status for status in statuses if status is not None), None)
        if error is not None:
            if backend is None:
                raise original_error
            raise RuntimeError(f"Distributed prediction finalization failed: {error}")

        if not ddp or self._rank == 0:
            try:
                merge_shards(self, self._world if ddp else 1)
            except Exception as exc:
                original_error = exc
                error = f"{type(exc).__name__}: {exc}"
        if backend is not None:
            status = [error]
            try:
                dist.broadcast_object_list(status, src=0)
            except Exception as exc:
                raise RuntimeError(
                    f"Distributed prediction finalization failed: merge-status exchange: {exc}"
                ) from exc
            error = status[0]
        if error is not None:
            if backend is None:
                raise original_error
            raise RuntimeError(f"Distributed prediction finalization failed: {error}")

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

            def forward(self, x):
                return self.model(x)

            def predict(self, x):
                return self.forward(x)

        wrapped_model = WrappedModel(model)
        wrapped_model.eval()
        model = wrapped_model
    elif model is None:
        self._log("⚠️ No model provided - using placeholder interpolation model.")
        placeholder_model = SRModelPL(factor=factor)
        placeholder_model.eval()
        model = placeholder_model
    else:
        raise NotImplementedError(
            "Model must be a LightningModule, torch.nn.Module, or None."
        )

    # set model variables
    model.log_dir = self.log_dir  # set log dir for previews
    model.model_dm_length = len(
        self.datamodule.dataset
    )  # set datamodule length for preview interval
    model = _attach_hooks(model)
    model._save_temp_folder = temp_folder
    model._save_windows = windows
    model._save_factor = factor
    model._save_output_units = units
    return model
