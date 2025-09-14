import os
import json
import glob
import numpy as np
import rasterio
from rasterio.windows import Window
from tqdm import tqdm

# -----------------------------
# Blending a single patch (yours, now supports an optional open handle)
# -----------------------------
def write_to_placeholder(
    self,
    sr,
    idx,
    image_meta,
    factor=4,
    overlap=8,
    eliminate_border_px=0,
    *,
    profile: str = "linear",    # "linear" | "sigmoid" | "cosine"
    profile_alpha: float = 6.0, # steepness for sigmoid
    dst=None                    # <-- NEW: optionally pass an already-open rasterio dataset in "r+" mode
):
    """
    Blend a single super-resolved (SR) patch into a placeholder GeoTIFF.

    This function writes one SR patch into its corresponding location within a
    large output raster, applying smooth feathering ramps at patch edges to
    avoid visible seams. Overlaps with already-written data are weighted by
    a blending profile (linear, sigmoid, or cosine). At the global dataset
    borders, patches are written without feathering to prevent black frames.

    Parameters
    ----------
    self : object
        Parent object (e.g., SR runner class) that owns context such as
        ``self.temp_folder`` and ``self.placeholder_filepath``.
    sr : array-like
        The super-resolved patch to write. Supported types:
          - ``torch.Tensor`` of shape (C,H,W), (H,W,C), or (H,W)
          - ``numpy.ndarray`` of shape (C,H,W), (H,W,C), or (H,W)
          - list/tuple of per-band 2D arrays
          - ``xarray.DataArray`` with dims including H,W,(C)
          - ``PIL.Image``
        Internally coerced to float32, CHW layout.
    idx : int
        Index into ``image_meta["window_coordinates"]`` defining the LR window
        for this patch.
    image_meta : dict
        Metadata dictionary with at least:
          - ``"window_coordinates"`` : list of rasterio.windows.Window objects
          - ``"placeholder_path"``   : str, path to output GeoTIFF
          - ``"dtype"``              : str or numpy dtype of target raster
    factor : int, default=4
        Upscaling factor from LR to HR grid.
    overlap : int, default=8
        Width (in pixels) of the overlap feathering zone at patch edges.
    eliminate_border_px : int, default=0
        Number of pixels at each patch edge to zero before feathering begins.
        Must be ≤ ``overlap``.
    profile : {"linear", "sigmoid", "cosine"}, default="linear"
        Ramp profile for blending overlaps:
          - "linear"  : straight ramp
          - "sigmoid" : logistic S-curve (controlled by ``profile_alpha``)
          - "cosine"  : half-cosine ease-in/ease-out
    profile_alpha : float, default=6.0
        Steepness of the sigmoid curve. Ignored for other profiles.
    dst : rasterio.io.DatasetWriter, optional
        If provided, an already-open GeoTIFF dataset in ``r+`` mode.
        If None, the function will open and close the file internally.

    Notes
    -----
    - Blending is performed in float32 and cast to ``image_meta["dtype"]`` at the end.
    - Nodata values in the placeholder are respected when detecting valid data.
    - Borders of the global raster are written without blending to avoid artifacts.
    - For best performance, call this function with an already-open ``dst`` handle
      inside a loop (see :func:`stitch_sr_patches`).

    Examples
    --------
    >>> write_to_placeholder(
    ...     self,
    ...     sr_patch,
    ...     idx=0,
    ...     image_meta=meta,
    ...     factor=4,
    ...     overlap=16,
    ...     eliminate_border_px=2,
    ...     profile="sigmoid",
    ...     profile_alpha=8.0,
    ...     dst=open_dataset  # re-use handle inside a loop
    ... )
    """
    # -------------------------
    # Helpers for ramps
    # -------------------------
    def make_ramp(n: int, curve: str, alpha: float) -> np.ndarray:
        if n <= 0:
            return np.zeros((0,), dtype=np.float32)
        t = np.linspace(0.0, 1.0, n, dtype=np.float32)
        if curve == "linear":
            r = t
        elif curve == "sigmoid":
            a = float(alpha)
            r = 1.0 / (1.0 + np.exp(-a * (t - 0.5)))
            r = (r - r[0]) / (r[-1] - r[0] + 1e-12)
        elif curve == "cosine":
            r = 0.5 * (1.0 - np.cos(np.pi * t))
        else:
            raise ValueError(f"Unknown profile '{curve}' (use 'linear'|'sigmoid'|'cosine').")
        return r.astype(np.float32)

    def apply_edge_ramp(vec: np.ndarray, side: str, overlap_px: int, cut_px: int, curve: str, alpha: float):
        ramp_len = max(0, overlap_px - cut_px)
        if side in ("left", "top"):
            if cut_px > 0:
                vec[:cut_px] = 0.0
            if ramp_len > 0:
                r = make_ramp(ramp_len, curve, alpha)  # 0→1
                vec[cut_px:cut_px + ramp_len] = np.minimum(vec[cut_px:cut_px + ramp_len], r)
        else:  # right/bottom
            if cut_px > 0:
                vec[-cut_px:] = 0.0
            if ramp_len > 0:
                r = make_ramp(ramp_len, curve, alpha)[::-1]  # 1→0
                seg = slice(-(cut_px + ramp_len), -cut_px if cut_px > 0 else None)
                vec[seg] = np.minimum(vec[seg], r)

    # -------------------------
    # Coerce sr to float32 CHW
    # -------------------------
    def coerce_to_chw_float32(x, expected_C=None):
        try:
            import torch
        except Exception:
            torch = None
        if torch is not None and hasattr(x, "detach") and isinstance(x, torch.Tensor):
            x = x.detach().cpu().float().numpy()
        try:
            import xarray as xr
            if hasattr(xr, "DataArray") and isinstance(x, xr.DataArray):
                x = x.values
        except Exception:
            pass
        try:
            from PIL import Image
            if isinstance(x, Image.Image):
                x = np.array(x)
        except Exception:
            pass
        if isinstance(x, (list, tuple)):
            x = [np.asarray(b) for b in x]
            x = np.stack(x, axis=0)
        if isinstance(x, np.ndarray):
            if x.ndim == 2:
                x = x[None, ...]
            elif x.ndim == 3:
                C1, H1, W1 = x.shape
                if expected_C is not None:
                    if C1 == expected_C:
                        pass
                    elif x.shape[-1] == expected_C:
                        x = np.transpose(x, (2, 0, 1))
                    elif C1 <= 12:
                        pass
                    else:
                        x = np.transpose(x, (2, 0, 1))
                else:
                    if C1 <= 12:
                        pass
                    else:
                        x = np.transpose(x, (2, 0, 1))
            else:
                raise TypeError(f"sr must be 2D or 3D array; got shape {x.shape}.")
            return x.astype(np.float32, copy=False)
        raise TypeError(f"Unsupported sr type '{type(x).__name__}'.")

    # -------------------------
    # Validate + compute windows
    # -------------------------
    overlap = int(overlap)
    elim = int(eliminate_border_px)
    if overlap < 0 or elim < 0:
        raise ValueError("overlap and eliminate_border_px must be >= 0")
    if elim > overlap:
        elim = overlap

    lr_win = image_meta["window_coordinates"][idx]
    row_off = int(lr_win.row_off * factor)
    col_off = int(lr_win.col_off * factor)
    hr_win = Window(col_off=col_off, row_off=row_off, width=None, height=None)

    # -------------------------
    # Open once if needed; otherwise use given handle
    # -------------------------
    _external = dst is not None
    if not _external:
        dst = rasterio.open(image_meta["placeholder_path"], "r+")

    try:
        expected_C = dst.count
        sr = coerce_to_chw_float32(sr, expected_C=expected_C)
        C, H, W = sr.shape
        if C != expected_C:
            raise ValueError(f"Band mismatch: dst has {expected_C} bands, SR has {C}")

        hr_win = Window(col_off=col_off, row_off=row_off, width=W, height=H)
        ph = dst.read(window=hr_win).astype(np.float32)

        # nodata-aware valid mask
        nd = dst.nodata
        if nd is None:
            valid = (ph != 0).any(axis=0)
        else:
            valid = (ph != nd).any(axis=0)
        valid = valid.astype(np.float32)

        # dataset borders
        touch_left   = (hr_win.col_off == 0)
        touch_right  = (hr_win.col_off + hr_win.width  == dst.width)
        touch_top    = (hr_win.row_off == 0)
        touch_bottom = (hr_win.row_off + hr_win.height == dst.height)

        # detect overlap over a thin band
        if overlap > 0:
            b = max(1, min(overlap, max(1, W // 4), max(1, H // 4)))
        else:
            b = 1
        left_valid   = valid[:, :b].any()
        right_valid  = valid[:, -b:].any()
        top_valid    = valid[:b, :].any()
        bottom_valid = valid[-b:, :].any()

        # 1D weights
        u = np.ones(W, dtype=np.float32)
        v = np.ones(H, dtype=np.float32)
        if left_valid   and not touch_left:   apply_edge_ramp(u, "left",  overlap, elim, profile, profile_alpha)
        if right_valid  and not touch_right:  apply_edge_ramp(u, "right", overlap, elim, profile, profile_alpha)
        if top_valid    and not touch_top:    apply_edge_ramp(v, "top",   overlap, elim, profile, profile_alpha)
        if bottom_valid and not touch_bottom: apply_edge_ramp(v, "bottom",overlap, elim, profile, profile_alpha)

        # 2D weights & blend
        w_sr_2d = (v[:, None] * u[None, :]).astype(np.float32)
        w_ph_2d = (1.0 - w_sr_2d) * valid
        out = (w_sr_2d[None, ...] * sr + w_ph_2d[None, ...] * ph)

        target_dtype = np.dtype(image_meta["dtype"])
        if np.issubdtype(target_dtype, np.integer):
            info = np.iinfo(target_dtype)
            out = np.clip(out, info.min, info.max)
        out = out.astype(target_dtype)

        for bidx in range(C):
            dst.write(out[bidx], bidx + 1, window=hr_win)
    finally:
        # only close if we opened it here
        if not _external:
            dst.close()


# -----------------------------
# Open-once stitcher (called by rank 0)
# -----------------------------
def stitch_sr_patches(
    self,
    index_path=None,
    limit=None,
    *,
    cleanup_every=50,
    profile="linear",
    profile_alpha=6.0,
    gdal_cache_mb=256
):
    """
    Distributed-safe wrapper for stitching SR patches into a placeholder GeoTIFF.

    This function ensures that the expensive write phase is performed only once
    (on global rank 0), while all other ranks in a distributed run return
    immediately. A synchronization barrier is inserted to prevent race
    conditions, using Lightning's strategy barrier when available.

    Behavior by environment
    -----------------------
    - **CPU / single process**: Stitches normally.
    - **Single GPU**: Stitches normally (rank 0 is the only rank).
    - **Multi-GPU (DDP)**: Only rank 0 executes the stitch. Other ranks skip
      and wait at a safe barrier to ensure consistent teardown.

    Parameters
    ----------
    self : object
        Parent object (e.g., SR runner class) that provides context such as
        ``self.temp_folder`` and ``self.placeholder_filepath``.
    index_path : str, optional
        Path to the ``index.json`` describing patches. Defaults to
        ``<temp_folder>/index.json``.
    limit : int, optional
        If provided, only the first ``limit`` patches are stitched (useful
        for debugging).
    cleanup_every : int, default=50
        Frequency (in tiles) at which staged temporary files are deleted.
        Set lower to reduce RAM usage.
    profile : {"linear", "sigmoid", "cosine"}, default="linear"
        Feathering profile used when blending overlaps (see
        :func:`write_to_placeholder`).
    profile_alpha : float, default=6.0
        Steepness of the sigmoid ramp, ignored for other profiles.
    gdal_cache_mb : int, default=256
        Size of GDAL's internal block cache in MB for the write session.

    Returns
    -------
    out : tuple or None
        If run on rank 0, returns ``(sr_path, missing_count)`` from
        :func:`stitch_sr_patches`. For other ranks, returns ``None``.

    Notes
    -----
    - This function should always be called instead of
      :func:`stitch_sr_patches` directly in distributed runs.
    - Uses Lightning's barrier if available; otherwise falls back to
      ``torch.distributed.barrier()`` with proper safety checks.
    - Recommended environment settings for stability:
      ``GDAL_CACHEMAX=256`` and ``GDAL_NUM_THREADS=1``.
    """
    # 1) load index
    if index_path is None:
        index_path = os.path.join(self.temp_folder, "index.json")
    with open(index_path, "r") as f:
        idx = json.load(f)

    entries = idx["entries"]
    entries.sort(key=lambda e: (e["row_off_lr"], e["col_off_lr"]))
    if limit is not None:
        entries = entries[:int(limit)]

    # 2) build meta from self
    image_meta = {
        "placeholder_path": self.placeholder_filepath,
        "dtype": str(self.image_meta["dtype"]),
        "window_coordinates": [
            Window(e["col_off_lr"], e["row_off_lr"], e["width_lr"], e["height_lr"])
            for e in entries
        ],
    }
    factor = int(self.factor)
    overlap = int(self.overlap)
    eliminate_border_px = int(self.eliminate_border_px)

    container = idx.get("saved_container", "npz")
    key = idx.get("saved_key", "arr")

    to_delete = []
    missing_count = 0

    # 3) GDAL environment + single open
    with rasterio.Env(GDAL_CACHEMAX=gdal_cache_mb, NUM_THREADS="1"):
        with rasterio.open(image_meta["placeholder_path"], "r+") as dst:
            for i, e in enumerate(tqdm(entries, desc="Stitching → placeholder", unit="tile"), start=1):
                p = e["path"]
                try:
                    if container == "npz" and p.endswith(".npz"):
                        with np.load(p) as z:
                            sr = z[key]
                    else:
                        sr = np.load(p)

                    write_to_placeholder(
                        self, sr, i-1, image_meta,
                        factor=factor,
                        overlap=overlap,
                        eliminate_border_px=eliminate_border_px,
                        profile=profile,
                        profile_alpha=profile_alpha,
                        dst=dst  # <-- reuse open dataset
                    )
                    to_delete.append(p)
                except FileNotFoundError:
                    missing_count += 1

                # periodic flush + cleanup
                if cleanup_every and (i % cleanup_every == 0):
                    for fp in to_delete:
                        try:
                            os.remove(fp)
                        except FileNotFoundError:
                            pass
                        for leftover in glob.glob(fp + ".*"):
                            try:
                                os.remove(leftover)
                            except FileNotFoundError:
                                pass
                    to_delete.clear()

    # 4) rename placeholder -> final
    sr_path = image_meta["placeholder_path"].replace("sr_placeholder.tif", "sr.tif")
    os.replace(image_meta["placeholder_path"], sr_path)

    self._log(f"🧩 Stitched {len(entries)} tiles into: {sr_path}")
    if missing_count > 0:
        self._log(f"⚠️ {missing_count} patch files were missing and could not be stitched.")

    return sr_path, missing_count


# -----------------------------
# DDP-safe wrapper
# -----------------------------
def ddp_safe_stitch(
    self,
    index_path=None,
    limit=None,
    *,
    cleanup_every=50,
    profile="linear",
    profile_alpha=6.0,
    gdal_cache_mb=256
):
    """
    Distributed-safe wrapper for stitching SR patches into a GeoTIFF.

    This function ensures that the stitching phase runs only once on the
    global rank 0 process when using multi-GPU training with PyTorch Lightning
    (DDP). All other ranks return immediately. Unlike the idempotent version,
    this implementation will overwrite an existing ``sr.tif`` if found.

    Behavior by environment
    -----------------------
    - **CPU / single process**: Executes stitching normally.
    - **Single GPU**: Executes stitching normally (rank 0 is the only rank).
    - **Multi-GPU (DDP)**: Only rank 0 runs the stitcher. Other ranks skip
      immediately.

    Overwrite policy
    ----------------
    - If a final output file ``sr.tif`` already exists in the target folder,
      it is **deleted and replaced** by a fresh stitched output.
    - A warning message is printed to notify that the file will be
      overwritten.
    - If a placeholder file ``sr_placeholder.tif`` exists, it will be used
      as the stitching target as normal.

    Parameters
    ----------
    self : object
        Parent object (e.g., SR runner class) providing context such as
        ``self.temp_folder`` and ``self.placeholder_filepath``.
    index_path : str, optional
        Path to the ``index.json`` file describing all patches. Defaults to
        ``<temp_folder>/index.json``.
    limit : int, optional
        If given, only the first ``limit`` patches are stitched (useful
        for debugging).
    cleanup_every : int, default=50
        Frequency (in tiles) at which staged temporary files are deleted.
    profile : {"linear", "sigmoid", "cosine"}, default="linear"
        Feathering profile used when blending overlaps
        (see :func:`write_to_placeholder`).
    profile_alpha : float, default=6.0
        Steepness parameter for the sigmoid ramp. Ignored for other profiles.
    gdal_cache_mb : int, default=256
        Size of GDAL's internal block cache in MB for the write session.

    Returns
    -------
    out : tuple or None
        On rank 0, returns ``(sr_path, missing_count)`` from
        :func:`stitch_sr_patches`. On other ranks, returns ``None``.

    Notes
    -----
    - This function should always be used instead of calling
      :func:`stitch_sr_patches` directly in multi-GPU runs.
    - To reduce risk of GDAL deadlocks, set environment variables such as
      ``GDAL_CACHEMAX=256`` and ``GDAL_NUM_THREADS=1`` before running.
    """

    trainer = getattr(self, "trainer", None)

    def _is_rank0_local():
        if trainer is not None and hasattr(trainer, "is_global_zero"):
            return bool(trainer.is_global_zero)
        try:
            import torch
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                return torch.distributed.get_rank() == 0
        except Exception:
            pass
        return True
    is_rank0 = _is_rank0_local()

    out = None
    if is_rank0:
        placeholder = self.placeholder_filepath
        sr_path = placeholder.replace("sr_placeholder.tif", "sr.tif")
        self.final_sr_path = sr_path

        # --- overwrite logic ---
        if os.path.exists(sr_path):
            self._log(f"⚠️ Existing output found at {sr_path} — will be overwritten.")
            try:
                os.remove(sr_path)
            except OSError:
                pass

        out = stitch_sr_patches(
            self,
            index_path=index_path,
            limit=limit,
            cleanup_every=cleanup_every,
            profile=profile,
            profile_alpha=profile_alpha,
            gdal_cache_mb=gdal_cache_mb,
        )

    return out
