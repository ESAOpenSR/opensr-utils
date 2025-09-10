import numpy as np
import rasterio
from rasterio.windows import Window
 
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
    profile_alpha: float = 6.0  # steepness for sigmoid
):
    """
    Write a super-resolved (SR) patch into a placeholder GeoTIFF with smooth
    feathering along overlapping edges.

    This function blends a single SR patch into a large output raster using
    separable horizontal and vertical weight ramps. Overlaps with already-
    written patches are linearly or nonlinearly feathered to avoid visible seams.
    Edges at the **global dataset border** are written without feathering to
    prevent black frames.

    Parameters
    ----------
    sr : array-like
        Super-resolved patch. Accepted formats:
          - torch.Tensor (C,H,W), (H,W,C), or (H,W)
          - numpy.ndarray (C,H,W), (H,W,C), or (H,W)
          - list/tuple of per-band 2D arrays
          - xarray.DataArray with dims including H,W,(C)
          - PIL.Image (converted to HWC)
        The patch is internally coerced to float32 CHW for blending.

    idx : int
        Index into ``image_meta["window_coordinates"]`` that defines the LR window
        corresponding to this patch.

    image_meta : dict
        Metadata dict containing at least:
          - ``"window_coordinates"`` : list of rasterio.windows.Window in LR grid
          - ``"placeholder_path"``   : str, path to the output GeoTIFF
          - ``"dtype"``              : str or numpy dtype, target raster dtype

    factor : int, default=4
        Scale factor from low-resolution to high-resolution grid.

    overlap : int, default=8
        Width (in pixels) of the overlap feathering zone at patch edges.

    eliminate_border_px : int, default=0
        Number of pixels at each patch edge to hard-zero (SR suppressed) before
        starting the feather ramp. Must be ≤ ``overlap``.

    profile : {"linear", "sigmoid", "cosine"}, default="linear"
        Shape of the feathering curve:
          - "linear"  : straight ramp
          - "sigmoid" : smooth logistic transition, steeper with larger alpha
          - "cosine"  : half-cosine ease-in/ease-out

    profile_alpha : float, default=6.0
        Steepness parameter for the sigmoid profile. Ignored otherwise.

    Notes
    -----
    - Feathering is only applied on sides that overlap *existing valid data* in
      the placeholder raster.
    - At the outermost dataset border, no ramping is applied to avoid black edges.
    - Blending is performed in float32 and cast once at the end to ``image_meta["dtype"]``.
    - ``nodata`` values in the placeholder are respected when detecting valid data.

    Examples
    --------
    >>> write_to_placeholder(
    ...     sr_patch, i, meta,
    ...     factor=4,
    ...     overlap=16,
    ...     eliminate_border_px=2,
    ...     profile="sigmoid",
    ...     profile_alpha=8.0
    ... )
    """
    
    
    import numpy as np
    import rasterio
    from rasterio.windows import Window

    # -------------------------
    # Helper: build 1D ramp
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
        import numpy as np
        try:
            import torch
        except Exception:
            torch = None
        if torch is not None and hasattr(x, "detach") and isinstance(x, torch.Tensor):
            x = x.detach().cpu().float().numpy()
        try:
            import xarray as xr
            if isinstance(x, xr.DataArray):
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
    # Validate inputs
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

    with rasterio.open(image_meta["placeholder_path"], "r+") as dst:
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

        # --- detect if patch touches dataset border ---
        touch_left   = (hr_win.col_off == 0)
        touch_right  = (hr_win.col_off + hr_win.width  == dst.width)
        touch_top    = (hr_win.row_off == 0)
        touch_bottom = (hr_win.row_off + hr_win.height == dst.height)

        # detect overlap over a small band
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

        # 2D weights
        w_sr_2d = (v[:, None] * u[None, :]).astype(np.float32)
        w_ph_2d = (1.0 - w_sr_2d) * valid

        # Blend
        out = (w_sr_2d[None, ...] * sr + w_ph_2d[None, ...] * ph)

        target_dtype = np.dtype(image_meta["dtype"])
        if np.issubdtype(target_dtype, np.integer):
            info = np.iinfo(target_dtype)
            out = np.clip(out, info.min, info.max)
        out = out.astype(target_dtype)

        for bidx in range(C):
            dst.write(out[bidx], bidx + 1, window=hr_win)
