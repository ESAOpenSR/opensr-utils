import numpy as np
import rasterio
from rasterio.windows import Window


def write_to_placeholder(self, sr, idx, image_meta, factor=4, overlap=8, eliminate_border_px=0):
    """
    Blend-write SR patch into the placeholder with linear feathering on overlapping edges
    and optional border-pixel elimination.
    Assumes sr is (C,H,W) either np.ndarray or torch.Tensor on CPU in [0,1] (or your scale).
    """
    # --- Inputs ---
    # info dict keys used:
    # - window_coordinates: list[Window] at LR grid
    # - sr_path: path to placeholder GeoTIFF
    # - factor: LR->HR scale (e.g., 4)
    # - overlap: even int (e.g., 10)
    # - eliminate_border_px: even int in [0..4]
    # - dtype: target dtype string (e.g., "uint16")
    factor = factor
    overlap = int(overlap)
    elim = int(eliminate_border_px)

    # tensor -> numpy
    if hasattr(sr, "detach"):
        sr = sr.detach().cpu().numpy()
    sr = np.asarray(sr)
    C, H, W = sr.shape

    # scale + dtype cast to match placeholder (keep if you store L2A-scaled)
    sr = sr.astype(image_meta["dtype"], copy=False)

    # compute HR window from LR window
    lr_win = image_meta["window_coordinates"][idx]
    row_off = int(lr_win.row_off * factor)
    col_off = int(lr_win.col_off * factor)
    hr_win = Window(col_off=col_off, row_off=row_off, width=W, height=H)

    with rasterio.open(image_meta["placeholder_path"], "r+") as dst:
        if dst.count != C:
            raise ValueError(f"Band mismatch: dst has {dst.count}, SR has {C}")

        # read placeholder patch (C,H,W)
        ph = dst.read(window=hr_win)

        # validity mask: where placeholder already has data in ANY band
        # (If your background is 0; adjust if you use a nodata value.)
        valid = (ph != 0).any(axis=0).astype(np.uint8)  # (H,W)

        # detect which sides overlap existing data by checking the **outermost** row/col
        left_valid   = valid[:, 0].any()
        right_valid  = valid[:, -1].any()
        top_valid    = valid[0, :].any()
        bottom_valid = valid[-1, :].any()

        # build separable 1D ramps (width=W, height=H)
        # start with all-ones (no taper), then apply ramps on overlapping sides
        u = np.ones(W, dtype=np.float32)  # horizontal weights
        v = np.ones(H, dtype=np.float32)  # vertical weights

        def apply_edge_ramp(vec, length, side, overlap_px, cut_px):
            """In-place ramp on a 1D vector (length), for given side:
               side='left'|'right'|'top'|'bottom'. overlap_px >= cut_px."""
            ramp_len = max(0, overlap_px - cut_px)
            if side in ("left", "top"):
                # eliminate: hard-zero the first cut_px (SR suppressed)
                if cut_px > 0:
                    vec[:cut_px] = 0.0
                # feather: linear 0→1 over ramp_len
                if ramp_len > 0:
                    r = np.linspace(0.0, 1.0, ramp_len, endpoint=True, dtype=np.float32)
                    vec[cut_px:cut_px + ramp_len] = np.maximum(vec[cut_px:cut_px + ramp_len], r)
            else:  # right / bottom
                if cut_px > 0:
                    vec[-cut_px:] = 0.0
                if ramp_len > 0:
                    r = np.linspace(1.0, 0.0, ramp_len, endpoint=True, dtype=np.float32)
                    vec[-(cut_px + ramp_len):-cut_px if cut_px > 0 else None] = np.maximum(
                        vec[-(cut_px + ramp_len):-cut_px if cut_px > 0 else None], r
                    )

        # apply ramps ONLY on sides that overlap existing data
        if left_valid:
            apply_edge_ramp(u, W, "left",  overlap_px=overlap, cut_px=elim)
        if right_valid:
            apply_edge_ramp(u, W, "right", overlap_px=overlap, cut_px=elim)
        if top_valid:
            apply_edge_ramp(v, H, "top",   overlap_px=overlap, cut_px=elim)
        if bottom_valid:
            apply_edge_ramp(v, H, "bottom",overlap_px=overlap, cut_px=elim)

        # make 2D SR weight by separable product (strong near center, tapered near any overlapping edge)
        w_sr_2d = (v[:, None] * u[None, :]).astype(np.float32)        # (H,W)
        # placeholder weight only where placeholder is valid
        w_ph_2d = (1.0 - w_sr_2d) * valid

        # expand to channels
        w_sr = np.broadcast_to(w_sr_2d, (C, H, W))
        w_ph = np.broadcast_to(w_ph_2d, (C, H, W))

        # blend
        out = (w_sr * sr.astype(np.float32) + w_ph * ph.astype(np.float32)).astype(image_meta["dtype"])

        # write back
        for b in range(C):
            dst.write(out[b], b + 1, window=hr_win)