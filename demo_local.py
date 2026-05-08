#!/usr/bin/env python3
"""Minimal bilinear overlap demo for opensr-utils.

The demo converts resources/example_sat_img.png into a 640x640 LR GeoTIFF,
runs the real tiled pipeline with a 4x bilinear model for three overlap
settings, and adds a small random RGB color shift to each predicted SR patch.
The comparison PNG shows how overlap blending removes those patch-edge seams.
"""

from pathlib import Path

import numpy as np
import rasterio
import torch
from PIL import Image, ImageDraw
from rasterio.transform import from_origin

import opensr_utils


OUT_DIR = Path("playground/bilinear_cross_overlap_demo")
INPUT_PNG = Path("resources/example_sat_img.png")

NATIVE_SIZE = 640
TILE_SIZE = 128
FACTOR = 4
WINDOW_SIZE = (TILE_SIZE, TILE_SIZE)
CASES = (
    ("overlap_0", 0, 0),
    ("overlap_10", 10, 0),
    ("overlap_30_edge_10", 30, 10),
)

LABEL_HEIGHT = 44
PANEL_GUTTER = 48
COMPARISON_PNG = OUT_DIR / "comparison_input_and_srs.png"
PATCH_COLOR_SHIFT = 0.16
PATCH_COLOR_SEED = 20240508


class TintedBilinearSRModel(torch.nn.Module):
    """4x bilinear super-resolution baseline with per-patch RGB shifts."""

    def __init__(self, factor, color_shift=PATCH_COLOR_SHIFT, seed=PATCH_COLOR_SEED):
        super().__init__()
        self.factor = int(factor)
        self.color_shift = float(color_shift)
        self.generator = torch.Generator()
        self.generator.manual_seed(int(seed))

    def forward(self, x):
        sr = torch.nn.functional.interpolate(
            x,
            scale_factor=self.factor,
            mode="bilinear",
            align_corners=False,
        )
        if sr.shape[1] == 1:
            sr = sr.repeat(1, 3, 1, 1)
        elif sr.shape[1] > 3:
            sr = sr[:, :3]

        shifts = torch.empty(
            (sr.shape[0], sr.shape[1], 1, 1),
            dtype=sr.dtype,
            device="cpu",
        )
        shifts.uniform_(-self.color_shift, self.color_shift, generator=self.generator)
        shifts = shifts.to(sr.device)
        return torch.clamp(sr + shifts, 0.0, 1.0)


def load_input_image():
    image = Image.open(INPUT_PNG).convert("RGB")
    if image.size != (NATIVE_SIZE, NATIVE_SIZE):
        image = image.resize((NATIVE_SIZE, NATIVE_SIZE), Image.Resampling.BICUBIC)
    return np.asarray(image, dtype=np.float32) / 255.0


def write_input_tif(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    if image.ndim == 2:
        data = image[None, ...]
    else:
        data = np.moveaxis(image[:, :, :3], -1, 0)
    data = np.round(data * 10000).astype(np.uint16)
    profile = {
        "driver": "GTiff",
        "height": image.shape[0],
        "width": image.shape[1],
        "count": data.shape[0],
        "dtype": "uint16",
        "crs": "EPSG:3857",
        "transform": from_origin(0, image.shape[0], 1, 1),
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)


def read_tif_unit(path):
    with rasterio.open(path) as src:
        data = np.clip(src.read().astype(np.float32) / 10000.0, 0.0, 1.0)
    if data.shape[0] == 1:
        return data[0]
    return np.moveaxis(data[:3], 0, -1)


def unit_image_to_pil(image):
    arr = (np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)
    if arr.ndim == 2:
        return Image.fromarray(arr, mode="L")
    return Image.fromarray(arr[:, :, :3], mode="RGB")


def run_case(name, overlap, eliminate_border_px, image):
    case_dir = OUT_DIR / name
    input_tif = case_dir / "input_lr_example_sat.tif"

    write_input_tif(input_tif, image)

    runner = opensr_utils.large_file_processing(
        root=str(input_tif),
        model=TintedBilinearSRModel(factor=FACTOR),
        window_size=WINDOW_SIZE,
        factor=FACTOR,
        overlap=overlap,
        eliminate_border_px=eliminate_border_px,
        device="cpu",
        gpus=None,
        batch_size=8,
        num_workers=0,
        cleanup=True,
        overwrite=True,
    )

    sr_tif = runner.run()
    sr = read_tif_unit(sr_tif)

    return {
        "name": name,
        "overlap": overlap,
        "eliminate_border_px": eliminate_border_px,
        "window_count": len(runner.image_meta["image_windows"]),
        "sr_tif": Path(sr_tif),
        "panel": labeled_panel(
            sr,
            f"Tinted 4x bilinear SR, overlap={overlap} px, edge removal={eliminate_border_px} px",
        ),
    }


def labeled_panel(image, label):
    panel = Image.new("RGB", (NATIVE_SIZE, NATIVE_SIZE + LABEL_HEIGHT), "white")
    draw = ImageDraw.Draw(panel)
    draw.text((12, 14), label, fill=(0, 0, 0))
    preview = unit_image_to_pil(image)
    if preview.size != (NATIVE_SIZE, NATIVE_SIZE):
        preview = preview.resize((NATIVE_SIZE, NATIVE_SIZE), Image.Resampling.BOX)
    panel.paste(preview.convert("RGB"), (0, LABEL_HEIGHT))
    return panel


def remove_stale_pngs():
    if not OUT_DIR.exists():
        return
    for path in OUT_DIR.rglob("*.png"):
        path.unlink()


def save_comparison(input_image, rows):
    panels = [
        labeled_panel(input_image, f"Input LR native ({NATIVE_SIZE}x{NATIVE_SIZE})")
    ]
    for row in rows:
        panels.append(row["panel"])

    comparison = Image.new(
        "RGB",
        (
            NATIVE_SIZE * len(panels) + PANEL_GUTTER * (len(panels) - 1),
            NATIVE_SIZE + LABEL_HEIGHT,
        ),
        "white",
    )
    x = 0
    for idx, panel in enumerate(panels):
        comparison.paste(panel, (x, 0))
        x += NATIVE_SIZE + PANEL_GUTTER
        if idx < len(panels) - 1:
            draw = ImageDraw.Draw(comparison)
            draw.line(
                [(x - PANEL_GUTTER // 2, 0), (x - PANEL_GUTTER // 2, comparison.height)],
                fill=(220, 220, 220),
                width=1,
            )
    comparison.save(COMPARISON_PNG)


def main():
    remove_stale_pngs()
    image = load_input_image()

    rows = []
    for name, overlap, eliminate_border_px in CASES:
        row = run_case(name, overlap, eliminate_border_px, image)
        rows.append(row)
        print(f"{name}")
        print(f"  overlap:       {overlap}px")
        print(f"  edge removal:  {eliminate_border_px}px")
        print(f"  LR input:      {NATIVE_SIZE}x{NATIVE_SIZE}px")
        print(f"  SR output:     {NATIVE_SIZE * FACTOR}x{NATIVE_SIZE * FACTOR}px")
        print(f"  LR tile size:  {TILE_SIZE}x{TILE_SIZE}px")
        print(f"  windows:       {row['window_count']}")
        print(f"  Export:        {row['sr_tif']}")

    save_comparison(image, rows)
    print(f"Comparison PNG: {COMPARISON_PNG}")


if __name__ == "__main__":
    main()
