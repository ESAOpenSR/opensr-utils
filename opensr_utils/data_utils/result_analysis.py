from typing import Tuple
import os
import numpy as np
import matplotlib.pyplot as plt
from rasterio.windows import Window
import rasterio
from skimage.transform import resize
from typing import Tuple


def stretch(arr: np.ndarray) -> np.ndarray:
    """Apply normalization and return RGB image (H, W, 3)"""
    if arr.max() > 256:
        arr = arr / 10000.0  # assuming input is in [0, 10000]
    rgb = arr[:3]
    rgb = rgb*3.5
    rgb = rgb.clip(0, 1)
    return np.clip(rgb.transpose(1, 2, 0), 0, 1)


def generate_side_by_side_previews(
    self,
    tif_path: str,
    out_dir: str,
    num_examples: int = 100,
    window_size: int = 2000,
    crop_size: int = 512,
    target_size: Tuple[int, int] = (512, 512),
):
    """Generate side-by-side PNG previews of random windows from a 4-band GeoTIFF."""
    os.makedirs(out_dir, exist_ok=True)

    with rasterio.open(tif_path) as src:
        W, H = src.width, src.height

        for i in range(num_examples):
            if W >= window_size and H >= window_size:
                x_off = np.random.randint(0, W - window_size)
                y_off = np.random.randint(0, H - window_size)
                win = Window(x_off, y_off, window_size, window_size)
                arr = src.read(window=win)
            else:
                arr = src.read()
                self._log(f"⚠️ Image too small for window, using full image for sample {i}")

            resized = resize(arr, (4, *target_size), order=1, preserve_range=True, anti_aliasing=True)

            h, w = arr.shape[1:]
            crop_h, crop_w = min(crop_size, h), min(crop_size, w)
            start_y = (h - crop_h) // 2
            start_x = (w - crop_w) // 2
            crop = arr[:, start_y:start_y+crop_h, start_x:start_x+crop_w]

            if crop.shape[1:] != target_size:
                pad = ((0, 0), (0, target_size[0] - crop.shape[1]), (0, target_size[1] - crop.shape[2]))
                crop = np.pad(crop, pad, mode='edge')

            left_img = stretch(resized)
            right_img = stretch(crop)

            fig, axes = plt.subplots(1, 2, figsize=(8, 4))
            axes[0].imshow(left_img)
            axes[0].set_title("Overview")
            axes[1].imshow(right_img)
            axes[1].set_title("Center Native Crop")
            for ax in axes:
                ax.axis('off')

            out_path = os.path.join(out_dir, f"example_{i:03d}.png")
            plt.tight_layout()
            plt.savefig(out_path, dpi=150)
            plt.close()
            self._log(f"✅ Saved preview PNG at {out_path}")


def crop_and_save_georeferenced_excerpt(
    self,
    tif_path: str,
    out_tif: str,
    random_crop_size: Tuple[int, int] = (512, 512),
) -> np.ndarray:
    """
    Randomly crops a window from the input GeoTIFF, saves it with proper geotransform, and returns a random crop from that window.

    Args:
        tif_path: Path to input .tif
        out_tif: Output .tif to save the cropped window
        patch_size: (width, height) of the window to extract from the full image
        random_crop_size: (width, height) of the in-memory random crop to return

    Returns:
        np.ndarray: (C, H, W) random crop from the extracted window
    """
    with rasterio.open(tif_path) as src:
        full_width, full_height = src.width, src.height
        patch_width, patch_height = random_crop_size

        # ✅ Ensure random window fits in image
        max_x = full_width - patch_width
        max_y = full_height - patch_height
        if max_x < 0 or max_y < 0:
            raise ValueError("Patch size larger than input image")

        # 🎲 Random top-left of patch
        x_off = np.random.randint(0, max_x + 1)
        y_off = np.random.randint(0, max_y + 1)

        win = Window(x_off, y_off, patch_width, patch_height)
        transform = src.window_transform(win)

        # 📥 Read and save cropped window
        meta = src.meta.copy()
        meta.update({
            "height": patch_height,
            "width": patch_width,
            "transform": transform
        })

        data = src.read(window=win)
        with rasterio.open(out_tif, 'w', **meta) as dst:
            dst.write(data)

    # ✂️ Extract random crop from saved patch
    C, H, W = data.shape
    crop_h, crop_w = random_crop_size
    if H < crop_h or W < crop_w:
        raise ValueError("Saved patch is too small for requested crop size")

    top = np.random.randint(0, H - crop_h + 1)
    left = np.random.randint(0, W - crop_w + 1)
    crop = data[:, top:top+crop_h, left:left+crop_w]

    self._log(f"✅ Saved georeferenced example SR patch of size {crop.shape} to {out_tif}")


if __name__ == "__main__":

    tif_path = "/data2/simon/mosaic/Q10_20240701_20240930_Global-S2GM-m36p8_STD_v2.0.5/S2GM_Q10_20240701_20240930_Global-S2GM-m36p8_STD_v2.0.5/tile_0/sr.tif"
    #tif_path = "/data2/simon/mosaic/individual_tile/sr.tif"

    crop_and_save_georeferenced_excerpt(
    tif_path=tif_path,
    out_tif="cropped_excerpt.tif",
    random_crop_size=(512, 512)
    )
    
    generate_side_by_side_previews(
    tif_path=tif_path,
    out_dir="playground/examples",
    num_examples=10
    )

