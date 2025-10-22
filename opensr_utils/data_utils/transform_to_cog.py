from tqdm import tqdm
import os
from pathlib import Path
import rasterio as rio
from rasterio.shutil import copy as rio_copy
from rasterio.enums import Resampling
from rio_cogeo.cogeo import cog_validate, cog_translate
from rio_cogeo.profiles import cog_profiles

ROOT = Path("/data2/simon/1msqkm/deep/tiles/done/")
DEBUG = False # Set to True to enable debug mode (no file writing)
REMOVE_ORIGINAL = True # Set to True to delete original files after successful COG creation

COG_PROFILE = cog_profiles.get("deflate").copy()
COG_PROFILE.update({
    "blocksize": 512,
    "compress": "deflate",
    "bigtiff": "IF_SAFER",
})

RASTERIO_CONFIG = {
    "NUM_THREADS": "ALL_CPUS",
    "GDAL_TIFF_OVR_BLOCKSIZE": "512",
}

# Use enum for Rasterio path, *string* for rio-cogeo path
OVR_ENUM = Resampling.average
OVR_STR = "average"

def make_cog_rio(in_tif: Path, out_tif: Path):
    with rio.Env(**RASTERIO_CONFIG):
        rio_copy(
            str(in_tif), str(out_tif),
            driver="COG",
            compress=COG_PROFILE["compress"],
            blockxsize=COG_PROFILE["blocksize"],
            blockysize=COG_PROFILE["blocksize"],
            bigtiff=COG_PROFILE["bigtiff"],
            overview_resampling=OVR_ENUM,   # enum OK here
            num_threads="all_cpus",
            tiled=True,
        )

def make_cog_rcg(in_tif: Path, out_tif: Path):
    with rio.Env(**RASTERIO_CONFIG):
        cog_translate(
            str(in_tif), str(out_tif),
            COG_PROFILE,
            in_memory=False,
            web_optimized=False,
            overview_level=None,               # auto
            overview_resampling=OVR_STR,       # <- string here fixes your error
            forward_band_tags=True,
            config=RASTERIO_CONFIG,
        )

def make_cog(in_tif: Path, out_tif: Path):
    try:
        make_cog_rio(in_tif, out_tif)
    except Exception:
        make_cog_rcg(in_tif, out_tif)

def readable_with_checksums(path: Path) -> bool:
    with rio.open(path) as ds:
        _ = [ds.checksum(b) for b in ds.indexes]
        return ds.width > 0 and ds.height > 0 and ds.count > 0


counter_success = 0
counter_fail = 0

with tqdm(ROOT.rglob("sr.tif"), desc="Processing TIFFs to COGs", unit="file") as pbar:
    for in_tif in pbar:
        out_tif = in_tif.with_name(in_tif.stem + "_cog.tif")

        if DEBUG:
            print(f"   [DEBUG] 🔍 Found file: {in_tif}")
            print(f"   [DEBUG] → Will create: {out_tif}")
            print(f"   [DEBUG] Using pip-only Rasterio/rio-cogeo workflow (no gdal_translate).")
            continue

        try:
            make_cog(in_tif, out_tif)

            is_valid, errors, warnings = cog_validate(out_tif)
            io_ok = readable_with_checksums(out_tif)

            if is_valid and io_ok:
                if REMOVE_ORIGINAL:
                    os.remove(in_tif)
                counter_success += 1
            else:
                counter_fail += 1
                if errors:   print("  Errors:", *errors, sep="\n    - ")
                if warnings: print("  Warnings:", *warnings, sep="\n    - ")
        except Exception as e:
            counter_fail += 1
            print(f"!! Error processing {in_tif}: {e}")

        pbar.set_postfix(success=counter_success, fail=counter_fail)    

