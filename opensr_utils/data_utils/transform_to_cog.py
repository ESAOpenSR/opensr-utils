import argparse
import os
from pathlib import Path

RASTERIO_CONFIG = {
    "NUM_THREADS": "ALL_CPUS",
    "GDAL_TIFF_OVR_BLOCKSIZE": "512",
}


def _cog_profile():
    from rio_cogeo.profiles import cog_profiles

    profile = cog_profiles.get("deflate").copy()
    profile.update({
        "blocksize": 512,
        "compress": "deflate",
        "bigtiff": "IF_SAFER",
    })
    return profile


def make_cog_rio(in_tif: Path, out_tif: Path):
    import rasterio as rio
    from rasterio.enums import Resampling
    from rasterio.shutil import copy as rio_copy

    profile = _cog_profile()
    with rio.Env(**RASTERIO_CONFIG):
        rio_copy(
            str(in_tif),
            str(out_tif),
            driver="COG",
            compress=profile["compress"],
            blockxsize=profile["blocksize"],
            blockysize=profile["blocksize"],
            bigtiff=profile["bigtiff"],
            overview_resampling=Resampling.average,
            num_threads="all_cpus",
            tiled=True,
        )


def make_cog_rcg(in_tif: Path, out_tif: Path):
    import rasterio as rio
    from rio_cogeo.cogeo import cog_translate

    with rio.Env(**RASTERIO_CONFIG):
        cog_translate(
            str(in_tif),
            str(out_tif),
            _cog_profile(),
            in_memory=False,
            web_optimized=False,
            overview_level=None,
            overview_resampling="average",
            forward_band_tags=True,
            config=RASTERIO_CONFIG,
        )


def make_cog(in_tif: Path, out_tif: Path):
    try:
        make_cog_rio(in_tif, out_tif)
    except Exception:
        make_cog_rcg(in_tif, out_tif)


def readable_with_checksums(path: Path) -> bool:
    import rasterio as rio

    with rio.open(path) as ds:
        _ = [ds.checksum(b) for b in ds.indexes]
        return ds.width > 0 and ds.height > 0 and ds.count > 0


def process_tree(root: Path, remove_original: bool = False, dry_run: bool = False):
    from rio_cogeo.cogeo import cog_validate
    from tqdm import tqdm

    counter_success = 0
    counter_fail = 0

    with tqdm(root.rglob("sr.tif"), desc="Processing TIFFs to COGs", unit="file") as pbar:
        for in_tif in pbar:
            out_tif = in_tif.with_name(in_tif.stem + "_cog.tif")

            if dry_run:
                print(f"Found: {in_tif}")
                print(f"Would create: {out_tif}")
                continue

            try:
                make_cog(in_tif, out_tif)
                is_valid, errors, warnings = cog_validate(out_tif)
                io_ok = readable_with_checksums(out_tif)

                if is_valid and io_ok:
                    if remove_original:
                        os.remove(in_tif)
                    counter_success += 1
                else:
                    counter_fail += 1
                    if errors:
                        print("  Errors:", *errors, sep="\n    - ")
                    if warnings:
                        print("  Warnings:", *warnings, sep="\n    - ")
            except Exception as e:
                counter_fail += 1
                print(f"Error processing {in_tif}: {e}")

            pbar.set_postfix(success=counter_success, fail=counter_fail)

    return counter_success, counter_fail


def main():
    parser = argparse.ArgumentParser(description="Convert sr.tif outputs below a directory to COGs.")
    parser.add_argument("root", type=Path, help="Root directory to search recursively")
    parser.add_argument("--remove_original", action="store_true", help="Delete sr.tif after a validated COG is written")
    parser.add_argument("--dry_run", action="store_true", help="Print planned work without writing files")
    args = parser.parse_args()

    ok, fail = process_tree(args.root, remove_original=args.remove_original, dry_run=args.dry_run)
    print(f"Summary: OK={ok} FAIL={fail}")


if __name__ == "__main__":
    main()
