#!/usr/bin/env python3
"""Super-resolve all ten SEN2SR bands using an LDSR-S2 RGB-NIR backbone."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


def _source_checkout_import():
    """Allow ``python /checkout/examples/run_sen2sr.py`` from any working directory."""
    try:
        import opensr_utils  # noqa: F401
    except ModuleNotFoundError:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input", type=Path, help="GeoTIFF, SAFE directory/archive, or S2GM folder"
    )
    parser.add_argument("-o", "--output", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument(
        "--download", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--backbone-checkpoint", type=Path, help="Local LDSR-S2 weights"
    )
    parser.add_argument("--backbone-config", type=Path, help="Matching LDSR-S2 YAML")
    parser.add_argument(
        "--sampling-steps", type=int, help="Override the backbone's diffusion steps"
    )
    parser.add_argument(
        "--bands",
        type=int,
        nargs=10,
        help="Ten 1-based GeoTIFF indexes in B02,B03,B04,B05,B06,B07,B08,B8A,B11,B12 order",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--factor", type=int, default=4)
    parser.add_argument(
        "--window-size",
        type=int,
        nargs=2,
        default=(128, 128),
        metavar=("WIDTH", "HEIGHT"),
    )
    parser.add_argument("--overlap", type=int, default=12)
    parser.add_argument("--eliminate-border-px", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if (args.backbone_checkpoint is None) != (args.backbone_config is None):
        parser.error(
            "--backbone-checkpoint and --backbone-config must be supplied together"
        )
    if not args.download and (
        args.checkpoint is None or args.backbone_checkpoint is None
    ):
        parser.error(
            "--no-download requires local SEN2SR --checkpoint and both LDSR-S2 backbone artifacts"
        )

    _source_checkout_import()
    from opensr_utils import super_resolve
    from opensr_utils.model_utils.get_models import get_model

    model = get_model(
        "sen2sr",
        device="cpu",
        checkpoint=args.checkpoint,
        cache_dir=args.cache_dir,
        variant="main",
        backbone="ldsrs2",
        backbone_checkpoint=args.backbone_checkpoint,
        backbone_config=args.backbone_config,
        sampling_steps=args.sampling_steps,
        download=args.download,
    )
    result = super_resolve(
        args.input,
        model=model,
        output=args.output,
        device=args.device,
        factor=args.factor,
        window_size=tuple(args.window_size),
        overlap=args.overlap,
        eliminate_border_px=args.eliminate_border_px,
        batch_size=args.batch_size,
        overwrite=args.overwrite,
        band_indexes=args.bands,
        # Native 20m red-edge/SWIR sources must be mapped to the RGB 10m grid.
        # WarpedVRT reads do this per window, without a full intermediate stack.
        resample_inputs=True,
    )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
