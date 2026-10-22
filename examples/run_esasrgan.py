#!/usr/bin/env python3
"""Run an ESA OpenSR SRGAN/ESASRGAN generator through OpenSR-Utils."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


def _source_checkout_import():
    """Allow ``python /checkout/examples/run_esasrgan.py`` from any working directory."""
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
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Local checkpoint; omit for the official RGB-NIR preset",
    )
    parser.add_argument(
        "--config", type=Path, help="Matching local model configuration"
    )
    parser.add_argument("--cache-dir", type=Path, help="Model artifact cache directory")
    parser.add_argument(
        "--download",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow downloading the official preset when no local checkpoint is supplied",
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
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.config is not None and args.checkpoint is None:
        parser.error("--config requires --checkpoint")
    if not args.download and args.checkpoint is None:
        parser.error("--no-download requires --checkpoint")

    _source_checkout_import()
    from opensr_utils import super_resolve
    from opensr_utils.model_utils.get_models import get_model

    model = get_model(
        "esasrgan",
        device="cpu",
        checkpoint=args.checkpoint,
        config=args.config,
        cache_dir=args.cache_dir,
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
    )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
