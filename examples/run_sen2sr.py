#!/usr/bin/env python3
"""Run a SEN2SR model through OpenSR-Utils' tiled geospatial pipeline."""

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


def main() -> int:
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
        "--variant", choices=("lite", "NonReference_RGBN_x4"), default="lite"
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
    args = parser.parse_args()

    _source_checkout_import()
    from opensr_utils import super_resolve
    from opensr_utils.model_utils.get_models import get_model

    model = get_model(
        "sen2sr",
        device="cpu",
        checkpoint=args.checkpoint,
        cache_dir=args.cache_dir,
        variant=args.variant,
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
