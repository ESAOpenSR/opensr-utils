#!/usr/bin/env python3
"""Super-resolve only Sentinel-2 RGB-NIR (B04, B03, B02, B08) with LDSR-S2."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input", type=Path, help="GeoTIFF, SAFE directory/archive, or S2GM folder"
    )
    parser.add_argument("-o", "--output", type=Path)
    parser.add_argument(
        "--checkpoint", type=Path, help="Local LDSR-S2 checkpoint (requires --config)"
    )
    parser.add_argument("--config", type=Path, help="Matching LDSR-S2 YAML")
    parser.add_argument(
        "--download", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--sampling-steps", type=int, help="Override the model's diffusion steps"
    )
    parser.add_argument(
        "--bands", type=int, nargs=4, help="1-based GeoTIFF indexes in R,G,B,NIR order"
    )
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--window-size",
        type=int,
        nargs=2,
        default=(128, 128),
        metavar=("WIDTH", "HEIGHT"),
    )
    parser.add_argument("--overlap", type=int, default=12)
    parser.add_argument("--eliminate-border-px", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if (args.checkpoint is None) != (args.config is None):
        parser.error("--checkpoint and --config must be supplied together")
    if not args.download and args.checkpoint is None:
        parser.error("--no-download requires --checkpoint and --config")

    # Keep --help independent of Torch and optional model installations.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from opensr_utils import super_resolve
    from opensr_utils.model_utils.ldsrs2 import load_ldsrs2

    model = load_ldsrs2(
        checkpoint=args.checkpoint,
        config_path=args.config,
        sampling_steps=args.sampling_steps,
        download=args.download,
    )
    output = super_resolve(
        args.input,
        model,
        args.output,
        device=args.device,
        batch_size=args.batch_size,
        window_size=tuple(args.window_size),
        overlap=args.overlap,
        eliminate_border_px=args.eliminate_border_px,
        band_indexes=args.bands,
        overwrite=args.overwrite,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
