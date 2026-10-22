#!/usr/bin/env python3
"""Instructional command-line demo for OpenSR-Utils.

Examples::

    python examples/run_model.py scene.tif --model none --factor 1 --output scene_sr.tif
    python examples/run_model.py scene.SAFE.zip --model sen2sr --checkpoint weights.ckpt
    python examples/run_model.py scene.tif --model esasrgan --device cuda --batch-size 8

``--help`` does not import torch, rasterio, or optional model packages.
Model-specific dependencies and weights are described in ``README.md`` in this directory.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence


# Explicit per-model defaults avoid inheriting the pipeline's generic scale.
MODEL_DEFAULTS = {
    "none": {"factor": 1, "window_size": (128, 128), "overlap": 0},
    "ldsrs2": {
        "factor": 4,
        "window_size": (128, 128),
        "overlap": 12,
        "eliminate_border_px": 2,
    },
    "sen2sr": {
        "factor": 4,
        "window_size": (128, 128),
        "overlap": 12,
        "eliminate_border_px": 2,
    },
    "srgan": {
        "factor": 4,
        "window_size": (128, 128),
        "overlap": 12,
        "eliminate_border_px": 2,
    },
    "esasrgan": {
        "factor": 4,
        "window_size": (128, 128),
        "overlap": 12,
        "eliminate_border_px": 2,
    },
}


def build_parser() -> argparse.ArgumentParser:
    """Build the parser without importing heavy or optional dependencies."""
    parser = argparse.ArgumentParser(
        description="Generate a tiled super-resolution GeoTIFF from a raster or SAFE product.",
        epilog="Weights are downloaded only when a model is selected; use --model none for a smoke test.",
    )
    parser.add_argument(
        "input", type=Path, help="GeoTIFF, SAFE directory/archive, or S2GM folder"
    )
    parser.add_argument(
        "--model",
        default="none",
        type=str.lower,
        choices=sorted(MODEL_DEFAULTS),
        help="Model loader (default: none; srgan and esasrgan are aliases)",
    )
    parser.add_argument(
        "-o", "--output", type=Path, help="Output GeoTIFF (default: beside input)"
    )
    parser.add_argument(
        "--checkpoint", type=Path, help="Local model checkpoint/configuration"
    )
    parser.add_argument(
        "--config", type=Path, help="Model configuration (used by SRGAN/ESASRGAN)"
    )
    parser.add_argument(
        "--variant",
        choices=("lite", "NonReference_RGBN_x4"),
        help="SEN2SR variant (default: lite)",
    )
    parser.add_argument(
        "--cache-dir", type=Path, help="Directory for downloaded model weights"
    )
    parser.add_argument(
        "--download",
        dest="download",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow model loaders to download missing weights (default: yes)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, or cuda (default: auto); cuda:N selects one GPU",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        nargs="+",
        metavar="ID",
        help="CUDA GPU IDs when using --device cuda",
    )
    parser.add_argument(
        "--factor", type=int, help="Output scale (default: selected model's scale)"
    )
    parser.add_argument(
        "--window-size",
        "--window_size",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        help="Low-resolution tile size (default: model-specific)",
    )
    parser.add_argument(
        "--overlap", type=int, help="Tile overlap in low-resolution pixels"
    )
    parser.add_argument(
        "--eliminate-border-px",
        "--eliminate_border_px",
        type=int,
        help="Discard this many pixels from each tile border",
    )
    parser.add_argument(
        "--bands",
        type=int,
        nargs="+",
        metavar="INDEX",
        help="1-based GeoTIFF band indexes in model input order (for example: --bands 4 3 2 8)",
    )
    parser.add_argument(
        "--batch-size",
        "--batch_size",
        type=int,
        default=16,
        help="Inference batch size (default: 16)",
    )
    parser.add_argument(
        "--num-workers",
        "--num_workers",
        type=int,
        default=0,
        help="Data loader workers (default: 0)",
    )
    parser.add_argument(
        "--output-units",
        "--output_units",
        choices=("normalized", "reflectance"),
        default="normalized",
        help="Model output units (default: normalized)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output after validation",
    )
    parser.add_argument(
        "--keep-temp",
        "--keep_temp",
        action="store_true",
        help="Retain prediction patches and manifest",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Infer at most 100 tiles and write only a manifest",
    )
    parser.add_argument(
        "--save-preview",
        "--save_preview",
        action="store_true",
        help="Write LR/SR preview images",
    )
    parser.add_argument(
        "--delete-input-zip",
        "--delete_input_zip",
        action="store_true",
        help="Delete a ZIP only after successful input validation",
    )
    return parser


def _resolved_options(args: argparse.Namespace, model=None) -> dict:
    defaults = MODEL_DEFAULTS[args.model]
    factor = (
        args.factor
        if args.factor is not None
        else getattr(model, "factor", defaults["factor"])
    )
    return {
        "factor": factor,
        "window_size": tuple(args.window_size)
        if args.window_size
        else defaults["window_size"],
        "overlap": args.overlap if args.overlap is not None else defaults["overlap"],
        "eliminate_border_px": (
            args.eliminate_border_px
            if args.eliminate_border_px is not None
            else defaults.get("eliminate_border_px", 0)
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments, load the selected model, and produce one product."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.model == "none" and args.checkpoint:
        parser.error(
            "--checkpoint requires --model; use --model none without a checkpoint"
        )
    if args.variant and args.model != "sen2sr":
        parser.error("--variant is only supported with --model sen2sr")
    if args.config and args.model not in ("srgan", "esasrgan"):
        parser.error(
            "--config is only supported with --model srgan or --model esasrgan"
        )
    if args.factor is not None and args.factor <= 0:
        parser.error("--factor must be a positive integer")
    if args.overlap is not None and args.overlap < 0:
        parser.error("--overlap must be non-negative")

    model = None
    if args.model != "none":
        try:
            from opensr_utils.model_utils.get_models import get_model

            model_options = {
                "checkpoint": args.checkpoint,
                "cache_dir": args.cache_dir,
                "download": args.download,
            }
            if args.config is not None:
                model_options["config"] = args.config
            if args.variant is not None:
                model_options["variant"] = args.variant
            model = get_model(args.model, device="cpu", **model_options)
        except Exception as exc:
            parser.error(
                f"could not load {args.model}: {exc}. Install its optional model extra or pass --checkpoint."
            )

    options = _resolved_options(args, model)
    gpus = args.gpus
    device = args.device
    if device.startswith("cuda:"):
        try:
            selected = int(device.split(":", 1)[1])
        except (ValueError, IndexError):
            parser.error("--device cuda:N requires a non-negative integer N")
        if selected < 0:
            parser.error("--device cuda:N requires a non-negative integer N")
        device = "cuda"
        if gpus is None:
            gpus = [selected]
    elif device not in ("auto", "cpu", "cuda"):
        parser.error("--device must be auto, cpu, cuda, or cuda:N")
    if device != "cuda" and gpus is not None:
        parser.error("--gpus requires --device cuda or cuda:N")

    try:
        from opensr_utils import super_resolve

        result = super_resolve(
            args.input,
            model=model,
            output=args.output,
            device=device,
            gpus=gpus,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            output_units=args.output_units,
            band_indexes=args.bands,
            overwrite=args.overwrite,
            cleanup=not args.keep_temp,
            debug=args.debug,
            save_preview=args.save_preview,
            delete_input_zip=args.delete_input_zip,
            **options,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        parser.error(f"processing failed: {exc}")
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
