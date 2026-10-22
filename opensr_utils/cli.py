"""Command-line interface; importing this module does not load ML dependencies."""

from opensr_utils.model_utils.get_models import MODEL_NAMES, normalize_model_name


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(
        description="Tiled super-resolution for rasters, SAFE archives and S2GM folders."
    )
    parser.add_argument(
        "root", help="Input raster, SAFE directory/archive or S2GM folder"
    )
    parser.add_argument(
        "model",
        nargs="?",
        default="None",
        type=normalize_model_name,
        choices=MODEL_NAMES,
        help="Pretrained model (SRGAN aliases ESASRGAN), or None for interpolation",
    )
    parser.add_argument(
        "-o", "--output", help="Output GeoTIFF (default: sr.tif beside input)"
    )
    parser.add_argument(
        "--window-size",
        "--window_size",
        type=int,
        nargs=2,
        default=(128, 128),
        help="LR window width and height (default: 128 128)",
    )
    parser.add_argument(
        "--factor",
        type=int,
        default=None,
        help="Positive integer output scale (default: selected model's scale, otherwise 4)",
    )
    parser.add_argument(
        "--overlap", type=int, default=8, help="Overlap in LR pixels (default: 8)"
    )
    parser.add_argument(
        "--eliminate-border-px",
        "--eliminate_border_px",
        type=int,
        default=0,
        help="Discard interior patch borders in LR pixels (default: 0)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Inference device (default: auto)",
    )
    parser.add_argument("--gpus", type=int, nargs="+", help="CUDA GPU IDs (default: 0)")
    parser.add_argument(
        "--output-units",
        "--output_units",
        choices=["normalized", "reflectance"],
        default=None,
        help="Model output units (default: model's declared units, otherwise normalized)",
    )
    parser.add_argument(
        "--save-preview",
        "--save_preview",
        action="store_true",
        help="Save a georeferenced crop and LR/SR previews",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Infer at most 100 windows, keep predictions, and create no final raster",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing output after successful validation",
    )
    parser.add_argument(
        "--keep-temp",
        "--keep_temp",
        action="store_true",
        help="Keep every prediction and the manifest",
    )
    parser.add_argument(
        "--delete-input-zip",
        "--delete_input_zip",
        action="store_true",
        help="Delete source ZIP only after usable input validation",
    )
    parser.add_argument(
        "--batch-size",
        "--batch_size",
        type=int,
        default=16,
        help="Prediction batch size (default: 16)",
    )
    parser.add_argument(
        "--num-workers",
        "--num_workers",
        type=int,
        default=0,
        help="Data loading workers (default: 0)",
    )
    parser.add_argument(
        "--prefetch-factor",
        "--prefetch_factor",
        type=int,
        default=2,
        help="Prefetch batches per worker (default: 2)",
    )
    parser.add_argument(
        "--compressed-patches",
        "--compressed_patches",
        action="store_true",
        help="Store predictions as compressed NPZ instead of NPY",
    )
    parser.add_argument(
        "--checkpoint",
        help="Local SEN2SR MLSTAC directory/export, or ESA SRGAN checkpoint",
    )
    parser.add_argument(
        "--config",
        "--model-config",
        help="ESA SRGAN YAML for an official local checkpoint",
    )
    parser.add_argument("--cache-dir", help="Model download/cache directory")
    parser.add_argument(
        "--variant", "--model-variant", help="SEN2SR RGBN variant (lite)"
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Use local model artifacts without network downloads",
    )
    parser.add_argument(
        "--bands",
        type=int,
        nargs="+",
        help="1-based GeoTIFF input band indexes in the model's expected order",
    )
    args = parser.parse_args(argv)

    # Load on CPU so Lightning can select and initialize CUDA processes itself.
    model = None
    if args.model != "None" or any((args.checkpoint, args.config, args.variant)):
        try:
            if args.model == "LDSRS2" and not any(
                (
                    args.checkpoint,
                    args.config,
                    args.cache_dir,
                    args.variant,
                    args.no_download,
                )
            ):
                from opensr_utils.model_utils.get_models import get_ldsrs2

                model = get_ldsrs2(device="cpu")
            else:
                from opensr_utils.model_utils.get_models import get_model

                model = get_model(
                    args.model,
                    device="cpu",
                    checkpoint=args.checkpoint,
                    config=args.config,
                    cache_dir=args.cache_dir,
                    variant=args.variant,
                    download=not args.no_download,
                )
        except Exception as exc:
            label = "LDSR-S2" if args.model == "LDSRS2" else args.model
            parser.exit(
                1,
                f"Could not load {label} model. Check its optional dependencies and model artifacts.\nError: {exc}\n",
            )
    factor = args.factor if args.factor is not None else getattr(model, "factor", 4)
    output_units = args.output_units or getattr(model, "output_units", "normalized")
    input_options = {"band_indexes": args.bands} if args.bands is not None else {}

    from opensr_utils.pipeline import large_file_processing

    runner = None
    try:
        runner = large_file_processing(
            root=args.root,
            model=model,
            output_path=args.output,
            output_units=output_units,
            window_size=tuple(args.window_size),
            factor=factor,
            overlap=args.overlap,
            eliminate_border_px=args.eliminate_border_px,
            device=args.device,
            gpus=args.gpus if args.device != "cpu" else None,
            save_preview=args.save_preview,
            debug=args.debug,
            cleanup=not args.keep_temp,
            overwrite=args.overwrite,
            delete_input_zip=args.delete_input_zip,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
            compressed_patches=args.compressed_patches,
            **input_options,
        )
        result = runner.run()
        print(result)
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        diagnostic = (
            f"\nDiagnostics: {runner.temp_folder}" if runner is not None else ""
        )
        parser.exit(1, f"Processing failed: {exc}{diagnostic}\n")
    finally:
        if runner is not None:
            runner.close()


if __name__ == "__main__":
    main()
