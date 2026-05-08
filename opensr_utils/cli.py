def main():
    import argparse
    from opensr_utils.pipeline import large_file_processing

    parser = argparse.ArgumentParser(
        description="🚀 Patch-based super-resolution for large geospatial rasters"
    )
    parser.add_argument("root", type=str, help="📂 Path to input file/folder")

    parser.add_argument(
        "model",
        type=str,
        choices=["LDSRS2", "None"],
        help="🤖 Model to run: 'LDSRS2' or 'None'",
    )

    parser.add_argument(
        "--window_size",
        type=int,
        nargs=2,
        default=(128, 128),
        help="🔲 LR window size (default: 128 128)",
    )
    parser.add_argument(
        "--factor",
        type=int,
        default=4,
        help="⬆️ Positive integer output scale (default: 4; use 1 for no upscaling)",
    )
    parser.add_argument(
        "--overlap", type=int, default=8, help="🤝 Overlap in LR pixels (default: 8)"
    )
    parser.add_argument(
        "--eliminate_border_px",
        type=int,
        default=0,
        help="✂️ Pixels to eliminate at patch borders (default: 0)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="⚡ Device for inference (default: cpu)",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        nargs="+",
        default=[0],
        help="💻 GPU IDs to use (default: 0)",
    )
    parser.add_argument(
        "--save_preview",
        action="store_true",
        help="🖼️ Also save one georeferenced crop and 10 LR/SR previews",
    )
    parser.add_argument(
        "--debug", action="store_true", help="🐞 Debug mode: process only ~100 windows"
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace an existing sr.tif output"
    )
    parser.add_argument(
        "--keep_temp",
        action="store_true",
        help="Keep temporary patch files after stitching",
    )
    parser.add_argument(
        "--delete_input_zip",
        action="store_true",
        help="Delete the source .zip after successful extraction",
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Prediction batch size (default: 16)"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="DataLoader worker count (default: 4)",
    )
    parser.add_argument(
        "--prefetch_factor",
        type=int,
        default=2,
        help="DataLoader prefetch factor when workers > 0 (default: 2)",
    )
    parser.add_argument(
        "--compressed_patches",
        action="store_true",
        help="Store temporary patches as compressed .npz files",
    )

    args = parser.parse_args()

    # --- Resolve model selection ---
    if args.model == "LDSRS2":
        # Lazy import to avoid CUDA init unless needed
        try:
            from opensr_utils.model_utils.get_models import get_ldsrs2

            device = "cpu"  # "cuda" or "cpu" - Dont use the torch automated detection here, it messes up the lightning trainer multi-GPU setup
            model = get_ldsrs2(device=device)

        except Exception as e:
            print("❌ Could not load LDSR-S2 model from 'opensr_model'.")
            print(f"   Error: {e}")
            return
    else:
        print(
            "ℹ️ From CL, you can only run the LDSR-S2 model. Using interpolation placeholder instead"
        )
        model = None  # placeholder/no-model mode (still stitches pipeline outputs if provided)

    # --- Run the pipeline ---
    runner = large_file_processing(
        root=args.root,
        model=model,
        window_size=tuple(args.window_size),
        factor=args.factor,
        overlap=args.overlap,
        eliminate_border_px=args.eliminate_border_px,
        device=args.device,
        gpus=args.gpus if args.device == "cuda" else None,
        save_preview=args.save_preview,
        debug=args.debug,
        cleanup=not args.keep_temp,
        overwrite=args.overwrite,
        delete_input_zip=args.delete_input_zip,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        compressed_patches=args.compressed_patches,
    )
    runner.run()


if __name__ == "__main__":
    main()
