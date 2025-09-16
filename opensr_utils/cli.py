def main():
    import argparse
    import math
    from opensr_utils.pipeline import large_file_processing


    parser = argparse.ArgumentParser(
        description="🚀 Patch-based super-resolution for large geospatial rasters"
    )
    parser.add_argument("root", type=str, help="📂 Path to input file/folder")

    parser.add_argument("model", type=str, choices=["LDSRS2", "None"],
                        help="🤖 Model to run: 'LDSRS2' or 'None'")

    parser.add_argument("--window_size", type=int, nargs=2, default=(128, 128),
                        help="🔲 LR window size (default: 128 128)")
    parser.add_argument("--factor", type=int, default=4, choices=[2, 4, 6, 8],
                        help="⬆️ Upscaling factor (default: 4)")
    parser.add_argument("--overlap", type=int, default=8,
                        help="🤝 Overlap in LR pixels (default: 8)")
    parser.add_argument("--eliminate_border_px", type=int, default=0,
                        help="✂️ Pixels to eliminate at patch borders (default: 0)")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"],
                        help="⚡ Device for inference (default: cpu)")
    parser.add_argument("--gpus", type=int, nargs="+", default=[0],
                        help="💻 GPU IDs to use (default: 0)")
    parser.add_argument("--save_preview", action="store_true",
                        help="🖼️ Also save one georeferenced crop and 10 LR/SR previews")
    parser.add_argument("--debug", action="store_true",
                        help="🐞 Debug mode: process only ~100 windows")

    args = parser.parse_args()

    # --- Resolve model selection ---
    if args.model == "LDSRS2":
        # Lazy import to avoid CUDA init unless needed
        try:
            from omegaconf import OmegaConf
            from io import StringIO
            import requests  # your model package
            import opensr_model  # your model package
            # Instantiate model
            device = "cpu"  # "cuda" or "cpu" - Dont use the torch automated detection here, it messes up the lightning trainer multi-GPU setup
            config_url = "https://raw.githubusercontent.com/ESAOpenSR/opensr-model/refs/heads/main/opensr_model/configs/config_10m.yaml"
            response = requests.get(config_url)
            config = OmegaConf.load(StringIO(response.text))
            model = opensr_model.SRLatentDiffusion(config, device=device)
            model.load_pretrained(config.ckpt_version)

        except Exception as e:
            print("❌ Could not load LDSR-S2 model from 'opensr_model'.")
            print(f"   Error: {e}")
            return
    else:
        print("ℹ️ From CL, you can only run the LDSR-S2 model. Using interpolation placeholder instead")
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
    )

if __name__ == "__main__":
    main()