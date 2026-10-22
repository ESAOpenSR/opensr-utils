# -*- coding: utf-8 -*-


"""Windowed inference with disk-backed blending and atomic output publication."""

import torch
from pytorch_lightning import LightningModule

# general imports
import os
import shutil
import json
from datetime import datetime
from pathlib import Path

# Geo
import rasterio
from rasterio.transform import Affine

# local imports
from opensr_utils.data_utils.datamodule import PredictionDataModule
from opensr_utils.model_utils.prepare_model import preprocess_model


class large_file_processing:
    """Prepare and execute memory-bounded super-resolution of geospatial inputs.

    Accept a raster, SAFE directory/ZIP or S2GM folder and a PyTorch/Lightning
    model (or ``None`` for bilinear interpolation). Construction discovers input,
    validates grids and prepares inference; call :meth:`run` to create a product,
    or use :func:`opensr_utils.super_resolve` for the one-call workflow.

    Windows are ``(width, height)`` in LR pixels. ``factor`` is a positive integer;
    overlap and discarded borders are also in LR pixels. Batches stream to disk
    and overlapping predictions blend through disk-backed float32 accumulators.
    Plain PyTorch modules and Lightning modules both support batching and DDP.

    ``device="auto"`` chooses CUDA when available, otherwise CPU. ``gpus`` selects
    CUDA IDs; ``batch_size``, ``num_workers``, ``prefetch_factor`` and
    ``compressed_patches`` allow throughput/storage tuning. Models load on CPU
    before Lightning places them on their execution devices.

    Model inputs are normalized to [0,1]. ``output_units="normalized"`` scales
    predictions by 10000; ``"reflectance"`` accepts existing [0,10000] values.
    Nonfinite outputs are errors; finite outputs are clipped, rounded and saved
    as uint16. The output's internal validity mask preserves input masks and
    nodata independently of predicted values, including valid zeros.

    ``output_path`` names a destination GeoTIFF; the default is ``sr.tif`` beside
    the input (inside S2GM folders). Source rasters cannot be destinations.
    Existing outputs require ``overwrite=True`` and are atomically replaced only
    after full coverage and mask validation. ``cleanup=False`` keeps every patch
    and the manifest. Failed runs retain diagnostics. Source ZIP deletion always
    requires ``delete_input_zip=True`` and successful input validation.

    ``debug=True`` infers at most 100 windows, retains patches and returns the
    manifest path without publishing a final raster. ``save_preview=True`` adds
    preview images/crops. Use this object as a context manager or call ``close``
    to release cached raster handles. ``auto_run`` exists for legacy callers;
    explicit ``run`` is the default.
    """

    def __init__(
        self,
        root: str,
        model=None,
        window_size: tuple = (128, 128),
        factor=None,
        overlap: int = 8,
        eliminate_border_px=0,
        device: str = "auto",
        gpus=None,
        save_preview: bool = False,
        debug=False,
        auto_run: bool = False,
        cleanup: bool = True,
        overwrite: bool = False,
        delete_input_zip: bool = False,
        batch_size: int = 16,
        num_workers: int = 0,
        prefetch_factor: int = 2,
        compressed_patches: bool = False,
        output_path=None,
        output_units=None,
        checkpoint=None,
        model_config=None,
        cache_dir=None,
        model_variant=None,
        download: bool = True,
        band_indexes=None,
    ):
        # 1) Validate public inputs. Do not use assert: optimized Python removes it.
        if isinstance(model, str):
            from opensr_utils.model_utils.get_models import normalize_model_name

            model = normalize_model_name(model)
        if model is not None and not isinstance(
            model, (str, LightningModule, torch.nn.Module)
        ):
            raise TypeError("Model must be a PyTorch module, LightningModule, or None.")
        if not isinstance(window_size, (tuple, list)) or len(window_size) != 2:
            raise TypeError(
                "window_size must be a tuple/list of two positive integers."
            )
        if not all(
            isinstance(v, int) and not isinstance(v, bool) and v > 0
            for v in window_size
        ):
            raise ValueError("window_size must contain two positive integers.")
        if (
            isinstance(overlap, bool)
            or not isinstance(overlap, int)
            or overlap < 0
            or overlap % 2 != 0
        ):
            raise ValueError("overlap must be a non-negative even integer.")
        if (
            not isinstance(eliminate_border_px, int)
            or isinstance(eliminate_border_px, bool)
            or eliminate_border_px < 0
            or eliminate_border_px % 2 != 0
        ):
            raise ValueError("eliminate_border_px must be a non-negative even integer.")
        if overlap == 0:
            if eliminate_border_px != 0:
                raise ValueError("eliminate_border_px must be 0 when overlap is 0.")
        elif eliminate_border_px >= overlap:
            raise ValueError("eliminate_border_px must be smaller than overlap.")
        if overlap > 0 and (overlap - eliminate_border_px) < 4:
            raise ValueError(
                "overlap must be at least 4 px bigger than eliminate_border_px."
            )
        if overlap >= min(window_size):
            raise ValueError("overlap must be smaller than both window dimensions.")
        explicit_factor = factor is not None
        if factor is None:
            factor = getattr(model, "factor", 4)
        if not isinstance(factor, int) or isinstance(factor, bool) or factor <= 0:
            raise ValueError("factor must be a positive integer.")
        if device not in ("auto", "cpu", "cuda"):
            raise ValueError("device must be 'auto', 'cpu' or 'cuda'.")
        if output_units not in (None, "normalized", "reflectance"):
            raise ValueError("output_units must be 'normalized' or 'reflectance'.")
        for name, value in {
            "batch_size": batch_size,
            "num_workers": num_workers,
            "prefetch_factor": prefetch_factor,
        }.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer.")
        if batch_size == 0:
            raise ValueError("batch_size must be greater than zero.")
        if num_workers and prefetch_factor == 0:
            raise ValueError(
                "prefetch_factor must be greater than zero when num_workers > 0."
            )

        self.output_path = (
            str(Path(output_path).expanduser().resolve())
            if output_path is not None
            else None
        )
        if self.output_path and Path(self.output_path).suffix.lower() not in (
            ".tif",
            ".tiff",
        ):
            raise ValueError("output_path must name a .tif or .tiff file.")
        self.output_units = output_units
        if band_indexes is not None:
            if (
                not isinstance(band_indexes, (list, tuple))
                or not band_indexes
                or any(
                    isinstance(index, bool) or not isinstance(index, int) or index <= 0
                    for index in band_indexes
                )
                or len(set(band_indexes)) != len(band_indexes)
            ):
                raise ValueError(
                    "band_indexes must be unique positive 1-based raster band indexes."
                )
            band_indexes = list(band_indexes)
        self._requested_band_indexes = band_indexes
        if not isinstance(model, str) and any(
            value is not None
            for value in (checkpoint, model_config, cache_dir, model_variant)
        ):
            raise ValueError(
                "Checkpoint/config/cache/variant options require a named model; pass an already loaded module without loader options."
            )

        # General Settings
        self.debug = bool(
            debug
        )  # if True, only process 100 windows globally (DDP-safe)
        self.root = str(Path(root).expanduser())
        self.window_size = tuple(window_size)  # window size of the LR image
        self.factor = int(factor)  # sr factor of the model
        self.overlap = overlap  # overlap in px of the LR image windows
        self.eliminate_border_px = eliminate_border_px  # border pixels to eliminate in px on each side of the SR image
        self.device = (
            ("cuda" if torch.cuda.is_available() else "cpu")
            if device == "auto"
            else device
        )
        self.save_preview = bool(save_preview)
        self.cleanup = bool(cleanup)
        self.overwrite = bool(overwrite)
        self.delete_input_zip = bool(delete_input_zip)
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.prefetch_factor = int(prefetch_factor)
        self.compressed_patches = bool(compressed_patches)
        existing_run_id = os.environ.get("OPENSR_RUN_ID")
        distributed = torch.distributed
        active_group = distributed.is_available() and distributed.is_initialized()
        self._external_world = (
            distributed.get_world_size()
            if active_group
            else int(os.environ.get("WORLD_SIZE", "1"))
        )
        launcher_env = (
            os.environ.get("WORLD_SIZE", "1") not in ("", "1")
            or os.environ.get("LOCAL_RANK") is not None
            or os.environ.get("RANK") is not None
        )
        if existing_run_id and launcher_env:
            self.run_id = existing_run_id
        else:
            self.run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{os.getpid()}"
            if active_group:
                shared_id = [self.run_id if distributed.get_rank() == 0 else None]
                distributed.broadcast_object_list(shared_id, src=0)
                self.run_id = shared_id[0]
            elif self._external_world > 1:
                raise ValueError(
                    "External distributed launches require a unique OPENSR_RUN_ID shared by all ranks. "
                    "Alternatively, let the pipeline launch workers with gpus=[0, 1]."
                )
            os.environ["OPENSR_RUN_ID"] = self.run_id
        if isinstance(gpus, int):  # pass GPU/s to Trainer as list
            self.gpus = [gpus]  # number of gpus or list of gpu ids to use
        elif gpus is None:
            self.gpus = None
        elif isinstance(gpus, (list, tuple)):
            self.gpus = list(gpus)
        else:
            raise TypeError("gpus must be None, an int, or a list/tuple of ints.")
        if self.gpus is not None and (
            not self.gpus
            or not all(
                isinstance(g, int) and not isinstance(g, bool) and g >= 0
                for g in self.gpus
            )
            or len(set(self.gpus)) != len(self.gpus)
        ):
            raise ValueError("gpus must contain unique non-negative integer GPU ids.")
        if self.device == "cpu":
            self.gpus = None
        elif self.gpus is None:
            local_world = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
            self.gpus = list(range(local_world)) if self._external_world > 1 else [0]

        # 3) Locals/init-only (no paths derived yet!) - leave empty
        self.image_meta = {}
        self.input_type = None
        self.placeholder_filepath = None
        self.placeholder_path = None
        self.log_dir = None
        self.temp_folder = None
        self.final_sr_path = None

        # 4) Verify input type FIRST (will also unzip and update self.root if needed)
        from opensr_utils.data_utils.reading_utils import verify_input_file_type

        verify_input_file_type(self, self.root)  # multi-GPU safe

        # 5) Create directories AFTER type detection
        from opensr_utils.data_utils.reading_utils import create_dirs

        create_dirs(self)  # multi-GPU safe

        # Get Image information based on input, including image coordinate windows
        self.get_image_meta(self.root)  # multi-GPU safe

        # Load on CPU; Lightning moves the model after selecting its execution strategy.
        if isinstance(model, str):
            from opensr_utils.model_utils.get_models import get_model

            model = get_model(
                model,
                device="cpu",
                checkpoint=checkpoint,
                config=model_config,
                cache_dir=cache_dir,
                variant=model_variant,
                download=download,
            )
        model_factor = getattr(model, "factor", None)
        if model_factor is not None:
            if explicit_factor and model_factor != self.factor:
                raise ValueError(
                    f"The selected model produces factor {model_factor}; requested factor={self.factor}."
                )
            self.factor = model_factor
        model_units = getattr(model, "output_units", None)
        if model_units and output_units is not None and output_units != model_units:
            raise ValueError(
                f"The model returns {model_units} output; use output_units={model_units!r}."
            )
        self.output_units = output_units or model_units or "normalized"
        self._configure_model_input(model)
        self.create_datamodule()

        # Make sure model is useable and in eval mode
        self.model = preprocess_model(
            self,
            model,
            temp_folder=self.temp_folder,
            windows=self.image_meta["image_windows"],
            factor=self.factor,
            output_units=self.output_units,
        )

        # Deletion is opt-in and happens only after source, channel, model and
        # window contracts have all validated successfully.
        pending_zip = getattr(self, "_pending_input_zip_delete", None)
        if pending_zip and self._is_rank0():
            pending_path = Path(pending_zip)
            if pending_path.exists():
                pending_path.unlink()
                self._log(f"Deleted validated archive: {pending_path.name}")
            self._pending_input_zip_delete = None

        # Print Status
        self._log("📊 Status: Model and Data ready for inference ✅")

        if auto_run:
            self.run()

    def _configure_model_input(self, model):
        """Select declared model bands without changing generic module inputs."""
        if getattr(model, "input_units", "normalized") != "normalized":
            raise ValueError(
                "The pipeline supplies normalized [0,1] inputs. Construct the adapter with input_units='normalized'."
            )
        declared_output_bands = getattr(model, "output_band_names", None)
        if declared_output_bands is not None:
            if (
                not isinstance(declared_output_bands, (tuple, list))
                or not declared_output_bands
                or any(
                    not isinstance(name, str) or not name
                    for name in declared_output_bands
                )
            ):
                raise ValueError(
                    "Model output_band_names must be a sequence of nonempty band names."
                )
            self.image_meta["output_band_names"] = tuple(declared_output_bands)
        required = tuple(getattr(model, "input_bands", ()))
        maximum = getattr(model, "max_window_size", None)
        if maximum is not None:
            actual = (
                min(self.window_size[0], self.image_meta["width"]),
                min(self.window_size[1], self.image_meta["height"]),
            )
            if any(size > limit for size, limit in zip(actual, maximum)):
                raise ValueError(
                    f"This model accepts windows up to {tuple(maximum)}; requested {actual}. Use window_size={tuple(maximum)}."
                )
        aliases = {
            "R": "B04",
            "RED": "B04",
            "G": "B03",
            "GREEN": "B03",
            "B": "B02",
            "BLUE": "B02",
            "NIR": "B08",
        }
        if self.input_type == "file":
            with rasterio.open(self.root) as source:
                indexes = self._requested_band_indexes
                if indexes is not None and max(indexes) > source.count:
                    raise ValueError(
                        f"band_indexes requests a channel beyond the raster's {source.count} bands."
                    )
                if required and indexes is None:
                    names = [
                        aliases.get(
                            (name or "").strip().upper(), (name or "").strip().upper()
                        )
                        for name in source.descriptions
                    ]
                    if all(name in names for name in required):
                        if any(names.count(name) != 1 for name in required):
                            raise ValueError(
                                f"Raster band descriptions must uniquely identify {required}."
                            )
                        indexes = [names.index(name) + 1 for name in required]
                    elif source.count != len(required) or any(
                        name in required for name in names
                    ):
                        raise ValueError(
                            f"Model requires bands {required}. Set GeoTIFF band descriptions or pass band_indexes in that order; unlabeled inputs must contain exactly {len(required)} bands."
                        )
                selected_count = len(indexes) if indexes is not None else source.count
                if required and selected_count != len(required):
                    raise ValueError(
                        f"Model requires {len(required)} bands in {required} order; selected {selected_count}."
                    )
                self.image_meta["band_indexes"] = indexes
                self.image_meta["bands"] = selected_count
            return
        if self._requested_band_indexes is not None:
            raise ValueError(
                "band_indexes is for single raster files; SAFE/S2GM model bands are selected by name."
            )
        if required:
            paths = self.image_meta["lr_file_dict"]
            by_band = {
                aliases.get(label.upper(), label.upper()): (label, path)
                for label, path in paths.items()
            }
            missing = [band for band in required if band not in by_band]
            if missing:
                raise FileNotFoundError(
                    f"Model requires missing spectral bands {missing}; available bands: {sorted(by_band)}."
                )
            self.image_meta["lr_file_dict"] = dict(by_band[band] for band in required)
            self.image_meta["bands"] = len(required)

    def create_datamodule(self, windows=None):
        """
        Create a PredictionDataModule for patch-based inference.

        The datamodule wraps the input type, root path, and sliding windows,
        and handles batching for PyTorch Lightning’s predict loop.
        """
        previous = getattr(self, "datamodule", None)
        if previous is not None:
            previous.dataset.close()
        dm = PredictionDataModule(
            input_type=self.input_type,
            root=self.root,
            windows=self.image_meta["image_windows"] if windows is None else windows,
            lr_file_dict=self.image_meta["lr_file_dict"],
            band_indexes=self.image_meta.get("band_indexes"),
            prefetch_factor=self.prefetch_factor,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
        dm.setup()
        self._log(f"📦 Created PredictionDataModule with {len(dm.dataset)} patches.")
        self.datamodule = dm

    def get_image_meta(self, root_dir):
        """
        Extract raster metadata (width, height, dtype, CRS, transform, band count).

        - For files: opens directly with rasterio.
        - For SAFE: locates JP2 bands B02, B03, B04, B08.
        - For S2GM: locates GeoTIFF bands B02, B03, B04, B08.

        Adds a list of sliding image windows to `self.image_meta`.
        """
        if self.input_type == "file":
            self.image_meta["lr_file_dict"] = {"file": root_dir}
            with rasterio.open(root_dir) as src:
                (
                    self.image_meta["width"],
                    self.image_meta["height"],
                    self.image_meta["dtype"],
                ) = (src.width, src.height, src.dtypes[0])
                self.image_meta["transform"] = src.transform
                self.image_meta["crs"] = src.crs
                self.image_meta["bands"] = src.count

        elif self.input_type == "SAFE":
            # Get LR image file paths
            jp2s = []
            for root, _, files in os.walk(self.root):
                if "IMG_DATA" in root:
                    for f in files:
                        if f.endswith(".jp2"):
                            jp2s.append(os.path.join(root, f))
            jp2s = sorted(jp2s)  # Sort to maintain consistent band order
            if not jp2s:
                raise FileNotFoundError(
                    f"No JP2 bands found in SAFE folder {self.root}."
                )

            def pick_safe_band(band):
                matches = [
                    p
                    for p in jp2s
                    if f"_{band}_" in os.path.basename(p)
                    or f"_{band}." in os.path.basename(p)
                ]
                if not matches:
                    raise FileNotFoundError(f"Could not locate SAFE band {band}.")
                ten_meter = [p for p in matches if "_10m" in os.path.basename(p)]
                return sorted(ten_meter or matches)[0]

            def safe_band_name(path):
                basename = os.path.basename(path)
                for part in basename.replace(".", "_").split("_"):
                    if part.startswith("B") and part[1:].isalnum():
                        return part
                return os.path.splitext(basename)[0]

            lr_file_dict = {}
            for label, band_name in (
                ("R", "B04"),
                ("G", "B03"),
                ("B", "B02"),
                ("NIR", "B08"),
            ):
                try:
                    lr_file_dict[label] = pick_safe_band(band_name)
                except FileNotFoundError:
                    pass
            used = set(lr_file_dict.values())
            for path in jp2s:
                if path not in used:
                    name = safe_band_name(path)
                    if name.startswith("B") and name[1:].isalnum():
                        lr_file_dict[name] = path
            if not lr_file_dict:
                raise FileNotFoundError(f"No usable SAFE bands found in {self.root}.")

            # Get image properties
            first_band_path = next(iter(lr_file_dict.values()))
            with rasterio.open(first_band_path) as src:
                (
                    self.image_meta["width"],
                    self.image_meta["height"],
                    self.image_meta["dtype"],
                ) = (src.width, src.height, src.dtypes[0])
                self.image_meta["transform"] = src.transform
                self.image_meta["crs"] = src.crs
            same_grid = {}
            for label, path in lr_file_dict.items():
                with rasterio.open(path) as src:
                    if (
                        src.width == self.image_meta["width"]
                        and src.height == self.image_meta["height"]
                    ):
                        if (
                            src.transform != self.image_meta["transform"]
                            or src.crs != self.image_meta["crs"]
                        ):
                            raise ValueError(
                                "SAFE bands have matching dimensions but are on "
                                "different grids (transform or CRS mismatch)."
                            )
                        same_grid[label] = path
            self.image_meta["lr_file_dict"] = same_grid
            self.image_meta["bands"] = len(self.image_meta["lr_file_dict"])
            if self.image_meta["bands"] <= 0:
                raise FileNotFoundError("No SAFE bands matched the selected grid size.")

        elif self.input_type == "S2GM":
            # Get LR image file paths
            tifs = []
            for root, _, files in os.walk(self.root):
                for f in files:
                    if f.lower().endswith((".tif", ".tiff")):
                        tifs.append(os.path.join(root, f))

            if not tifs:
                raise FileNotFoundError(
                    f"No GeoTIFF bands found in S2GM folder {self.root}."
                )

            by_name = {
                os.path.splitext(os.path.basename(p))[0].upper(): p
                for p in sorted(tifs)
            }
            lr_file_dict = {}
            for label, band_name in (
                ("R", "B04"),
                ("G", "B03"),
                ("B", "B02"),
                ("NIR", "B08"),
            ):
                if band_name in by_name:
                    lr_file_dict[label] = by_name[band_name]
            used = set(lr_file_dict.values())
            for band_name, path in by_name.items():
                if (
                    path not in used
                    and band_name.startswith("B")
                    and band_name[1:].isalnum()
                ):
                    lr_file_dict[band_name] = path
            self.image_meta["lr_file_dict"] = lr_file_dict

            if not lr_file_dict:
                raise FileNotFoundError(
                    f"No usable S2GM bands found in {self.root}; "
                    "auxiliary rasters such as AOT/WVP are not spectral bands."
                )

            # Get image properties
            with rasterio.open(
                next(iter(self.image_meta["lr_file_dict"].values()))
            ) as src:
                self.image_meta["width"] = src.width
                self.image_meta["height"] = src.height
                self.image_meta["dtype"] = src.dtypes[0]
                self.image_meta["transform"] = src.transform
                self.image_meta["crs"] = src.crs
            same_grid = {}
            for label, path in self.image_meta["lr_file_dict"].items():
                with rasterio.open(path) as src:
                    if (
                        src.width == self.image_meta["width"]
                        and src.height == self.image_meta["height"]
                    ):
                        if (
                            src.transform != self.image_meta["transform"]
                            or src.crs != self.image_meta["crs"]
                        ):
                            raise ValueError(
                                "S2GM bands have matching dimensions but are on "
                                "different grids (transform or CRS mismatch)."
                            )
                        same_grid[label] = path
            self.image_meta["lr_file_dict"] = same_grid
            self.image_meta["bands"] = len(self.image_meta["lr_file_dict"])
            if self.image_meta["bands"] <= 0:
                raise FileNotFoundError("No S2GM bands matched the selected grid size.")

        # Protect every selected source band, including directory inputs.  The
        # samefile check catches an existing hardlink even when its pathname is
        # different from the source pathname.
        requested_output = getattr(self, "output_path", None)
        if requested_output and self.input_type in {"SAFE", "S2GM"}:
            output_path = os.path.abspath(os.path.expanduser(str(requested_output)))
            for source_path in self.image_meta["lr_file_dict"].values():
                source_path = os.path.abspath(str(source_path))
                try:
                    same_file = os.path.samefile(output_path, source_path)
                except (FileNotFoundError, OSError):
                    same_file = False
                if (
                    os.path.realpath(output_path) == os.path.realpath(source_path)
                    or same_file
                ):
                    raise ValueError(
                        "Output path must differ from every selected input band; "
                        "writing in place is not supported."
                    )

        # finally, add image windows to metadata
        self.image_meta["image_windows"] = self.create_image_windows()
        if not self.image_meta["image_windows"]:
            raise ValueError(
                "No image windows were created; check image size, window_size, and overlap."
            )

    def create_placeholder_file(
        self,
        force: bool = False,
        wait_timeout_s: float = 600.0,
        wait_sleep_s: float = 0.2,
        band_count=None,
        dtype=None,
        factor=None,
    ):
        """
        Create an empty HR GeoTIFF placeholder for stitched SR results.

        - Uses a run-specific staging path beside the final destination.
        - Only rank 0 creates/writes the file and ensures `temp/` exists.
        - Other ranks wait until the placeholder is readable, then return.
        - Safe on CPU and single-GPU (treated as rank 0).
        """
        import time

        def _is_rank0_env() -> bool:
            # Works before torch.distributed is initialized
            r = os.environ.get("RANK", "")
            ws = os.environ.get("WORLD_SIZE", "1")
            if ws not in ("", "1"):
                return int(r or os.environ.get("LOCAL_RANK", "0")) == 0
            return True  # single process → rank0

        def _placeholder_readable() -> bool:
            """True if placeholder exists and can be opened by rasterio (not truncated)."""
            if not os.path.exists(self.placeholder_filepath):
                return False
            try:
                with rasterio.open(self.placeholder_filepath, "r"):
                    return True
            except Exception:
                return False

        is_r0 = _is_rank0_env()

        if is_r0:
            # Ensure temp dir exists (rank 0 only)
            os.makedirs(self.temp_folder, exist_ok=True)
            self._log(f"📂 Ensured temporary folder: {self.temp_folder}")

            # If exists and not forcing, bail early
            if _placeholder_readable() and not force:
                self._log("⚠️ Placeholder file already exists — skipping creation.")
                return

            # Create (or overwrite) placeholder
            scale = int(factor if factor is not None else self.factor)
            nb = int(
                band_count
                if band_count is not None
                else self.image_meta.get("output_bands") or self.image_meta["bands"]
            )
            if nb <= 0:
                raise ValueError("Placeholder band count must be a positive integer.")
            out_dtype = str(
                dtype
                if dtype is not None
                else self.image_meta.get("output_dtype") or self.image_meta["dtype"]
            )
            W = int(self.image_meta["width"] * scale)
            H = int(self.image_meta["height"] * scale)
            tr = self.image_meta["transform"]
            save_transform = tr * Affine.scale(1 / scale)

            profile = {
                "driver": "GTiff",
                "dtype": out_dtype,
                "count": nb,
                "width": W,
                "height": H,
                "crs": self.image_meta["crs"],
                "transform": save_transform,
                "compress": "none",  # keep uncompressed while many writes happen
                "tiled": True,
                "blockxsize": 512,
                "blockysize": 512,
                "bigtiff": "YES",
                "sparse_ok": True,
                "nodata": 0,
            }

            # The private staging path is never a published final product.
            with rasterio.open(self.output_file_path, "w", **profile) as destination:
                names = self.image_meta.get("output_band_names")
                if names is not None:
                    if len(names) != nb:
                        raise ValueError(
                            "Declared output band names do not match the predicted channel count."
                        )
                    destination.descriptions = names

            self._log(
                f"💾 Saved empty placeholder SR image at: {self.output_file_path}"
            )

        else:
            # Non-rank0: wait until rank0 created a usable file and temp/
            t0 = time.time()
            while True:
                if os.path.isdir(self.temp_folder) and _placeholder_readable():
                    break
                if time.time() - t0 > wait_timeout_s:
                    raise TimeoutError(
                        f"Timed out waiting for placeholder at {self.placeholder_filepath} "
                        f"and temp dir {self.temp_folder} created by rank 0."
                    )
                time.sleep(wait_sleep_s)
            # No logging here (avoid concurrent writes); rank 0 logs already.

    def create_image_windows(self):
        """
        Generate overlapping rasterio Windows covering the input image.

        - Windows are defined at LR scale with specified size and overlap.
        - Extra windows are added at image borders to ensure full coverage.
        - Returns a list of rasterio.windows.Window objects.

        Returns
        -------
        list of rasterio.windows.Window
            Coordinates of all LR patches to process.
        """
        overlap = int(self.overlap)
        width = int(self.image_meta["width"])
        height = int(self.image_meta["height"])
        win_w = min(int(self.window_size[0]), width)
        win_h = min(int(self.window_size[1]), height)
        if width <= 0 or height <= 0:
            raise ValueError(
                f"Input image dimensions must be positive, got {width}x{height}."
            )

        def axis_offsets(length, window):
            if length <= window:
                return [0]
            stride = max(1, int(window) - overlap)
            offsets = list(range(0, length - window + 1, stride))
            final = length - window
            if offsets[-1] != final:
                offsets.append(final)
            return offsets

        window_coordinates = []
        for row_off in axis_offsets(height, win_h):
            for col_off in axis_offsets(width, win_w):
                window_coordinates.append(
                    rasterio.windows.Window(col_off, row_off, win_w, win_h)
                )

        return window_coordinates

    def delete_LR_temp(self):
        """
        Delete the temporary folder created for LR patches and intermediate files.

        Cleans up after prediction and stitching.
        """
        temp_folder = os.path.abspath(self.temp_folder)
        if not os.path.basename(temp_folder).startswith("temp_"):
            raise RuntimeError(
                f"Refusing to delete unexpected temp folder: {temp_folder}"
            )
        if os.path.isdir(temp_folder):
            shutil.rmtree(temp_folder)
            self._log(f"🗑️📂 Deleted temporary folder at: {temp_folder}")

    def run(self, debug=None, save_preview=None, cleanup=None, overwrite=None):
        """
        Execute inference, stitch the patches, and optionally write preview outputs.

        Construction prepares the object and validates inputs; this method performs
        the expensive and destructive work.
        """
        debug = self.debug if debug is None else bool(debug)
        save_preview = self.save_preview if save_preview is None else bool(save_preview)
        cleanup = self.cleanup if cleanup is None else bool(cleanup)
        overwrite = self.overwrite if overwrite is None else bool(overwrite)

        if not debug and os.path.exists(self.final_sr_path) and not overwrite:
            raise FileExistsError(
                f"Output already exists at {self.final_sr_path}. "
                "Pass overwrite=True to replace it."
            )

        self.start_super_resolution(debug=debug, save_preview=save_preview)

        if debug:
            # A truncated run is diagnostic data, never a complete SR product.
            index_path = os.path.join(self.temp_folder, "index.json")
            self._log(
                f"Debug predictions retained in {index_path}; no final raster created."
            )
            return index_path

        trainer = getattr(self, "trainer", None)
        if self._is_rank0(trainer):
            self._log("🪡 Stitching results into final sr.tif...")
            self.write_to_file(overwrite=overwrite, cleanup=cleanup)

            if cleanup:
                self.delete_LR_temp()

            if save_preview:
                from opensr_utils.data_utils.result_analysis import (
                    crop_and_save_georeferenced_excerpt,
                    generate_side_by_side_previews,
                )

                tif_path = self.final_sr_path
                crop_and_save_georeferenced_excerpt(
                    self,
                    tif_path=tif_path,
                    out_tif=os.path.join(self.log_dir, "cropped_excerpt.tif"),
                    random_crop_size=(512, 512),
                )
                self._log(
                    "✅ Saved an example SR patch to cropped_excerpt.tif in logs folder."
                )
                num_examples = 10
                generate_side_by_side_previews(
                    self,
                    tif_path=tif_path,
                    out_dir=self.log_dir,
                    num_examples=num_examples,
                )
                self._log(
                    f"✅ Saved {num_examples} side-by-side preview images to logs folder."
                )

        self.close()
        self._log("🎉 Processing done! SR process exited.")
        return self.final_sr_path

    def start_super_resolution(self, debug: bool = False, save_preview: bool = False):
        """
        Run super-resolution inference across all LR windows.

        - Creates a fresh PredictionDataModule for the selected windows.
        - Passes temp folder and window metadata to the model.
        - Uses a PyTorch Lightning Trainer to run `predict_step` in inference mode.
        - Saves SR patches into `temp/` as `.npy`/`.npz`.
        """
        import inspect
        import torch
        from pytorch_lightning import Trainer

        # Pick windows (debug trims to first 100 globally; DDP sampler will shard those across ranks)
        self._log("🚀 Running SR...")
        windows_all = self.image_meta["image_windows"]
        windows_run = windows_all[:100] if debug else windows_all
        self.create_datamodule(windows=windows_run)
        self.model.model_dm_length = len(self.datamodule.dataset)

        # Hand the hook context to the model (consumed inside predict_step)
        self.model._save_temp_folder = self.temp_folder
        self.model._save_windows = windows_run
        self.model._save_factor = int(self.factor)
        self.model._save_compressed = bool(self.compressed_patches)
        self.model._save_progress_preview = bool(save_preview)

        # Devices: list[int] or 1 for CPU/single-GPU
        local_world = int(os.environ.get("LOCAL_WORLD_SIZE", str(self._external_world)))
        devices = self.gpus if self.gpus is not None else local_world

        # Build Trainer kwargs robustly across PL versions
        trainer_kwargs = dict(
            accelerator=self.device,  # "cpu" or "cuda"
            devices=devices,  # int or list of GPU ids
            logger=False,
            enable_checkpointing=False,
            enable_model_summary=False,
            inference_mode=True,  # no_grad + eval
            enable_progress_bar=True,  # optional: cleaner logs
        )

        # Use DDP only when we truly have >1 CUDA devices
        if self._external_world > 1 or (
            isinstance(devices, (list, tuple))
            and len(devices) > 1
            and self.device == "cuda"
        ):
            trainer_kwargs["strategy"] = "ddp"
            trainer_kwargs["num_nodes"] = max(
                1, self._external_world // max(1, local_world)
            )

        # Make sure PL doesn't override our custom sampler
        sig = inspect.signature(Trainer)
        if "replace_sampler_ddp" in sig.parameters:  # PL 1.x
            trainer_kwargs["replace_sampler_ddp"] = False
        if "use_distributed_sampler" in sig.parameters:  # PL 2.x
            trainer_kwargs["use_distributed_sampler"] = False

        trainer = Trainer(**trainer_kwargs)
        self.trainer = (
            trainer  # let _log() and later code see rank via trainer.is_global_zero
        )

        # Stream predictions to disk inside predict_step (we return None there)
        try:
            trainer.predict(
                self.model, datamodule=self.datamodule, return_predictions=False
            )
        except RuntimeError as e:
            msg = str(e)
            if (
                "Lightning can't create new processes if CUDA is already initialized"
                in msg
            ):
                # Stop the workflow cleanly (no noisy traceback for users)
                self._log(
                    "🚨🔥 STOPPING WORKFLOW 🔥🚨\n"
                    "⚠️  CUDA was already initialized before launching DDP!\n"
                    "🖥️  This is a multi-GPU processing limitation of PyTorch Lightning.\n"
                    "🤔  This can happen if you pass 'None' as model with gpus>1.\n"
                    "🔄  Please restart the Python kernel or switch to single-GPU (devices=1).\n"
                    "✅  After restarting, re-run your workflow."
                )
                raise SystemExit(3)
            raise  # unrelated error: re-raise

        # Rank-aware status message
        is_ddp = torch.distributed.is_available() and torch.distributed.is_initialized()
        rank = torch.distributed.get_rank() if is_ddp else 0
        if rank == 0:
            self._log(
                f"✅ Prediction complete! SR patches saved in 📂: {self.temp_folder}"
            )
            if debug:
                self._log(
                    f"Debug mode processed {len(windows_run)} of {len(windows_all)} windows."
                )

    def _prepare_placeholder_from_prediction_index(self, index_path=None):
        """
        Create a fresh placeholder whose band count and dtype match saved patches.
        """
        if index_path is None:
            index_path = os.path.join(self.temp_folder, "index.json")
        if not os.path.exists(index_path):
            return

        with open(index_path, "r") as f:
            idx = json.load(f)

        entries = idx.get("entries", [])
        output_bands = idx.get("output_bands")
        if not output_bands:
            entry_bands = {int(e["bands"]) for e in entries if e.get("bands")}
            if len(entry_bands) == 1:
                output_bands = entry_bands.pop()

        if not output_bands and entries:
            first_path = entries[0].get("path")
            if first_path and os.path.exists(first_path):
                import numpy as np

                container = idx.get("saved_container", "npz")
                key = idx.get("saved_key", "arr")
                if container == "npz" and first_path.endswith(".npz"):
                    with np.load(first_path) as z:
                        output_bands = int(z[key].shape[0])
                else:
                    output_bands = int(np.load(first_path, mmap_mode="r").shape[0])

        if not output_bands:
            output_bands = int(self.image_meta["bands"])

        output_dtype = idx.get("saved_dtype") or self.image_meta["dtype"]
        output_factor = int(idx.get("factor", self.factor))
        self.image_meta["output_bands"] = int(output_bands)
        self.image_meta["output_dtype"] = str(output_dtype)
        self.image_meta["output_factor"] = output_factor

        # Create this after inference so output channel count can differ from input.
        self.create_placeholder_file(
            force=True,
            band_count=int(output_bands),
            dtype=str(output_dtype),
            factor=output_factor,
        )

    def write_to_file(self, index_path=None, limit=None, overwrite=None, cleanup=None):
        """Stitch on global rank zero and atomically publish a validated raster.

        Other ranks return immediately. Failed writes retain the previous output,
        patches and diagnostics; ``cleanup=False`` also retains successful patches.
        """
        from opensr_utils.data_utils.writing_utils import ddp_safe_stitch

        if overwrite is not None:
            self.overwrite = bool(overwrite)
        if self._is_rank0(getattr(self, "trainer", None)):
            self._prepare_placeholder_from_prediction_index(index_path=index_path)
        return ddp_safe_stitch(
            self,
            index_path=index_path,
            limit=limit,
            cleanup=getattr(self, "cleanup", True)
            if cleanup is None
            else bool(cleanup),
            profile="sigmoid",  # "linear"|"sigmoid"|"cosine"
            profile_alpha=6.0,
            gdal_cache_mb=256,
        )

    def close(self):
        """Release cached raster handles; retained patches and logs stay on disk."""
        datamodule = getattr(self, "datamodule", None)
        if datamodule is not None:
            datamodule.dataset.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def _is_rank0(self, trainer=None):
        """Return True only for the process that should print (rank 0)."""
        # When Trainer exists, trust it
        if trainer is not None and hasattr(trainer, "is_global_zero"):
            return bool(trainer.is_global_zero)

        # If torch.distributed already initialized, trust it
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            try:
                return torch.distributed.get_rank() == 0
            except Exception:
                pass

        # Pre-init: rely on env set by the launcher
        r = os.environ.get("RANK", "")
        lr = os.environ.get("LOCAL_RANK", "")
        ws = os.environ.get("WORLD_SIZE", "")
        if ws not in ("", "1"):  # multi-process expected
            return int(r or lr or "0") == 0

        # Single process
        return True

    def _log(self, message: str):
        """
        Rank-0 logging with print + file write and backlog support.

        Behavior
        --------
        - Always prints the message to stdout if called on rank 0.
        - Buffers messages in-memory (backlog) until `self.log_file` is set.
        - Once `self.log_file` exists (created in `create_dirs()`), all backlog
        messages are flushed and new entries are appended to the log file.
        - If writing fails (e.g. permissions), the message is preserved in backlog.

        Notes
        -----
        - Ensures no logs are lost, even if logging starts before the log file exists.
        - Timestamp is automatically prepended to every message.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] {message}"

        # always print on rank 0
        if self._is_rank0(getattr(self, "trainer", None)):
            print(line)

        # ensure backlog buffer exists
        if not hasattr(self, "_log_backlog"):
            self._log_backlog = []

        # if log_file not ready yet → backlog
        if (
            not hasattr(self, "log_file")
            or not self.log_file
            or not os.path.exists(os.path.dirname(self.log_file))
        ):
            self._log_backlog.append(line)
            return

        # if log_file ready → flush backlog first
        try:
            with open(self.log_file, "a") as f:
                if self._log_backlog:
                    for bl_line in self._log_backlog:
                        f.write(bl_line + "\n")
                    self._log_backlog.clear()
                f.write(line + "\n")
        except OSError:
            # in case file is not writable, keep line in backlog
            self._log_backlog.append(line)
