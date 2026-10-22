# OpenSR Utils

Super-resolution for large geospatial rasters with PyTorch or PyTorch Lightning
models. OpenSR Utils discovers input bands, batches overlapping windows, streams
predictions to disk, and blends them into a georeferenced GeoTIFF. Plain PyTorch
modules and Lightning modules both support batched CPU, CUDA and multi-GPU inference.

[![DOI](https://zenodo.org/badge/716535395.svg)](https://doi.org/10.5281/zenodo.17599309)

![OpenSR workflow](resources/utils_poster.png)

Install the package, or include the optional pretrained model loader:

```bash
pip install opensr-utils
pip install 'opensr-utils[model]'
```

The CI matrix covers Python **3.9–3.14**, with pinned older PyTorch/Lightning
stacks and current compatible dependencies. The full suite is also exercised
locally on Python 3.12 and 3.14. Use `opensr-utils[cog]` for COG conversion and
`opensr-utils[dev,cog]` for development. For current model installations, Python
**3.12–3.14** supports both SEN2SR and ESA SRGAN. Python 3.15 is not in the tested
matrix because [released PyTorch wheels](https://pypi.org/project/torch/#files)
currently cover interpreters only through 3.14.

## Python usage

Provide an input, model and output. The device, batching, tiling, temporary
storage, stitching and cleanup have defaults:

```python
from opensr_utils import super_resolve

output = super_resolve('scene.SAFE.zip', model='LDSRS2', output='results/scene_sr.tif')
```

Pass your own `torch.nn.Module` or `LightningModule` instead of a model name.
`model=None` performs bilinear interpolation and requires no pretrained weights.
The scale is taken from the model's declared `factor`, falling back to 4.
Use `factor=1` for inference without upscaling; an explicit factor must match a
named model's actual scale.

```python
output = super_resolve(
    'scene.tif', model=my_model, output='results/sr.tif',
    factor=2, batch_size=8, device='cpu', cleanup=False,
)
```

For preparation, inspection and explicit execution, the existing class remains
available. Construction prepares the run without invoking the model:

```python
from opensr_utils import large_file_processing

with large_file_processing(
    root='scene.tif', model=my_model, output_path='results/sr.tif', factor=2,
) as job:
    print(len(job.image_meta['image_windows']))
    output = job.run()
```

The one-call API always executes once and closes raster handles. The class also
has `close()` for callers who do not use a context manager.

## SEN2SR and ESA SRGAN

The named loaders return adapters for the models' actual inference contracts:

| Name | Supported pretrained model | Python | Input/output |
| --- | --- | --- | --- |
| `SEN2SR` | SEN2SRLite `NonReference_RGBN_x4` | 3.10–3.14 | Four RGB/NIR bands, 4× scale. |
| `ESASRGAN` or `SRGAN` | ESA OpenSR SRGAN `RGB-NIR` preset | 3.12–3.14 | Four RGB/NIR bands, 4× scale. |
| `LDSRS2` | ESA LDSR-S2 | Upstream package compatibility | Four RGB/NIR bands, 4× scale. |

```bash
python -m pip install 'opensr-utils[sen2sr]'
python -m pip install 'opensr-utils[srgan]'
```

```python
from opensr_utils import super_resolve

super_resolve('scene.SAFE', 'SEN2SR', 'results/sen2sr.tif', batch_size=1)
super_resolve('scene.SAFE', 'ESASRGAN', 'results/srgan.tif', batch_size=4)
```

SEN2SRLite expects fixed 128×128 tiles. The adapter replicate-pads smaller edge
windows and crops predictions back to their true footprint. Larger windows are
rejected clearly. The four-band adapter does not load the ten-band/full Mamba
variants. The ESA SRGAN adapter preserves the official `predict_step` path,
including histogram matching and EMA behavior; pass the adapter rather than an
unwrapped upstream Lightning module to retain those semantics. All adapters use
normalized values, explicit scales and named output bands.

SAFE/S2GM model inputs select and order the required **B04, B03, B02, B08** bands,
ignoring other spectral channels. GeoTIFF band descriptions can identify those
bands automatically; otherwise provide four native RGB/NIR channels or explicit
1-based indexes, for example `band_indexes=[4, 5, 2, 1]` (`--bands 4 5 2 1` on CLI).
Output GeoTIFFs retain the adapter's spectral band names.

See **[model demos and download instructions](examples/README.md)** for automatic
and offline use of official weights. The instruction script has complete help:

```bash
python demo.py --help
python demo.py scene.tif --model none --factor 1 --output results/check.tif
python examples/run_sen2sr.py scene.SAFE.zip --batch-size 1 --output results/sen2sr.tif
python examples/run_esasrgan.py scene.tif --batch-size 4 --output results/srgan.tif
```

`demo.py` performs no inference on import and accepts device/GPU selection,
checkpoints, configs, download/cache controls, band indexes, tiling, batching,
overwrite, retention and previews. `--device cuda:1` is a convenience for selecting
a single GPU; `--device cuda --gpus 0 1` selects multiple GPUs.

## Command line

```bash
# Pretrained LDSR-S2; model installation/download access is required.
opensr-run scene.SAFE.zip LDSRS2 -o results/scene_sr.tif

# Four-band pretrained models, with optional model extras installed.
opensr-run scene.SAFE SEN2SR -o results/sen2sr.tif --batch-size 1
opensr-run scene.tif SRGAN -o results/srgan.tif --batch-size 4

# Interpolation, using the automatic device and default 4× scale.
opensr-run scene.tif -o results/interpolated.tif

# Advanced throughput and blending settings.
opensr-run scene.tif LDSRS2 -o results/sr.tif \
  --device cuda --gpus 0 1 --batch-size 8 \
  --window-size 128 128 --overlap 12 --eliminate-border-px 2 --save-preview
```

`python -m opensr_utils` provides the same interface. Existing underscore options
such as `--window_size` and `--keep_temp` remain aliases. Model-loading and
processing failures report a diagnostic and return a nonzero exit status.

| Setting | Default | Meaning |
| --- | --- | --- |
| `output` / `-o` | `sr.tif` beside resolved input | Python class uses `output_path`; S2GM defaults inside its folder. |
| `window_size` / `--window-size` | `(128, 128)` | LR window width and height, clipped for small inputs. |
| `factor` / `--factor` | Model scale, otherwise `4` | Positive integer model scale; declared model scales are validated. |
| `overlap` / `--overlap` | `8` | Even overlap in LR pixels, smaller than both window dimensions. |
| `eliminate_border_px` / `--eliminate-border-px` | `0` | Even interior border discard in LR pixels; overlap minus discard must be at least 4 when overlap is nonzero. |
| `device` / `--device` | `auto` | CUDA when available, otherwise CPU; explicit `cpu` and `cuda` are supported. |
| `gpus` / `--gpus` | GPU 0 for CUDA | GPU index or list of indices in Python; list of IDs on CLI. |
| `batch_size` / `--batch-size` | `16` | Tune for model memory use; incomplete final batches are preserved. |
| `num_workers` / `--num-workers` | `0` | Increase for parallel raster reads when I/O limits throughput. |
| `prefetch_factor` / `--prefetch-factor` | `2` | Positive batches per worker; ignored with zero workers. |
| `output_units` / `--output-units` | Model units, otherwise `normalized` | Explicit prediction units, described below. |
| `band_indexes` / `--bands` | Model descriptions or native order | Select 1-based channels from a file in model input order. |
| `checkpoint` / `--checkpoint` | Download official preset | Local SEN2SR MLSTAC directory/export or ESA SRGAN weights. |
| `model_config` / `--config` | Preset config | ESA SRGAN YAML accompanying local weights. |
| `cache_dir` / `--cache-dir` | Loader cache | Directory for downloaded artifacts. |
| `download=False` / `--no-download` | Download permitted | Require local model artifacts. |
| `compressed_patches` / `--compressed-patches` | `False` | Smaller NPZ intermediates; NPY avoids compression overhead. |
| `cleanup=False` / `--keep-temp` | Cleanup enabled | Keep every prediction and its manifest after a successful run. |
| `overwrite` / `--overwrite` | `False` | Allow atomic replacement of a previous successful output. |
| `delete_input_zip` / `--delete-input-zip` | `False` | Delete the source ZIP only after usable input validation. |
| `save_preview` / `--save-preview` | `False` | Save inference previews, a georeferenced crop and LR/SR comparisons. |
| `debug` / `--debug` | `False` | Predict at most 100 windows, keep diagnostics, and create no final product. |

## Input and output contracts

Supported inputs are Rasterio-readable files, Sentinel-2 `.SAFE` directories,
ZIP archives containing a SAFE product, and S2GM band folders. SAFE/S2GM common
channels are ordered **R, G, B, NIR** (B04, B03, B02, B08), followed by additional
spectral channels on the selected grid. Auxiliary bands such as AOT, WVP and TCI
are excluded. Generic PyTorch models retain native file band order and channel count; adapters
with declared spectral inputs select named or explicitly indexed bands.
SAFE discovery prefers 10 m common bands. Bands on other-sized grids are omitted;
same-sized bands with different transforms or CRSs are rejected. No implicit
reprojection occurs. Models must accept the discovered channels; model output
channel counts may differ from input counts. LDSR-S2 expects Sentinel-2 RGB/NIR
inputs; use a model suited to mosaics for S2GM data.

Inputs are float32 values divided by 10000 and clipped to [0,1]. Masked or
nonfinite input pixels are filled with zero for the model. Predictions must be
finite tensors shaped `(batch, channels, height * factor, width * factor)` with
matching batch and metadata sizes. They are rounded and saved as uint16:

- `output_units='normalized'`: clip to [0,1] and multiply by 10000.
- `output_units='reflectance'`: clip existing digital-number values to [0,10000].

The former brightness-based units heuristic has been removed. A dark model
output of `5` means 5 DN only with `output_units='reflectance'`; the normalized
contract clips it to 1 and saves 10000 DN. Input normalization is unchanged.

The output preserves CRS and the complete affine footprint, including rotation
and shear. Source validity is the intersection of every input band's Rasterio
mask and finite values. That validity is upscaled by nearest-neighbor mapping
into an **internal GeoTIFF mask**, independently of prediction values. Valid
zero predictions remain valid. A common source nodata value is retained if
representable as uint16; otherwise nodata metadata is cleared. The internal
mask is authoritative; values under masked pixels are unspecified. Applications
should read masks, for example with `rasterio.open(output).read(masked=True)`.

## Output safety and diagnostics

Source rasters are protected against output-path, symlink and hardlink
collisions. A successful run atomically publishes a staging file beside the
final destination, after every patch and full raster coverage have validated.
An overwrite failure leaves the previous output intact. Publishing without
overwrite also protects against another job creating the destination during
stitching. Successful overwrites remove stale external mask/overview sidecars. Failed runs keep
patches and diagnostics for inspection. Retention keeps the manifest and every
patch it references; use that manifest to inspect coordinates, channels and units.
Logs and temporary files have separate run-specific directories beside the output.

Debug runs return the retained `index.json` path, even if fewer than 100 windows
exist. They never publish or replace a final raster. Run again with `debug=False`
for a complete product; normal full-coverage checks still apply.

For multiple GPUs, pass `gpus=[0, 1]` and let Lightning launch processes. Distributed
sampling never pads or duplicates windows and permits empty ranks. Externally
launched jobs must share an output filesystem and a unique `OPENSR_RUN_ID` across
ranks before construction (an already initialized process group can share it
automatically). `LOCAL_WORLD_SIZE` describes devices per node; global rank owns
publication. Distributed tests exercise real CPU/Gloo communication, with optional
GPU coverage where hardware is available; they do not establish multi-node
filesystem or GPU-driver compatibility for every deployment.

## Blending and development

Overlaps use weighted accumulation so four-way intersections and valid zeros
are handled consistently. Interior border discard does not remove the outer
image boundary. The lower-level stitcher supports linear, sigmoid and cosine
profiles; the pipeline uses sigmoid weighting.

![Overlap weighting](resources/overlay_weights.png)

See [tests/README.md](tests/README.md) for behavior tests, branch coverage,
real JP2 fixtures, distributed checks and model-adapter contracts. The runnable
[examples](examples/README.md) explain pretrained downloads and SR product creation.

## Citation

```bibtex
@software{Donike_OpenSR_Utils_2025,
  author = {Donike, Simon and Gomez-Chova, Luis},
  title = {OpenSR-Utils},
  license = {MIT},
  url = {https://github.com/ESAOpenSR/opensr-utils},
  year = {2025},
  doi = {10.5281/zenodo.17599310}
}
```
