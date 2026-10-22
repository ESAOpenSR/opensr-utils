# Behavior tests

These tests define observable contracts for future refactors. They use small
local rasters, known pixel values and deterministic models rather than pretrained
weights. HTTP requests through `requests` are blocked by an autouse fixture;
model-loading tests mock the download/checkpoint boundary. No GPU is required.

## Run locally

Use a virtual environment with Python and the package's runtime dependencies:

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[dev,cog]'
python -m pytest -q
```

In the current workspace the prepared environment is `/work/envs/opensr`:

```bash
/work/envs/opensr/bin/python -m pytest -q
```

Runtime dependencies are required: collection should fail if they are missing,
rather than allowing CI to pass by skipping most tests. CI runs the full suite
on CPU. To select cases:

```bash
python -m pytest -m integration -q
python -m pytest -m 'not integration and not known_bug' -q
python -m pytest -m known_bug --runxfail -q
python -m ruff check tests
python -m ruff format --check tests
```

`known_bug` tests define desired behavior that production code does not yet
provide. They use strict `xfail`: fixing the behavior causes CI to fail until the
marker is removed. Failure types are constrained to the reproduced defect.
`--runxfail` disables those markers to expose the actual failures. The defect
backlog and remaining design decisions are in [TODO.md](../TODO.md).

## Contracts and scope

| Area | Contract |
| --- | --- |
| Construction | Validate arguments before creating files; inference is explicit by default. |
| Band selection | Common channels are R/G/B/NIR; preserve additional spectral channels on the selected grid; ignore auxiliary files. |
| Input values | Read as float32, divide by 10000 and clamp to [0,1]; retain file channel order. |
| Tiling | Deterministic, unique, bounded windows cover every pixel, including narrow images and final edges. |
| Sampling/batching | Every index appears exactly once across ranks; never pad distributed inference or drop a final partial batch. |
| Model execution | Evaluation/inference mode; output channels may differ from input channels; dimensions obey the scale factor. |
| Patch persistence | NPY and compressed NPZ carry uint16 predictions and manifests with consistent LR/HR coordinates and channel metadata. |
| Blending | All three profiles handle valid zero values, preserve outer borders and support symmetric horizontal/vertical overlaps. |
| Output | Preserve CRS and spatial footprint; compare real tiled output against independently calculated expected pixels. |
| File safety | Protect source data and existing outputs; reject incomplete products; retain diagnostics after failures. |
| CLI/model loader | Forward options, reject malformed arguments, validate HTTP responses and stop after failed loading stages. |
| Auxiliary tools | Cold imports avoid loading ML dependencies or writing files; COGs and excerpts retain pixels and spatial metadata. |

The full CPU inference tests use nearest-neighbor models so expected output is
computed independently with NumPy repeats. A one-DN absolute tolerance accounts
for float32 weighted accumulation followed by integer casting. Array/layout and
manifest tests otherwise check exact values or explicit numerical tolerances.

SAFE discovery fixtures use GeoTIFF payloads with `.jp2` filenames; Rasterio
detects the actual format. They exercise naming, resolution preference and band
selection, not JP2 codec support. COG conversion uses the real converter and
validator. Distributed manifest tests simulate rank state and on-disk shards;
they do not replace multi-process/GPU testing.

Prefer new tests at public boundaries, using hand-calculated expectations and
failure-preservation checks. Mock external services and rank/process context;
keep the raster I/O, models, persistence and stitching real where inexpensive.
Use `pipeline_factory` / `dataset_factory` for automatic raster-handle cleanup.
Keep unresolved units, nodata and debug-output choices in the backlog until
their intended behavior is decided.
