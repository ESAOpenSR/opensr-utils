# Behavior tests

Tests use local rasters, known pixel values and deterministic models. Pretrained
weights and HTTP access are unnecessary: an autouse fixture blocks `requests`
network calls, and model-loader tests mock the download/checkpoint boundary.
Runtime dependencies are required during collection. Fixed maintenance regressions
run as normal tests, with no expected-failure markers.

## Run locally

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[dev,cog]'
python -m pytest -q
python -m ruff check opensr_utils tests examples setup.py demo.py
python -m ruff format --check opensr_utils tests examples setup.py demo.py
```

The prepared workspace environment is `/work/envs/opensr/bin/python`. The real
process tests need permission to open local loopback sockets; restricted sandboxes
must enable local networking for those tests. They do not contact external services.
CPU/Gloo workers use bounded process-group and process-join timeouts. The Lightning
check runs in a bounded subprocess with a dynamically selected port.

```bash
python -m pytest -m integration -q
python -m pytest tests/test_distributed.py tests/test_distributed_trainer.py -q
python -m pytest --cov=opensr_utils --cov-branch --cov-report=term-missing --cov-report=html
```

CI covers Python 3.9–3.14. Its oldest interpreter uses the constrained versions in
[constraints-oldest.txt](../.github/constraints-oldest.txt); other jobs use current
compatible dependencies. CPU PyTorch and real COG dependencies are installed.
Branch-aware XML coverage reports are uploaded per interpreter. Coverage is used
to identify missing behavior cases; no arbitrary percentage gate masks the
importance of individual branches. Child-process execution is validated through
observable files and process outcomes; its lines may not appear in the parent
coverage report.

The model tests keep official downloads outside the normal test suite. Adapter
tests exercise tensor values, batch shapes, tile padding/cropping, normalization,
ESA histogram-inference delegation, band selection and labeled GeoTIFF output.
Pretrained smoke runs use the published SEN2SRLite RGBN and ESA RGB-NIR artifacts
following [the model examples](../examples/README.md). Optional model dependencies
can be installed with `.[sen2sr]` (Python 3.10+) and `.[srgan]` (Python 3.12–3.14).

## Contracts and scope

| Area | Observable contract |
| --- | --- |
| API and CLI | One-call execution, explicit output destinations, automatic device defaults, nonzero failure status, and handle cleanup. |
| Construction | Validate arguments before inference; protect source/output collisions and preserve existing outputs. |
| Discovery | R/G/B/NIR order, auxiliary exclusion, preferred resolution, selected-grid channels, and rejection of spatial mismatches. |
| Input values | Float32 normalized/clipped values, native file channel order, and zero-filled masked/nonfinite inputs. |
| Tiling | Unique deterministic windows cover image edges, rectangular/tiny/narrow rasters and every pixel. |
| Sampling | Every index appears once across ranks; no padded inference samples or dropped partial batches. |
| Model execution | Evaluation/inference mode, dynamic output channels, explicit units, finite values and complete batch validation. |
| Persistence | NPY/NPZ uint16 values and manifests with consistent LR/HR offsets, channels and units. |
| Blending | Linear/sigmoid/cosine weighting, four-way intersections, valid zeros, outer borders and symmetric overlaps. |
| Spatial output | Correct CRS, full affine footprint including rotation/shear, dimensions and independently calculated pixels. |
| Masks | Intersect input validity, nearest-neighbor upscaling, internal masks, source nodata metadata and valid zero predictions. |
| Safety | Atomic publication, failed overwrite preservation, full coverage, retained diagnostics, complete retention and ZIP validation. |
| Debug | At most 100 retained windows, no final product, and a subsequent exhaustive full run. |
| Distributed | Real Gloo communication, empty ranks, rank identities, shard/write/merge errors, lost peers and recovery. |
| Auxiliary tools | Lightweight cold imports, crop georeferencing and real COG conversion/validation. |

Full CPU inference tests compare nearest-neighbor models with independent NumPy
repeats. A one-DN tolerance accounts for weighted float32 accumulation followed by
integer casting. Other tests use exact values or explicit numerical tolerances.
Mask tests include partial output blocks larger than 512 pixels, factor-three
upscaling, differing source masks and nonfinite values.

Most SAFE discovery fixtures store GeoTIFF payloads under SAFE filenames for
speed. A separate fixture creates and reads actual JP2OpenJPEG bands, verifies
the driver and tolerates lossy-codec quantization. It skips only when that driver
is unavailable; other discovery tests still run. The small generated products
cannot represent every vendor/product-generation variant.

Real Gloo workers and a two-rank CPU Lightning run complement the simulated shard
unit tests. The optional two-GPU NCCL test skips without suitable hardware. These
checks do not establish all CUDA-driver or multi-node shared-filesystem behavior.

Prefer tests at public boundaries with hand-calculated expectations and failure
preservation checks. `pipeline_factory` and `dataset_factory` close raster handles
automatically. Keep mocks at external boundaries and use real raster I/O, model
execution, persistence and stitching when inexpensive.
