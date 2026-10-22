# Model examples

These scripts use the same tiled, georeferenced pipeline as the public API.
They support CPU and CUDA inference. Model downloads happen only when a model
is selected and `--no-download` is absent.

## SEN2SR RGB-NIR

The supported adapter is SEN2SRLite `NonReference_RGBN_x4`: a 4x model trained
on fixed 128x128 low-resolution tiles. Smaller edge tiles are padded and
cropped by the adapter. It expects channels in `B04, B03, B02, B08` order.
For an unlabeled BGRN GeoTIFF, select that order explicitly with the root demo:

```bash
python -m pip install 'opensr-utils[sen2sr]' huggingface_hub

# Automatic model-artifact download and CPU inference.
python examples/run_sen2sr.py input.SAFE.zip --output sen2sr.tif
python examples/run_sen2sr.py input.tif --device cuda --output sen2sr.tif

# Optional offline cache. The directory contains the downloaded mlm.json.
hf download tacofoundation/SEN2SR \
  --include 'SEN2SRLite/NonReference_RGBN_x4/**' \
  --local-dir models/SEN2SRLite_RGBN
python examples/run_sen2sr.py input.SAFE.zip \
  --checkpoint models/SEN2SRLite_RGBN/SEN2SRLite/NonReference_RGBN_x4 \
  --no-download --output sen2sr.tif

# If a standalone raster is stored as B, G, R, N, reorder it to R, G, B, N.
python demo.py input_bgrn.tif --model sen2sr --bands 3 2 1 4 \
  --output sen2sr.tif
```

SEN2SR documentation: <https://github.com/ESAOpenSR/SEN2SR>.

## SRGAN / ESASRGAN RGB-NIR

The supported preset is the 4-band RGB-NIR generator. Install the published
extra (Python 3.12 through 3.14), then choose either automatic or local
artifacts:

```bash
python -m pip install 'opensr-utils[srgan]>=0.5.1'

# Automatic preset download and CPU/GPU inference.
python examples/run_esasrgan.py input.tif --output esasrgan.tif
python examples/run_esasrgan.py input.tif --device cuda --output esasrgan.tif

# Offline local checkpoint and matching configuration.
hf download simon-donike/SR-GAN config_RGB-NIR.yaml RGB-NIR_4band_inference.ckpt \
  --local-dir models/SR-GAN
python examples/run_esasrgan.py input.tif \
  --checkpoint models/SR-GAN/RGB-NIR_4band_inference.ckpt \
  --config models/SR-GAN/config_RGB-NIR.yaml --no-download \
  --output esasrgan.tif
```

SRGAN documentation: <https://srgan.opensr.eu/api-reference/inference/>.
Source: <https://github.com/ESAOpenSR/SRGAN>.

Both examples accept `--device`, `--batch-size`, `--window-size`, `--overlap`,
`--eliminate-border-px`, and `--overwrite`. The model scale is 4x by default;
use `--factor` for a compatible checkpoint with another scale. OpenSR-Utils
normalizes model inputs to [0, 1] and writes uint16 GeoTIFF products.
