# Model examples

The examples run the public tiled pipeline and write labeled, georeferenced
uint16 GeoTIFFs. Run any script with `--help` before downloading weights. Use
Python 3.12–3.14 to install all model families in one environment.

## All-band SEN2SR with an LDSR-S2 backbone

`run_sen2sr.py` combines SEN2SRLite's learned red-edge/SWIR fusion with **LDSR-S2
as its RGB-NIR backbone**. It reads and writes ten bands in this exact order:

```text
B02 B03 B04 B05 B06 B07 B08 B8A B11 B12
```

B01, B09 and B10 are outside the upstream model's supported bands. Native
SAFE/S2GM folders can contain both 10 m and 20 m sources: windowed
nearest-neighbor reads map them to the RGB 10 m reference grid before SEN2SR's
learned 20 m → 10 m → 2.5 m stages. The output has ten bands at 2.5 m. Standalone
GeoTIFFs must already have a common grid; descriptions select bands by name, or
use `--bands` with ten 1-based indexes in the order above.

```bash
python -m pip install 'opensr-utils[sen2sr-ldsr]' huggingface_hub

# Automatic SEN2SR fusion and LDSR-S2 backbone downloads.
python examples/run_sen2sr.py scene.SAFE.zip --device cuda --output sen2sr_all.tif

# Diffusion batch size defaults to 1. Sampling steps control inference cost;
# omit this option to use the backbone's configured quality/speed settings.
python examples/run_sen2sr.py aligned_10band.tif --sampling-steps 20 \
  --batch-size 1 --device cuda --output sen2sr_all.tif

# Download every fusion artifact, including the 20m/10m fusion weights.
hf download tacofoundation/SEN2SR --include 'SEN2SRLite/main/**' \
  --local-dir models/SEN2SR
hf download simon-donike/RS-SR-LTDF opensr-ldsrs2_v1_0_0.ckpt \
  --local-dir models/LDSRS2
```

The LDSR-S2 checkpoint also needs a matching configuration. Obtain
[config_10m.yaml from opensr-model](https://github.com/ESAOpenSR/opensr-model/blob/main/opensr_model/configs/config_10m.yaml)
and save it as `models/LDSRS2/config_10m.yaml`. Keep the checkpoint/config pair
from the same model release. Fully offline inference then uses:

```bash
python examples/run_sen2sr.py scene.SAFE.zip --device cuda --no-download \
  --checkpoint models/SEN2SR/SEN2SRLite/main \
  --backbone-checkpoint models/LDSRS2/opensr-ldsrs2_v1_0_0.ckpt \
  --backbone-config models/LDSRS2/config_10m.yaml --output sen2sr_all.tif
```

This composes the official [SEN2SR](https://github.com/ESAOpenSR/SEN2SR) fusion
model with [LDSR-S2](https://github.com/ESAOpenSR/opensr-model). Smaller windows
are replicate-padded to 128×128 and cropped after inference. All Fourier masks
and diffusion device state follow CPU/CUDA placement.

## LDSR-S2 RGB-NIR only

`run_ldsrs2.py` runs the diffusion model alone and writes four bands in
**B04, B03, B02, B08** (R, G, B, NIR) order at 4× scale. SAFE/S2GM inputs select
those bands automatically. For an unlabeled B, G, R, NIR GeoTIFF, use
`--bands 3 2 1 4`.

```bash
python -m pip install 'opensr-utils[model]'
python examples/run_ldsrs2.py scene.SAFE.zip --device cuda --output ldsrs2_rgbn.tif
python examples/run_ldsrs2.py input_bgrn.tif --bands 3 2 1 4 \
  --device cpu --output ldsrs2_rgbn.tif

# Reuse the local checkpoint/config pair downloaded above.
python examples/run_ldsrs2.py scene.SAFE.zip --device cuda --no-download \
  --checkpoint models/LDSRS2/opensr-ldsrs2_v1_0_0.ckpt \
  --config models/LDSRS2/config_10m.yaml --output ldsrs2_rgbn.tif
```

`--sampling-steps` overrides the backbone's configured diffusion steps. CPU is
supported but diffusion inference benefits substantially from CUDA.

## ESA SRGAN RGB-NIR

`run_esasrgan.py` uses the official four-band `RGB-NIR` preset, including its
histogram matching and EMA inference path. Its optional package supports Python
3.12–3.14.

```bash
python -m pip install 'opensr-utils[srgan]' huggingface_hub
python examples/run_esasrgan.py input.tif --device cuda --output esasrgan.tif

hf download simon-donike/SR-GAN config_RGB-NIR.yaml RGB-NIR_4band_inference.ckpt \
  --local-dir models/SR-GAN
python examples/run_esasrgan.py input.tif --no-download \
  --checkpoint models/SR-GAN/RGB-NIR_4band_inference.ckpt \
  --config models/SR-GAN/config_RGB-NIR.yaml --output esasrgan.tif
```

Upstream [inference documentation](https://srgan.opensr.eu/api-reference/inference/)
and [source](https://github.com/ESAOpenSR/SRGAN).

All dedicated examples accept `--device`, `--batch-size`, `--window-size`,
`--overlap`, `--eliminate-border-px`, and `--overwrite`. `run_model.py` exposes
additional generic controls for interpolation, multiple GPUs, previews,
diagnostic runs and retaining predictions. Examples are safe to import and
`--help` does not import Torch or download model artifacts.
