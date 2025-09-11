# opensr-utils - Efficient Multi-GPU Super-Resolution of Large Scale S2-EO Data

#### 🚀 Fast multi-GPU super-resolution for massive Sentinel-2 imagery 🌍 — seamless weighted blending ✨ removes patch artifacts, while flexible input support (.SAFE 📂, S2GM 🛰️, GeoTIFF 🗺️) makes generating high-quality SR products effortless.

*WARNING*   
- This Repo is currently udnergoing major revision
- It is functional for RGB+NIR images currently.
- Functionalities coming in the future:
	- Pass more file types
	- Run SR for a selectable amount of bands
    
![img1](resources/utils_poster.png)

# Description  
This package performs super-resolution with any PyTorch or PyTorch lighning model for the Sentinel-2 10m bands (R-G-B-NIR).  

### Functionalities: 
- The Input can be either:  
	- a **".SAFE"** folder, zipped or unzipped.
	- a **"S2GM"** folder as available from the S2 Global Mosaic Hub
	- any **".tif"** file or similar that can be laoded by rasterio
- The following is performed automatically:  
	- Patching of input images by selectable size (eg 128x128)
	- Super-Resolution of individual patches with provided model
	- writing of georeferenced output raster
	- overlapping and sigmoid weightning of patches by selectable pixel amount to reduce patching artifacts - See image
	- CPU, GPU and multi-GPU inference is supported via PL-Lightning
- Supported Models:  
	- 'LightningModule': Any PL Lightning model with a .predict() or .forward() function. If this model type is passed, multi-GPU and multi-batch processing is activated, which leads to a significant inference speed increase.
	- 'torch.nn.Module': Any SR model with a .forward() function can be passed. The drawback is that for this model type, multi-GPU and multi-batch processing is not supported. This is therefore considerably slower.


## Usage example:
First, download a .SAFE tile from the Copernicus Browser, a Mosaic folder from S2G, or get your RGB-NIR .tif file ready.

#### 1. install libraries
```sql
pip install opensr-utils
pip install opensr-model
```
#### 2. Create Model - in this case our LDSR-S2 Model
```python
# Instanciate Model
import opensr_model # import pachage
model = opensr_model.SRLatentDiffusion(config, device=device) # create model
model.load_pretrained(config.ckpt_version) # load checkpoint
```
#### 3. Run large-scale Inference

```python

#
import opensr_utils
sr_object = opensr_utils.large_file_processing( 
				root=path, # File or Folder path
				model=None, # your SR model
				window_size=(128, 128), # LR window size for patching
				factor=4, # SR factor
				overlap=12, # amount of overlapping pixels that are weighted to avoid artifacts
				eliminate_border_px=2, # amount of pixels taht are discarded along the edges
				device="cuda", # set 'cuda' for GPU-accelerated inference
				gpus=0, # pass GPU ID or list of GPUs
				)
sr_object.start_super_resolution()
```

## Overlapping Strategy
In order to avoid patching artifacts that are present in many SR products, we perform a weighting based on the distance to the edge of the patches. It can be a linear weighting, but what works best is a sigmoid weight curve which puts leads to a more even edge. Additionally, in order to eliminate edge-artifacts that are present in many SR models, it is also possible to discard a fixed number of pixels along the edges of all patches.
![img3](resources/overlay_weights.png)
![img4](resources/overlay_matrix.png)

Example of Patching Artifacts in other SR models:  
![img5](resources/artifact_example.png)






[![Downloads](https://static.pepy.tech/badge/opensr-utils)](https://pepy.tech/project/opensr-utils)
