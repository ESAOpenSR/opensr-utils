# opensr-utils - Supplementary Code for the ESA OpenSR project

*WARNING*  
The 'eliminate_pixel' functionality is currently experimental and can in some cases return unexpected results.  
Only the combination of overlap:40 and eliminate_pixel:20 is proven to work. Set to 0 to disable.  
  
  
*UPDATE*  
The current version supports PyTorch Lightning multi-GPU inference, but the documentation hasnt updated yet.
  
  

*Description*  
This package performs super-resolution with any PyTorch or PyTorch lighning model for Sentinel-2 10m and 20m bands.  
This package provides useful functions to perform super-resolution of raw Sentinel-2 tiles.  
Funcitonalities:
- The Input can be either:  
	- a ".SAFE" folder, the format straight from the Copenricus Hub download.  Then, stacking of the 10 and 20m bands of Sentinel-2 '.SAFE' file format is performed (works with Sen2 downloads straight out of the box)
	- any ".tif" file or similar that can be laoded by rasterio. Either 4- or 6-band for the different models.
- The following is performed automatically:  
	- Patching of input images by selectable size (eg 128x128)
	- Super-Resolution of individual patches with provided model
	- writing of georeferenced output raster
	- overlapping and linear weightning of patches by selectable quantity to reduce patching artifacts
	- Processing is performed on the same device as the model that is passed to the funciton
- Supported Models:  
	- 'torch.nn.Module': Any SR model with a .forward() function can be passed. The drawback is that for this model type, multi-GPU and multi-batch processing is not supported. This is therefore considerably slower.
	- 'LightningModule': Any PL Lightning model with a .predict() function. If this model type is passed, multi-GPU and multi-batch processing is activated, which lkeads to a significant inference speed increase.

Usage example:
```python
!pip install opensr-utils
import opensr_utils
from opensr_utils.main import windowed_SR_and_saving 

# Create SR Object
file_path = "/yourfilepath/S2A_MSIL2A_20230729T100031_N0509_R122_T33TUG_20230729T134559.SAFE/" # define unzipped folder location of .SAFE format
sr_obj = windowed_SR_and_saving(file_path) # create required class object

# Create Model
from yourmodel import sr_model_10m,sr_model_20m
model_10m = sr_model_10m()
model_20m = sr_model_20m()

# perform windowed SR - 10m
sr_obj.start_super_resolution(band_selection="10m",model=model_10m,forward_call="forward",overlap=20, eliminate_border_px=10)
# perform windowed SR - 20m
sr_obj.start_super_resolution(band_selection="20m",model=model_20m,forward_call="forward",overlap=20, eliminate_border_px=10)
```
To start the Super-Resolution, you need to pass a model to the 'start_super_resolution' function of the 'windowed_SR_and_saving' object.  
If the call model to SR is different than 'forward' for torch.nn.Module types, you can pass the name of the call as an argument. If the input is a PyTorch Lightning model, the .predict() funciton is called.

For more information, this is the doctring of the only important function for now (not up to date):
```
	Class that performs windowed super-resolution on a Sentinel-2 image and saves the result. Steps:
        - Copies the 10m and 20m bands to new tiff files in the input directory.
        - 10m and 20m bands can be called separately and preformed with different models.
        - SR Results are saved with an averaged overlap and georeferenced in the input folder.

        Inputs:
            - folder_path (string): path to folder containing S2 SAFE data format
            - window_size (tuple): window size of the LR image
            - factor (int): SR factor
            - overlap (int): Overlap of images when writing SR results to avoid patching artifacts
            - keep_lr_stack (bool): decide wether to delete the LR stack after SR is done

        Outputs:
            - None

        Functions:
            - start_super_resolution: starts the super-resolution process. Takes model and band selection as inputs.
              Call this separately for 10m or 20m bands, sequentially
            - delete_LR_stack: deletes the LR stack after SR is done, call if not selected to do it automatically.

        Usage Example:
            # create instance of class
            sr_obj = windowed_SR_and_saving(folder_path,keep_lr_stack=True)
            # perform super-resolution on 20m bands
            sr_obj.start_super_resolution(band_selection="20m")
            # perform super-resolution on 10m bands
            sr_obj.start_super_resolution(band_selection="10m")
            # delete LR stack
            sr_obj.delete_LR_stack()
```


[![Downloads](https://static.pepy.tech/badge/opensr-utils)](https://pepy.tech/project/opensr-utils)
