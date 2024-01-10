# opensr-utils - Supplementary Code for the ESA OpenSR project
This package provides useful functions to perform super-resolution of raw Sentinel-2 tiles.  
Funcitonalities:
- Reading and stacking of the 10 and 20m bands of Sentinel-2 '.SAFE' file format (worrks with Sen2 downloads straight out of the box)
- Patching of input images by selectable size (eg 128x128)
- Super-Resolution of individual patches
- writing of georeferenced output raster
- overlapping and averaging of patches by sleectable quantity to reduce patching artifacts
- Processing is performed on the same device as the model that is passed to the funciton

Usage example:
```python
!pip install opensr-utils
import opensr_utils
from opensr_utils.main import windowed_SR_and_saving 

file_path = "/yourfilepath/S2A_MSIL2A_20230729T100031_N0509_R122_T33TUG_20230729T134559.SAFE/" # define unzipped folder location of .SAFE format
sr_obj = windowed_SR_and_saving(file_path) # create required class object

# perform windowed SR - 10m
sr_obj.start_super_resolution(band_selection="10m",model=None,forward_call="forward")
# perform windowed SR - 20m
sr_obj.start_super_resolution(band_selection="20m",model=None,forward_call="forward")
```
To start the Super-Resolution, you need to pass a model to the 'start_super_resolution' function of the 'windowed_SR_and_saving' object.  
If the call model to SR is different than 'forward',such as PyTorch lightnings 'predict' you can pass the name of the call as an argument.

For more information, this is the doctring of the only important function for now:
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
## TODo:
- Enable multi-batch calculation of the SR. Currently, the tool only super-resolutes one image at a time (1,4,128,128)