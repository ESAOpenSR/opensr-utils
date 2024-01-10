# opensr-utils - Supplementary Code for the ESA OpenSR project
This package provides useful functions to perform super-resolution of raw Sentinel-2 tiles.  
Funcitonalities:
- Reading and stacking of the 10 and 20m bands of Senitnel-2 .SAFE file format (worrks with Sen2 downloads straight out of the box)
- Patching of input images by selectable size (eg 128x128)
- Super-Resolution of individual patches
- writing of georeferenced output raster
- overlapping and averaging of patches by sleectable quantity to reduce patching artifacts

Usage example:
```python
from main import windowed_SR_and_saving # import package
file_path="/data2/simon/test_s2/S2A_MSIL2A_20230729T100031_N0509_R122_T33TUG_20230729T134559.SAFE/" # define unzipped folder location of .SAFE format
sr_obj = windowed_SR_and_saving(file_path) # create required class object

# perform windowed SR - 10m
sr_obj.start_super_resolution(band_selection="10m",model=None,forward_call="forward")
# perform windowed SR - 20m
sr_obj.start_super_resolution(band_selection="20m",model=None,forward_call="forward")
```
To start the Super-Resolution, you need to pass a model to the start_super_resolution function. It the call to SR is different than 'forward', you can pass the name of the call as an argument.
