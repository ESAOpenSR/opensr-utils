""" 
----------------------------------------------------------------
1. Instanciate Models - 10 & 20m with opensr-model
----------------------------------------------------------------
"""


import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import time



import opensr_model
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"


# 10m Model
model_10m = opensr_model.SRLatentDiffusionLightning(bands="10m",device=device) 
model_10m.load_pretrained("opensr_10m_v4_v6.ckpt")

# 20m Model
#model_20m = opensr_model.SRLatentDiffusionLightning(bands="20m",device=device)
#model_20m.load_pretrained("opensr_20m_v4_v3.ckpt")

""" 
----------------------------------------------------------------
2. Run xAI - 10m Uncertainty Estimation - Example
----------------------------------------------------------------
"""
from opensr_utils import windowed_SR_and_saving
file_path = "/data1/simon/datasets/inf_data/S2B_MSIL2A_20241031T105109_N0511_R051_T30SYJ_20241031T133016.SAFE/"
sr_obj = windowed_SR_and_saving(file_path, window_size=(128, 128), factor=4, keep_lr_stack=True,mode="SR")
sr_obj.start_super_resolution(band_selection="10m",model=model_10m,forward_call="forward",overlap=20, eliminate_border_px=0)

from opensr_utils import windowed_SR_and_saving
file_path = "/data1/simon/datasets/inf_data/S2A_MSIL2A_20241026T105131_N0511_R051_T30SYJ_20241026T150453.SAFE/"
sr_obj = windowed_SR_and_saving(file_path, window_size=(128, 128), factor=4, keep_lr_stack=True,mode="SR")
sr_obj.start_super_resolution(band_selection="10m",model=model_10m,forward_call="forward",overlap=20, eliminate_border_px=0)
exit()


from opensr_utils import windowed_SR_and_saving
file_path = "/data1/simon/datasets/inf_data/S2A_MSIL2A_20241026T105131_N0511_R051_T30SYJ_20241026T150453.SAFE"
sr_obj = windowed_SR_and_saving(file_path, window_size=(128, 128), factor=4, keep_lr_stack=True,mode="SR")
sr_obj.start_super_resolution(band_selection="10m",model=model_10m,forward_call="forward",overlap=20, eliminate_border_px=0)




""" 
----------------------------------------------------------------
2. Run SR - 10 & 20m with opensr-utils - Examples
----------------------------------------------------------------
"""
"""
# Test on straight tiff
from opensr_utils import windowed_SR_and_saving
file_path = "/data1/simon/datasets/val_s2_tiles/xai_testing/stacked_10m.tif"
sr_obj = windowed_SR_and_saving(file_path, window_size=(128, 128), factor=4, keep_lr_stack=True,mode="SR")
sr_obj.start_super_resolution(band_selection="10m",model=model_10m,forward_call="forward",overlap=20, eliminate_border_px=0)



# 2. perform SR with utils package
from opensr_utils import windowed_SR_and_saving
# create object that holds window coordinates etc
file_path = "/data1/simon/datasets/val_s2_tiles/S2B_MSIL2A_20230830T162839_N0509_R083_T16SEH_20230830T204046.SAFE/"
sr_obj = windowed_SR_and_saving(file_path, window_size=(128, 128), factor=4, keep_lr_stack=True)
# perform windowed SR - 10m
sr_obj.start_super_resolution(band_selection="10m",model=model_10m,forward_call="forward",overlap=20, eliminate_border_px=0)
# perform windowed SR - 20m
sr_obj.start_super_resolution(band_selection="20m",model=None,forward_call="forward",overlap=2, eliminate_border_px=0)
"""




""" 
----------------------------------------------------------------
2. Run several SRs with opensr-utils - Validation Tiles
----------------------------------------------------------------
"""

from opensr_utils import windowed_SR_and_saving
# Tile 1
file_path = "/data1/simon/datasets/val_s2_tiles/S2B_MSIL2A_20230830T162839_N0509_R083_T16SEH_20230830T204046.SAFE/"
print("SRing Tile 1, path:",file_path)
try:
    sr_obj = windowed_SR_and_saving(file_path, window_size=(128, 128), factor=4, keep_lr_stack=True)
    sr_obj.start_super_resolution(band_selection="10m",model=model_10m,forward_call="forward",overlap=20, eliminate_border_px=0)
except:
    print("Error in File path:",file_path)

# Tile 2
file_path = "/data1/simon/datasets/val_s2_tiles/S2A_MSIL2A_20230811T134711_N0509_R024_T21HUB_20230811T212155.SAFE/"
print("SRing Tile 1, path:",file_path)
try:
    sr_obj = windowed_SR_and_saving(file_path, window_size=(128, 128), factor=4, keep_lr_stack=True)
    sr_obj.start_super_resolution(band_selection="10m",model=model_10m,forward_call="forward",overlap=20, eliminate_border_px=0)
except:
    print("Error in File path:",file_path)

# Tile 3
file_path = "/data1/simon/datasets/val_s2_tiles/S2A_MSIL2A_20230909T103631_N0509_R008_T31TDG_20230909T155201.SAFE/"
print("SRing Tile 1, path:",file_path)
try:
    sr_obj = windowed_SR_and_saving(file_path, window_size=(128, 128), factor=4, keep_lr_stack=True)
    sr_obj.start_super_resolution(band_selection="10m",model=model_10m,forward_call="forward",overlap=20, eliminate_border_px=0)
except:
    print("Error in File path:",file_path)

# Tile 4
file_path = "/data1/simon/datasets/val_s2_tiles/S2A_MSIL2A_20230916T084611_N0509_R107_T35SQD_20230916T131358.SAFE/"
print("SRing Tile 1, path:",file_path)
try:
    sr_obj = windowed_SR_and_saving(file_path, window_size=(128, 128), factor=4, keep_lr_stack=True)
    sr_obj.start_super_resolution(band_selection="10m",model=model_10m,forward_call="forward",overlap=20, eliminate_border_px=0)
except:
    print("Error in File path:",file_path)

# Tile 5
file_path = "/data1/simon/datasets/val_s2_tiles/S2B_MSIL1C_20230820T094549_N0509_R079_T34VFP_20230820T115157.SAFE/"
print("SRing Tile 1, path:",file_path)
try:
    sr_obj = windowed_SR_and_saving(file_path, window_size=(128, 128), factor=4, keep_lr_stack=True)
    sr_obj.start_super_resolution(band_selection="10m",model=model_10m,forward_call="forward",overlap=20, eliminate_border_px=0)
except:
    print("Error in File path:",file_path)

# Tile 6
file_path = "/data1/simon/datasets/val_s2_tiles/S2B_MSIL2A_20230916T000229_N0509_R030_T56HKH_20230916T013320.SAFE/"
print("SRing Tile 1, path:",file_path)
try:
    sr_obj = windowed_SR_and_saving(file_path, window_size=(128, 128), factor=4, keep_lr_stack=True)
    sr_obj.start_super_resolution(band_selection="10m",model=model_10m,forward_call="forward",overlap=20, eliminate_border_px=0)
except:
    print("Error in File path:",file_path)
