""" 
----------------------------------------------------------------
1. Instanciate Models - 10 & 20m with opensr-model
----------------------------------------------------------------
"""
import opensr_model
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"

# 10m Model
model_10m = opensr_model.SRLatentDiffusionLightning(bands="10m",device=device) 
model_10m.load_pretrained("opensr_10m_v4_v3.ckpt")

# 20m Model
#model_20m = opensr_model.SRLatentDiffusionLightning(bands="20m",device=device)
#model_20m.load_pretrained("opensr_20m_v4_v3.ckpt")


""" 
----------------------------------------------------------------
1. Run SR - 10 & 20m with opensr-utils
----------------------------------------------------------------
"""
# 2. perform SR with utils package
from opensr_utils import windowed_SR_and_saving
# create object that holds window coordinates etc
file_path = "/data1/simon/datasets/val_s2_tiles/S2B_MSIL2A_20230830T162839_N0509_R083_T16SEH_20230830T204046.SAFE/"
sr_obj = windowed_SR_and_saving(file_path, window_size=(128, 128), factor=4, keep_lr_stack=True)
# perform windowed SR - 10m
sr_obj.start_super_resolution(band_selection="10m",model=None,forward_call="forward",overlap=20, eliminate_border_px=10)
# perform windowed SR - 20m
sr_obj.start_super_resolution(band_selection="20m",model=None,forward_call="forward",overlap=2, eliminate_border_px=0)
