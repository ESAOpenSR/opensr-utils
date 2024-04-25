


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

def save_metrics_as_json(metrics_dict, file_path):
    import json
    with open(file_path, 'w') as json_file:
        json.dump(metrics_dict, json_file)


from opensr_utils import windowed_SR_and_saving
# Tile 1
file_path = "/data1/simon/datasets/val_s2_tiles/S2B_MSIL2A_20230830T162839_N0509_R083_T16SEH_20230830T204046.SAFE/"
try:
    sr_obj = windowed_SR_and_saving(file_path, window_size=(128, 128), factor=4, keep_lr_stack=True,mode="Metrics")
    sr_obj.start_super_resolution(band_selection="10m",model=model_10m,forward_call="forward",custom_steps=200)
    save_metrics_as_json(sr_obj.metrics, file_path+"metrics.json")
except:
    print("Error in",file_path)

# Tile 2
file_path = "/data1/simon/datasets/val_s2_tiles/S2A_MSIL2A_20230811T134711_N0509_R024_T21HUB_20230811T212155.SAFE/"
try:
    sr_obj = windowed_SR_and_saving(file_path, window_size=(128, 128), factor=4, keep_lr_stack=True,mode="Metrics")
    sr_obj.start_super_resolution(band_selection="10m",model=model_10m,forward_call="forward",custom_steps=200)
    save_metrics_as_json(sr_obj.metrics, file_path+"metrics.json")
except:
    print("Error in",file_path)

# Tile 3
file_path = "/data1/simon/datasets/val_s2_tiles/S2A_MSIL2A_20230909T103631_N0509_R008_T31TDG_20230909T155201.SAFE/"
try:
    sr_obj = windowed_SR_and_saving(file_path, window_size=(128, 128), factor=4, keep_lr_stack=True,mode="Metrics")
    sr_obj.start_super_resolution(band_selection="10m",model=model_10m,forward_call="forward",custom_steps=200)
    save_metrics_as_json(sr_obj.metrics, file_path+"metrics.json")
except:
    print("Error in",file_path)

# Tile 4
file_path = "/data1/simon/datasets/val_s2_tiles/S2A_MSIL2A_20230916T084611_N0509_R107_T35SQD_20230916T131358.SAFE/"
try:
    sr_obj = windowed_SR_and_saving(file_path, window_size=(128, 128), factor=4, keep_lr_stack=True,mode="Metrics")
    sr_obj.start_super_resolution(band_selection="10m",model=model_10m,forward_call="forward",custom_steps=200)
    save_metrics_as_json(sr_obj.metrics, file_path+"metrics.json")
except:
    print("Error in",file_path)

# Tile 5
file_path = "/data1/simon/datasets/val_s2_tiles/S2B_MSIL1C_20230820T094549_N0509_R079_T34VFP_20230820T115157.SAFE/"
try:
    sr_obj = windowed_SR_and_saving(file_path, window_size=(128, 128), factor=4, keep_lr_stack=True,mode="Metrics")
    sr_obj.start_super_resolution(band_selection="10m",model=model_10m,forward_call="forward",custom_steps=200)
    save_metrics_as_json(sr_obj.metrics, file_path+"metrics.json")
except:
    print("Error in",file_path)


# Tile 6
file_path = "/data1/simon/datasets/val_s2_tiles/S2B_MSIL2A_20230916T000229_N0509_R030_T56HKH_20230916T013320.SAFE/"
try:
    sr_obj = windowed_SR_and_saving(file_path, window_size=(128, 128), factor=4, keep_lr_stack=True,mode="Metrics")
    sr_obj.start_super_resolution(band_selection="10m",model=model_10m,forward_call="forward",custom_steps=200)
    save_metrics_as_json(sr_obj.metrics, file_path+"metrics_6.json")
except:
    print("Error in",file_path)




