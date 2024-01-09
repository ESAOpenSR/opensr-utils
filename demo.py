
# if we have raw S2 download, create stacked image
from utils.bands10m_stacked_from_S2_folder   import extract_10mbands_from_S2_folder
from utils.bands20m_stacked_from_S2_folder import extract_20mbands_from_S2_folder

stack_needed=False
if stack_needed:
    folder_path = "/data2/simon/test_s2/S2A_MSIL2A_20230729T100031_N0509_R122_T33TUG_20230729T134559.SAFE"
    extract_10mbands_from_S2_folder(folder_path)
    extract_20mbands_from_S2_folder(folder_path)


# create object that holds window coordinates etc
from main import windowed_SR_and_saving
file_path="/data2/simon/test_s2/S2A_MSIL2A_20230729T100031_N0509_R122_T33TUG_20230729T134559.SAFE/"
sr_obj = windowed_SR_and_saving(file_path)


# perform windowed SR - 10m
sr_obj.start_super_resolution(band_selection="10m",model=None,forward_call="forward")
# perform windowed SR - 20m
sr_obj.start_super_resolution(band_selection="20m",model=None,forward_call="forward")

