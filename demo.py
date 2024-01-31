from opensr_utils import windowed_SR_and_saving

# create object that holds window coordinates etc
file_path="/data2/simon/test_s2/S2A_MSIL2A_20230729T100031_N0509_R122_T33TUG_20230729T134559.SAFE/"
sr_obj = windowed_SR_and_saving(file_path, window_size=(128, 128), factor=4, keep_lr_stack=True)


# perform windowed SR - 10m
sr_obj.start_super_resolution(band_selection="10m",model=None,forward_call="forward",overlap=20, eliminate_border_px=0)
# perform windowed SR - 20m
sr_obj.start_super_resolution(band_selection="20m",model=None,forward_call="forward",overlap=20, eliminate_border_px=0)

