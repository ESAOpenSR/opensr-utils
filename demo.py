from opensr_utils import windowed_SR_and_saving

# create object that holds window coordinates etc
file_path="/data2/simon/test_s2/S2A_MSIL2A_20230729T100031_N0509_R122_T33TUG_20230729T134559.SAFE/"
sr_obj = windowed_SR_and_saving(file_path, window_size=(128, 128), factor=4, keep_lr_stack=True)


# perform windowed SR - 10m
sr_obj.start_super_resolution(band_selection="10m",model=None,forward_call="forward",overlap=20, eliminate_border_px=10)
# perform windowed SR - 20m
sr_obj.start_super_resolution(band_selection="20m",model=None,forward_call="forward",overlap=40, eliminate_border_px=20)

# get DataLoader object from sr object
from torch.utils.data import Dataset, DataLoader
from opensr_utils.main import windowed_SR_and_saving_dataset
folder_path = "/data2/simon/test_s2/S2A_MSIL2A_20230729T100031_N0509_R122_T33TUG_20230729T134559.SAFE/"
ds = windowed_SR_and_saving_dataset(folder_path=folder_path, band_selection="10m",
                    overlap=20,eliminate_border_px=10,
                    window_size=(128, 128), factor=4,keep_lr_stack=True)
dl = DataLoader(ds,batch_size=10,shuffle=False)
