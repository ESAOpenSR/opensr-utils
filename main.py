import rasterio
from einops import rearrange
from rasterio.transform import Affine
import json
from torchvision import transforms
import numpy as np
import torch
from tqdm import tqdm

# local imports
from utils.denormalize_image_per_band_batch import denormalize_image_per_band_batch as denorm
from utils.stretching import hq_histogram_matching


class windowed_SR_and_saving():
    
    def __init__(self, file_path, window_size=(128, 128),apply_norm=True,
                         dataset_type="S2NAIP",
                         band_stats_path="utils/band_statistics.json"):
        self.file_path = file_path
        self.window_size = window_size
        self.bands = [0,1,2,3]# Sen2: [4,3,2,8]
        self.apply_norm = apply_norm
        self.factor=4
        self.overlap=8
        
        # Get File information
        with rasterio.open(self.file_path) as src:
            self.img_width, self.img_height = src.width, src.height
            # Extract the affine transformation matrix
            self.geo_transform = src.transform
            # Extract the CRS
            self.crs = src.crs
            
        # get list of windows for windowed reading
        self.window_coordinates = self.create_window_coordinates_overlap()
        
        # define tensor to be filled with SR
        self.sr_tensor_placeholder = np.zeros((len(self.bands), 
                                               self.img_width*self.factor,
                                               self.img_height*self.factor),dtype=np.float64)
        
        # initialize transformations
        self.init_transforms(apply_transforms=True)
        
        # set denorm params
        self.denorm = denorm
        
    def create_window_coordinates_overlap(self):
        overlap = self.overlap
        # Calculate the number of windows in each dimension
        self.n_windows_x = (self.img_width - overlap) // (self.window_size[0] - overlap)
        self.n_windows_y = (self.img_height - overlap) // (self.window_size[1] - overlap)

        # Create list of batch windows coordinates
        window_coordinates = []
        for win_y in range(self.n_windows_y):
            for win_x in range(self.n_windows_x):
                # Define window to read with overlap
                window = rasterio.windows.Window(
                    win_x * (self.window_size[0] - overlap),
                    win_y * (self.window_size[1] - overlap),
                    self.window_size[0],
                    self.window_size[1]
                )
                window_coordinates.append(window)

        # Check for any remaining space after the sliding window approach
        final_x = self.img_width - self.window_size[0]
        final_y = self.img_height - self.window_size[1]

        # Add extra windows for the edges if there's remaining space
        # Adjust the check to handle the overlap correctly
        if final_x % (self.window_size[0] - overlap) > 0:
            for win_y in range(self.n_windows_y):
                window = rasterio.windows.Window(
                    self.img_width - self.window_size[0],
                    win_y * (self.window_size[1] - overlap),
                    self.window_size[0],
                    self.window_size[1]
                )
                window_coordinates.append(window)

        if final_y % (self.window_size[1] - overlap) > 0:
            for win_x in range(self.n_windows_x):
                window = rasterio.windows.Window(
                    win_x * (self.window_size[0] - overlap),
                    self.img_height - self.window_size[1],
                    self.window_size[0],
                    self.window_size[1]
                )
                window_coordinates.append(window)

        # Handle corner case if both x and y have remaining space
        if (final_x % (self.window_size[0] - overlap) > 0 and
                final_y % (self.window_size[1] - overlap) > 0):
            window = rasterio.windows.Window(
                self.img_width - self.window_size[0],
                self.img_height - self.window_size[1],
                self.window_size[0],
                self.window_size[1]
            )
            window_coordinates.append(window)

        # Return filled list of coordinates
        return window_coordinates
    
    
    def __len__(self):
        return(len(self.window_coordinates))
    
    def get_window(self,idx):
        # TODO: perform batched SR instead of single image per batch
        # assert number required is valid
        assert idx>=0 and idx<len(self.window_coordinates), "idx not in range of windows"
        # get window of current idx
        current_window = self.window_coordinates[idx]
        
        # open file SRC
        with rasterio.open(self.file_path) as src:
            data = src.read(window=current_window)
            # select bands
            data = data[self.bands,:,:]
        
        # TRANSFORMS
        # bring to range 0..1
        data = data/10000
        # bring data to range -1..+1
        data = (data*2)-1
        # apply norm
        if self.apply_norm:
            data = self.apply_transforms(data)
        
        # return array of window that has been read
        return(torch.Tensor(data))
            
        
    def fill_SR_overlap(self, sr, idx):
        # If input nor np.array, transform to array
        sr = sr.numpy() if isinstance(sr, torch.Tensor) else sr
        
        overlap = self.overlap

        # Get coor of idx window
        current_window = self.window_coordinates[idx]
        row_off, col_off = current_window.row_off * 4, current_window.col_off * 4

        # Get sr shape info
        num_channels, win_height, win_width = sr.shape

        # Define the area that will be filled with new data
        fill_area = (slice(None), 
                     slice(row_off, row_off + win_height), 
                     slice(col_off, col_off + win_width))

        # Calculate the mask of non-zero (already filled) values in the placeholder tensor
        existing_mask = self.sr_tensor_placeholder[fill_area] != 0

        # Calculate the average for the overlapping area
        # We use the mask to only average where the placeholder tensor has non-zero values
        # Since existing_mask is a boolean mask, we need to use it to selectively
        # update values in the placeholder tensor
        new_values = sr.copy()
        new_values[existing_mask] = (self.sr_tensor_placeholder[fill_area][existing_mask] + sr[existing_mask]) / 2.

        # Place the averaged or new values into the placeholder tensor
        self.sr_tensor_placeholder[fill_area] = new_values

        
    def save_SR(self):
        # define path of SR saving
        out_file = self.sr_tensor_placeholder
        self.out_path = self.file_path[:self.file_path.rfind(".")]+"_SR.tif"
        # alter original geotransform to reflect new SR image
        # Updated transform after super-resolution
        save_transform = Affine(
                self.geo_transform.a / self.factor, 
                self.geo_transform.b, 
                self.geo_transform.c, 
                self.geo_transform.d, 
                self.geo_transform.e / self.factor, 
                self.geo_transform.f
            )

        
        # write to tiff file
        with rasterio.open(
            self.out_path, 
            "w", 
            driver="GTiff", 
            height=out_file.shape[1], 
            width=out_file.shape[2], 
            count=out_file.shape[0], 
            dtype=out_file.dtype, 
            crs=self.crs, 
            transform=save_transform
        ) as dest:
            dest.write(out_file)
        
    
    def init_transforms(self,dataset_type="S2NAIP",band_stats_path="utils/band_statistics.json",apply_transforms=True):
        
        if apply_transforms==False:
            self.transform_PIL = transforms.ToPILImage()
            self.transform = transforms.ToTensor()
        
        if apply_transforms==True:
            try:
                with open(band_stats_path, 'r') as file:
                    # Load the JSON data into a dictionary
                    data_dict = json.load(file)
                mean = [data_dict["norm_dict"][dataset_type]["mean"]["0"],
                        data_dict["norm_dict"][dataset_type]["mean"]["1"],
                        data_dict["norm_dict"][dataset_type]["mean"]["2"],
                        data_dict["norm_dict"][dataset_type]["mean"]["3"]]

                std = [data_dict["norm_dict"][dataset_type]["std"]["0"],
                        data_dict["norm_dict"][dataset_type]["std"]["1"],
                        data_dict["norm_dict"][dataset_type]["std"]["2"],
                        data_dict["norm_dict"][dataset_type]["std"]["3"]]
                transform_ = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(mean=mean, std=std),
                         ])
                self.transform_PIL = transforms.ToPILImage()
                self.transform = transform_
                
                # set info for whole class
                self.means = mean
                self.stds = std
            except Exception as e:
                print("Error in reading norm properties:", str(e))
                self.transform_PIL = transforms.ToPILImage()
                self.transform = transforms.ToTensor()

    def apply_transforms(self,im):
        """
        in: - torch tensor of shape W H C
        out: normalized tensor
        """
        # check if we need to transpose
        transpose_needed = False
        if im.shape[0]<im.shape[-1]: # True if W H B
            transpose_needed = True
            im = rearrange(im,"c w h -> w h c")
            
        
        im = im.astype(np.float32)
        im = self.transform(im)


        if im.shape[-1]<im.shape[0]:
            im = rearrange(im,"w h c -> c w h")
            
        return(im)
    
    
    def super_resolute_whole_image(self,model=None,forward_call="forward",custom_steps=200):
        
        """
        Super-resolutes the entire image of the class using a specified or default super-resolution model.

        Parameters:
        -----------
        model : object, optional
            A PyTorch model instance that performs super-resolution. If not specified, 
            a default SR model using bilinear interpolation is used.

        forward_call : str, optional
            The name of the method to call on the model for performing super-resolution.
            Defaults to "forward". You can specify any custom method that your model has for
            super-resolution.

        Usage:
        ------
        >>> sr_instance.super_resolute_whole_image(model=my_model, forward_call="custom_forward")

        Notes:
        ------
        - The model provided (or the default one) should have a method corresponding to the name 
          passed in `forward_call` that takes a low-res image and returns a super-resoluted version.

        - SR tensor is created in memory, might take up a lot of space

        - Once super-resolution is complete, the image will be saved in the same directory.

        Returns:
        --------
        None : Saves the super-resoluted image and updates the calling instance.

        """
        # Get interpolation model if model not specified
        if model==None:
            # Create SR mock-up
            class sr_model():
                def __init__(self,custom_steps=200):
                    pass
                def forward(self,lr,custom_steps=200):
                    sr = torch.nn.functional.interpolate(lr, size=(512, 512), mode='bilinear', align_corners=False)
                    return(sr)
            model = sr_model()
            
        # allow custom defined forward/SR call on model
        model_sr_call = getattr(model, forward_call,custom_steps)
            
        
        # iterate over image batches
        print("Super-Resoluting whole image..")
        for idx in tqdm(range(len(self)),ascii=False):
            # get image from S2 image
            im = self.get_window(idx)
            # batch image
            im = im.unsqueeze(0)
            
            # if ddpm, prepare for dictionary input
            if forward_call == "perform_custom_inf_step":
                im = {"LR_image":im,"image":torch.rand(im.shape[0],im.shape[1],512,512)}
            
            # super-resolute image
            sr = model_sr_call(im,custom_steps=200)
            # denorm image
            if self.apply_norm:
                sr = self.denorm(sr,self.means,self.stds)
                sr = (sr+1)/2
                lr = self.denorm(im,self.means,self.stds)
                lr = (lr+1)/2
            # post-processing: hist match
            sr = hq_histogram_matching(sr,lr)
            
            # save SR into image
            self.fill_SR_overlap(sr[0],idx)

        # when done, save array into same directory
        print("Saving SR image...")
        self.save_SR()
        print("Finished. SR image saved at",self.out_path)
        
def main(file_path): 
    sr_obj = windowed_SR_and_saving(file_path)
    sr_obj.super_resolute_whole_image(model=None,forward_call="forward",custom_steps=200)
    pass

if __name__ == "__main__":
    
    # if we need to stack first, call function
    stack_needed = False
    if stack_needed:
        from utils.rgbnir_stacked_from_S2_folder import extract_rgbnir_from_S2_folder
        folder_path = "/data2/simon/test_s2/S2A_MSIL2A_20230729T100031_N0509_R122_T33TUG_20230729T134559.SAFE"
        extract_rgbnir_from_S2_folder(folder_path)
        
    #TODO: add CLArgs
    file_path = "/data2/simon/test_s2/S2A_MSIL2A_20230729T100031_N0509_R122_T33TUG_20230729T134559.SAFE/stacked_RGBNIR.tif"
    main()