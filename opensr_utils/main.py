import rasterio
from einops import rearrange
from rasterio.transform import Affine
import numpy as np
import torch
from tqdm import tqdm
import os
from pytorch_lightning import LightningModule


# local imports
from opensr_utils.denormalize_image_per_band_batch import denormalize_image_per_band_batch as denorm
from opensr_utils.stretching import hq_histogram_matching
from opensr_utils.bands10m_stacked_from_S2_folder import extract_10mbands_from_S2_folder
from opensr_utils.bands20m_stacked_from_S2_folder import extract_20mbands_from_S2_folder
from opensr_utils.weighted_overlap import weighted_overlap


class windowed_SR_and_saving():
    
    def __init__(self, folder_path, window_size=(128, 128), factor=4, keep_lr_stack=True, mode="xAI"):
        """
        Class that performs windowed super-resolution on a Sentinel-2 image and saves the result. Steps:
        - Copies the 10m and 20m bands to new tiff files in the input directory.
        - 10m and 20m bands can be called separately and preformed with different models.
        - SR Results are saved with an averaged overlap and georeferenced in the input folder.

        Inputs:
            - folder_path (string): path to folder containing S2 SAFE data format
            - window_size (tuple): window size of the LR image
            - factor (int): SR factor
            - keep_lr_stack (bool): decide wether to delete the LR stack after SR is done
            - custom_steps (int): number of steps to perform for Diffusion models

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
            sr_obj.start_super_resolution(band_selection="20m",model=model,forward_call="forward",custom_steps=100)
            # perform super-resolution on 10m bands
            sr_obj.start_super_resolution(band_selection="10m",model=model,forward_call="forward",custom_steps=100)
            # delete LR stack
            sr_obj.delete_LR_stack()
        """

        # General Settings
        self.folder_path = folder_path # path to folder containing S2 SAFE data format
        self.window_size = window_size # window size of the LR image
        self.factor = factor # sr factor of the model
        self.hist_match = False # wether we want to perform hist matching here
        self.keep_lr_stack = keep_lr_stack # decide wether to delete the LR stack after SR is done
        self.mode = mode # select SR or uncertainty calculation

        # check that folder path exists, and that it's the correct type
        assert os.path.exists(self.folder_path), "Input folder/file path does not exist"
        assert self.mode in ["SR","xAI"], "Mode not in ['SR','xAI']"
        print("Working in ",self.mode,"mode!")


    def create_and_save_placeholder_SR_files(self,info_dict,out_name):
        """
        Saves a georeferenced placeholder SR file in the input folder.
        """
        # create placeholder tensor in memory
        # If we're in xAI, we only need and want 1 band of SR
        if self.mode=="xAI":
            no_bands = 1
        if self.mode=="SR":
            no_bands = len(info_dict["bands"])    
        # create tensor
        sr_tensor_placeholder = np.zeros((no_bands, 
                                               info_dict["img_width"]*self.factor,
                                               info_dict["img_height"]*self.factor),dtype=info_dict["dtype"])
        
        # change geotransform to reflect smaller SR pixel size
        save_transform = Affine(
                info_dict["geo_transform"].a / self.factor, 
                info_dict["geo_transform"].b, 
                info_dict["geo_transform"].c, 
                info_dict["geo_transform"].d, 
                info_dict["geo_transform"].e / self.factor, 
                info_dict["geo_transform"].f
            )

        # create Metadata for rasterio saveing
        meta = {
        'driver': 'GTiff',
        'dtype': info_dict["dtype"],  # Ensure dtype matches your array's dtype
        'nodata': None,  # Set to your no-data value, if applicable
        'width': info_dict["img_width"]*self.factor,
        'height': info_dict["img_height"]*self.factor,
        'count': no_bands,  # Number of bands; adjust if your array has multiple bands
        'crs': info_dict["crs"],  # CRS (Coordinate Reference System); set as needed
        'transform': save_transform,  # Adjust as needed
                }


        # work with file lock in order to not do this multiple times during Multi-GPU inference
        # Attempt to acquire an exclusive lock on a temporary lock file
        import os
        import fcntl
        output_file_path = os.path.join(self.folder_path,out_name)
        lock_file_path = output_file_path + ".lock"
        with open(lock_file_path, 'w') as lock_file:
            try:
                fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
                # This block now runs only in the process that acquired the lock
                if os.path.exists(output_file_path): # remove file if it exists
                    os.remove(output_file_path)
                    print("Overwriting existing placeholder file...")
                if not os.path.exists(output_file_path): # create file
                    # FILE CREATION LOGIC -----------------------------------------------------------------------
                    # Create and write SR placeholder to the raster file
                    with rasterio.open(os.path.join(self.folder_path,out_name), 'w', **meta) as dst:
                        # Assuming 'your_array' is 2D, write it to the first band
                        for band in range(sr_tensor_placeholder.shape[0]):
                            dst.write(sr_tensor_placeholder[band, :, :], band + 1)
                    print("Saved empty placeholder SR image at: ",os.path.join(self.folder_path,out_name))
                    # END FILE CREATION LOGIC -------------------------------------------------------------------
                fcntl.flock(lock_file, fcntl.LOCK_UN) # release lock
            except IOError:
                # Another process has the lock; skip file creation
                pass
        # delete the lock file
        #os.remove(lock_file_path)

        # return file path of placeholder
        return os.path.join(self.folder_path,out_name)
        
    
    def create_window_coordinates_overlap(self,info_dict):
        """
        Creates a list of window coordinates for the input image. The windows overlap by a specified amount.
        Output type is a list of rasterio.windows.Window objects.
        """
        # get amount of overlap
        overlap = info_dict["overlap"]
        
        # Calculate the number of windows in each dimension
        n_windows_x = (info_dict["img_width"] - overlap) // (self.window_size[0] - overlap)
        n_windows_y = (info_dict["img_height"] - overlap) // (self.window_size[1] - overlap)

        # Create list of batch windows coordinates
        window_coordinates = []
        for win_y in range(n_windows_y):
            for win_x in range(n_windows_x):
                # Define window to read with overlap
                window = rasterio.windows.Window(
                    win_x * (self.window_size[0] - overlap),
                    win_y * (self.window_size[1] - overlap),
                    self.window_size[0],
                    self.window_size[1]
                )
                window_coordinates.append(window)

        # Check for any remaining space after the sliding window approach
        final_x = info_dict["img_width"] - self.window_size[0]
        final_y = info_dict["img_height"] - self.window_size[1]

        # Add extra windows for the edges if there's remaining space
        # Adjust the check to handle the overlap correctly
        if final_x % (self.window_size[0] - overlap) > 0:
            for win_y in range(n_windows_y):
                window = rasterio.windows.Window(
                    info_dict["img_width"] - self.window_size[0],
                    win_y * (self.window_size[1] - overlap),
                    self.window_size[0],
                    self.window_size[1]
                )
                window_coordinates.append(window)

        if final_y % (self.window_size[1] - overlap) > 0:
            for win_x in range(n_windows_x):
                window = rasterio.windows.Window(
                    win_x * (self.window_size[0] - overlap),
                    info_dict["img_height"] - self.window_size[1],
                    self.window_size[0],
                    self.window_size[1]
                )
                window_coordinates.append(window)

        # Handle corner case if both x and y have remaining space
        if (final_x % (self.window_size[0] - overlap) > 0 and
                final_y % (self.window_size[1] - overlap) > 0):
            window = rasterio.windows.Window(
                info_dict["img_width"] - self.window_size[0],
                info_dict["img_height"] - self.window_size[1],
                self.window_size[0],
                self.window_size[1]
            )
            window_coordinates.append(window)

        # Return filled list of coordinates
        return window_coordinates
    
    
    def get_window(self,idx,info_dict):
        """
        Loads a window of the input image and returns it as a tensor.
        """
        # TODO: perform batched SR instead of single image per batch
        # assert number required is valid
        assert idx>=0 and idx<len(info_dict["window_coordinates"]), "idx not in range of windows"
        # get window of current idx
        current_window = info_dict["window_coordinates"][idx]
        
        # open file SRC
        with rasterio.open(info_dict["lr_path"]) as src:
            data = src.read(window=current_window)
            # select bands
            data = data[info_dict["bands"],:,:]
        
        data = data/10000 # bring to 0..1
        data = torch.from_numpy(data)
        
        # return array of window that has been read
        return(data)
    
    def delete_LR_stack(self,info_dict):
        # delete LR stack
        os.remove(info_dict["lr_path"])
        print("Deleted stacked image at",info_dict["lr_path"])

    def delete_lock_files(self):
        # delete all lock files in the input directory to clean up
        # Construct the pattern to match all .lock files
        import glob
        directory = self.folder_path
        pattern = os.path.join(directory, '*.lock')
        
        # Find all files in the directory matching the pattern
        lock_files = glob.glob(pattern)
        
        # Iterate over the list of file paths & remove each file
        for file_path in lock_files:
            try:
                os.remove(file_path)
                print(f"Deleted lock file: {file_path}")
            except OSError as e:
                print(f"Error deleting lock file: {file_path} : {e.strerror}")
        
    def fill_SR_overlap(self, sr, idx, info_dict):
        """
        Fills the SR placeholder image with the super-resoluted window at the correct location of the image. Performs windowed writing via rasterio.
        """
        # If input not np.array, transform to array
        sr = sr.numpy() if isinstance(sr, torch.Tensor) else sr

        # Get coor of idx window, create new rasterio Windoow in which it should be saved
        current_window = info_dict["window_coordinates"][idx]
        row_off, col_off = current_window.row_off * 4, current_window.col_off * 4
        sr_file_window = rasterio.windows.Window(col_off,row_off, sr.shape[-2], sr.shape[-1])

        # Get sr shape info
        num_channels, win_height, win_width = sr.shape

        # change sr data range and dtype to correspond with original
        sr = sr*10000
        sr = np.array(sr, dtype=info_dict["dtype"])
        sr = sr.astype(info_dict["dtype"])

        # save to placehodler .tiff on disk
        # Open the TIFF file in 'r+' mode (read/write mode)
        with rasterio.open(info_dict["sr_path"], 'r+') as dst:
            # Check if the number of bands in the tensor matches the TIFF file
            if dst.count != sr.shape[0]:
                raise ValueError("The number of bands in the tensor does not match the TIFF file.")
            
            # read data already in the placeholder image
            placeholder_image = dst.read(window=sr_file_window)

            # perform weighted average between placeholder and SR image
            sr = weighted_overlap(sr=sr, placeholder=placeholder_image,
                                  overlap=info_dict["overlap"], 
                                  pixels_eliminate=info_dict["eliminate_border_px"],
                                  hr_size=sr.shape[-1])

            # Write each band of the tensor to the corresponding band in the raster
            for band in range(sr.shape[0]):
                dst.write(sr[band, :, :], band + 1, window=sr_file_window)

    
    def super_resolute_bands(self,info_dict,model=None,forward_call="forward",custom_steps=100):
        
        """
        Super-resolutes the entire image of the class using a specified or default super-resolution model.

        Parameters:
        -----------
        info_dict : dict
            A dictionary containing information about the image to super-resolute. This is generated automatically
            if the class has been initialized and called correctly (via the start_super_resolution).

        model : object, optional
            A PyTorch model instance that performs super-resolution. If not specified, 
            a default SR model using bilinear interpolation is used.

        forward_call : str, optional
            The name of the method to call on the model for performing super-resolution.
            Defaults to "forward". You can specify any custom method that your model has for
            super-resolution.

        Usage:
        ------
        sr_instance.super_resolute_bands(info_dict, model=my_model, forward_call="custom_forward")

        Notes:
        ------
        - The model provided (or the default one) should have a method corresponding to the name 
          passed in `forward_call` that takes a low-res image and returns a super-resoluted version.

        Returns:
        --------
        None : Saves the super-resoluted image in the SR placeholder on the disk via windowed writing.

        """
        # Get interpolation model if model not specified
        if model==None:
            class SRModelPL(LightningModule): # placeholder interpolation model for testing
                def __init__(self):
                    super(SRModelPL, self).__init__()
                def forward(self, x, custom_steps=100):
                    sr = torch.nn.functional.interpolate(x, size=(512, 512), mode='nearest')
                    return sr
                def predict(self,x,custom_steps=100):
                    return self.forward(x)
            model = SRModelPL()
            
        # allow custom defined forward/SR call on model
        model_sr_call = getattr(model, forward_call,custom_steps)
        
        # iterate over image batches
        for idx in tqdm(range(len(info_dict["window_coordinates"])),ascii=False,desc="Super-Resoluting"):
            # get image from S2 image
            im = self.get_window(idx,info_dict)
            # batch image
            im = im.unsqueeze(0)
            # turn into wanted dtype (double)
            im = im.float()
            # send to device
            im = im.to(model.device)
            # if ddpm, prepare for dictionary input
            if forward_call == "perform_custom_inf_step":
                im = {"LR_image":im,"image":torch.rand(im.shape[0],im.shape[1],512,512)}
            
            # super-resolute image
            if self.mode=="SR":
                sr = model_sr_call(im,custom_steps=custom_steps)
                sr = sr.squeeze(0)
            if self.mode=="xAI":
                sr = self.calculate_uncertainty(im,model_sr_call,custom_steps=custom_steps)

            # try to move to CPU, might be already on CPU in some cases, therefore try catch block
            try:
                sr = sr.detach().cpu()
            except:
                pass
            
            # save SR into image
            self.fill_SR_overlap(sr,idx,info_dict)

        # when done, save array into same directory
        print("Finished. SR image saved at",info_dict["sr_path"])

    def calculate_uncertainty(self,im,model_sr_call,custom_steps=200):
        
        device = "cuda:0"
        print("Device info:",device)
        
        if len(im.shape)==3:
            im = im.unsqueeze(0)
        im = im.float()
        im = im.to(device)
        
        no_uncertainty = 20 # amount of images to SR
        variations = []
        for i in range(no_uncertainty):
            sr = model_sr_call(im)
            sr = sr.squeeze(0)
            variations.append(sr.detach().cpu())
        variations = torch.stack(variations)

        # Get statistics for xAI
        srs_mean = variations.mean(dim=0)
        srs_stdev = variations.std(dim=0)
        lower_bound = srs_mean-srs_stdev
        upper_bound = srs_mean+srs_stdev
        interval_size = srs_stdev*2
        interval_size = interval_size.mean(dim=0).unsqueeze(0)
        return interval_size

    
    def initialize_info_dicts(self,band_selection="10m",overlap=8, eliminate_border_px=0):
        # select info dictionary:
        # 1 .Get file info from Rasterio
        # 2. Create window coordinates for selected bands
        # 3. Create and save placeholder SR file

        # check wether input is in .SAFE format or is traight to file
        if self.folder_path.replace("/","")[-5:] == ".SAFE":
            safe_format = True
        else:
            safe_format = False
            from opensr_utils.utils import can_read_directly_with_rasterio
            assert can_read_directly_with_rasterio(self.folder_path) == True


        if band_selection=="10m":

            # If Safe format, perform band extraction ect
            if safe_format:
                # work with file lock in order to not do this multiple times during Multi-GPU inference
                # Attempt to acquire an exclusive lock on a temporary lock file
                import os
                import fcntl
                stack_output_path = os.path.join(self.folder_path,"stacked_10m.tif")
                lock_file_path = stack_output_path + ".lock"
                with open(lock_file_path, 'w') as lock_file:
                    try:
                        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        # This block now runs only in the process that acquired the lock
                        if os.path.exists(stack_output_path): # remove file if it exists
                            pass
                            #os.remove(stack_output_path)
                            #print("Overwriting existing 10 bands file...")
                        else: # create file
                            # FILE CREATION LOGIC -----------------------------------------------------------------------
                            print("Creating stacked 10m bands file ...")
                            extract_10mbands_from_S2_folder(self.folder_path)
                            # END FILE CREATION LOGIC -------------------------------------------------------------------
                        fcntl.flock(lock_file, fcntl.LOCK_UN) # release lock
                    except IOError:
                        # Another process has the lock; skip file creation
                        pass
                self.b10m_file_path = os.path.join(self.folder_path,"stacked_10m.tif")
            if safe_format==False:
                self.b10m_file_path = os.path.join(self.folder_path)



            # Get File information - 10m bands
            
            self.b10m_info = {}
            self.b10m_info["lr_path"] = self.b10m_file_path
            self.b10m_info["bands"] = [0,1,2,3]
            self.b10m_info["eliminate_border_px"] = eliminate_border_px
            self.b10m_info["overlap"] = overlap
            with rasterio.open(self.b10m_file_path) as src:
                self.b10m_info["img_width"], self.b10m_info["img_height"],self.b10m_info["dtype"] = src.width, src.height,src.dtypes[0]
                # Extract the affine transformation matrix
                self.b10m_info["geo_transform"] = src.transform
                # Extract the CRS
                self.b10m_info["crs"] = src.crs
            
            # call local functions: get windxw coordinates and create placeholder SR file
            self.b10m_info["window_coordinates"] = self.create_window_coordinates_overlap(self.b10m_info)
            out_name = str(self.mode+"_10mbands.tif")
            self.b10m_info["sr_path"] = self.create_and_save_placeholder_SR_files(self.b10m_info,out_name=out_name)
            info_dict = self.b10m_info
            return(info_dict)
        
        # for 20m case
        if band_selection=="20m":
            if safe_format:
                # work with file lock in order to not do this multiple times during Multi-GPU inference
                # Attempt to acquire an exclusive lock on a temporary lock file
                import os
                import fcntl
                stack_output_path = os.path.join(self.folder_path,"stacked_20m.tif")
                lock_file_path = stack_output_path + ".lock"
                with open(lock_file_path, 'w') as lock_file:
                    try:
                        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        # This block now runs only in the process that acquired the lock
                        if os.path.exists(stack_output_path): # remove file if it exists
                            pass
                            #os.remove(stack_output_path)
                            #print("Overwriting existing 20m bands  file...")
                        else:
                            # FILE CREATION LOGIC -----------------------------------------------------------------------
                            print("Creating stacked 20m bands file ...")
                            extract_20mbands_from_S2_folder(self.folder_path)
                            # END FILE CREATION LOGIC -------------------------------------------------------------------
                        fcntl.flock(lock_file, fcntl.LOCK_UN) # release lock
                    except IOError:
                        # Another process has the lock; skip file creation
                        pass
                self.b20m_file_path = os.path.join(self.folder_path,"stacked_10m.tif")
            if safe_format==False:
                self.b20m_file_path = os.path.join(self.folder_path)



            # Get File information - 20m bands 
            self.b20m_file_path = os.path.join(self.folder_path,"stacked_20m.tif")
            self.b20m_info = {}
            self.b20m_info["lr_path"] = self.b20m_file_path
            self.b20m_info["bands"] = [0,1,2,3,4,5]
            self.b20m_info["eliminate_border_px"] = eliminate_border_px
            self.b20m_info["overlap"] = overlap
            with rasterio.open(self.b20m_file_path) as src:
                self.b20m_info["img_width"], self.b20m_info["img_height"],self.b20m_info["dtype"] = src.width, src.height, src.dtypes[0]
                # Extract the affine transformation matrix
                self.b20m_info["geo_transform"] = src.transform
                # Extract the CRS
                self.b20m_info["crs"] = src.crs

            # call local functions: get windxw coordinates and create placeholder SR file
            self.b20m_info["window_coordinates"] = self.create_window_coordinates_overlap(self.b20m_info)
            out_name = str(self.mode+"_20mbands.tif")
            self.b20m_info["sr_path"] = self.create_and_save_placeholder_SR_files(self.b20m_info,out_name=out_name)
            info_dict = self.b20m_info
            return(info_dict)

    
    def start_super_resolution(self,band_selection="10m",model=None, forward_call="forward", custom_steps=100, overlap=8, eliminate_border_px=0):
        # TODO: then, profit from these settings from the PL dataset class
        # assert band selection in available implemented methods
        assert band_selection in ["10m","20m"], "band_selection not in ['10m','20m']"    
        info_dict = self.initialize_info_dicts(band_selection=band_selection,overlap=overlap, eliminate_border_px=eliminate_border_px)    

        # If model is a torch.nn.Module, do 1-batch SR with patching on the fly
        if isinstance(model, LightningModule):
            print("Lightning Model detected, performing multi-batched inference with PyTorch Lightning.")
            from opensr_utils.pl_utils import predict_pl_workflow
            args = {
                "band_selection": band_selection,
                "overlap": overlap,
                "eliminate_border_px": eliminate_border_px,
                "num_workers": 64,
                "batch_size": 24,
                "prefetch_factor": 4,
                "accelerator": "gpu",
                "devices": 1,
                "strategy": "ddp",
                "custom_steps": custom_steps,
                "mode": self.mode,
                "window_size": self.window_size}
            predict_pl_workflow(input_file=self.folder_path,model=model,**args)
        elif isinstance(model, torch.nn.Module):
            if self.mode!="xAI":
                print("Model is torch.NN.Module, performing 1-batched inference with patching on the fly. For faster inference, provide a PyTorch Lightning module.")
            self.super_resolute_bands(info_dict,model, forward_call=forward_call, custom_steps=custom_steps)
        elif model==None: # test case
            print("No model passed. Performing interpolation instead of SR for testing purposes.")
            self.super_resolute_bands(info_dict,model, forward_call=forward_call, custom_steps=custom_steps)
        else:
            raise NotImplementedError("Model type not recognized. Please provide a PyTorch Lightning model, PyTorch Model, or for testing purposes 'None'.")
       
        # CLEANUP
        self.delete_lock_files() # delete lock files created for multiprocessing

        # if wanted, delete LR stack
        if not self.keep_lr_stack:
            self.delete_LR_stack(info_dict)

from torch.utils.data import Dataset, DataLoader
class windowed_SR_and_saving_dataset(Dataset):
    """
    This wraps aroudn the object class to profit from
    the multi-threaded approach of Lightning and to 
    embedd it in the workflow with a PL SR model.
    """
    def __init__(self, folder_path, band_selection="10m",
                 overlap=20,eliminate_border_px=10,
                 window_size=(128, 128), factor=4,keep_lr_stack=True):
        # create object
        self.sr_obj = windowed_SR_and_saving(folder_path, window_size=window_size,
                                             factor=factor, keep_lr_stack=keep_lr_stack)
        # initialze information
        self.info_dict = self.sr_obj.initialize_info_dicts(band_selection=band_selection,
                                          overlap=overlap,
                                          eliminate_border_px=eliminate_border_px) 

    def __len__(self):
        return(len(self.info_dict["window_coordinates"]))
    
    def __getitem__(self,idx):
        im = self.sr_obj.get_window(idx,self.info_dict)
        im = im.float()
        return(im)

"""
# logic to create PyTorch Dataset and DataLoader for this dataset object
if __name__ == "__main__":
    folder_path = "/data2/simon/test_s2/S2A_MSIL2A_20230729T100031_N0509_R122_T33TUG_20230729T134559.SAFE/"
    ds = windowed_SR_and_saving_dataset(folder_path=folder_path, band_selection="20m",
                    overlap=20,eliminate_border_px=10,
                    window_size=(128, 128), factor=4,keep_lr_stack=False)
    dl = DataLoader(ds,batch_size=10,shuffle=False)

if __name__ == "__main__":
            
    folder_path = "/data2/simon/test_s2/S2A_MSIL2A_20230729T100031_N0509_R122_T33TUG_20230729T134559.SAFE/"

    sr_obj = windowed_SR_and_saving(folder_path,keep_lr_stack=True,mode="xAI")

    #sr_obj.start_super_resolution(band_selection="20m")
    #sr_obj.start_super_resolution(band_selection="10m")
"""
