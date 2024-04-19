import torch
import torch
import os
from pytorch_lightning.callbacks import BasePredictionWriter
from tqdm import tqdm

def predict_pl_workflow(input_file,model,**kwargs):
    band_selection = kwargs.get('band_selection', "20m")
    overlap = kwargs.get('overlap', 40)
    eliminate_border_px = kwargs.get('eliminate_border_px', 0)
    num_workers = kwargs.get('num_workers', 64)
    batch_size = kwargs.get('batch_size', 24)
    prefetch_factor = kwargs.get('prefetch_factor', 4)
    accelerator = kwargs.get('accelerator', "gpu")
    devices = kwargs.get('devices', -1)
    strategy = kwargs.get('strategy', "ddp")
    custom_steps = kwargs.get('custom_steps', 100)
    mode =  kwargs.get('mode', "SR")
    window_size = kwargs.get('window_size', (128,128))
    
    # -----------------------------------------------------------------------------
    # Create PyTorch Lighnting Workflow for Multi-GPU processing
    import torch
    torch.set_float32_matmul_precision('medium')
    from opensr_utils.main import windowed_SR_and_saving_dataset 

    # create DataLoader object from opensr_utils
    from opensr_utils.main import windowed_SR_and_saving_dataset
    from torch.utils.data import Dataset, DataLoader
    ds = windowed_SR_and_saving_dataset(folder_path=input_file, band_selection=band_selection,
                                        overlap=overlap,eliminate_border_px=eliminate_border_px,
                                        window_size=window_size,keep_lr_stack=False)
    dl = DataLoader(ds,num_workers=num_workers, batch_size=batch_size,prefetch_factor=prefetch_factor)

    # Create custom writer that writes pl_model outputs to placeholder
    #import CustomWriter
    writer_callback = CustomWriter(ds)

    # create Trainer - here for initialization of multi-GPU processing
    from pytorch_lightning import Trainer
    trainer = Trainer(accelerator=accelerator, devices=devices,strategy=strategy,callbacks=[writer_callback],logger=False)

    # initialize models
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pl_model = model
    # set properties for model
    model.mode = mode

    # run prediction
    trainer.predict(pl_model, dataloaders=dl,return_predictions=False)


class CustomWriter(BasePredictionWriter):
    """
    - Applied after each predict step by the predict() of PL model
    - this class needs as input the SR object created by opensr-utils
    - it writes the results to a temporary folder
    - it writes the results to the placeholder file at the end
    """
    def __init__(self, sr_obj, write_interval="batch"):
        super().__init__(write_interval)
        self.sr_obj = sr_obj # save SR obj in class

        # create folder and info to save temporary results
        base_path = os.path.dirname(self.sr_obj.info_dict["sr_path"])
        self.temp_path = os.path.join(base_path,"tmp_sr")
        os.makedirs(self.temp_path, exist_ok=True)

        # if files exist in folder, remove them
        files = os.listdir(self.temp_path)
        for file in files:
            os.remove(os.path.join(self.temp_path,file))

    def write_on_batch_end(self, trainer,
                           pl_module, prediction,
                           batch_indices, batch,
                           batch_idx, dataloader_idx):
        try:
            prediction = prediction.cpu()
        except:
            pass

        # iterate over predictions and save them accordingly
        #for pred,idx in zip(prediction,batch_indices):
        #    self.sr_obj.sr_obj.fill_SR_overlap(pred,idx,self.sr_obj.info_dict)

        # create filename
        formatted_number_1 = "{:06}".format(batch_indices[0])
        formatted_number_2 = "{:04d}".format(len(batch_indices))
        filename = "batch{}__{}batches.pt".format(formatted_number_1,formatted_number_2)
        # create dictionary out of results and save to disk
        results_dict = {"sr":prediction,"batch_indices":batch_indices}
        torch.save(results_dict,os.path.join(self.temp_path,filename))
        
        # return nothing in order not to accumulate results
        return None
    
    #def write_all_to_placeholder(self):
    def on_predict_end(self, trainer, pl_module):
        """
        Custom function to be called after all predictions are done
        """
        # run this only on one GPU
        if trainer.global_rank == 0:
            print("Running weighted Patching on Worker:",str(trainer.global_rank),"...")
            # read all files in folder
            files = os.listdir(self.temp_path)
            files = [file for file in files if file.endswith(".pt")] # keep only .pt files

            # iterate over files and save them to placeholder
            for file in tqdm(files,desc="Saving to PH"):
                # load file
                results_dict = torch.load(os.path.join(self.temp_path,file))
                # iterate over results and save them
                for pred,idx in zip(results_dict["sr"],results_dict["batch_indices"]):
                    self.sr_obj.sr_obj.fill_SR_overlap(pred,idx,self.sr_obj.info_dict)
            
            # delete temporary folder and all its contents
            # Make sure the path exists and is a directory
            if os.path.isdir(self.temp_path):
                try:
                    import time
                    import shutil
                    time.sleep(5)
                    shutil.rmtree(self.temp_path)
                    print("Data written to SR output, temp folder deleted. Finished.")
                except:
                    print("Data written to SR output, temp folder not deleted due to Error in deletion process.")
            else:
                print("Data written to SR output, temp folder not deleted.")
            return None
