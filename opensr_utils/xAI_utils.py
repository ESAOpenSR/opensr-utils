"""
WARNING
This code is currently unused, instead the xAI part is directly embedded in main
"""


import torch
from tqdm import tqdm
import numpy as np
import properscoring as ps
import os

def calc_xai(srs,hr):
     # calculate mean and std of tensor
    srs_mean = srs.mean(dim=0)
    srs_stdev = srs.std(dim=0)

    lower_bound = srs_mean-srs_stdev
    upper_bound = srs_mean+srs_stdev

    error = torch.abs(hr - srs_mean)
    interval_size = srs_stdev*2

    return interval_size

def calculate_crps_for_tensors(observation: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    """Calculates the CRPS score for a tensor of observations and predictions.

    Args:
        observation (np.ndarray): Tensor of observations (C, H, W)
        predictions (np.ndarray): Tensor of predictions (T, C, H, W)

    Returns:
        np.ndarray: Tensor of CRPS scores (C, H, W)
    """

    C, H, W = observation.shape
    crps_scores = np.zeros((C, H, W))
    for c in range(C):
        for h in range(H):
            for w in range(W):
                obs = observation[c, h, w]
                fcst = predictions[:, c, h, w]
                crps_score = ps.crps_ensemble(obs, fcst)
                crps_scores[c, h, w] = crps_score

    return crps_scores


def sr_probabilities(self,info_dict,pickle_path,model=None,forward_call="forward",custom_steps=100):
        # set object to self to keep workflow intact

        # Get interpolation model if model not specified
        if model==None:
            # Create SR mock-up
            class sr_model():
                def __init__(self,custom_steps=200):
                    self.device="cpu"
                    pass
                def forward(self,lr,custom_steps=200):
                    sr = torch.nn.functional.interpolate(lr, size=(512, 512), mode='bilinear', align_corners=False)
                    return(sr)
            model = sr_model()
            
        # allow custom defined forward/SR call on model
        model_sr_call = getattr(model, forward_call,custom_steps)

        crps_list = []

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
            # set holders for multiple srs
            sr_probs = []
            for r in range(10):
                sr = model_sr_call(im,custom_steps=custom_steps)
                try:
                    sr = sr.cpu()
                except:
                    pass
                sr_probs.append(sr)
            sr_probs = torch.stack(sr_probs) # turn to tensor

            # get inteval_size
            #interval_size = calc_xai(sr_probs,im)
            # get crps
            im = im.squeeze()
            sr_probs = sr_probs.squeeze()
            crps = calculate_crps_for_tensors(im.cpu().numpy(),sr_probs.cpu().numpy())

            # append result to list
            crps_list.append(crps.mean())

            # save iamge crps to rest of list
            import pickle
            #pickle_path = os.path.join(self.folder_path,"crps_list_10m.pickle")
            with open(pickle_path, 'wb') as file:
                pickle.dump(crps_list, file)

            self.fill_SR_overlap(crps,idx,info_dict)

        return(crps_list)