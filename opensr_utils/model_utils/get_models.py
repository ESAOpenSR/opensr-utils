import requests
from omegaconf import OmegaConf
from io import StringIO
import opensr_model  # your model package

def get_ldsrs2(device: str = "cpu"):
    """
    Load the LDSR-S2 model on a specified device.

    Parameters
    ----------
    device : str, optional
        Device to load the model on. 
        Options: "cpu", "cuda", "cuda:0", etc. 
        Default is "cpu".
    
    Returns
    -------
    opensr_model.SRLatentDiffusion
        The initialized and pretrained LDSR-S2 model.
    """
    # config
    config_url = "https://raw.githubusercontent.com/ESAOpenSR/opensr-model/refs/heads/main/opensr_model/configs/config_10m.yaml"
    response = requests.get(config_url)
    config = OmegaConf.load(StringIO(response.text))

    # model
    model = opensr_model.SRLatentDiffusion(config, device=device)
    model.load_pretrained(config.ckpt_version)
    return model