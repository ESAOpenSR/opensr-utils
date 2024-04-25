import torch
import opensr_test

def compute_metrics(lr,sr):
    try:
        lr = lr.detach()
    except:
        pass
    try:
        sr = sr.detach()
    except:
        pass

    if len(sr.shape)>3:
        sr = sr.squeeze(0)
    if len(lr.shape)>3:
        lr = lr.squeeze(0)

    metrics = opensr_test.Metrics()
    m = metrics.compute(lr=lr, sr=sr, hr=sr)
    return(m)

def append_dicts(original_dict, new_dict):
    """
    Append values from keys of a new dictionary to the original dictionary.

    Args:
        original_dict (dict): The original dictionary to which values will be appended.
        new_dict (dict): The new dictionary containing values to be appended.

    Returns:
        dict: The original dictionary with values appended.
    """
    for key, value in new_dict.items():
        if key in original_dict:
            if isinstance(original_dict[key], list):
                original_dict[key].append(value)
            else:
                original_dict[key] = [original_dict[key], value]
        else:
            original_dict[key] = value if isinstance(value, list) else [value]
    return original_dict

