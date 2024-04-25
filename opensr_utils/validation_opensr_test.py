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

    # squeeze batch dim if needed
    if len(sr.shape)>3:
        sr = sr.squeeze(0)
    if len(lr.shape)>3:
        lr = lr.squeeze(0)

    # permute to (C,H,W) if needed
    if sr.shape[-1]<sr.shape[0]:
        sr = sr.permute(1,2,0)
    if lr.shape[-1]<lr.shape[0]:
        lr = lr.permute(1,2,0)

    # opensr_test cant take 32x32 patches, interpoalte if needed
    if lr.shape[0]<64:
        lr = torch.nn.functional.interpolate(lr.unsqueeze(0), scale_factor=2, mode='linear', align_corners=False).squeeze(0)
        sr = torch.nn.functional.interpolate(sr.unsqueeze(0), scale_factor=2, mode='linear', align_corners=False).squeeze(0)

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

