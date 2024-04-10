import numpy as np
import torch
import torch.nn.functional as F 


def weighted_overlap(sr, placeholder,overlap=10,pixels_eliminate=0,hr_size=512):
    
    # assert logical necessities for overlap
    assert overlap % 2 == 0, "Overlap in weighted overlap needs to be even"
    assert pixels_eliminate % 2 == 0, "Boundary Pixels removal in weighted overlap needs to be even"
    assert overlap > pixels_eliminate, "Overlap must be bigger than amount of pixels to be removed in weighted overlap"
    
    def calculate_distance_to_edge(height, width):
        # Create an empty array for distances
        distance_array = np.zeros((height, width))

        for y in range(height):
            for x in range(width):
                # Distance to the nearest horizontal edge
                dist_to_horizontal_edge = min(y, height - y - 1)

                # Distance to the nearest vertical edge
                dist_to_vertical_edge = min(x, width - x - 1)

                # Nearest edge distance is the minimum of the two
                distance_array[y, x] = min(dist_to_horizontal_edge, dist_to_vertical_edge)+1

        return distance_array

    # rename images
    ph = placeholder
    im = sr
    
    linear_length = overlap-pixels_eliminate

    # mask that says where the image is valid
    ph_validity_mask = (ph != 0).astype(bool)
    
    # Create different version for if we want to eliminate pixels
    if pixels_eliminate>0:
        ph_validity_mask = F.interpolate(torch.Tensor(ph_validity_mask).unsqueeze(0),(hr_size-pixels_eliminate,hr_size-pixels_eliminate),mode='nearest').squeeze(0).numpy()

    # info necessary
    num_channels = sr.shape[0] # amount of bands
    distance_to_edge = calculate_distance_to_edge(ph_validity_mask.shape[-1], ph_validity_mask.shape[-2]) # get distance to edge for each pixel
    distance_to_edge[distance_to_edge>linear_length] = linear_length # set maximum value to overlap amount
    distance_to_edge = distance_to_edge/linear_length # set weight as fraction from overlap
    distance_to_edge_inverse = 1.-distance_to_edge # set inverse factional weight for SR placeholder

    # if wanted, padd validity mask again to original dimension to remove edge pixels
    if True:#pixels_eliminate>0:
        # padd each border cardinality accordingly
        # find out if each cardinality is valid
        half_pixel = ph_validity_mask.shape[-1]//2
        left_valid = ph_validity_mask[0, half_pixel, 0] > 0.5
        right_valid = ph_validity_mask[0, half_pixel, -1] > 0.5
        top_valid = ph_validity_mask[0, 0, half_pixel] > 0.5
        bottom_valid = ph_validity_mask[0, -1, half_pixel] > 0.5
        #print("Cardinalities valid (L-R-T-B):",left_valid,right_valid,top_valid,bottom_valid)
        
        if left_valid:
            left_pad = pixels_eliminate
        else:
            left_pad=0
            
        if right_valid:
            right_pad = pixels_eliminate
        else:
            right_pad=0
        
        if top_valid:
            top_pad = pixels_eliminate
        else:
            top_pad=0
        
        if bottom_valid:
            bottom_pad = pixels_eliminate
        else: 
            bottom_pad=0
            
        # if 2 cardinalities in same direction are cropped, half the cropping area
        if left_valid and right_valid:
            left_pad = pixels_eliminate//2
            right_pad = pixels_eliminate//2
        if top_valid and bottom_valid:
            top_pad = pixels_eliminate//2
            bottom_pad = pixels_eliminate//2
        #print("Pad amounts per cardinality (L-R-T-B):",left_pad,right_pad,top_pad,bottom_pad)
            
        # if none is valid, interpolate mask back to original dimensions without padding
        if True not in [left_valid,right_valid,top_valid,bottom_valid]:
            ph_validity_mask = F.interpolate(torch.Tensor(ph_validity_mask).unsqueeze(0),(hr_size,hr_size),mode='nearest').squeeze(0).numpy()
            # recalculate gradients for full use of pixels
            distance_to_edge = calculate_distance_to_edge(ph_validity_mask.shape[-1], ph_validity_mask.shape[-2]) # get distance to edge for each pixel
            distance_to_edge[distance_to_edge>overlap] = overlap # set maximum value to overlap amount
            distance_to_edge = distance_to_edge/overlap # set weight as fraction from overlap
            distance_to_edge_inverse = 1.-distance_to_edge # set inverse factional weight for SR placeholder
        else: # if we can eliminate pixels, we will via the padding
            # padd each direction with 0s or 1s if its valid or not
            ph_validity_mask = torch.nn.functional.pad(torch.Tensor(ph_validity_mask), pad=(left_pad, right_pad, top_pad, bottom_pad), mode='constant', value=1.0).numpy()            
            distance_to_edge = torch.nn.functional.pad(torch.Tensor(distance_to_edge), pad=(left_pad, right_pad, top_pad, bottom_pad), mode='constant', value=0.0).numpy()
            distance_to_edge_inverse = torch.nn.functional.pad(torch.Tensor(distance_to_edge_inverse), pad=(left_pad, right_pad, top_pad, bottom_pad), mode='constant', value=1.0).numpy()            
            
            # now, if only 1 side of each cardinality is padded, we need to pad the other side with either 0s or 1s according to the original validity mask
            # basically this is for the edges + first row. Edge artifacts introduced by this will be removed later
            if left_pad>0 and right_pad==0  and ph_validity_mask.shape!=im.shape:
                distance_to_edge = F.pad(torch.Tensor(distance_to_edge), pad=(0, 0, pixels_eliminate, 0), mode='constant', value=1.0).numpy()
                distance_to_edge_inverse = F.pad(torch.Tensor(distance_to_edge_inverse), pad=(0, 0, pixels_eliminate, 0), mode='constant', value=0.0).numpy()
                ph_validity_mask = F.pad(torch.Tensor(ph_validity_mask), pad=(0, 0, pixels_eliminate, 0), mode='constant', value=0.0).numpy()
            if left_pad==0 and right_pad>0  and ph_validity_mask.shape!=im.shape:
                distance_to_edge = F.pad(torch.Tensor(distance_to_edge), pad=(0, 0, pixels_eliminate, 0), mode='constant', value=1.0).numpy()
                distance_to_edge_inverse = F.pad(torch.Tensor(distance_to_edge_inverse), pad=(0, 0, pixels_eliminate, 0), mode='constant', value=0.0).numpy()
                ph_validity_mask = F.pad(torch.Tensor(ph_validity_mask), pad=(0, 0, pixels_eliminate, 0), mode='constant', value=0.0).numpy()
            if top_pad>0 and bottom_pad==0  and ph_validity_mask.shape!=im.shape:
                distance_to_edge = F.pad(torch.Tensor(distance_to_edge), pad=(0, pixels_eliminate, 0, 0), mode='constant', value=1.0).numpy()
                distance_to_edge_inverse = F.pad(torch.Tensor(distance_to_edge_inverse), pad=(0, pixels_eliminate, 0, 0), mode='constant', value=0.0).numpy()
                ph_validity_mask = F.pad(torch.Tensor(ph_validity_mask), pad=(0, pixels_eliminate, 0, 0), mode='constant', value=0.0).numpy()
            if top_pad==0 and bottom_pad>0 and ph_validity_mask.shape!=im.shape:
                distance_to_edge = F.pad(torch.Tensor(distance_to_edge), pad=(0, pixels_eliminate, 0, 0), mode='constant', value=1.0).numpy()
                distance_to_edge_inverse = F.pad(torch.Tensor(distance_to_edge_inverse), pad=(0, pixels_eliminate, 0, 0), mode='constant', value=0.0).numpy()
                ph_validity_mask = F.pad(torch.Tensor(ph_validity_mask), pad=(0, pixels_eliminate, 0, 0), mode='constant', value=0.0).numpy()
               
    # repeat in band dimension and clip with validity mask to put linear weights only if cardinality is valid
    weights = np.repeat(np.expand_dims(distance_to_edge,axis=0), repeats=num_channels, axis=0) # repeat axis to archieve same shape as original tensor
    weights[ph_validity_mask == 0.] = 1. # take validity mask of original into account
    # inverse the weights to serve as multiplicator for PH tensor
    weights_inverse = np.repeat(np.expand_dims(distance_to_edge_inverse,axis=0), repeats=num_channels, axis=0) # repeat axis to archieve same shape as original tensor
    weights_inverse[1-ph_validity_mask == 1.] = 0. # clip area with validity mask to set rest of values to either 0 or 1
    
    # now, get rid of corner artifacts
    # first, extract middle edge squares
    center_start = ((hr_size//2)-overlap)
    center_end = ((hr_size//2)+overlap)
    offset_squares = overlap*2
    # normal weights extraction
    top_center = weights[:, 0:offset_squares, center_start:center_end]
    bottom_center = weights[:, -offset_squares:, center_start:center_end]
    left_center = weights[:, center_start:center_end,:offset_squares]
    right_center = weights[:, center_start:center_end,-offset_squares:]
    # inverse corner extraction
    top_center_inverse = weights_inverse[:, 0:offset_squares, center_start:center_end]
    bottom_center_inverse = weights_inverse[:, -offset_squares:, center_start:center_end]
    left_center_inverse = weights_inverse[:, center_start:center_end,:offset_squares]
    right_center_inverse = weights_inverse[:, center_start:center_end,-offset_squares:]
    
    # perform replacement based on all possible cardinality cases
    
    if top_valid and not right_valid:
        weights[:, 0:offset_squares, -offset_squares:] = top_center
        weights_inverse[:, 0:offset_squares, -offset_squares:] = top_center_inverse
    if top_valid and not left_valid:
        weights[:,0:offset_squares, 0:offset_squares] = top_center
        weights_inverse[:,0:offset_squares, 0:offset_squares] = top_center_inverse
    if left_valid and not top_valid:
        weights[:,0:offset_squares, 0:offset_squares] = left_center
        weights_inverse[:,0:offset_squares, 0:offset_squares] = left_center_inverse
    if left_valid and not bottom_valid:
        weights[:, -offset_squares:, 0:offset_squares] = left_center
        weights_inverse[:, -offset_squares:, 0:offset_squares] = left_center_inverse
    if bottom_valid and not left_valid:
        weights[:, -offset_squares:, 0:offset_squares] = bottom_center
        weights_inverse[:, -offset_squares:, 0:offset_squares] = bottom_center_inverse
    if bottom_valid and not right_valid:
        weights[:, -offset_squares:, -offset_squares:] = bottom_center
        weights_inverse[:, -offset_squares:, -offset_squares:] = bottom_center_inverse
    if right_valid and not bottom_valid:
        weights[:, -offset_squares:, -offset_squares:] = right_center
        weights_inverse[:, -offset_squares:, -offset_squares:] = right_center_inverse
    if right_valid and not top_valid:
        weights[:, 0:offset_squares, -offset_squares:] = right_center
        weights_inverse[:, 0:offset_squares, -offset_squares:] = right_center_inverse
    
        
    # reset NaN values to 0 from earlier
    weights = np.nan_to_num(weights)
    weights_inverse =  np.nan_to_num(weights_inverse)
    
    
    try:
        assert np.all(weights+weights_inverse)==1., "weights dont sum um to 1. for every single pixel. Abort."
    except AssertionError:
        import warnings
        warnings.warn("Weight matrix in valid in weighted overlap. Returning zero tensor.", UserWarning)
    
    # perform weighting
    try:
        weighted_image = (weights*im) + (weights_inverse*ph)
        return(weighted_image)
    except:
        import warnings
        warnings.warn("Weighted overlap failed due to weight dimensions. Returning zero tensor.", UserWarning)
        return(torch.zeros_like(torch.Tensor(sr)))
    