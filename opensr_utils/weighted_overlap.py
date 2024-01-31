import numpy as np
import torch
import torch.nn.functional as F 

"""def weighted_overlap(sr, placeholder,overlap=8,pixels_eliminate=0,hr_size=512):
    
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

    # mask that says where the image is valid
    ph_validity_mask = (ph != 0).astype(bool)

    # info necessary
    overlap = overlap # define the amount of pixel overlap
    num_channels = sr.shape[0] # amount of bands

    distance_to_edge = calculate_distance_to_edge(hr_size-pixels_eliminate, hr_size-pixels_eliminate) # get distance to edge for each pixel
    distance_to_edge = np.pad(distance_to_edge, (pixels_eliminate // 2, pixels_eliminate // 2), 'constant', constant_values=np.NaN) # pad back with 0s around the edges
    distance_to_edge[distance_to_edge>overlap] = overlap # set maximum value to overlap amount
    distance_to_edge = distance_to_edge/overlap # set weight as fraction from overlap
    distance_to_edge_inverse = 1.-distance_to_edge # set inverse factional weight for SR placeholder

    weights = np.repeat(np.expand_dims(distance_to_edge,axis=0), repeats=num_channels, axis=0) # repeat axis to archieve same shape as original tensor
    #print(weights.shape,ph_validity_mask.shape)
    weights[ph_validity_mask == 0.] = 1. # take validity mask of original into account
    #weights = np.pad(weights, ((0, 0), (pixels_eliminate//2, pixels_eliminate//2), (pixels_eliminate//2, pixels_eliminate//2)), 'constant', constant_values=(0)) # pad back to original dimensions

    # inverse the weights to serve as multiplicator for PH tensor
    weights_inverse = np.repeat(np.expand_dims(distance_to_edge_inverse,axis=0), repeats=num_channels, axis=0) # repeat axis to archieve same shape as original tensor
    weights_inverse[~ph_validity_mask == 1.] = 0. # clip area with validity mask to set rest of values to either 0 or 1
    #weights_inverse = np.pad(weights_inverse, ((0, 0), (pixels_eliminate//2, pixels_eliminate//2), (pixels_eliminate//2, pixels_eliminate//2)), 'constant', constant_values=(0)) # pad back to original dimension
    
    # reset NaN values to 0 from earlier
    weights = np.nan_to_num(weights)
    weights_inverse =  np.nan_to_num(weights_inverse)
    
    # perform weighting
    weighted_image = (weights*im) + (weights_inverse*ph)

    return weighted_image#weighted_image

"""

def weighted_overlap(sr, placeholder,overlap=8,pixels_eliminate=0,hr_size=512):
    
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

    # mask that says where the image is valid
    ph_validity_mask = (ph != 0).astype(bool)
    
    # Create different version for if we want to eliminate pixels
    if pixels_eliminate>0:
        ph_validity_mask = F.interpolate(torch.Tensor(ph_validity_mask).unsqueeze(0),(hr_size-pixels_eliminate,hr_size-pixels_eliminate),mode='nearest').squeeze(0).numpy()
    

    # info necessary
    num_channels = sr.shape[0] # amount of bands

    distance_to_edge = calculate_distance_to_edge(ph_validity_mask.shape[-1], ph_validity_mask.shape[-2]) # get distance to edge for each pixel
    distance_to_edge[distance_to_edge>overlap] = overlap # set maximum value to overlap amount
    distance_to_edge = distance_to_edge/overlap # set weight as fraction from overlap
    distance_to_edge_inverse = 1.-distance_to_edge # set inverse factional weight for SR placeholder
    
    # if wanted, padd validity mask again to original dimension to remove edge pixels
    if pixels_eliminate>0:
        # padd each border cardinality accordingly
        # find out if each cardinality is valid
        half_pixel = ph_validity_mask.shape[-1]//2
        left_valid = ph_validity_mask[0,:,:][0,half_pixel] >0.5
        right_valid = ph_validity_mask[0,:,:][-1,half_pixel] >0.5
        top_valid = ph_validity_mask[0,:,:][half_pixel,0] >0.5
        bottom_valid = ph_validity_mask[0,:,:][half_pixel,-1] >0.5

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
            
        # ib 2 cardinalities in same direction are cropped, half the cropping area
        if left_valid and right_valid:
            left_pad = pixels_eliminate//2
            right_pad = pixels_eliminate//2
        if top_valid and bottom_valid:
            top_pad = pixels_eliminate//2
            bottom_pad = pixels_eliminate//2 
            
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
            ph_validity_mask = np.pad(ph_validity_mask,  ((0,0),(left_pad, right_pad), (top_pad, bottom_pad)), mode='constant',constant_values  = 1.)
            distance_to_edge = np.pad(distance_to_edge,  ((left_pad, right_pad), (top_pad, bottom_pad)), mode='constant',constant_values  = 1.)
            distance_to_edge_inverse = np.pad(distance_to_edge_inverse,  ((left_pad, right_pad), (top_pad, bottom_pad)), mode='constant',constant_values  = 1.)

            
            # now, if only 1 side of each cardinality is padded, we need to pad the other side with either 0s or 1s according to the original validity mask
            if left_pad>0 and right_pad==0  and ph_validity_mask.shape!=im.shape:
                distance_to_edge = np.pad(distance_to_edge,  ((0,0),(0, pixels_eliminate)), mode='constant',constant_values  = 1.)
                distance_to_edge_inverse = np.pad(distance_to_edge_inverse,  ((0,0),(0, pixels_eliminate)), mode='constant',constant_values  = 0.)
                ph_validity_mask = np.pad(ph_validity_mask,  ((0,0,),(0,0),(0, pixels_eliminate)), mode='constant',constant_values  = 0.)
            if left_pad==0 and right_pad>0  and ph_validity_mask.shape!=im.shape:
                distance_to_edge = np.pad(distance_to_edge,  ((0,0),(pixels_eliminate)), mode='constant',constant_values  = 1.)
                distance_to_edge_inverse = np.pad(distance_to_edge_inverse,  ((0,0),(pixels_eliminate)), mode='constant',constant_values  = 0.)
                ph_validity_mask = np.pad(ph_validity_mask,  ((0,0,),(0,0),(pixels_eliminate)), mode='constant',constant_values  = 0.)
            if top_pad>0 and bottom_pad==0  and ph_validity_mask.shape!=im.shape:
                distance_to_edge = np.pad(distance_to_edge,  ((0,pixels_eliminate),(0,0)), mode='constant',constant_values  = 1.)
                distance_to_edge_inverse = np.pad(distance_to_edge_inverse,  ((0,pixels_eliminate),(0,0)), mode='constant',constant_values  = 0.)
                ph_validity_mask = np.pad(ph_validity_mask,  ((0,0,),(0,pixels_eliminate),(0,0)), mode='constant',constant_values  = 0.)
            if top_pad==0 and bottom_pad>0 and ph_validity_mask.shape!=im.shape:
                distance_to_edge = np.pad(distance_to_edge,  ((pixels_eliminate,0),(0,0)), mode='constant',constant_values  = 1.)
                distance_to_edge_inverse = np.pad(distance_to_edge_inverse,  ((pixels_eliminate,0),(0,0)), mode='constant',constant_values  = 0.)
                ph_validity_mask = np.pad(ph_validity_mask,  ((0,0,),(pixels_eliminate,0),(0,0)), mode='constant',constant_values  = 0.)

    weights = np.repeat(np.expand_dims(distance_to_edge,axis=0), repeats=num_channels, axis=0) # repeat axis to archieve same shape as original tensor
    weights[ph_validity_mask == 0.] = 1. # take validity mask of original into account
    # inverse the weights to serve as multiplicator for PH tensor
    weights_inverse = np.repeat(np.expand_dims(distance_to_edge_inverse,axis=0), repeats=num_channels, axis=0) # repeat axis to archieve same shape as original tensor
    weights_inverse[1-ph_validity_mask == 1.] = 0. # clip area with validity mask to set rest of values to either 0 or 1
    
    # reset NaN values to 0 from earlier
    weights = np.nan_to_num(weights)
    weights_inverse =  np.nan_to_num(weights_inverse)
    #print("pads",left_pad,right_pad,top_pad,bottom_pad) # 10 0 10 0
    #print("image shapes:",weights.shape,weights_inverse.shape,im.shape) 
    
    assert weights.shape == weights_inverse.shape and weights_inverse.shape == im.shape, "In overlap calculation: images arent of the same shape as weights. Abort."
    
    # perform weighting
    weighted_image = (weights*im) + (weights_inverse*ph)

    return weighted_image