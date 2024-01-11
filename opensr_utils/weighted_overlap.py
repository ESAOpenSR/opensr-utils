import numpy as np
import torch


def weighted_overlap(sr, placeholder,overlap):

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

    distance_to_edge = calculate_distance_to_edge(512, 512) # get distance to edge for each pixel
    distance_to_edge[distance_to_edge > overlap] = overlap # set maximum value to overlap amount
    distance_to_edge = distance_to_edge/overlap # set weight as fraction from overlap
    distance_to_edge_inverse = 1.-distance_to_edge # set inverse factional weight for SR placeholder

    # repeat channels for later addition
    weights = np.repeat(np.expand_dims(distance_to_edge,axis=0), repeats=num_channels, axis=0) # repeat axis to archieve same shape as original tensor
    weights_inverse = np.repeat(np.expand_dims(distance_to_edge_inverse,axis=0), repeats=num_channels, axis=0) # repeat axis to archieve same shape as original tensor

    # clip area with validity mask to set rest of values to either 0 or 1
    weights[ph_validity_mask == 0.] = 1.
    weights_inverse[~ph_validity_mask == 1.] = 0.

    # perform weighting
    weighted_image = (weights*im) + (weights_inverse*ph)

    return(weighted_image)