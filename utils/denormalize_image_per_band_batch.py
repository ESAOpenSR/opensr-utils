import torch
import einops
from einops import rearrange

def denormalize_image_per_band_batch(normalized_image_tensor,means,stds):
    """
    Denormalizes a batch of multi-band image tensors using the given means and standard deviations per band.

    This function takes a batch of multi-band image tensors that have been normalized, along with the
    corresponding means and standard deviations for each band used during normalization. It denormalizes
    each element in the batch independently, restoring the pixel values of each band back to their
    original scale.

    Note: The input tensors should have the shape (batch, bands, height, width), where 'batch' is the
    number of image samples in the batch, 'bands' is the number of bands in each image, 'height' is the
    height of the image, and 'width' is the width of the image.

    Note: The means and stdevs are extracted from the pytorch datamodule, so they need to be set when creating
    the dataloader object. Example: pl_datamodule.denormalization_parameters = {"means":[0,0,0,0],"stds":[0,0,0,0]}

    Args:
        normalized_image_tensor (torch.Tensor): A batch of multi-band normalized image tensors with shape
            (batch, bands, height, width).

    Returns:
        torch.Tensor: A tensor containing the denormalized multi-band image for each batch element.
            The shape of the output tensor will be (batch, bands, height, width).
    """

    # rearrange to B C W H if shape is B W H C
    if normalized_image_tensor.shape[1]>normalized_image_tensor.shape[-1]:
        rearrange_needed = True
        normalized_image_tensor = rearrange(normalized_image_tensor,"b w h c -> b c w h")
    else:
        rearrange_needed = False

    # normalized_image_tensor shape: (batch, bands, height, width)
    # means and stds are tensors with shape (bands,)
    batch_size, num_bands, height, width = normalized_image_tensor.shape

    denormalized_images = []
    for i in range(batch_size):
        batch_element = normalized_image_tensor[i]  # Get the i-th element from the batch
        denormalized_bands = []
        for band, mean, std in zip(batch_element, means, stds):
            denormalized_band = (band * std) + mean
            denormalized_bands.append(denormalized_band)
        denormalized_image = torch.stack(denormalized_bands, dim=0)
        denormalized_images.append(denormalized_image)

    #stack to tensor
    denormalized_images = torch.stack(denormalized_images, dim=0)

    # return to original dimensions B C W H 
    if rearrange_needed:
        denormalized_images = rearrange(denormalized_images,"b c w h -> b w h c")

    # return stacked tensors
    return denormalized_images