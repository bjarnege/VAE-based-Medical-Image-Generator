import torch


def calc_reconstruction_loss_bce(input_images: torch.Tensor, output_images: torch.Tensor, reduction: str = 'mean') -> torch.Tensor:
    """
    Computes the binary cross-entropy (BCE) reconstruction loss between input and output images.

    The BCE loss measures the difference between the input images and their reconstructed versions, where each pixel of the images
    is treated as a binary classification problem (belonging to the foreground or background). The reduction argument specifies
    how to reduce the loss over the batch, either by taking the mean or sum.

    Args:
        input_images (torch.Tensor): The input images.
        output_images (torch.Tensor): The reconstructed images.
        reduction (str, optional): The reduction method for the loss, either "mean" or "sum". Defaults to "mean".

    Returns:
        torch.Tensor: The BCE reconstruction loss between the input and output images.
    """
    return torch.nn.functional.binary_cross_entropy(output_images, input_images, reduction=reduction)


def calc_reconstruction_loss_mse(input_images: torch.Tensor, output_images: torch.Tensor) -> torch.Tensor:
    """
    Computes the mean squared error (MSE) reconstruction loss between input and output images.

    The MSE loss measures the difference between the input images and their reconstructed versions, where each pixel of the images
    is treated as a continuous regression problem (predicting the pixel value). 

    Args:
        input_images (torch.Tensor): The input images.
        output_images (torch.Tensor): The reconstructed images.

    Returns:
        torch.Tensor: The MSE reconstruction loss between the input and output images.
    """
    return torch.nn.functional.mse_loss(output_images, input_images)