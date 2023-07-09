import torch
import torch.nn

from vae_based_medical_image_generator.metrics.kl_divergence import calc_kl_divergence
from vae_based_medical_image_generator.metrics.reconstruction import calc_reconstruction_loss_bce


def vae_loss(x: torch.Tensor, x_hat: torch.Tensor, mean: torch.Tensor, log_variance: torch.Tensor) -> torch.Tensor:
    """
    Computes the variational autoencoder (VAE) loss for a batch of input images.

    The VAE loss is the sum of the binary cross-entropy (BCE) reconstruction loss and the Kullback-Leibler (KL) divergence loss,
    where the reconstruction loss measures the difference between the input images and their reconstructed versions, and the KL divergence
    loss measures the difference between the learned latent distribution and a prior distribution (here, a standard normal distribution).
    The mean and log_variance tensors are the output of the VAE encoder and are used to compute the KL divergence loss.

    Args:
        x (torch.Tensor): The input images.
        x_hat (torch.Tensor): The reconstructed images.
        mean (torch.Tensor): The mean of the learned latent distribution.
        log_variance (torch.Tensor): The log variance of the learned latent distribution.

    Returns:
        torch.Tensor: The VAE loss for the batch of input images.

    Raises:
        ValueError: If the input tensors have different shapes.
    """
    return calc_reconstruction_loss_bce(x, x_hat, reduction='sum') + calc_kl_divergence(mean, log_variance)
