import torch


def calc_kl_divergence(mean: torch.Tensor, log_variance: torch.Tensor) -> torch.Tensor:
    """
    Computes the Kullback-Leibler (KL) divergence between a learned latent distribution and a prior distribution.

    The KL divergence measures the difference between two probability distributions, in this case the learned latent distribution
    and a prior distribution (here, a standard normal distribution). The mean and log_variance tensors are the output of the VAE encoder.

    Args:
        mean (torch.Tensor): The mean of the learned latent distribution.
        log_variance (torch.Tensor): The log variance of the learned latent distribution.

    Returns:
        torch.Tensor: The KL divergence between the learned latent distribution and the prior distribution.
    """
    return -0.5 * torch.sum(1 + log_variance - mean.pow(2) - log_variance.exp())
