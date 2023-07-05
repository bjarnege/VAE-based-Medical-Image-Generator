import torch
import torch.nn as nn
import numpy as np


class VariationalAutoencoder(nn.Module):

    def __init__(self, image_channels: int, latent_dimension: int, device: torch.device):
        """
        Constructs a Variational Autoencoder (VAE) neural network architecture for image generation.

        Args:
            image_channels (int): The number of channels in the input and output images.
            n_labels (int): The number of labels.
            latent_dimension (int): The dimensionality of the learned latent space.
            device (torch.device): The device on which to run the VAE (e.g., 'cpu', 'cuda').

        Attributes:
            encoder (nn.Sequential): The encoder network.
            mean_fc (nn.Linear): The fully-connected layer for the mean of the learned latent distribution.
            log_variance_fc (nn.Linear): The fully-connected layer for the log variance of the learned latent distribution.
            decoder (nn.Sequential): The decoder network.
        """
        super(VariationalAutoencoder, self).__init__()
        self.device = device

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=image_channels, out_channels=16, kernel_size=3, stride=1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Flatten(),
        )

        hidden_dimension = self.get_encoder_hidden_dimension(image_channels)

        self.mean_fc = nn.Linear(hidden_dimension, latent_dimension)
        self.log_variance_fc = nn.Linear(hidden_dimension, latent_dimension)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dimension, hidden_dimension),
            torch.nn.Unflatten(1, torch.Size([64, int(np.sqrt(hidden_dimension // 64)),
                                              int(np.sqrt(hidden_dimension // 64))])),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.ConvTranspose2d(in_channels=16, out_channels=image_channels, kernel_size=3, stride=1),
            nn.Sigmoid(),
        )

    def get_encoder_hidden_dimension(self, image_channels: int):
        """
        Returns the dimensionality of the encoder's output feature maps.

        Args:
            image_channels (int): The number of channels in the input images.

        Returns:
            int: The dimensionality of the encoder's output feature maps.
        """
        return self.encoder(torch.zeros(1, image_channels, 28, 28)).shape[-1]

    def reparameterization(self, mean, log_variance):
        """
        Performs the reparameterization trick to sample from the learned latent distribution.

        The reparameterization trick is used to sample from the learned latent distribution by introducing random
        noise to the mean and log variance of the distribution, which allows backpropagation through the sampling
        operation.

        Args:
            mean (torch.Tensor): The mean of the learned latent distribution.
            log_variance (torch.Tensor): The log variance of the learned latent distribution.

        Returns:
            torch.Tensor: A tensor of samples from the learned latent distribution.
        """
        standard_deviation = torch.exp(0.5 * log_variance)
        epsilon = torch.randn_like(standard_deviation).to(self.device)
        return mean + standard_deviation * epsilon

    def forward(self, x):
        """
        Computes the forward pass of the VAE.

        The forward pass of the VAE consists of encoding the input images into the learned latent space, sampling
        from the learned latent distribution, and decoding the samples back into reconstructed images. The VAE loss
        is the sum of the binary cross-entropy (BCE) reconstruction loss and the Kullback-Leibler (KL) divergence
        loss, where the reconstruction loss measures the difference between the input images and their reconstructed
        versions, and the KL divergence loss measures the difference between the learned latent distribution and a
        prior distribution (here, a standard normal distribution).

        Args:
            x (torch.Tensor): The input images with dimensions (batch_size, image_channels, image_height, image_width).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple of the reconstructed images, the mean of the learned
            latent distribution, and the log variance of the learned latent distribution.
        """
        hidden = self.encoder(x)
        mean, log_variance = self.mean_fc(hidden), self.log_variance_fc(hidden)
        sampled_latent_variable = self.reparameterization(mean, log_variance)
        return self.decoder(sampled_latent_variable), mean, log_variance