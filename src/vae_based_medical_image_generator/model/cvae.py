import torch
import torch.nn as nn
import numpy as np


class ConditionalVariationalAutoencoder(nn.Module):

    def __init__(self, image_channels: int, n_labels: int, latent_dimension: int, device: torch.device):
        super(ConditionalVariationalAutoencoder, self).__init__()
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

        self.mean_fc = nn.Linear(hidden_dimension + n_labels, latent_dimension)
        self.log_variance_fc = nn.Linear(hidden_dimension + n_labels, latent_dimension)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dimension + n_labels, hidden_dimension),
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
        return self.encoder(torch.zeros(1, image_channels, 28, 28)).shape[-1]

    def reparameterization(self, mean, log_variance):
        standard_deviation = torch.exp(0.5 * log_variance)
        epsilon = torch.randn_like(standard_deviation).to(self.device)
        return mean + standard_deviation * epsilon

    def forward(self, x, c):
        hidden = self.encoder(x)
        hidden_with_labels = torch.cat([hidden, c], 1)

        mean, log_variance = self.mean_fc(hidden_with_labels), self.log_variance_fc(hidden_with_labels)
        sampled_latent_variable = self.reparameterization(mean, log_variance)
        sampled_latent_variable_with_labels = torch.cat([sampled_latent_variable, c], 1)
        return self.decoder(sampled_latent_variable_with_labels), mean, log_variance


def vae_loss(x, x_hat, mean, log_variance):
    reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
    kld_regularizer = -0.5 * torch.sum(1 + log_variance - mean.pow(2) - log_variance.exp())
    return reproduction_loss + kld_regularizer