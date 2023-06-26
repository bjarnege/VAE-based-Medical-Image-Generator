import torch
import torch.nn as nn


# Definieren der Encoder Klasse
class EncoderVAE(nn.Module):

    def __init__(self, img_channels, feature_dim, latent_dim):
        super(EncoderVAE, self).__init__()
        self.encConv = nn.Conv2d(img_channels, 16, 5)
        self.encConv2 = nn.Conv2d(16, 32, 5)
        self.FC_mean = nn.Linear(feature_dim, latent_dim)
        self.FC_var = nn.Linear(feature_dim, latent_dim)
        self.LeakyReLU = nn.LeakyReLU(0.2)
        self.feature_dim = feature_dim

    # Forward pass durch den Encoder -> Bild zu mean und log_var
    def forward(self, x):
        h_ = self.LeakyReLU(self.encConv(x))
        h_ = self.LeakyReLU(self.encConv2(h_))
        h_ = h_.view(-1, self.feature_dim)

        mean = self.FC_mean(h_)
        log_var = self.FC_var(h_)  # encoder produces mean and log of variance
        #             (i.e., parateters of simple tractable normal distribution "q"

        return mean, log_var


# Definieren der Decoder-Klasse
class DecoderVAE(nn.Module):

    def __init__(self, img_channels, feature_dim, latent_dim):
        super(DecoderVAE, self).__init__()
        self.decFC1 = nn.Linear(latent_dim, feature_dim)
        self.decConv = nn.ConvTranspose2d(32, 16, 5)
        self.decConv2 = nn.ConvTranspose2d(16, img_channels, 5)
        self.LeakyReLU = nn.LeakyReLU(0.2)

    def forward(self, z):
        # Forward pass durch den Decoder -> latenter z-Vektor zu Bild
        h = self.LeakyReLU(self.decFC1(z))
        h = h.view(-1, 32, 20, 20)
        h = self.LeakyReLU(self.decConv(h))
        x_hat = torch.sigmoid(self.decConv2(h))

        return x_hat


# Zusammenführen beider Modelle im VAE
class VariationalAutoencoder(nn.Module):

    def __init__(self, Encoder, Decoder, device):
        super(VariationalAutoencoder, self).__init__()
        self.Encoder = Encoder
        self.Decoder = Decoder
        self.device = device

    def reparameterization(self, mean, var):
        # Sampling des latenten Vektors z mittels mean und var
        epsilon = torch.randn_like(var).to(self.device)  # sampling epsilon
        z = mean + var * epsilon  # reparameterization trick
        return z

    def forward(self, x):
        mean, log_var = self.Encoder(x)
        # Samplen der Daten aus Gausverteilung mit mean und var
        z = self.reparameterization(mean, torch.exp(0.5 * log_var))  # takes exponential function (log var -> var)
        x_hat = self.Decoder(z)

        return x_hat, mean, log_var


# define vae loss function that will be optimized
def vae_loss(x, x_hat, mean, log_var):
    reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
    kld_regularizer = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

    return reproduction_loss + kld_regularizer