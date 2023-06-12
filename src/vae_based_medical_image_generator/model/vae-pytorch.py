import torch
import torch.nn as nn

# Definieren der Encoder Klasse
class EncoderVAE(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(EncoderVAE, self).__init__()

        self.FC_input = nn.Linear(input_dim, hidden_dim)
        self.FC_input2 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_input3 = nn.Linear(hidden_dim, hidden_dim)

        self.FC_mean  = nn.Linear(hidden_dim, latent_dim)
        self.FC_var   = nn.Linear (hidden_dim, latent_dim)
        
        self.LeakyReLU = nn.LeakyReLU(0.2)
        self.training = True
    
    # Forward pass durch den Encoder -> Bild zu mean und log_var
    def forward(self, x):
        h_  = self.LeakyReLU(self.FC_input(x))
        h_  = self.LeakyReLU(self.FC_input2(h_))
        h_  = self.LeakyReLU(self.FC_input3(h_))
        mean = self.FC_mean(h_)
        log_var  = self.FC_var(h_)                     # encoder produces mean and log of variance 
                                                       #             (i.e., parateters of simple tractable normal distribution "q"
        
        return mean, log_var

# Definieren der Decoder-Klasse
class DecoderVAE(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(DecoderVAE, self).__init__()
        self.FC_hidden = nn.Linear(latent_dim, hidden_dim)
        self.FC_hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_hidden3 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_output = nn.Linear(hidden_dim, output_dim)
        
        self.LeakyReLU = nn.LeakyReLU(0.2)
        
    def forward(self, x):
        # Forward pass durch den Decoder -> latenter z-Vekto zu Bild
        h = self.LeakyReLU(self.FC_hidden(x))
        h = self.LeakyReLU(self.FC_hidden2(h))
        h = self.LeakyReLU(self.FC_hidden3(h))
              
        x_hat = torch.sigmoid(self.FC_output(h))
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
        epsilon = torch.randn_like(var).to(self.device)        # sampling epsilon        
        z = mean + var*epsilon                          # reparameterization trick
        return z
        
                
    def forward(self, x):
        mean, log_var = self.Encoder(x)
        # Samplen der Daten aus Gausverteilung mit mean und var
        z = self.reparameterization(mean, torch.exp(0.5 * log_var)) # takes exponential function (log var -> var)
        x_hat            = self.Decoder(z)
        
        return x_hat, mean, log_var