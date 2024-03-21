import pandas as pd
from ctgan import CTGANSynthesizer
from sdv.tabular import TVAE
import torch
from torch import nn

# Load your dataset
data = pd.read_csv('data.csv')

# Define the number of synthetic samples you want to generate
num_samples = len(data)

# Initialize and train CTGAN
ctgan = CTGANSynthesizer(epochs=5)  # Use more epochs for better training
ctgan.fit(data)

# Initialize and train TVAE from SDV
tvae = TVAE(epochs=5)  # Use more epochs for better training
tvae.fit(data)

# Define the TVAE model in PyTorch
class PyTorchTVAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(PyTorchTVAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim * 2)  # Output mean and log variance
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
            nn.Sigmoid()
        )

    def encode(self, x):
        mean_logvar = self.encoder(x)
        mean, logvar = mean_logvar.chunk(2, dim=-1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mean)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        return self.decode(z), mean, logvar

    def loss_function(self, recon_x, x, mean, logvar):
        recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        return recon_loss + kl_loss

# Prepare data for PyTorch model
input_dim = data.shape[1]
latent_dim = 10  # Size of the latent space
pytorch_tvae = PyTorchTVAE(input_dim, latent_dim)
optimizer = torch.optim.Adam(pytorch_tvae.parameters(), lr=1e-3)

data_tensor = torch.tensor(data.values, dtype=torch.float32)

# Training loop for our PyTorch TVAE
epochs = 5  # Use more epochs for better training
for epoch in range(epochs):
    pytorch_tvae.train()
    optimizer.zero_grad()
    reconstructed, mean, logvar = pytorch_tvae(data_tensor)
    loss = pytorch_tvae.loss_function(reconstructed, data_tensor, mean, logvar)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

# Generate synthetic data
ctgan_data = ctgan.sample(num_samples)
tvae_data = tvae.sample(num_samples)

# Generate data from PyTorch TVAE
pytorch_tvae.eval()
with torch.no_grad():
    z_sample = torch.randn(num_samples, latent_dim)
    vae_data = pytorch_tvae.decode(z_sample).numpy()

# Combine the synthetic data
synthetic_data = pd.concat([ctgan_data, pd.DataFrame(tvae_data), pd.DataFrame(vae_data)])

# You can now use synthetic_data for your further analysis or modeling
