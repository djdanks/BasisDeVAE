# Based on https://github.com/kasparmartens/BasisVAE/blob/master/BasisVAE/encoder.py
# Original author: Kaspar Märtens

import torch
import torch.nn as nn

from torch.nn.functional import softplus

class Encoder(nn.Module):

    def __init__(self, data_dim, hidden_dim, z_dim, nonlinearity=torch.nn.ReLU,
                 device="cpu"):
        """
        Encoder for the VAE (neural network that maps P-dimensional data to [mu_z, sigma_z])
        :param data_dim:
        :param hidden_dim:
        :param z_dim:
        :param nonlinearity:
        """
        super().__init__()

        if device == "gpu":
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            print(f"Encoder: {device} specified, {self.device} used")
        else:
            self.device = torch.device("cpu")
            print(f"Encoder: {device} specified, {self.device} used")

        self.z_dim = z_dim

        self.mapping = torch.nn.Sequential(
            torch.nn.Linear(data_dim, hidden_dim),
            nonlinearity(),
            torch.nn.Linear(hidden_dim, 2*z_dim)
        )

    def forward(self, Y):

        out = self.mapping(Y)

        mu = out[:, 0:self.z_dim]
        sigma = 1e-6 + softplus(out[:, self.z_dim:(2 * self.z_dim)])
        return mu, sigma
