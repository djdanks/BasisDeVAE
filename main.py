# Demonstration of training BasisDeVAE on the synthetic data, obtaining fits
# corresponding to Figure 4, bottom left panel of the paper.

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sys

from collections import OrderedDict

from VAE import VAE
from decoder import BasisDecoder, BasisODEDecoder, ODEDecoder, StandardDecoder
from encoder import Encoder

from torch.utils.data import TensorDataset, DataLoader

import synth_data_gen

seed = 0
np.random.seed(seed)
torch.manual_seed(seed)

training = True
# training = False
device = "gpu"

# Import data, add noise and load into dataloader
df = pd.read_csv(os.path.join(os.getcwd(),"synth_x.csv"))
dfz = pd.read_csv(os.path.join(os.getcwd(),"synth_t.csv"))
z = dfz.values.squeeze()
idcs = np.argsort(z)
D = df.values
data_noise = 0.1
D_ = D + data_noise * np.random.randn(*D.shape)
Y = torch.Tensor(D_)
dataset = TensorDataset(Y)
batch_size = 32
shuffle = training
data_loader = DataLoader(dataset, batch_size=batch_size, pin_memory=True, shuffle=shuffle)

# Specify encoder and decoder
z_dim = 1
hidden_dim = 64
data_dim = Y.shape[1]
encoder = Encoder(data_dim, hidden_dim, z_dim, device=device)
decoder = BasisODEDecoder(data_dim, hidden_dim, z_dim, 3,
                       likelihood="Gaussian",
                       nonlinearity = torch.nn.Tanh,
                       alpha=0.1,
                       x0init=torch.mean(Y,0,True).numpy(),
                       device=device)

model = VAE(encoder, decoder, lr=0.005)

if training:
    model.optimize(data_loader, n_epochs=300, logging_freq=10)
    torch.save(model.state_dict(), "basisdevae")
    sys.exit(0)
else:
    state_dict = torch.load("basisdevae")
    model.load_state_dict(state_dict)
    model.eval()


# Only run if training == False

# Get cluster assignments
with torch.no_grad():
    cluster_probs = decoder.get_phi()
w = cluster_probs.cpu().numpy()
w_argmax = w.argmax(1)

# Reconstruct data via VAE
idx = np.arange(data_dim)
z = np.array([])
Y_pred = np.array([]).reshape(0,data_dim)
for batch_idx, (Y_subset, ) in enumerate(data_loader):
    z_ = encoder(Y_subset.to(decoder.device))[0].detach().cpu().numpy().squeeze()
    Y_ = model.decoder(torch.tensor(z_,device=decoder.device)[:,None])[0].detach().cpu().numpy()[:,idx,w_argmax]
    z = np.concatenate((z,z_),axis=None)
    Y_pred = np.concatenate((Y_pred,Y_),axis=0)
z_argsort = np.argsort(z)
z_grid = z[z_argsort]
Y_pred = Y_pred[z_argsort,:]


plt.rcParams.update({'font.size': 16})
plt.figure()
colors = ['red','blue','green']
for j in range(data_dim):
    plt.plot(z_grid, Y_pred[:, j], c=colors[w_argmax[j]])
plt.xlabel('z')
plt.ylabel('x(z)')
plt.ylim([-2.,2.])
plt.title('BasisDeVAE')
plt.tight_layout()
plt.savefig(f"synth_fit.pdf", bbox_inches='tight')
