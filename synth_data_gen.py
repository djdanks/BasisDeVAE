# Code to generate the synthetic data used in the paper (noise added in main.py)

import matplotlib.pyplot as plt
import numpy as np

np.random.seed(0)

def gaussian(A, t0, sigma, t):
    """Expects numpy arrays with shapes:

    A: (d,)
    t0: (d,)
    sigma: (d,)
    t: (N,)

    Returns (N,d) numpy array.
    """
    t_ = t[:,None]
    t0_ = t0[None,:]
    A_ = A[None,:]
    sigma_ = sigma[None,:]
    return A_ * np.exp(-0.5*(t_ - t0_)**2/sigma**2)

def scaled_softplus(t, t_shift, A):
    """Expects numpy arrays with shapes:

    A: (d,)
    t_shift: (d,)
    t: (N,)

    Returns (N,d) numpy array."""
    return A[None,:]*0.4*np.logaddexp(0, t[:,None] + t_shift[None,:])


t = np.linspace(-3,3,500) # Times considered (ground truth pseudotimes)

# Gaussian components
A = np.linspace(0.4,1.6,30)
np.random.shuffle(A)
t0 = np.linspace(-1.5,1.5,30)
np.random.shuffle(t0)
sigma = np.ones(30)
gaussians = gaussian(A,t0,sigma,t)

# Monotonically increasing components
t_shift_inc = np.linspace(-1.5,1.5,10)
A_inc = np.linspace(0.4,1.6,10)
np.random.shuffle(A_inc)
np.random.shuffle(t_shift_inc)
monoinc = scaled_softplus(t, t_shift_inc, A_inc)
# Monotonically decreasing components
t_shift_dec = np.linspace(-1.5,1.5,10)
A_dec = np.linspace(0.4,1.6,10)
np.random.shuffle(A_dec)
np.random.shuffle(t_shift_dec)
monodec = -scaled_softplus(t, t_shift_dec, A_dec)

# plt.plot(t,gaussians,c='red')
# plt.plot(t,monoinc,c='green')
# plt.plot(t,monodec,c='blue')
# plt.savefig('synth_features.png')

np.savetxt("synth_x.csv", np.concatenate((gaussians,monoinc,monodec),1), delimiter=",")
np.savetxt("synth_t.csv", t, delimiter=",")
