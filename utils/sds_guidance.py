import torch
from torch import tensor
from einops import rearrange
import numpy as np


def rbf_kernel(X, Y, gamma=-1, ad=1):
    # X and Y should be tensors with shape (batch_size, num_channels, height, width)
    # gamma is a hyperparameter controlling the width of the RBF kernel

    # Reshape X and Y to have shape (batch_size, num_channels*height*width)
    X_flat = X.view(X.size(0), -1)
    Y_flat = Y.view(Y.size(0), -1)

    # Compute the pairwise squared Euclidean distances between the samples
    with torch.cuda.amp.autocast():
        dists = torch.cdist(X_flat, Y_flat, p=2)**2

    if gamma <0: # use median trick
        gamma = torch.median(dists)
        gamma = torch.sqrt(0.5 * gamma / np.log(dists.size(0) + 1))
        gamma = 1 / (2 * gamma**2)
        # print(gamma)

    gamma = gamma * ad 
    # gamma = torch.max(gamma, torch.tensor(1e-3))
    # Compute the RBF kernel using the squared distances and gamma
    K = torch.exp(-gamma * dists)
    dK = -2 * gamma * K.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * (X.unsqueeze(1) - Y.unsqueeze(0))
    dK_dX = torch.sum(dK, dim=1)

    return K, dK_dX