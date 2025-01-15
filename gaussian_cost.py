import numpy as np
def transport_cost_gaussians(A, B, T):
    """
    Compute the transportation cost between two Gaussian distributions
    with covariance matrices A and B under the linear transport map T.

    Args:
    A (numpy.ndarray): Covariance matrix of the source Gaussian distribution μ.
    B (numpy.ndarray): Covariance matrix of the target Gaussian distribution ν.
    T (numpy.ndarray): The linear transport map matrix.

    Returns:
    float: The expected transportation cost.
    """
    # Compute traces
    trA = np.trace(A)
    trB = np.trace(B)
    trTA_AT = np.trace(T @ A + A.T @ T)  # Using the property that trace(AB) = trace(BA)

    # Compute the transportation cost
    cost = trA + trB - trTA_AT
    return cost

import torch

def transport_cost_gaussians_torch(A, B, T):
    """
    Compute the transportation cost between two Gaussian distributions
    with covariance matrices A and B under the linear transport map T using PyTorch.

    Args:
    A (torch.Tensor): Covariance matrix of the source Gaussian distribution μ.
    B (torch.Tensor): Covariance matrix of the target Gaussian distribution ν.
    T (torch.Tensor): The linear transport map matrix.

    Returns:
    torch.Tensor: The expected transportation cost.
    """
    # Compute traces
    trA = torch.trace(A)
    trB = torch.trace(B)
    trTA_AT = torch.trace(T @ A + A.T @ T)  # Using the property that trace(AB) = trace(BA)

    # Compute the transportation cost
    cost = trA + trB - trTA_AT
    return cost