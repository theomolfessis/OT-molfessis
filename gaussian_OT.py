import numpy as np
from scipy.linalg import sqrtm,inv

def gaussian_OT(Sigma1, Sigma2, m1, m2):
    """
    Compute the optimal transport map and cost for Gaussian distributions.
    
    Parameters:
    Sigma1 (np.array): Covariance matrix of the source Gaussian distribution.
    Sigma2 (np.array): Covariance matrix of the target Gaussian distribution.
    m1 (np.array): Mean of the source Gaussian distribution.
    m2 (np.array): Mean of the target Gaussian distribution.
    
    Returns:
    tuple: A tuple containing the optimal transport map matrix and the transport cost.
    """
    # Calculate the middle matrix for the transformation
    middle_matrix = sqrtm(Sigma1) @ Sigma2 @ sqrtm(Sigma1)
    # Compute the square root of the middle matrix
    middle_sqrt = sqrtm(middle_matrix)
    # Compute the optimal transport map matrix A
    A = np.linalg.inv(sqrtm(Sigma1)) @ middle_sqrt @ np.linalg.inv(sqrtm(Sigma1))
    # Compute the transport cost using the Bures-Wasserstein metric
    transport_cost = np.linalg.norm(m1 - m2)**2 + np.trace(Sigma1 + Sigma2 - 2 * middle_sqrt)

    return A, transport_cost


import torch
from torch.linalg import inv, matrix_power

def sqrtm_torch(A):
    """
    Compute the matrix square root of A using PyTorch's SVD.
    """
    U, S, Vh = torch.linalg.svd(A)
    return U @ torch.diag(torch.sqrt(S)) @ Vh

def gaussian_OT_torch(Sigma1, Sigma2):
    """
    Compute the optimal transport map and cost for Gaussian distributions using PyTorch.
    
    Parameters:
    Sigma1 (torch.Tensor): Covariance matrix of the source Gaussian distribution.
    Sigma2 (torch.Tensor): Covariance matrix of the target Gaussian distribution.
    m1 (torch.Tensor): Mean of the source Gaussian distribution.
    m2 (torch.Tensor): Mean of the target Gaussian distribution.
    
    Returns:
    tuple: A tuple containing the optimal transport map matrix and the transport cost.
    """
    # Calculate the middle matrix for the transformation
    middle_matrix = sqrtm_torch(Sigma1) @ Sigma2 @ sqrtm_torch(Sigma1)
    # Compute the square root of the middle matrix
    middle_sqrt = sqrtm_torch(middle_matrix)
    # Compute the optimal transport map matrix A
    A = inv(sqrtm_torch(Sigma1)) @ middle_sqrt @ inv(sqrtm_torch(Sigma1))
    # Compute the transport cost using the Bures-Wasserstein metric
    transport_cost =  torch.trace(Sigma1 + Sigma2 - 2 * middle_sqrt)

    return A, transport_cost
