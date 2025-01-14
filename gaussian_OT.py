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


