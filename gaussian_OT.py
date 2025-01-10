import numpy as np

def gaussian_OT(A, B,m1,m2):
    """
    Compute the optimal transport map and its cost between two Gaussian distributions.
    
    Parameters:
    A (numpy.ndarray): Covariance matrix of the source Gaussian distribution.
    B (numpy.ndarray): Covariance matrix of the target Gaussian distribution.
    m1 (numpy.ndarray): Mean vector of the source Gaussian distribution.
    m2 (numpy.ndarray): Mean vector of the target Gaussian distribution.
    Returns:
    numpy.ndarray: The matrix representing the optimal transport map.
    float: The squared cost of transporting the source distribution to the target distribution.
    """
    # Compute the square root and inverse of the square root of A
    A_sqrt = np.linalg.cholesky(A)
    A_inv_sqrt = np.linalg.inv(A_sqrt)
    
    # Compute the middle term (A^{1/2} B A^{1/2})
    middle_matrix = A_sqrt @ B @ A_sqrt
    
    # Compute the square root of the middle term
    middle_sqrt = np.linalg.matrix_power(middle_matrix, 1/2)
    
    # Compute the optimal transport map matrix
    T = A_inv_sqrt @ middle_sqrt @ A_inv_sqrt
    
    # Compute the transport cost
    sq_cost = np.linalg.norm(m1-m2)**2 + np.trace(A + B - 2 * middle_sqrt)
    
    return T, sq_cost


