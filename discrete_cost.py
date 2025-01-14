import numpy as np

import numpy as np

def discrete_cost(T, X, Y):
    """
    Compute the discrete transport cost using the transport plan T between sets X and Y.
    
    Parameters:
    T (np.array): Transport matrix (N x N), binary, with T[i, j] indicating transport from X[i] to Y[j].
    X (np.array): Source points (N x d).
    Y (np.array): Target points (N x d).
    
    Returns:
    float: The total transport cost calculated as the sum of squared Euclidean distances.
    """
    N = X.shape[0]
    # Reorder Y according to the transport plan T
    Y_reordered = Y[np.argmax(T, axis=1)]  # np.argmax(T, axis=1) gives the index of the 1 in each row of T

    # Compute the transport cost in the original space using the reordered Y
    # Using broadcasting to subtract and then sum the squares of the differences
    cost_matrix_full = np.sum((X - Y_reordered) ** 2, axis=1)
    total_cost = np.sum(cost_matrix_full)

    return total_cost/N