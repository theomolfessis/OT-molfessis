import numpy as np

def discrete_cost(T, X, Y):
    N = X.shape[0]
    Y_reordered = Y[np.argmax(T, axis=1)]  # Reorder full-dimensional Y according to the OT map

    # Compute the transport cost in the original space using the reordered Y
    cost_matrix_full = np.linalg.norm(X[:, np.newaxis, :] - Y_reordered[np.newaxis, :, :], axis=2)**2
    total_cost = np.sum(T * cost_matrix_full)

    return total_cost