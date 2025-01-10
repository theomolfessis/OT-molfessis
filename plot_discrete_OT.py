import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_discrete_OT(X, Y, T, dimension=2):
    """
    Plot the optimal transport between two sets of points in 2D or 3D.
    
    Parameters:
    X (numpy.ndarray): Source points (N x dimension).
    Y (numpy.ndarray): Target points (N x dimension).
    T (numpy.ndarray): Transport matrix (N x N).
    dimension (int): Dimension of the data (2 or 3).
    """
    if dimension not in [2, 3]:
        raise ValueError("Dimension must be 2 or 3")
    
    fig = plt.figure()
    if dimension == 3:
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], color='blue', marker='o', label='Source')
        ax.scatter(Y[:, 0], Y[:, 1], Y[:, 2], color='red', marker='^', label='Target')
    else:
        ax = fig.add_subplot(111)
        ax.scatter(X[:, 0], X[:, 1], color='blue', marker='o', label='Source')
        ax.scatter(Y[:, 0], Y[:, 1], color='red', marker='^', label='Target')
    
    # Draw lines for the transport plan
    for i in range(len(X)):
        for j in range(len(Y)):
            if T[i, j] > 0:  # There's a transport from i to j
                # Extract starting and ending points
                start_point = X[i, :dimension]
                end_point = Y[j, :dimension]
                # Draw a line between them
                ax.plot(*zip(start_point, end_point), color='gray', linewidth=0.5, alpha=0.5)
    
    ax.legend()
    ax.grid(True)
    plt.title('Optimal Transport Plan Visualization')
    plt.show()

# Example usage
N = 5
dimension = 3  # Change to 3 for 3D visualization
X = np.random.rand(N, dimension) * 10  # Source points
Y = np.random.rand(N, dimension) * 10  # Target points

# Example transport plan (e.g., identity for simplicity)
T = np.eye(N)

plot_optimal_transport(X, Y, T, dimension=dimension)
