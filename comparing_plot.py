import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.linalg import sqrtm, inv
from mpl_toolkits.mplot3d import Axes3D

def calculate_A(Sigma1, Sigma2):
    """ Compute the optimal transport map matrix A for Gaussian distributions """
    middle_matrix = sqrtm(Sigma1) @ Sigma2 @ sqrtm(Sigma1)
    middle_sqrt = sqrtm(middle_matrix)
    A = np.linalg.inv(sqrtm(Sigma1)) @ middle_sqrt @ np.linalg.inv(sqrtm(Sigma1))
    return A

def comparing_plot(m1, Sigma1, m2, Sigma2, X, Y, T1, T2):
    """ Plot the optimal transport map and two discrete transports with 2D, 3D and higher dimensional considerations """
    dimension = X.shape[1]
    if dimension == 2:
        fig, ax = plt.subplots(figsize=(10, 8))
    elif dimension == 3:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
    elif dimension > 3:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')


    A = calculate_A(Sigma1, Sigma2)  # Compute the transport matrix A for 2D and 3D

    # Plot trajectories based on the optimal transport map for Gaussian distributions
    for i, x in enumerate(X):
        trajectory = [(1-t) * x + (m2 + A @ (x - m1)) * t for t in np.linspace(0, 1, 100)]
        trajectory = np.array(trajectory)
        if dimension >= 3:
            ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 'k--', alpha=0.2, label='Gaussian Trajectory' if i == 0 else "")
        else:
            ax.plot(trajectory[:, 0], trajectory[:, 1], 'k--', alpha=0.2, label='Gaussian Trajectory' if i == 0 else "")

    bijection1 = np.argmax(T1, axis=1)
    bijection2 = np.argmax(T2, axis=1)

    # Plot discrete transports from X to Y
    for i, j in enumerate(bijection1):
        if dimension >= 3:
            ax.plot([X[i, 0], Y[j, 0]], [X[i, 1], Y[j, 1]], [X[i, 2], Y[j, 2]] if dimension >= 3 else [], 'r-', alpha=0.5, label='Discrete optimal, high dim' if i == 0 else "")
        else:
            ax.plot([X[i, 0], Y[j, 0]], [X[i, 1], Y[j, 1]], 'r-', alpha=0.5, label='Discrete optimal, high dim' if i == 0 else "")

    for i, j in enumerate(bijection2):
        if dimension >= 3:
            ax.plot([X[i, 0], Y[j, 0]], [X[i, 1], Y[j, 1]], [X[i, 2], Y[j, 2]] if dimension >= 3 else [], 'b--', alpha=0.5, label='Discrete optimal, subspace' if i == 0 else "")
        else:
            ax.plot([X[i, 0], Y[j, 0]], [X[i, 1], Y[j, 1]], 'b--', alpha=0.5, label='Discrete optimal, subspace' if i == 0 else "")

    # Plot the sampled points from X and Y
    if dimension >= 3:
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], c='green', s=100, zorder=5, edgecolors='k', label='Points X')
        ax.scatter(Y[:, 0], Y[:, 1], Y[:, 2], c='orange', s=100, zorder=5, edgecolors='k', label='Points Y')
    else:
        ax.scatter(X[:, 0], X[:, 1], c='green', s=100, zorder=5, edgecolors='k', label='Points X')
        ax.scatter(Y[:, 0], Y[:, 1], c='orange', s=100, zorder=5, edgecolors='k', label='Points Y')

    ax.set_title("Optimal Transport and Trajectories")
    ax.legend()
    plt.show()

# Example parameters needed to be defined for your specific case
