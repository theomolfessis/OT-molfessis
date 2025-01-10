import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.linalg import sqrtm, inv

def calculate_A(Sigma1, Sigma2):
    """ Compute the optimal transport map matrix A for Gaussian distributions """
    middle_matrix = sqrtm(sqrtm(Sigma2) @ Sigma1 @ sqrtm(Sigma2))
    return sqrtm(Sigma2) @ inv(middle_matrix) @ sqrtm(Sigma2)

def plot_transport_and_trajectories(m1, Sigma1, m2, Sigma2, X, Y, bijection1, bijection2):
    """ Plot the optimal transport map and two discrete transports """
    fig, ax = plt.subplots(figsize=(10, 8))

    A = calculate_A(Sigma1, Sigma2)  # Compute the transport matrix A

    # Plot trajectories based on the optimal transport map for Gaussian distributions
    ii=0
    for x in X:
        trajectory = [(1-t) * x + (m2 + A @ (x - m1)) * t for t in np.linspace(0, 1, 100)]
        trajectory = np.array(trajectory)
        if ii== 0:
            ax.plot(trajectory[:, 0], trajectory[:, 1], 'k--', alpha=0.2, label='Gaussian Trajectory')
        else : 
            ax.plot(trajectory[:, 0], trajectory[:, 1], 'k--', alpha=0.2)
        ii +=1

    # Plot discrete transports from X to Y
    for i, j in enumerate(bijection1):
        ax.plot([X[i, 0], Y[j, 0]], [X[i, 1], Y[j, 1]], 'r-', alpha=0.5, label='Bijection 1' if i == 0 else "")
    for i, j in enumerate(bijection2):
        ax.plot([X[i, 0], Y[j, 0]], [X[i, 1], Y[j, 1]], 'b--', alpha=0.5, label='Bijection 2' if i == 0 else "")

    # Plot the sampled points from X and Y
    ax.scatter(X[:, 0], X[:, 1], c='green', s=100, zorder=5, edgecolors='k', label='Points X')
    ax.scatter(Y[:, 0], Y[:, 1], c='orange', s=100, zorder=5, edgecolors='k', label='Points Y')

    ax.set_title("Optimal Transport and Trajectories")
    ax.legend()
    plt.show()

# Example usage
m1 = np.array([0, 0])
Sigma1 = np.array([[2, 0.5], [0.5, 1]])
m2 = np.array([5, 5])
Sigma2 = np.array([[1, 0], [0, 3]])
X = np.random.multivariate_normal(m1, Sigma1, 10)  # Sample 10 points from the first Gaussian
Y = np.random.multivariate_normal(m2, Sigma2, 10)  # Target points for the discrete transport

bijection1 = np.arange(10)  # Example bijections
bijection2 = np.arange(10)

plot_transport_and_trajectories(m1, Sigma1, m2, Sigma2, X, Y, bijection1, bijection2)
