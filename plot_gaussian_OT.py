import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.linalg import sqrtm
from mpl_toolkits.mplot3d import Axes3D

def geodesic_path(m1, Sigma1, m2, Sigma2, t):
    """ Compute the mean and covariance of the geodesic path at time t """
    mean_t = (1 - t) * m1 + t * m2
    cov_t = (1 - t)**2 * Sigma1 + t**2 * Sigma2 + \
            t * (1 - t) * (sqrtm(sqrtm(Sigma1) @ Sigma2 @ sqrtm(Sigma1))) + \
            t * (1 - t) * (sqrtm(sqrtm(Sigma2) @ Sigma1 @ sqrtm(Sigma2)))
    return mean_t, cov_t

def plot_geodesic(m1, Sigma1, m2, Sigma2, steps):
    """ Plot the evolution of the Gaussian distribution along the geodesic path in 2D or 3D """
    dimension = len(m1)
    fig = plt.figure(figsize=(18, 4))

    for i, t in enumerate(steps):
        if dimension == 3:
            ax = fig.add_subplot(1, len(steps), i + 1, projection='3d')
            grid_extent = np.linspace(-3, 3, 100)
            x, y, z = np.meshgrid(grid_extent, grid_extent, grid_extent)
            pos = np.empty(x.shape + (3,))
            pos[..., 0] = x; pos[..., 1] = y; pos[..., 2] = z
        else:
            ax = fig.add_subplot(1, len(steps), i + 1)
            grid_extent = np.linspace(-3, 3, 100)
            x, y = np.meshgrid(grid_extent, grid_extent)
            pos = np.empty(x.shape + (2,))
            pos[..., 0] = x; pos[..., 1] = y

        mean_t, cov_t = geodesic_path(m1, Sigma1, m2, Sigma2, t)
        rv = multivariate_normal(mean_t, cov_t)
        if dimension == 3:
            pdf = rv.pdf(pos)
            ax.contourf(x, y, z, pdf, zdir='z', offset=-0.1, levels=50, cmap='viridis')
        else:
            pdf = rv.pdf(pos)
            ax.contourf(x, y, pdf, levels=50, cmap='viridis')

        ax.set_title(f"t={t*100:.0f}%")
        if dimension == 3:
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('PDF')
            ax.set_xlim([-3, 3])
            ax.set_ylim([-3, 3])
            ax.set_zlim([-0.1, 0.1])
        else:
            ax.set_xlabel('X')
            ax.set_ylabel('Y')

    plt.show()



