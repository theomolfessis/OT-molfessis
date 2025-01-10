import numpy as np

def generate_covariance_matrix(d, lambda_, decay_rate):
    # Generate eigenvalues that decay geometrically
    eigenvalues = np.array([lambda_ * (decay_rate ** i) for i in range(d)])
    
    # Generate a random matrix and perform QR decomposition to obtain orthogonal matrix Q
    random_matrix = np.random.randn(d, d)
    Q, _ = np.linalg.qr(random_matrix)
    
    # Construct the covariance matrix using the spectral decomposition
    Lambda = np.diag(eigenvalues)
    covariance_matrix = Q @ Lambda @ Q.T
    
    return covariance_matrix

def generate_gaussian_samples(n, covariance_matrix):
    d = covariance_matrix.shape[0]
    mean = np.zeros(d)
    return np.random.multivariate_normal(mean, covariance_matrix, n)

# Parameters
d = 5  # Dimensionality of the Gaussian
lambda_ = 1  # Largest eigenvalue
decay_rate = 0.5  # Decay rate of eigenvalues
n = 1000  # Number of samples

# Generate the covariance matrix
covariance_matrix = generate_covariance_matrix(d, lambda_, decay_rate)
print("Covariance Matrix:")
print(covariance_matrix)

# Generate samples from the Gaussian distribution
samples = generate_gaussian_samples(n, covariance_matrix)
print("Sample Shape:", samples.shape)
