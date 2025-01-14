import numpy as np
def generate_covariance_matrix(d, lambda_, decay_rate, diagonal=True, decay_type='geometric'):
    # Generate eigenvalues
    if decay_type == 'geometric':
        eigenvalues = np.array([lambda_ * (decay_rate ** i) for i in range(d)])
    elif decay_type == 'linear':
        eigenvalues = np.array([lambda_ - decay_rate * i for i in range(d) if lambda_ - decay_rate * i > 0])
    else:
        raise ValueError("Invalid decay type. Use 'geometric' or 'linear'.")

    # Ensure eigenvalues do not become non-positive in case of linear decay
    eigenvalues = np.maximum(eigenvalues, 1e-5)  # replace near-zero or negative values with a small positive number

    # Generate the covariance matrix
    if diagonal:
        covariance_matrix = np.diag(eigenvalues)
    else:
        # Generate a random matrix and perform QR decomposition to obtain an orthogonal matrix Q
        np.random.seed(42)
        random_matrix = np.random.randn(d, d)
        Q, _ = np.linalg.qr(random_matrix)
        # Construct the covariance matrix using the spectral decomposition
        Lambda = np.diag(eigenvalues)
        covariance_matrix = Q @ Lambda @ Q.T

    return covariance_matrix , eigenvalues


def generate_gaussian_samples(n, covariance_matrix,mean= None):
    d = covariance_matrix.shape[0]
    if mean is None:
        mean = np.zeros(d)
    return np.random.multivariate_normal(mean, covariance_matrix, n)

def generate_samples(d,lambda1,lambda2,decay1,decay2,decay_type1,decay_type2,n,mean2):
    S ,eigenS= generate_covariance_matrix(d,lambda1,decay1,decay_type=decay_type1)
    Sbis ,eigenSbis= generate_covariance_matrix(d,lambda2,decay2,diagonal=False,decay_type=decay_type2)
    X = generate_gaussian_samples(n,S)
    Y = generate_gaussian_samples(n,Sbis,mean=mean2)
    return X, Y, S , Sbis,eigenS,eigenSbis
