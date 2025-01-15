import torch
from gaussian_OT import gaussian_OT_torch


def compute_orthogonal_complement(V):
    """
    Compute the orthogonal complement of the column space of V using PyTorch.
    
    Args:
    V (torch.Tensor): An n x k matrix with orthonormal columns.

    Returns:
    torch.Tensor: An n x (n-k) matrix whose columns form an orthonormal basis for the orthogonal complement of V.
    """
    n = V.shape[0]
    I = torch.eye(n, device=V.device)  # Identity matrix
    P = V @ V.T  # Projection matrix onto the column space of V
    Q, R = torch.linalg.qr(I - P)  # QR decomposition of I - P
    
    return Q[:, :n - V.shape[1]]  # Return the first n-k columns of Q which span the orthogonal complement

def schur_complement(A, V):
    """
    Compute the Schur complement of A with respect to the subspace spanned by the columns of V using PyTorch.
    
    Args:
    A (torch.Tensor): The square matrix from which to compute the Schur complement.
    V (torch.Tensor): An n x k matrix whose columns form an orthonormal basis for the subspace E.

    Returns:
    torch.Tensor: The Schur complement of A with respect to E.
    """
    V_perp = compute_orthogonal_complement(V)
    AE = V.T @ A @ V
    AEE_perp = V.T @ A @ V_perp
    AE_perp = V_perp.T @ A @ V_perp

    AE_inv = torch.linalg.inv(AE)
    schur = AE_perp - AEE_perp.T @ AE_inv @ AEE_perp
    
    return schur
def decompose_matrix_torch(A, V, V_perp):
    """
    Decompose the matrix A into blocks based on V and V_perp using PyTorch.
    
    Args:
    A (torch.Tensor): The matrix to decompose.
    V (torch.Tensor): Subspace matrix.
    V_perp (torch.Tensor): Orthogonal complement matrix.

    Returns:
    tuple: Tuple of decomposed matrix components (AE, AEE_perp, AE_perp, AE_perpE).
    """
    AE = V.T @ A @ V
    AEE_perp = V.T @ A @ V_perp
    AE_perp = V_perp.T @ A @ V_perp
    AE_perpE = V_perp.T @ A @ V  # Should be transpose of AEE_perp if A is symmetric

    return AE, AEE_perp, AE_perp, AE_perpE
def MK_gaussian_torch(A, B, V):
    """
    Compute the Monge-Knothe transport matrix for subspaces defined by V in PyTorch.
    
    Args:
    A (torch.Tensor): Covariance matrix A
    B (torch.Tensor): Covariance matrix B
    V (torch.Tensor): An n x k matrix with orthonormal columns defining the subspace.

    Returns:
    torch.Tensor: The Monge-Knothe transport matrix.
    """
    k = V.shape[1]
    d = A.shape[0]
    V_perp = compute_orthogonal_complement(V)
    AE, AEE_perp, AE_perp, _ = decompose_matrix_torch(A, V, V_perp)
    BE, BEE_perp, BE_perp, _ = decompose_matrix_torch(B, V, V_perp)

    AE_inv = torch.linalg.inv(AE)
    BE_inv = torch.linalg.inv(BE)
    
    Aschur = AE_perp - AEE_perp.T @ AE_inv @ AEE_perp
    Bschur = BE_perp - BEE_perp.T @ BE_inv @ BEE_perp
    Tee, _ = gaussian_OT_torch(AE, BE)
    Tschur, _ = gaussian_OT_torch(Aschur, Bschur)

    T_AE_BE_inv = torch.linalg.inv(Tee)

    zero_matrix = torch.zeros((k, d - k), device=A.device)

    lower_left = (BEE_perp.T @ T_AE_BE_inv - Tschur @ AEE_perp.T) @ AE_inv
    T_MK = torch.cat([
        torch.cat([Tee, zero_matrix], dim=1),
        torch.cat([lower_left, Tschur], dim=1)
    ], dim=0)
    
    return T_MK



