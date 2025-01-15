import numpy as np
from gaussian_OT import gaussian_OT

def compute_orthogonal_complement(V):
    """
    Compute the orthogonal complement of the column space of V.
    
    Args:
    V (numpy.ndarray): An n x k matrix with orthonormal columns.

    Returns:
    numpy.ndarray: An n x (n-k) matrix whose columns form an orthonormal basis for the orthogonal complement of V.
    """
    n = V.shape[0]
    I = np.eye(n)  # Identity matrix
    P = V @ V.T  # Projection matrix onto the column space of V
    Q, R = np.linalg.qr(I - P)  # QR decomposition of I - P
    
    V_perp= Q[:, :n - V.shape[1]]
  

    return V_perp  # Return the first n-k columns of Q which span the orthogonal complement

def schur_complement(A, V):
    """
    Compute the Schur complement of A with respect to the subspace spanned by the columns of V.
    
    Args:
    A (numpy.ndarray): The square matrix from which to compute the Schur complement.
    V (numpy.ndarray): An n x k matrix whose columns form an orthonormal basis for the subspace E.

    Returns:
    numpy.ndarray: The Schur complement of A with respect to E.
    """
    # Calculate the orthogonal complement of V
    V_perp = compute_orthogonal_complement(V)
    # Submatrices
    AE = V.T @ A @ V
    AEE_perp = V.T @ A @ V_perp
    AE_perp = V_perp.T @ A @ V_perp

    # Compute the Schur complement
    try:
        AE_inv = np.linalg.inv(AE)
    except np.linalg.LinAlgError:
        raise ValueError("The submatrix AE is not invertible, cannot compute the Schur complement.")

    schur = AE_perp - AEE_perp.T @ AE_inv @ AEE_perp
    
    return schur

def decompose_matrix(A, V, V_perp):
    # Decompose the matrix A into blocks based on V and V_perp
    AE = V.T @ A @ V
    AEE_perp = V.T @ A @ V_perp
    AE_perp = V_perp.T @ A @ V_perp
    AE_perpE = V_perp.T @ A @ V  # This should be the transpose of AEE_perp if A is symmetric

    return AE, AEE_perp, AE_perp, AE_perpE

def MK_gaussian(A,B,V):
    k = V.shape[1]
    d = A.shape[0]
    V_perp = compute_orthogonal_complement(V)



    AE, AEE_perp, AE_perp, AE_perpE = decompose_matrix(A,V,V_perp)
    BE, BEE_perp, BE_perp, BE_perpE = decompose_matrix(B,V,V_perp)
    try:
        AE_inv = np.linalg.inv(AE)
    except np.linalg.LinAlgError:
        raise ValueError("The submatrix AE is not invertible, cannot compute the Schur complement.")
    try:
        BE_inv = np.linalg.inv(BE)
    except np.linalg.LinAlgError:
        raise ValueError("The submatrix BE is not invertible, cannot compute the Schur complement.")
    
    Aschur = AE_perp - AEE_perp.T @ AE_inv @ AEE_perp
    Bschur = BE_perp - BEE_perp.T @ BE_inv @ BEE_perp
    T_AE_BE , _ = gaussian_OT(AE,BE,np.zeros(AE.shape[0]),np.zeros(BE.shape[0]))
    T_Aschur_Bschur = gaussian_OT(Aschur,Bschur,np.zeros(Aschur.shape[0]),np.zeros(Bschur.shape[0]))

    T_AE_BE_inv = np.linalg.inv(T_AE_BE)

    # Dimensions for the zero matrix
    zero_matrix = np.zeros((k, d - k))

    # Assemble T_MK using numpy.block
    lower_left = (BEE_perp.T @ T_AE_BE_inv - T_Aschur_Bschur @ AEE_perp.T) @ AE_inv
    T_MK = np.block([
        [T_AE_BE, zero_matrix],
        [lower_left, T_Aschur_Bschur]
])
    return T_MK
