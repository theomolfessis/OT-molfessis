import numpy as np
from MK_torch import  MK_gaussian_torch
from gaussian_cost import transport_cost_gaussians, transport_cost_gaussians_torch
from gaussian_OT import gaussian_OT_torch
import torch
from torch.linalg import svd, inv

def polar_decomposition(A):
    """
    Compute the polar decomposition of a matrix A, returning the unitary and
    positive semi-definite parts.

    Args:
    A (numpy.ndarray): The input matrix for which to compute the polar decomposition.

    Returns:
    the unitary matrix U
    """
    U, S, V = svd(A)
    return U @ V.T




def algorithm_one(A, B, eta, num_iterations, k):
    """
    Perform gradient descent on the Grassmann manifold to find the optimal transport matrix using PyTorch.

    Args:
    A, B (torch.Tensor): Covariance matrices of the source and target Gaussian distributions.
    eta (float): Learning rate.
    num_iterations (int): Number of iterations.
    k (int): Chosen dimension for the subspace.

    Returns:
    torch.Tensor: Optimized subspace matrix V.
    """
    # Initialization
    V = polar_decomposition(A @ B)
    cost_hist = []

    for i in range(num_iterations):
        Vk = V[:, :k]

        # Here you need to define MK_gaussian to compute T_MK and possibly compute its gradients
        T_MK, _ = MK_gaussian_torch(A, B, Vk)  # Assuming MK_gaussian returns the transport matrix and cost

        # Calculate the cost - adjust as needed if transport_cost_gaussians_torch needs different parameters
        MK_cost = transport_cost_gaussians_torch(Vk.T @ A @ Vk, Vk.T @ B @ Vk, T_MK)
        cost_hist.append(MK_cost.item())

        # Computing the gradient of V manually in respect to the cost
        V.requires_grad_(True)
        MK_cost.backward()

        # Gradient update on the Grassmann manifold (retracting to the manifold)
        with torch.no_grad():
            gradient = V.grad
            V -= eta * gradient  # simple gradient descent update
            V, _ = torch.linalg.qr(V)  # re-orthonormalize to stay on the Grassmann manifold

        # Optionally print the cost to monitor convergence
        if i % 10 == 0:
            print(f"Iteration {i}, Cost: {MK_cost.item()}")

    return V[:, :k], cost_hist

def MK_torch(A, B, k=2):
    d = A.shape[0]
  
    Ae = A[:k, :k]
    Aeet = A[:k, k:]
    Aet = A[k:, k:]
  
    schurA = Aet - Aeet.t().mm(torch.inverse(Ae)).mm(Aeet)
  
    Be = B[:k, :k]
    Beet = B[:k, k:]
    Bet = B[k:, k:]
  
    schurB = Bet - Beet.t().mm(torch.inverse(Be)).mm(Beet)
  
    Tee = gaussian_OT_torch(Ae, Be)
    Tschur = gaussian_OT_torch(schurA, schurB)
 
    return torch.cat((torch.cat((Tee, Beet.t().mm(torch.inverse(Be)).mm(Tee) - Tschur.mm(Aeet.t()).mm(torch.inverse(Ae))), 0), 
                      torch.cat((torch.zeros((k, d-k)), Tschur), 0)), 1)

def MK_dist_torch(A, B, k = 2):
    T = MK_torch(A, B, k)
    return torch.trace(A + B - (T.mm(A) + A.mm(T.t())))


def subspace_gd(A, B, k, lr = 5e-5, niter = 401, minimize=True, verbose=False):

    d = A.shape[0]

    losses = []
    Ps = []

#    L = torch.randn((d, d))
#    S = L.mm(L.t()) / d

    S = A.mm(B)

    with torch.no_grad():
        S.data = torch.from_numpy(polar_decomposition(S.data))

#    S = torch.eye(d)
    S.requires_grad = True

    optimizer = torch.optim.SGD([S], lr = lr)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.98, last_epoch=-1)


    for i in range(niter):

        SAS = S.t().mm(A).mm(S)
        SBS = S.t().mm(B).mm(S)


        loss = (2 * minimize - 1) * MK_dist_torch(SAS, SBS, k)

        if loss.item() != loss.item():
            print('Nan loss')
            break

        losses.append(loss.item())
        Ps.append(S.detach())

        if i % 50 == 0 and verbose:
            print(('iteration {} : loss {}').format(i, torch.abs(loss)))


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            S.data = torch.from_numpy(polar_decomposition(S.data))

    best_iter = np.argmin(losses)
    P_opt = Ps[best_iter]
    
    return P_opt, losses[best_iter]