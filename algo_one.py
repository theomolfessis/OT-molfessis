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
    V, _, W = np.linalg.svd(A)
    return V.dot(W.T)

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
  
    Tee , _ = gaussian_OT_torch(Ae, Be)
    Tschur, _ = gaussian_OT_torch(schurA, schurB)
 
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