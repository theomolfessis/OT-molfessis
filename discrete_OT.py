import numpy as np
import cvxpy as cp

def discrete_OT(X, Y):
    N = X.shape[0]
    cost_matrix = np.linalg.norm(X[:, np.newaxis, :] - Y[np.newaxis, :, :], axis=2)**2

    T = cp.Variable((N, N), boolean=True)
    constraint_rows = cp.sum(T, axis=1) == 1
    constraint_cols = cp.sum(T, axis=0) == 1
    objective = cp.Minimize(cp.sum(cp.multiply(T, cost_matrix)))
    problem = cp.Problem(objective, [constraint_rows, constraint_cols])
    problem.solve()

    return T.value, cost_matrix