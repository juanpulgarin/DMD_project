import numpy as np
from scipy.linalg import svd, eig, pinv


def custom_hankel_embedding(X, delay):
    n, m = X.shape
    hankel_matrix = np.vstack([
        X[:, i: m - delay + i + 1] for i in range(delay) ])
    return hankel_matrix


def perform_dmd(H, rank=None):
    X1 = H[:, :-1]
    X2 = H[:, 1:]

    U, S, Vh = svd(X1, full_matrices=False)

    if rank is not None:
        U_tilde  = U[:, :rank]
        S_tilde  = S[:rank]
        Vh_tilde = Vh[:rank, :]
    else:
        U_tilde  = U
        S_tilde  = S
        Vh_tilde = Vh

    Sigma_inv = np.diag(1 / (S_tilde+0.000001))

    A_tilde = U_tilde.T @ X2 @ Vh_tilde.T @ Sigma_inv
    eigvals, W = eig(A_tilde)

    Phi = X2 @ Vh_tilde.T @ Sigma_inv @ W

    return eigvals,Phi,S
