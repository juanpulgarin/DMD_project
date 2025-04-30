import numpy as np
import scipy.linalg as LA
from scipy.linalg import svd, eig, pinv


def custom_hankel_embedding(X, delay):
    n, m = X.shape
    hankel_matrix = np.vstack([
        X[:, i: m - delay + i + 1] for i in range(delay)
    ])
    return hankel_matrix

def perform_dmd(H, rank=None, d=1, dt=1):
    #print(np.shape(H))
    X1 = H[:, :-1]
    X2 = H[:, 1:]

    U, S, Vh = svd(X1, full_matrices=False)

    if rank is not None:
        U_r  = U[:, :rank]
        S_r  = S[:rank]
        Vh_r = Vh[:rank, :]
    else:
        U_r  = U[:, :]
        S_r  = S[:]
        Vh_r = Vh[:, :]

    Sigma_inv = np.diag(1 / (S_r + 0.000001 ))
    A_tilde = U_r.T @ X2 @ Vh_r.T @ Sigma_inv
    eigvals, W = LA.eig(A_tilde)
    Phi = X2 @ Vh_r.T @ Sigma_inv @ W
    #if d > 1:
    Φ = np.average( Phi.reshape(d,Phi.shape[0] // d, Phi.shape[1],),axis=0,)


    return S,eigvals,Phi, Φ
    #x0 = X1[:, 0]
    #b = pinv(Phi) @ x0
    #omega = np.log(eigvals) / dt
    #timesteps = np.arange(X1.shape[1]) * dt + t0
    #print(timesteps)
    #time_dynamics = np.array([b * np.exp(omega * t) for t in timesteps]).T

    #X_dmd = Phi @ time_dynamics
    #return eigvals, Phi, X_dmd.real, timesteps

def Dynamics(H,Eigvals,Modes,nt,d):
    x1=H[:,0]

    #if d > 1:
    Φ = np.average( Modes.reshape(d,Modes.shape[0] // d, Modes.shape[1],),axis=0,)

    time_steps = np.linspace(0,nt-d,nt-d+1,dtype=int)
    temp = np.repeat( Eigvals[:, None], time_steps.shape[0], axis=1 )
    tpow = ( time_steps - 0 ) / ( time_steps[1] - time_steps[0] )

    amplitudes = LA.pinv(Modes) @ x1
    dynamics = np.power(temp, tpow) * amplitudes[:, None]

    return temp, amplitudes, dynamics , np.dot(Φ,dynamics)

def LEAST_OUTER(D, Phi, mu, tspan, dt, r):
    # Get dimensions
    n_rows, len_tspan = np.shape(D)
    dim_outer_space = len_tspan * n_rows

    # Flatten real and imaginary parts of D
    xbig_real = np.real(D).flatten()
    xbig_imag = np.imag(D).flatten()

    # Concatenate real and imaginary parts into x_big
    x_big = np.concatenate((xbig_real, xbig_imag))

    # Precompute exponentials
    t_diff = tspan - tspan[0]
    exp_terms = np.exp(np.log(mu[:, np.newaxis] ** (1 / dt)) * t_diff[np.newaxis, :])

    # Initialize Gbig_real and Gbig_imag
    Gbig_real = np.zeros((dim_outer_space, r))
    Gbig_imag = np.zeros((dim_outer_space, r))

    # Fill Gbig_real and Gbig_imag using broadcasting and reshaping
    for nu_i in range(r):
        Phi_exp = Phi[:, nu_i][:, np.newaxis] * exp_terms[nu_i]
        Gbig_real[:, nu_i] = np.real(Phi_exp).flatten()
        Gbig_imag[:, nu_i] = np.imag(Phi_exp).flatten()

    # Assemble Gbig
    Gbig = np.zeros((2 * dim_outer_space, 2 * r))
    Gbig[0:dim_outer_space, 0:r] = Gbig_real
    Gbig[dim_outer_space:, r:2*r] = Gbig_real
    Gbig[0:dim_outer_space, r:2*r] = -Gbig_imag
    Gbig[dim_outer_space:, 0:r] = Gbig_imag

    # Solve the least squares problem
    sol = lsq_linear(Gbig, x_big)

    # Construct the result
    b_new = sol.x[:r] + 1.0j * sol.x[r:2*r]

    return b_new
