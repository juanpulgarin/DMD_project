import numpy as np
from scipy.integrate import solve_ivp
import scipy.linalg as LA

#========================================================================================
#============================      System      ==========================================
def eps_vb(k, J, delta):
    return  2. * J * np.cos(k) - 2. * J - delta / 2.

def eps_cb(k, J, delta):
    return -2. * J * np.cos(k) + 2. * J + delta / 2.

def delta_epsilon(k, J, delta):
    return eps_cb(k,J, delta) - eps_vb(k,J, delta)

#========================================================================================
#============================   Light-Matter   ==========================================
def D(k):
    return k

def E(t, sigma, E0, ω=0,case=1, t0=None):
    if t0 == None:
        t0=4.5*sigma

    if case==1:
        return E0 / (np.sqrt(2 * np.pi * sigma ** 2)) * np.exp((-(t - t0) ** 2) / (2 * sigma ** 2))
    if case==2:
        return E0 / (np.sqrt(2 * np.pi * sigma ** 2)) * np.exp((-(t - t0) ** 2) / (2 * sigma ** 2)) * np.cos(ω * (t - t0))

def V(t, k, sigma, E0, ω=0,case=1):
    return -E(t, sigma, E0, ω, case) #* D(k)

#========================================================================================
#============================  Density Operator =========================================

def A(t, k, J, delta, sigma, E0, ω, case):
    v = V(t, k, sigma, E0, ω, case)
    dE = delta_epsilon(k, J, delta)
    return np.array([
        [0,     0,     -1j * v,   1j * v],
        [0,     0,      1j * v,  -1j * v],
        [-1j * v, 1j * v, -1j * dE,     0],
        [1j * v, -1j * v,     0,  -1j * dE]
    ], dtype=np.complex64)

def rhs(t, y, k, J, delta, sigma, E0, ω=0,case=1):
    return A(t, k, J, delta, sigma, E0, ω,case) @ y

#========================================================================================
#====================================  DMD ==============================================

def perform_dmd(X, Xp, t, r):
    U, S, Vh = np.linalg.svd(X, full_matrices=False)
    U_r, S_r, Vh_r = U[:, :r], np.diag(S[:r]), Vh[:r, :]
    A_tilde = U_r.T @ Xp @ Vh_r.T @ np.linalg.inv(S_r)
    eigvals, W = np.linalg.eig(A_tilde)


    Phi = Xp @ Vh_r.T @ np.linalg.inv(S_r) @ W


    dt = t[1] - t[0]
    omega = np.log(eigvals) / dt
    b = np.linalg.pinv(Phi) @ X[:, 0]
    time_dynamics = np.array([b * np.exp(omega * T) for T in t]).T
    X_dmd = Phi @ time_dynamics
    #return eigvals, X_dmd.real
    return eigvals, Phi, r, S



def DMD(X, Y, power, orden_truncado, truncate=True, correction=True):
    U2,Sig2,Vh2 = LA.svd(X, full_matrices=False) # SVD of input matrix

    if truncate==True:
        r = np.where(np.log10(Sig2/Sig2[0])<=orden_truncado)[0][0]
    
    if power==False:
        r=orden_truncado

    U = U2[:,:r]
    Sig = np.diag(Sig2)[:r,:r]
    V = Vh2.conj().T[:,:r]

    Atil = np.dot(np.dot(np.dot(U.conj().T, Y), V), LA.inv(Sig)) # build A tilde
    mu,W = LA.eig(Atil)

    Phi = np.dot(np.dot(np.dot(Y, V), LA.inv(Sig)), W) # build DMD modes
    
    return mu, Phi, r, Sig2




# DMD Reconstruction Function
def DMDRebuild(X, Y, t, orden_truncado, truncate=True, correction=True):
    U2, Sig2, Vh2 = LA.svd(X, full_matrices=False)  # SVD of input matrix

    if truncate:
        r = np.where(np.log10(Sig2 / Sig2[0]) <= orden_truncado)[0][0]
    else:
        r = len(Sig2)

    U = U2[:, :r]
    Sig = np.diag(Sig2[:r])
    V = Vh2.conj().T[:, :r]

    Atil = U.conj().T @ Y @ V @ LA.inv(Sig)
    mu, W = LA.eig(Atil)
    Phi = Y @ V @ LA.inv(Sig) @ W

    dt = t[1] - t[0]
    omega = np.log(mu) / dt
    b = np.linalg.pinv(Phi) @ X[:, 0]

    time_dynamics = np.array([b * np.exp(omega * T) for T in t]).T  # shape (r, len(t)-1)
    X_dmd = Phi @ time_dynamics  # shape (n, len(t)-1)

    return mu, X_dmd



