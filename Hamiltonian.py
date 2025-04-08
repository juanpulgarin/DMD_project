import numpy as np
from scipy.integrate import solve_ivp

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

