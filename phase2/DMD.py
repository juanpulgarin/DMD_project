#import simulate
import numpy as np
import matplotlib.pyplot as plt
#from scipy.integrate import solve_ivp





#-------------  DMD ---------
def static_full_system_dmd_plot(r, k_index, filename="/Users/leomayques/Desktop/semester_VI/BT/scripts/phase2/density_vs_k.txt"):

# Loading data from the .txt file
    def load_full_k_dmd_data(filename):
        data = np.loadtxt(filename)
        k_vals = np.unique(data[:, 0]) 
        t_vals = np.unique(data[:, 1])
        num_k = len(k_vals)
        num_t = len(t_vals)

        rho_vb = data[:, 2].reshape(num_k, num_t)
        rho_cb = data[:, 3].reshape(num_k, num_t)
        X_full = np.vstack([rho_vb, rho_cb])  

        return X_full[:, :-1], X_full[:, 1:], t_vals[:-1], rho_vb, rho_cb, k_vals



#Note: for a detailed procedure on DMD algorithm using SVD, see: 
## "Using Dynamic Model decomposition to Predict The Dynamics of a two-time Non-equilibrium Green's function", Yin et al. 
    def perform_dmd(X, Xp, t, r):
# --------- Decomposition part of DMD:
        U, S, Vh = np.linalg.svd(X, full_matrices=False)
        U_r, S_r, Vh_r = U[:, :r], np.diag(S[:r]), Vh[:r, :]
        A_tilde = U_r.T @ Xp @ Vh_r.T @ np.linalg.inv(S_r)
        eigvals, W = np.linalg.eig(A_tilde)

# --------- Reconstruction part of DMD:
        Phi = Xp @ Vh_r.T @ np.linalg.inv(S_r) @ W
        dt = t[1] - t[0]
        omega = np.log(eigvals) / dt
        b = np.linalg.pinv(Phi) @ X[:, 0]
        time_dynamics = np.array([b * np.exp(omega * T) for T in t]).T
        X_dmd = Phi @ time_dynamics
        return eigvals, X_dmd.real

    # Load data and perform DMD
    X, Xp, t, rho_vb_orig, rho_cb_orig, k_vals = load_full_k_dmd_data(filename)
    eigvals, X_dmd = perform_dmd(X, Xp, t, r)
    num_k = len(k_vals)

    rho_vb_dmd = X_dmd[:num_k, :]
    rho_cb_dmd = X_dmd[num_k:, :]

    # Static plot for selected k_index
    plt.figure(figsize=(8, 4))
    plt.plot(t, rho_vb_orig[k_index, :-1], label="Original VB")
    plt.plot(t, rho_vb_dmd[k_index], '--', label="DMD VB")
    plt.plot(t, rho_cb_orig[k_index, :-1], label="Original CB")
    plt.plot(t, rho_cb_dmd[k_index], '--', label="DMD CB")

    plt.title(f"DMD Reconstruction at k={k_index*np.pi/32}, for r={r}")
    plt.xlabel("Time")
    plt.ylabel("Population")
    plt.grid(True)
    plt.legend()
    plt.show()


#----------------- Display ---------------

#DMD vs original simulation
static_full_system_dmd_plot(r=30, k_index=0)