import numpy as np
import scipy.linalg as LA
from scipy.linalg import svd, eig, pinv
from scipy.optimize import lsq_linear # New import for lsq_linear
import matplotlib.pyplot as plt

# 1. User's provided function for Hankel embedding
def custom_hankel_embedding(X, delay):
    """
    Performs Hankel embedding on the input data X.

    Parameters:
    X (np.ndarray): The input data matrix (spatial_points, time_steps).
                    Each column is a snapshot.
    delay (int): The embedding dimension (number of time delays).

    Returns:
    np.ndarray: The Hankel embedded matrix.
                Shape: (spatial_points * delay, time_steps - delay + 1).
    """
    if delay == 0: # Handle the case where no embedding is requested
        return X

    n_spatial, n_time = X.shape

    if n_time < delay:
        raise ValueError(f"Number of time steps ({n_time}) is less than the delay embedding dimension ({delay}). Cannot perform Hankel embedding.")

    hankel_matrix_rows = []
    # Iterate 'delay' times to create the stacked Hankel blocks
    for i in range(delay):
        hankel_matrix_rows.append(X[:, i : n_time - delay + i + 1])

    hankel_matrix = np.vstack(hankel_matrix_rows)
    return hankel_matrix

# 2. User's provided function for performing DMD
def perform_dmd(H, rank=None, delay_embedding_dim=1, tikhonov_regularization=None):
    """
    Performs Dynamic Mode Decomposition (DMD) on a Hankel embedded matrix.

    Parameters:
    H (np.ndarray): The Hankel embedded data matrix.
    rank (int, optional): The truncation rank for SVD. If None, rank is determined automatically.
    delay_embedding_dim (int): The embedding dimension used in Hankel embedding.
                               Used to reshape modes for physical interpretation.
                               Note: This parameter's default value will be overridden if
                               the calling context (e.g., main script or class) passes
                               a different value. It's used here for reshaping Phi_physical.
    tikhonov_regularization (float, optional): Regularization parameter for the
                                                pseudoinverse (modifies singular values).

    Returns:
    tuple: (S, eigvals, Phi_full, Phi_physical)
        S (np.ndarray): Singular values from SVD of X1.
        eigvals (np.ndarray): DMD eigenvalues (discrete-time).
        Phi_full (np.ndarray): Full DMD modes in the embedded (Hankel) space.
        Phi_physical (np.ndarray): Averaged/physical DMD modes (reshaped to original spatial dim).
    """

    X1 = H[:, :-1]
    X2 = H[:, 1:]

    # Singular Value Decomposition (SVD) of X1
    # U: Left singular vectors (spatial basis for X1)
    # S: Singular values
    # Vh: Conjugate transpose of right singular vectors (temporal basis for X1)
    U, S, Vh = np.linalg.svd(X1, full_matrices=False)

    V = Vh.conj().T

    # Determine truncation rank if not explicitly provided
    if rank is None:
        # A simple heuristic: select rank that captures a significant portion of energy
        r_threshold = 0.01
        rank = np.sum(S > S[0] * r_threshold)
        if rank == 0: # Ensure at least one mode is selected
            rank = 1
        print(f"Automatically determined rank for DMD: {rank}")

    U_r  = U[:, :rank]
    S_r  = S[:rank]
    V_r = V[:, :rank]

    # Apply Tikhonov regularization if a parameter is provided
    if tikhonov_regularization is not None:
        norm_X = np.linalg.norm(X1, 'fro') # Use Frobenius norm for X1
        # Modify singular values for regularization: S_r_reg = (S_r**2 + lambda * ||X||_F) / S_r
        S_r_reg = (S_r**2 + tikhonov_regularization * norm_X) / S_r
        # Create inverse of regularized singular values, adding a small epsilon for stability
        Sigma_inv = np.diag(1.0 / (S_r_reg + 1e-10))
    else:
        Sigma_inv = np.diag(1.0 / (S_r + 1e-10))

    # Compute the Koopman operator (or its finite-dimensional approximation, A_tilde)
    # A_tilde = U_r^* X2 V_r Sigma_inv
    A_tilde = U_r.conj().T @ X2 @ V_r @ Sigma_inv

    eigvals, W = LA.eig(A_tilde)

    # Compute the full DMD modes (Phi_full) in the embedded (Hankel) state space
    #Phi_full = X2 @ V_r @ Sigma_inv @ W
    Phi_full = U_r.dot(W)

    # The actual delay used for embedding might be 1 even if delay_embedding_dim was 0,
    # as H would be the original matrix. For reshaping Phi_physical, we need to know
    # how many "blocks" there are in Phi_full that correspond to the *original* spatial dimension.
    # If delay_embedding_dim was 0, it means no actual embedding, so it's a single "block".
    effective_delay_for_physical_modes = delay_embedding_dim if delay_embedding_dim > 0 else 1

    num_spatial_points_original = Phi_full.shape[0] // effective_delay_for_physical_modes
    Phi_physical = np.average(
        Phi_full.reshape(effective_delay_for_physical_modes, num_spatial_points_original, Phi_full.shape[1]),
        axis=0
    )

    return S, eigvals, Phi_full, Phi_physical

# New function for least squares amplitude calculation (LEAST_OUTER)
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
    # CORRECTED LINE: Scale t_diff by dt to get the number of discrete steps
    exp_terms = np.exp(np.log(mu[:, np.newaxis]) * (t_diff[np.newaxis, :] / dt))

    # Initialize Gbig_real and Gbig_imag
    Gbig_real = np.zeros((dim_outer_space, r))
    Gbig_imag = np.zeros((dim_outer_space, r))

    # Fill Gbig_real and Gbig_imag using broadcasting and reshaping
    for nu_i in range(r):
        # Phi is Modes (i.e., Phi_full in the main script)
        # It has shape (embedded_spatial_dim, rank)
        Phi_exp = Phi[:, nu_i][:, np.newaxis] * exp_terms[nu_i]
        Gbig_real[:, nu_i] = np.real(Phi_exp).flatten()
        Gbig_imag[:, nu_i] = np.imag(Phi_exp).flatten()

    # Assemble Gbig
    Gbig = np.zeros((2 * dim_outer_space, 2 * r), dtype=float)
    Gbig[0:dim_outer_space, 0:r] = Gbig_real
    Gbig[dim_outer_space:, r:2*r] = Gbig_real
    Gbig[0:dim_outer_space, r:2*r] = -Gbig_imag
    Gbig[dim_outer_space:, 0:r] = Gbig_imag

    # Solve the least squares problem
    sol = lsq_linear(Gbig, x_big)

    # Construct the result
    b_new = sol.x[:r] + 1.0j * sol.x[r:2*r]

    return b_new


# 3. User's provided function for reconstructing dynamics
def Dynamics(H, Eigvals, Modes, nt, d, original_time_tspan, amplitudes_method='pinv'):
    """
    Reconstructs the dynamics using DMD modes and eigenvalues, along with initial amplitudes.
    This function specifically replicates the logic and variable usage from the user's provided
    Dynamics function, with a correction for the 'dynamics' calculation.

    Parameters:
    H (np.ndarray): The full Hankel embedded data matrix. Its first column (H[:,0])
                    is used as the initial state.
    Eigvals (np.ndarray): DMD eigenvalues (discrete-time).
    Modes (np.ndarray): Full DMD modes in the embedded space (Phi_full from perform_dmd).
    nt (int): The number of time steps (length of the original time array).
              Used to determine the length of the reconstruction.
    d (int): The embedding dimension used in Hankel embedding. Used for slicing 'physical_modes'.
    original_time_tspan (np.ndarray): The original time array for the full data.
    amplitudes_method (str): Method to calculate amplitudes. 'pinv' (default) or 'least_outer'.

    Returns:
    tuple: (dynamics, amplitudes, reconstructed_data)
        dynamics (np.ndarray): The time dynamics of each DMD mode.
        amplitudes (np.ndarray): The initial amplitudes of each DMD mode.
        reconstructed_data (np.ndarray): The reconstructed data in the original spatial dimension.
    """
    x1 = H[:, 0] # Initial state of the embedded system (first column of Hankel matrix)

    # Replicate user's 'physical_modes' calculation (first block of Modes)
    # Note: 'd' here is the delay_embedding_dim. If d was 0, it means no embedding,
    # so the effective d for slicing would be 1 (original spatial dimension)
    effective_d_for_physical_modes = d if d > 0 else 1
    num_spatial_points_original = Modes.shape[0] // effective_d_for_physical_modes
    physical_modes = Modes[0:num_spatial_points_original, :]

    # Replicate user's 'time_steps' and 'tpow' calculation
    time_steps = np.linspace(0, nt - 1, nt, dtype=int)
    if nt > 1:
        tpow = (time_steps - 0) / (time_steps[1] - time_steps[0])
    else:
        tpow = np.array([0])

    # Calculate amplitudes based on chosen method
    if amplitudes_method == 'least_outer':
        # Calculate dt from the original time array
        dt_val = original_time_tspan[1] - original_time_tspan[0] if len(original_time_tspan) > 1 else 1.0
        # Pass the full embedded matrix H as D, and Phi_full (Modes) as Phi
        amplitudes = LEAST_OUTER(
            D=H, # Pass the full embedded matrix H as D for LEAST_OUTER
            Phi=Modes, # Pass the full embedded modes (Modes)
            mu=Eigvals,
            tspan=original_time_tspan[:H.shape[1]], # CORRECTED LINE: Slice tspan to match H's time dimension
            dt=dt_val,
            r=len(Eigvals) # Rank 'r' is the number of eigenvalues
        )
    else: # Default or 'pinv' method
        amplitudes = LA.pinv(Modes) @ x1

    dynamics = np.power(Eigvals[:, None], tpow[None, :]) * amplitudes[:, None]

    reconstructed_data = np.dot(physical_modes, dynamics)

    return dynamics, amplitudes, reconstructed_data
