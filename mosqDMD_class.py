import numpy as np
import scipy.linalg as LA
from scipy.linalg import svd, eig, pinv
from scipy.optimize import lsq_linear
import matplotlib.pyplot as plt
from typing import NamedTuple

# Define the NamedTuple for TLSQ return type
class TLSQ(NamedTuple):
    X_denoised: np.ndarray
    Y_denoised: np.ndarray

# Function to compute Total Least Squares (TLSQ) denoising
def compute_tlsq(
    X: np.ndarray, Y: np.ndarray, tlsq_rank: int
) -> NamedTuple(
    "TLSQ", [("X_denoised", np.ndarray), ("Y_denoised", np.ndarray)]
):
    """
    Compute Total Least Square.

    :param X: the first matrix;
    :type X: np.ndarray
    :param Y: the second matrix;
    :type Y: np.ndarray
    :param tlsq_rank: the rank for the truncation; If 0, the method
        does not compute any noise reduction; if positive number, the
        method uses the argument for the SVD truncation used in the TLSQ
        method.
    :type tlsq_rank: int
    :return: the denoised matrix X, the denoised matrix Y
    :rtype: NamedTuple("TLSQ", [('X_denoised', np.ndarray),
                                 ('Y_denoised', np.ndarray)])

    References:
    https://arxiv.org/pdf/1703.11004.pdf
    https://arxiv.org/pdf/1502.03854.pdf
    """
    # Do not perform tlsq
    if tlsq_rank == 0:
        return TLSQ(X, Y) # Return as NamedTuple for consistency

    # Concatenate X and Y vertically for SVD
    combined_xy = np.append(X, Y, axis=0)

    # Perform SVD on the combined matrix
    # We only need Vh (the last element of the tuple returned by svd)
    V = np.linalg.svd(combined_xy, full_matrices=False)[-1]

    # Determine the truncation rank for TLSQ
    # It's the minimum of the requested tlsq_rank and the available dimensions in V
    rank = min(tlsq_rank, V.shape[0])

    # Construct the projection matrix VV from the truncated singular vectors
    # V[:rank, :] selects the top 'rank' rows of V
    # .conj().T takes the conjugate transpose
    # .dot performs matrix multiplication
    VV = V[:rank, :].conj().T.dot(V[:rank, :])

    # Project the original X and Y onto the denoised subspace
    return TLSQ(X.dot(VV), Y.dot(VV))


class DMD:
    def __init__(self, rank=None, delay_embedding_dim=0, tikhonov_regularization=None, amplitudes_method='pinv', frequencies=None, tlsq_rank=0):
        self.rank = rank
        self.delay_embedding_dim = delay_embedding_dim
        self.tikhonov_regularization = tikhonov_regularization
        self.amplitudes_method = amplitudes_method
        self.frequencies = frequencies # User-defined frequency range for reconstruction
        self.tlsq_rank = tlsq_rank # New parameter for TLSQ rank

        # Attributes to store results
        self.S = None
        self.eigvals = None
        self.Phi_full = None
        self.Phi_physical = None
        self.amplitudes_custom = None
        self.b_custom = None
        self.dynamics_custom = None
        self.reconstructed_X = None
        self.reconstructed_omega = None
        self.full_data_matrix = None
        self.time_array = None
        self.reconstruction_time_array = None

    def _custom_hankel_embedding(self, X):
        n_spatial, n_time = X.shape
        delay = self.delay_embedding_dim

        if n_time < delay:
             raise ValueError(f"Number of time steps ({n_time}) is less than the delay embedding dimension ({delay}). Cannot perform Hankel embedding.")

        hankel_matrix_rows = []
        for i in range(delay):
            hankel_matrix_rows.append(X[:, i : n_time - delay + i + 1])
        hankel_matrix = np.vstack(hankel_matrix_rows)
        return hankel_matrix

    def _perform_dmd_core(self, H, delay_embedding_dim_for_modes):
        X1 = H[:, :-1]
        X2 = H[:, 1:]

        # Apply TLSQ denoising if tlsq_rank is specified
        if self.tlsq_rank > 0:
            print(f"Applying TLSQ denoising with rank={self.tlsq_rank}...")
            denoised_matrices = compute_tlsq(X1, X2, self.tlsq_rank)
            X1_processed = denoised_matrices.X_denoised
            X2_processed = denoised_matrices.Y_denoised
        else:
            X1_processed = X1
            X2_processed = X2

        U, S, Vh = np.linalg.svd(X1_processed, full_matrices=False) # Use processed X1
        V = Vh.conj().T

        rank = self.rank
        if rank is None:
            r_threshold = 0.01
            rank = np.sum(S > S[0] * r_threshold)
            if rank == 0:
                rank = 1
            print(f"Automatically determined rank for DMD: {rank}")
            self.rank = rank

        U_r  = U[:, :rank]
        S_r  = S[:rank]
        V_r = V[:, :rank]

        if self.tikhonov_regularization is not None:
            norm_X = np.linalg.norm(X1_processed, 'fro') # Use processed X1

            # Create a copy to avoid modifying the original S_r
            S_r_reg = S_r.copy()

            # Find the singular values that are below the threshold
            # Using a logical mask for a vectorized, efficient operation
            small_singular_values_mask = S_r / S_r[0] <= 1e-7

            # Apply the regularization formula only to the singular values
            # where the mask is True.
            S_r_reg[small_singular_values_mask] = (
                S_r[small_singular_values_mask]**2 + self.tikhonov_regularization * norm_X
            ) / S_r[small_singular_values_mask]

            Sigma_inv = np.diag(1.0 / (S_r_reg + 1e-10))
        else:
            Sigma_inv = np.diag(1.0 / (S_r + 1e-10))

        A_tilde = U_r.conj().T @ X2_processed @ V_r @ Sigma_inv # Use processed X2

        eigvals, W = LA.eig(A_tilde)

        Phi_full = U_r.dot(W)

        effective_delay_for_physical_modes = delay_embedding_dim_for_modes if delay_embedding_dim_for_modes > 0 else 1

        num_spatial_points_original_flat = Phi_full.shape[0] // effective_delay_for_physical_modes

        Phi_physical = np.average(
            Phi_full.reshape(effective_delay_for_physical_modes, num_spatial_points_original_flat, Phi_full.shape[1]),
            axis=0
        )

        return S, eigvals, Phi_full, Phi_physical

    def _least_outer_amplitudes(self, D, Phi, mu, tspan, dt, r):
        n_rows, len_tspan = np.shape(D)
        dim_outer_space = len_tspan * n_rows

        xbig_real = np.real(D).flatten()
        xbig_imag = np.imag(D).flatten()

        x_big = np.concatenate((xbig_real, xbig_imag))

        t_diff = tspan - tspan[0]
        exp_terms = np.exp(np.log(mu[:, np.newaxis]) * (t_diff[np.newaxis, :] / dt))

        Gbig_real = np.zeros((dim_outer_space, r))
        Gbig_imag = np.zeros((dim_outer_space, r))

        for nu_i in range(r):
            Phi_exp = Phi[:, nu_i][:, np.newaxis] * exp_terms[nu_i]
            Gbig_real[:, nu_i] = np.real(Phi_exp).flatten()
            Gbig_imag[:, nu_i] = np.imag(Phi_exp).flatten()

        Gbig = np.zeros((2 * dim_outer_space, 2 * r), dtype=float)
        Gbig[0:dim_outer_space, 0:r] = Gbig_real
        Gbig[dim_outer_space:, r:2*r] = Gbig_real
        Gbig[0:dim_outer_space, r:2*r] = -Gbig_imag
        Gbig[dim_outer_space:, 0:r] = Gbig_imag

        sol = lsq_linear(Gbig, x_big)

        b_new = sol.x[:r] + 1.0j * sol.x[r:2*r]

        return b_new

    def _delta_frequency(self, target_frequencies, dmd_frequencies, σ):
        return 1./(np.sqrt(2.*np.pi)*σ)*np.exp(-(target_frequencies[None,:] - dmd_frequencies[:,None])**2/(2*σ**2))

    def _dynamics_core(self, H, Eigvals, Modes, nt_original, d, original_time_tspan, num_future_steps):
        x1 = H[:, 0]

        effective_d_for_physical_modes = d if d > 0 else 1
        num_spatial_points_original_flat = Modes.shape[0] // effective_d_for_physical_modes

        physical_modes = Modes[0:num_spatial_points_original_flat, :]

        total_reconstruction_steps = nt_original + num_future_steps

        dt_original = original_time_tspan[1] - original_time_tspan[0] if nt_original > 1 else 1.0
        self.reconstruction_time_array = np.linspace(
            original_time_tspan[0],
            original_time_tspan[0] + (total_reconstruction_steps - 1) * dt_original,
            total_reconstruction_steps
        )

        time_steps_indices = np.arange(total_reconstruction_steps)

        if total_reconstruction_steps > 1:
            tpow = (time_steps_indices - time_steps_indices[0]) / (time_steps_indices[1] - time_steps_indices[0])
        else:
            tpow = np.array([0])

        if self.amplitudes_method == 'least_outer':
            dt_val = original_time_tspan[1] - original_time_tspan[0] if len(original_time_tspan) > 1 else 1.0
            amplitudes = self._least_outer_amplitudes(
                D=H,
                Phi=Modes,
                mu=Eigvals,
                tspan=original_time_tspan[:H.shape[1]],
                dt=dt_val,
                r=len(Eigvals)
            )
        else:
            amplitudes = LA.pinv(Modes) @ x1

        dynamics = np.power(Eigvals[:, None], tpow[None, :]) * amplitudes[:, None]

        reconstructed_data = np.dot(physical_modes, dynamics)

        #Cholesky decomposition of modes and amplitudes



        data_frequency = None
        if self.frequencies is not None:
            dt_val = original_time_tspan[1] - original_time_tspan[0] if len(original_time_tspan) > 1 else 1.0
            omega_dmd_continuous_freqs = np.log(Eigvals) / dt_val

            #plt.figure()
            #plt.plot(np.real(omega_dmd_continuous_freqs),np.imag(omega_dmd_continuous_freqs),'.')
            #plt.title('DMD Continuous-Time Frequencies (Real vs Imaginary)')
            #plt.xlabel('Growth/Decay Rate (Real Part)')
            #plt.ylabel('Angular Frequency (Imaginary Part)')
            #plt.grid(True)
            #plt.show()

            # Calculate sigma based on frequency resolution
            if len(self.frequencies) > 1:
                freq_resolution = (self.frequencies[-1] - self.frequencies[0]) / (len(self.frequencies) - 1)
            else:
                freq_resolution = 1.0 # Default if only one frequency point

            sigma_for_delta_freq = freq_resolution # Set sigma to the frequency resolution

            delta_omega = self._delta_frequency(self.frequencies, omega_dmd_continuous_freqs.imag, σ=sigma_for_delta_freq)

            data_frequency = np.sqrt(2.*np.pi) * np.einsum('kl,l,lw->kw', physical_modes, amplitudes, delta_omega)

            print(f"Omega_DMD_Continuous_Freqs: {omega_dmd_continuous_freqs}")
            print(f"Max Delta_Omega: {np.max(delta_omega)}")
            print(f"dt_val: {dt_val}")
            print(f"Calculated Sigma for Delta_Frequency: {sigma_for_delta_freq}") # Print the calculated sigma
            print(f"Max Data_Frequency: {np.max(data_frequency)}")

        return dynamics, amplitudes, reconstructed_data, data_frequency

    def Cholesky_decomposition_modes_amplitudes(self):
        if self.Phi_physical is None or self.amplitudes_custom is None:
            raise ValueError("DMD must be fitted before performing Cholesky decomposition.")
        
        # Cholesky decomposition of modes
        for k in range(self.Phi_physical.shape[0]):
            mode_covariance = np.outer(self.Phi_physical[k, :], self.amplitudes_custom)
            L, D, _ = LA.cholesky(mode_covariance)
            


    def fit(self, full_data_matrix, time_array, num_future_steps=0):
        self.full_data_matrix = full_data_matrix
        self.time_array = time_array
        num_original_time_steps = full_data_matrix.shape[1]

        actual_delay_for_calculations = self.delay_embedding_dim if self.delay_embedding_dim > 0 else 1

        if self.delay_embedding_dim > 0:
            H = self._custom_hankel_embedding(full_data_matrix)
        else:
            H = full_data_matrix

        self.S, self.eigvals, self.Phi_full, self.Phi_physical = self._perform_dmd_core(
            H=H,
            delay_embedding_dim_for_modes=actual_delay_for_calculations
        )

        self.dynamics_custom, self.amplitudes_custom, self.reconstructed_X, self.reconstructed_omega = self._dynamics_core(
            H=H,
            Eigvals=self.eigvals,
            Modes=self.Phi_full,
            nt_original=num_original_time_steps,
            d=actual_delay_for_calculations,
            original_time_tspan=self.time_array,
            num_future_steps=num_future_steps
        )
        self.b_custom = self.amplitudes_custom
