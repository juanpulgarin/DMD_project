import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp



# ------------ Simulation ------------

def eps_vb(J, delta, k): return 2 * J * np.cos(k) - 2 * J - delta / 2
def eps_cb(J, delta, k): return -2 * J * np.cos(k) + 2 * J + delta / 2
def delta_epsilon(k, J, delta): return eps_cb(J, delta, k) - eps_vb(J, delta, k)
def D(k): return (k -3.14)                                                 ##### CENTERING THE K for maximal effect at k=0

def E(t, sigma, E0, t0=0.5):    #0.5                                                 ##### DECIDE THE STARTING POINT OF THE PULSE HERE
    return E0 / (np.sqrt(2 * np.pi * sigma ** 2)) * np.exp((-(t - t0) ** 2) / (2 * sigma ** 2))

def V(t, k, sigma, E0): return -E(t, sigma, E0) * D(k)

def A(t, k, J, delta, sigma, E0):
    v = V(t, k, sigma, E0)
    dE = delta_epsilon(k, J, delta)
    return np.array([
        [0,     0,     -1j * v,   1j * v],
        [0,     0,      1j * v,  -1j * v],
        [-1j * v, 1j * v, -1j * dE,     0],
        [1j * v, -1j * v,     0,  -1j * dE]
    ], dtype=complex)

def rhs(t, y, k, J, delta, sigma, E0):
    return A(t, k, J, delta, sigma, E0) @ y




#-------------------  Display -------------

# Function 1: Band structure plot
def plot_band_structure(J_list, delta_list):
    k_vals = np.linspace(-np.pi, np.pi, 200)

    plt.figure()
    for J in J_list:
        for delta in delta_list:
            vb = eps_vb(J, delta, k_vals)
            cb = eps_cb(J, delta, k_vals)
            plt.plot(k_vals, vb, label=f"VB: J={J}, Δ={delta}")
            plt.plot(k_vals, cb, label=f"CB: J={J}, Δ={delta}")
    
    plt.title("Band Structure")
    plt.xlabel("k")
    plt.ylabel("Energy")
    plt.grid(True)
    plt.legend()
    plt.show()


# Function 2: Electric field pulse plot
# Tune sigma to tune the width of the pulse
def plot_electric_pulse(sigma, E0):
    t = np.linspace(0, 20, 1000)
    efield = E(t, sigma, E0)

    plt.figure()
    plt.plot(t, efield, label="Electric Field Pulse E(t)")
    plt.title("Electric Field Pulse")
    plt.xlabel("Time")
    plt.ylabel("E(t)")
    plt.grid(True)
    plt.legend()
    plt.show()

# Function 3: Density matrix evolution
def plot_density_matrix_evolution(J, k_list, delta, sigma, E0):
    t_span = (0, 20)
    t_eval = np.linspace(*t_span, 1000)

    plt.figure()
    for k in k_list:
        y0 = np.array([1.0, 0.0, 0.0, 0.0], dtype=complex)
        sol = solve_ivp(lambda t, y: rhs(t, y, k, J, delta, sigma, E0), t_span, y0, t_eval=t_eval)
        plt.plot(sol.t, sol.y[0].real, label=fr'Re($\rho_{{VB}}$), k={k:.2f}')
        plt.plot(sol.t, sol.y[1].real, label=fr'Re($\rho_{{CB}}$), k={k:.2f}')

    plt.title(f"Density Matrix Evolution (J={J}, Δ={delta})")
    plt.xlabel("Time")
    plt.ylabel("Population")
    plt.grid(True)
    #plt.legend() uncomment if you want to see the legend
    plt.show()

#-------------  calling the display functions ---------
plot_band_structure(J_list=[2.45, 3.0], delta_list=[4.26, 5.0])
plot_electric_pulse(sigma=0.12, E0=0.2)
plot_density_matrix_evolution(J=2.45, k_list=[-0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75], delta=6.26, sigma=0.12, E0=0.2)




#------------- Creating the .txt file to store the data from the simulation 

# ------------ Parameters -----------
J = 2.45
delta = 4.26
sigma = 0.12
E0 = 0.2
a = 1

# ----------- Simulation for single k -----------
def simulate_density_single_k(J, k, delta, sigma, E0):
    y0 = np.array([1.0, 0.0, 0.0, 0.0], dtype=complex)
    t_span = (0, 20)
    t_eval = np.linspace(*t_span, 1000)

    sol = solve_ivp(lambda t, y: rhs(t, y, k, J, delta, sigma, E0),
                    t_span, y0, t_eval=t_eval, method="RK45")

    return np.column_stack((np.full_like(sol.t, k), sol.t, sol.y[0].real, sol.y[1].real))

# ----------- Main export function -----------
def generate_density_vs_k_file(filename="density_vs_k.txt"):
    k_range = np.linspace(-np.pi / a, np.pi / a, 64)
    all_data = [simulate_density_single_k(J, k, delta, sigma, E0) for k in k_range]
    all_data = np.vstack(all_data)
    np.savetxt(filename, all_data, header="k time Re(rho_VB) Re(rho_CB)")

# Call it
generate_density_vs_k_file("density_vs_k.txt")