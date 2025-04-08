import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import Hamiltonian as Hamilton
import constants as cst

def plot_band_structure(k_vals, J_list, delta_list,vb,cb, limits,fout=''):
    fig, axs=plt.subplots(1, figsize=(7,5))

    param_pairs = [(J, delta) for J in J_list for delta in delta_list]
    colors = cm.get_cmap('tab10', len(param_pairs))

    for idx, (J, delta) in enumerate(param_pairs):
        color = colors(idx)
        axs.plot(k_vals, vb[J,delta]*cst.Ry, label=f"J={J*cst.Ry:.2f}, Δ={delta*cst.Ry:.2f}",color=color)
        axs.plot(k_vals, cb[J,delta]*cst.Ry, color=color)

    axs.legend(loc='best', fontsize=15,fancybox=True,framealpha=1)

    axs.set_xlim(limits[0,0],limits[0,1])
    axs.set_ylim(limits[1,0],limits[1,1])

    #axs.set_title("Band Structure",fontsize=15)
    axs.set_xlabel("$k$ (units of $1/a$)",fontsize=15)
    axs.set_ylabel("Energy (eV)",fontsize=15)

    axs.tick_params(axis="x", labelsize=15)
    axs.tick_params(axis="y", labelsize=15)

    axs.spines['top'].set_visible(False)
    axs.spines['right'].set_visible(False)

    if len(fout) > 0:
        plt.savefig(fout+'.png', bbox_inches='tight', transparent=True)
        plt.savefig(fout+'.pdf', bbox_inches='tight', transparent=True)
    else:
        plt.show()
    return 0

def plot_electric_pulse(t, E_t, limits,fout=''):

    fig, axs=plt.subplots(1, figsize=(7,5))

    axs.plot(t, E_t, label=f"Electric Field $E(t)$",color='black')

    axs.legend(loc='best', fontsize=15,fancybox=True,framealpha=1)

    axs.set_xlim(limits[0,0],limits[0,1])
    axs.set_ylim(limits[1,0],limits[1,1])

    #axs.set_title("Electric Field Pulse")
    axs.set_xlabel("Time (fs)",fontsize=15)
    axs.set_ylabel("Amplitude (au)",fontsize=15)

    axs.tick_params(axis="x", labelsize=15)
    axs.tick_params(axis="y", labelsize=15)

    axs.spines['top'].set_visible(False)
    axs.spines['right'].set_visible(False)

    if len(fout) > 0:
        plt.savefig(fout+'.png', bbox_inches='tight', transparent=True)
        plt.savefig(fout+'.pdf', bbox_inches='tight', transparent=True)
    else:
        plt.show()
    return 0

def plot_population_field(t_eval,electric_field,k_list,solutions,limits,fout=''):
    fig, axs=plt.subplots( 2,1, figsize=(7,6),sharex=True)

    axs[0].plot(t_eval,electric_field,color='black')

    axs[0].set_ylabel("Electric Field (au)",fontsize=15)
    axs[0].tick_params(axis="y", labelsize=15)

    axs[0].spines['top'].set_visible(False)
    axs[0].spines['right'].set_visible(False)


    ax1_secondary = axs[1].twinx()  # Create a twin y-axis sharing the same x-axis

    for k_i,k in enumerate(k_list):
        ax1_secondary.plot(t_eval, solutions[k_i,0,:], label=f'$k=${k:.3f}',color='red', alpha=1./(k_i+1.))

        axs[1].plot(t_eval, solutions[k_i,0,:],color='red',  alpha=1./(k_i+1.))
        axs[1].plot(t_eval, solutions[k_i,1,:],color='blue', alpha=1./(k_i+1.))

    ax1_secondary.set_ylabel(f'$\\rho_{{V}} (t)$', fontsize=15, color='black')  # Change the label for the second y-axis
    ax1_secondary.tick_params(axis="y", labelsize=15, labelcolor='red')  # Set y-axis tick color to red



    axs[1].set_xlim(limits[0,0],limits[0,1])
    axs[1].set_ylim(limits[1,0],limits[1,1])
    ax1_secondary.set_ylim(1.0-limits[1,1],1.0-limits[1,0])

    #plt.xlim(0,1500)
    axs[1].set_xlabel("Time (fs)",fontsize=15)
    axs[1].set_ylabel(f'$\\rho_{{C}} (t)$',fontsize=15)
    #axs[1].legend(fancybox=True,framealpha=1.0, fontsize=12)

    axs[1].tick_params(axis="x", labelsize=15)
    axs[1].tick_params(axis="y", labelsize=15,labelcolor='blue')

    axs[1].spines['top'].set_visible(False)
    axs[1].spines['right'].set_visible(False)
    ax1_secondary.spines['top'].set_visible(False)
    ax1_secondary.spines['right'].set_visible(False)
    ax1_secondary.spines['right'].set_linewidth(1.5)

    plt.subplots_adjust(hspace=0.01)

    if len(fout) > 0:
        plt.savefig(fout+'.png', bbox_inches='tight', transparent=True)
        plt.savefig(fout+'.pdf', bbox_inches='tight', transparent=True)
    else:
        plt.show()
    return 0


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
