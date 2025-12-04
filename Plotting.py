import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D

import Hamiltonian as Hamilton
import constants as cst

def plot_band_structure(k_vals, J_list, delta_list,vb,cb, limits,fout=''):
    fig, axs=plt.subplots(1, figsize=(6,4))

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
    fig, axs=plt.subplots(1, figsize=(6,4))

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

def plot_singular_values(sigma,rank,fout=''):
    fig, axs=plt.subplots(1, figsize=(6,4))

    axs.plot(sigma/sigma[0],'.',color='blue',label='Singular Values')

    axs.axhline(y=sigma[rank]/sigma[0],ls='--',color='gray',label='cutoff')


    axs.set_xlabel("index $i$",fontsize=15)
    axs.set_ylabel(f"$\\sigma_i/\\sigma_0$",fontsize=15)
    axs.set_yscale('log')

    axs.legend(loc='best', fontsize=15,fancybox=True,framealpha=1.0)

    axs.set_xlim(-0.1,len(sigma)+0.1)

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

def plot_eigenvalues(mu,fout=''):
    fig, axs=plt.subplots(1, figsize=(6,4))

    axs.plot([0.],[0.],'o',color='black')

    axs.plot(mu.real,mu.imag,'.',color='red')

    theta = np.linspace(0, 2 * np.pi, 300)
    x = np.cos(theta)
    y = np.sin(theta)
    axs.plot(x, y, 'k--')


    axs.set_xlim(-1.2,1.2)
    axs.set_ylim(-1.2,1.2)

    axs.set_xlabel(r"$\Re\{\lambda_{i}\}$",fontsize=15)
    axs.set_ylabel(r"$\Re\{\lambda_{i}\}$",fontsize=15)

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

def plot_amplitudes_b(amplitudes_costum,fout=''):
    fig, axs=plt.subplots(1, figsize=(6,4))


    for i in range(len(amplitudes_costum)):
        axs.plot(amplitudes_costum[i].real, amplitudes_costum[i].imag, 'o',label=f"$b_{i+1}$")

    axs.legend(loc='best', fontsize=15,fancybox=True,framealpha=1.0)

    axs.set_xlabel(r"$\Re \{ b_{i} \}$",fontsize=15)
    axs.set_ylabel(r"$\Im \{ b_{i} \}$",fontsize=15)

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

def plot_modes_vs_time(b_costum,index,limits,fout=''):
    fig, axs=plt.subplots(1, figsize=(6,4))

    axs.plot(b_costum[index,:].real,color='blue',label=f"$\\Re \\left( e^{{ \\omega_{index+1}t }} \\right)$")
    axs.plot(b_costum[index,:].imag,color='red', label=f"$\\Im \\left( e^{{ \\omega_{index+1}t }} \\right)$")

    axs.legend(loc='best', fontsize=15,fancybox=True,framealpha=1.0)

    axs.set_xlim(limits[0,0],limits[0,1])
    axs.set_ylim(limits[1,0],limits[1,1])


    axs.set_xlabel(f"snapshot",fontsize=15)
    axs.set_ylabel("Amplitude (au)",fontsize=15)

    axs.tick_params(axis="x", labelsize=15)
    axs.tick_params(axis="y", labelsize=15)

    axs.spines['top'].set_visible(False)
    axs.spines['right'].set_visible(False)

    if len(fout) > 0:
        plt.savefig(fout+f'index={index}.png', bbox_inches='tight', transparent=True)
        plt.savefig(fout+f'index={index}.pdf', bbox_inches='tight', transparent=True)

def plot_modes_vs_grid(Φ,index,limits,kx,ky=None,dimension=1,fout=''):
    fig, axs=plt.subplots(1, figsize=(6,4))

    if dimension==1:
        axs.plot(kx,Φ[:,index].real,color='blue',label=f"$\\Re \\left( \\Phi_{{ {index+1} }} \\right)$")
        axs.plot(kx,Φ[:,index].imag,color='red',label=f"$\\Im \\left( \\Phi_{index+1} \\right)$")

        axs.legend(loc='best', fontsize=15,fancybox=True,framealpha=1.0)

        axs.set_xlim(limits[0,0],limits[0,1])
        axs.set_ylim(limits[1,0],limits[1,1])


        axs.set_xlabel(f"$k$ (au)",fontsize=15)
        axs.set_ylabel("Amplitude (au)",fontsize=15)

        axs.tick_params(axis="x", labelsize=15)
        axs.tick_params(axis="y", labelsize=15)

        axs.spines['top'].set_visible(False)
        axs.spines['right'].set_visible(False)

        if len(fout) > 0:
            plt.savefig(fout+f'index={index}.png', bbox_inches='tight', transparent=True)
            plt.savefig(fout+f'index={index}.pdf', bbox_inches='tight', transparent=True)

    if dimension==2:
        vmax = np.max(Φ[:,:,index].real)
        vmin = -vmax

        im = axs.imshow(Φ[:,:,index].real, extent=(kx[0],kx[-1],ky[0],ky[-1]),\
                   #cmap='seismic',aspect='auto',interpolation='spline36')
                   cmap=plt.cm.seismic,aspect='auto',interpolation='bicubic',\
                    vmin=vmin, vmax=vmax)


        axs.set_xlim(limits[0,0],limits[0,1])
        axs.set_ylim(limits[1,0],limits[1,1])

        axs.set_title(r"$\Re \{ \Phi_{i} \}$",fontsize=15)
        axs.set_xlabel(f'$k_x$ (au)',fontsize=15)
        axs.set_ylabel(f'$k_y$ (au)',fontsize=15)

        cs = fig.colorbar(im)

        cs.ax.tick_params(labelsize=15)

        if len(fout) > 0:
            plt.savefig(fout+'_real.png', bbox_inches='tight', transparent=True)
            plt.savefig(fout+'_real.pdf', bbox_inches='tight', transparent=True)

        fig, axs=plt.subplots(1, figsize=(6,4))

        vmax = np.max(Φ[:,:,index].imag)
        vmin = -vmax

        im = axs.imshow(Φ[:,:,index].imag, extent=(kx[0],kx[-1],ky[0],ky[-1]),\
                   #cmap='seismic',aspect='auto',interpolation='spline36')
                   cmap=plt.cm.seismic,aspect='auto',interpolation='bicubic',\
                    vmin=vmin, vmax=vmax)


        axs.set_xlim(limits[0,0],limits[0,1])
        axs.set_ylim(limits[1,0],limits[1,1])

        axs.set_title(r"$\Im \{ \Phi_{i} \}$",fontsize=15)
        axs.set_xlabel(f'$k_x$ (au)',fontsize=15)
        axs.set_ylabel(f'$k_y$ (au)',fontsize=15)

        cs = fig.colorbar(im)

        cs.ax.tick_params(labelsize=15)

        if len(fout) > 0:
            plt.savefig(fout+'_imag.png', bbox_inches='tight', transparent=True)
            plt.savefig(fout+'_imag.pdf', bbox_inches='tight', transparent=True)

def plot_comparison_data_1d(original_data,reconstruction_costum,index,limits,fout=''):
    fig, axs=plt.subplots(1, figsize=(6,4))

    axs.plot(reconstruction_costum[index,:].real,'-',color='blue',label='DMD')
    axs.plot(original_data[index,:].real,'--',color='red',label='Original Data')

    axs.legend(loc='best', title=f'$x_{{ {index+1} }}(t)$', fontsize=15,fancybox=True,framealpha=1.0)

    axs.set_xlim(limits[0,0],limits[0,1])
    axs.set_ylim(limits[1,0],limits[1,1])

    axs.set_title("Real part",fontsize=15)
    axs.set_ylabel(f'Amplitude (au)',fontsize=15)
    axs.set_xlabel(f'Snapshot',fontsize=15)

    axs.tick_params(axis="x", labelsize=15)
    axs.tick_params(axis="y", labelsize=15)

    axs.spines['top'].set_visible(False)
    axs.spines['right'].set_visible(False)

    if len(fout) > 0:
        plt.savefig(fout+f'index={index}_real.png', bbox_inches='tight', transparent=True)
        plt.savefig(fout+f'index={index}_real.pdf', bbox_inches='tight', transparent=True)


    fig, axs=plt.subplots(1, figsize=(6,4))

    axs.plot(reconstruction_costum[index,:].imag,'-',color='green',label='DMD')
    axs.plot(original_data[index,:].imag,'--',color='orange',label='Original Data')

    axs.legend(loc='best', title=f'$x_{{ {index+1} }}(t)$', fontsize=13,fancybox=True,framealpha=1.0)

    axs.set_xlim(limits[0,0],limits[0,1])
    axs.set_ylim(limits[1,0],limits[1,1])

    axs.set_title("Imaginary part",fontsize=15)
    axs.set_ylabel(f'Amplitude (au)',fontsize=15)
    axs.set_xlabel(f'Snapshot',fontsize=15)

    axs.tick_params(axis="x", labelsize=15)
    axs.tick_params(axis="y", labelsize=15)

    axs.spines['top'].set_visible(False)
    axs.spines['right'].set_visible(False)

    if len(fout) > 0:
        plt.savefig(fout+f'index={index}_imag.png', bbox_inches='tight', transparent=True)
        plt.savefig(fout+f'index={index}_imag.pdf', bbox_inches='tight', transparent=True)

    return 0

def plot_comparison_data_2d(original_data,reconstruction_costum,kx,time,limits,kind='Real',fout=''):

    extent = 0, np.shape(original_data[:,:])[1], np.min(kx), np.max(kx)

    fig, axs=plt.subplots(1,2, figsize=(12,4))

    if kind == "Real" or kind == "real":
        im = axs[0].imshow(np.real(original_data[:,:]),cmap=plt.cm.gnuplot,aspect='auto',interpolation='bicubic',extent=extent)
    if kind == "Imag" or kind == "imag":
        im = axs[0].imshow(np.imag(original_data[:,:]),cmap=plt.cm.gnuplot,aspect='auto',interpolation='bicubic',extent=extent)
    if kind == "Abs" or kind == "abs":
        im = axs[0].imshow(np.abs(original_data[:,:]),cmap=plt.cm.gnuplot,aspect='auto',interpolation='bicubic',extent=extent)

    axs[0].set_title(f"{kind} part (Original)",fontsize=15)
    axs[0].set_ylabel(f'k (au)',fontsize=15)
    axs[0].set_xlabel(f'Snapshot',fontsize=15)

    axs[0].set_xlim(limits[0,0],limits[0,1])
    axs[0].set_ylim(limits[1,0],limits[1,1])

    axs[0].tick_params(axis="x", labelsize=15)
    axs[0].tick_params(axis="y", labelsize=15)

    cs = fig.colorbar(im)

    cs.ax.tick_params(labelsize=15)

    extent = 0, np.shape(reconstruction_costum[:,:])[1], np.min(kx), np.max(kx)

    if kind == "Real" or kind == "real":
        im = axs[1].imshow(np.real(reconstruction_costum[:,:]),cmap=plt.cm.gnuplot,aspect='auto',interpolation='bicubic',extent=extent)
    if kind == "Imag" or kind == "imag":
        im = axs[1].imshow(np.imag(reconstruction_costum[:,:]),cmap=plt.cm.gnuplot,aspect='auto',interpolation='bicubic',extent=extent)
    if kind == "Abs" or kind == "abs":
        im = axs[1].imshow(np.abs(reconstruction_costum[:,:]),cmap=plt.cm.gnuplot,aspect='auto',interpolation='bicubic',extent=extent)

    axs[1].set_title(f"{kind} part (DMD)",fontsize=15)
    axs[1].set_ylabel(f'k (au)',fontsize=15)
    axs[1].set_xlabel(f'Snapshot',fontsize=15)

    axs[1].set_xlim(limits[0,0],limits[0,1])
    axs[1].set_ylim(limits[1,0],limits[1,1])

    axs[1].tick_params(axis="x", labelsize=15)
    axs[1].tick_params(axis="y", labelsize=15)

    cs = fig.colorbar(im)

    cs.ax.tick_params(labelsize=15)

    if len(fout) > 0:
        plt.savefig(fout+"_original.png", bbox_inches='tight', transparent=True)
        plt.savefig(fout+"_original.pdf", bbox_inches='tight', transparent=True)

    return 0

def plot_frequencies_1d(reconstruction_costum,index,frequency,limits,fout=''):
    fig, axs=plt.subplots(1, figsize=(6,4))

    axs.plot(frequency,np.real(reconstruction_costum[index,:]),'-',color='blue',label='Real')
    axs.plot(frequency,np.imag(reconstruction_costum[index,:]),'--',color='red',label='Imag')

    axs.legend(loc='best', title=f'$x_{{ {index+1} }}(t)$', fontsize=13,fancybox=True,framealpha=1.0)

    axs.set_xlim(limits[0,0],limits[0,1])
    axs.set_ylim(limits[1,0],limits[1,1])

    axs.set_ylabel(f'Amplitude (au)',fontsize=15)
    axs.set_xlabel(f'Frequency',fontsize=15)

    axs.tick_params(axis="x", labelsize=15)
    axs.tick_params(axis="y", labelsize=15)

    axs.spines['top'].set_visible(False)
    axs.spines['right'].set_visible(False)

    if len(fout) > 0:
        plt.savefig(fout+f'index={index}.png', bbox_inches='tight', transparent=True)
        plt.savefig(fout+f'index={index}.pdf', bbox_inches='tight', transparent=True)

    return 0

def plot_frequencies_2d(reconstruction_costum,kx,frequency,limits,fout=''):

    extent = np.min(frequency), np.max(frequency), np.min(kx), np.max(kx)

    fig, axs=plt.subplots(1,2, figsize=(12,4))

    im = axs[0].imshow(np.real(reconstruction_costum[:,:]),cmap=plt.cm.gnuplot,aspect='auto',interpolation='bicubic',extent=extent)


    axs[0].set_title(f"Real part",fontsize=15)
    axs[0].set_ylabel(f'k (au)',fontsize=15)
    axs[0].set_xlabel(f'Frequency',fontsize=15)

    axs[0].set_xlim(limits[0,0],limits[0,1])
    axs[0].set_ylim(limits[1,0],limits[1,1])

    axs[0].tick_params(axis="x", labelsize=15)
    axs[0].tick_params(axis="y", labelsize=15)

    cs = fig.colorbar(im)

    cs.ax.tick_params(labelsize=15)


    im = axs[1].imshow(np.imag(reconstruction_costum[:,:]),cmap=plt.cm.gnuplot,aspect='auto',interpolation='bicubic',extent=extent)


    axs[1].set_title(f"Imag part",fontsize=15)
    axs[1].set_ylabel(f'k (au)',fontsize=15)
    axs[1].set_xlabel(f'Frequency',fontsize=15)

    axs[1].set_xlim(limits[0,0],limits[0,1])
    axs[1].set_ylim(limits[1,0],limits[1,1])

    axs[1].tick_params(axis="x", labelsize=15)
    axs[1].tick_params(axis="y", labelsize=15)

    cs = fig.colorbar(im)

    cs.ax.tick_params(labelsize=15)

    if len(fout) > 0:
        plt.savefig(fout+"_original.png", bbox_inches='tight', transparent=True)
        plt.savefig(fout+"_original.pdf", bbox_inches='tight', transparent=True)

    return 0


###Crear una funcion para hacer el plot de los modos en embeding, por ejemplo, mode[:,0], para ver en el eje-x real, eje-y imag, como una espcie de estrella


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
