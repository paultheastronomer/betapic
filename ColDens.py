import numpy as np
import matplotlib.pyplot as plt

def main(): 

    dat_directory = "/home/paw/science/betapic/data/calc/"
    ISM, D      = np.genfromtxt(dat_directory+'ColDens.txt',unpack=True)

    # Plot the results
    fig = plt.figure(figsize=(6,4.5))
    #fig = plt.figure(figsize=(14,5))
    fontlabel_size  = 18
    tick_size       = 18
    params = {'backend': 'wxAgg', 'lines.markersize' : 2, 'axes.labelsize': fontlabel_size, 'font.size': fontlabel_size, 'legend.fontsize': 15, 'xtick.labelsize': tick_size, 'ytick.labelsize': tick_size, 'text.usetex': True}
    plt.rcParams.update(params)
    plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    plt.rcParams['text.usetex'] = True
    plt.rcParams['text.latex.unicode'] = True   
    
    plt.axhspan(19.3, 19.5,color="#D3D3D3")
    plt.scatter(ISM,D,color="black",zorder=3)

    
    #x = np.arange(16,10,0)
    #plt.xlabel(r'Wavelength (\AA)')

    
    plt.plot([16,20],[19.4,19.4],'--',color="#4682b4",lw=2.0)
    plt.plot([18.4,18.4],[18.0,20.0],'--',color="#4682b4",lw=2.0)
    plt.plot([16,20],[19.5,19.5],'--k')
    plt.plot([16,20],[19.3,19.3],'--k')
    plt.xlim(16.9,19.45)
    plt.ylim(18.8,19.55)
    #plt.legend(loc='lower left', numpoints=1)
    plt.minorticks_on()
    plt.xlabel(r'$\log(N_{\mathrm{H}})_{\mathrm{ISM}}$')
    plt.ylabel(r'$\log(N_{\mathrm{H}})_{\mathrm{disk}}$')
    fig.tight_layout()
    #plt.savefig('../plots/ism_bp.pdf', bbox_inches='tight', pad_inches=0.1,dpi=300)
    plt.show()

if __name__ == '__main__':
    main()
