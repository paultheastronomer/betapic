import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.ticker import LinearLocator

from matplotlib import cm


def wave2RV(Wave,rest_wavelength,RV_BP):
    c = 299792458
    rest_wavelength = rest_wavelength*(RV_BP*1.e3)/c + rest_wavelength # Convert to beta pic reference frame
    delta_wavelength = Wave-rest_wavelength
    RV = ((delta_wavelength/rest_wavelength)*c)/1.e3	# km/s
    return RV

def main():    

    #W1, RV1, F0_01, E0_01, AG0, AG0err  = np.genfromtxt('/home/paw/science/betapic/data/HST/dat/B_2014.dat',unpack=True,skiprows=7000,skip_footer=6000)
    Wo, Fo, Eo                          = np.genfromtxt('Ly-alpha.dat',unpack=True)
    W, F, E                             = np.genfromtxt('Ly-alpha.dat',skiprows=900,skip_footer=145,unpack=True)
    

    LyA             = 1215.6702
    RV = wave2RV(W,LyA,0.0)     # Heliocentric frame
    RVo = wave2RV(Wo,LyA,0.0)
    
    fig = plt.figure(figsize=(8.5,4))
    fontlabel_size  = 18
    tick_size       = 18
    params = {'backend': 'wxAgg', 'lines.markersize' : 2, 'axes.labelsize': fontlabel_size, 'font.size': fontlabel_size, 'legend.fontsize': fontlabel_size, 'xtick.labelsize': tick_size, 'ytick.labelsize': tick_size, 'text.usetex': True}
    plt.rcParams.update(params)
    plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    plt.rcParams['text.usetex'] = True
    plt.rcParams['text.latex.unicode'] = True    


 

    dir_contents = os.listdir('.')
    
    ax1 = plt.subplot('121')
    files = sorted([fn for fn in dir_contents if fn.startswith('b') and fn.endswith('.dat')])

    colors = iter(cm.Blues(np.linspace(0.1, 1, len(files))))

    for i in range(len(files)):
        v, fit = np.genfromtxt(files[i],unpack=True)
        
        ax1.plot(v,fit,lw=2,color=next(colors))


    plt.scatter(RVo,Fo,color='black',alpha=0.25,label='Data not used for fit')
    plt.scatter(RV,F,color='black',label='Data used for fit')
    plt.ylabel('Flux (erg/s/cm$^2$/\AA)')
    plt.xlabel(r'RV [km/s]')
    plt.xlim(-700,435)
    plt.ylim(0.0,0.34e-13)

    ax2 = plt.subplot('122') 
    files = sorted([fn for fn in dir_contents if fn.startswith('nh') and fn.endswith('.dat')])

    colors = iter(cm.gist_heat(np.linspace(0.8, 0.3, len(files))))

    for i in range(len(files)):
        v, fit = np.genfromtxt(files[i],unpack=True)
        
        ax2.plot(v,fit,lw=2,color=next(colors))
        ax2.get_yaxis().set_ticklabels([])


    plt.scatter(RVo,Fo,color='black',alpha=0.25,label='Data not used for fit')
    plt.scatter(RV,F,color='black',label='Data used for fit')    
    plt.xlabel(r'RV [km/s]')


    plt.xlim(-700,435)
    plt.ylim(0.0,0.34e-13)

    fig.tight_layout()
    plt.savefig('Ly_alpha_vary_param.pdf', bbox_inches='tight', pad_inches=0.1,dpi=300)
    plt.show()

if __name__ == '__main__':
    main()
