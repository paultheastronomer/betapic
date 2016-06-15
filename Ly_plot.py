import numpy as np
import matplotlib.pyplot as plt
import sys

def Bin_data(x,y1,e1,bin_pnts):
    bin_size    = int(len(x)/bin_pnts)
    bins        = np.linspace(x[0], x[-1], bin_size)
    digitized   = np.digitize(x, bins)
    bin_y       = np.array([y1[digitized == i].mean() for i in range(0, len(bins))])
    bin_e       = np.array([e1[digitized == i].mean() for i in range(0, len(bins))])
    return bins, bin_y ,bin_e/np.sqrt(bin_pnts)

def wave2RV(Wave,rest_wavelength,RV_BP):
    c = 299792458
    rest_wavelength = rest_wavelength*(RV_BP*1.e3)/c + rest_wavelength # Convert to beta pic reference frame
    delta_wavelength = Wave-rest_wavelength
    RV = ((delta_wavelength/rest_wavelength)*c)/1.e3	# km/s
    return RV

def main():    

    dat_directory = "/home/paw/science/betapic/data/HST/dat/" 

    #Wc, Fc, Ec  = np.genfromtxt(dat_directory+'Ly-alpha.dat',skiprows=830,skip_footer=0,unpack=True)
    W0, F0, E0  = np.genfromtxt(dat_directory+'Ly-alpha_B_2014.dat',skip_header=8500,skip_footer=6500,unpack=True)
    W1, F1, E1  = np.genfromtxt(dat_directory+'Ly-alpha_B_10Dec.dat',skip_header=8500,skip_footer=6500,unpack=True)
    W2, F2, E2  = np.genfromtxt(dat_directory+'Ly-alpha_B_24Dec.dat',skip_header=8500,skip_footer=6500,unpack=True)
    W3, F3, E3  = np.genfromtxt(dat_directory+'Ly-alpha_B_30Jan.dat',skip_header=8500,skip_footer=6500,unpack=True)
    
    #W, F, E     = np.genfromtxt(dat_directory+'Ly_sky_subtracted.txt',unpack=True)
        
    rest_wavelength = 1215.6702 
    RV_BP           = 20.5
    bin_pnts    = 7
    
    # Convert to RV with beta Pic as reference frame
    #RVc                = wave2RV(Wc,rest_wavelength,RV_BP)
    #RV                 = wave2RV(W,rest_wavelength,20.5)
    #RVb, Fb, Eb        =   Bin_data(RV,F,E,bin_pnts)
    
    RV0         = wave2RV(W0,rest_wavelength,RV_BP)
    RV1         = wave2RV(W1,rest_wavelength,RV_BP)
    RV2         = wave2RV(W2,rest_wavelength,RV_BP)
    RV3         = wave2RV(W3,rest_wavelength,RV_BP)

    RV0b, F0b, E0b   =   Bin_data(RV0,F0,E0,bin_pnts)
    RV1b, F1b, E1b   =   Bin_data(RV1,F1,E1,bin_pnts)
    RV2b, F2b, E2b   =   Bin_data(RV2,F2,E2,bin_pnts)
    RV3b, F3b, E3b   =   Bin_data(RV3,F3,E3,bin_pnts)

    # Plot the results
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharex=True, sharey=True,figsize=(6,8))
    fontlabel_size  = 18
    tick_size       = 18
    params = {'backend': 'wxAgg', 'lines.markersize' : 2, 'axes.labelsize': fontlabel_size, 'font.size': fontlabel_size, 'legend.fontsize': 15, 'xtick.labelsize': tick_size, 'ytick.labelsize': tick_size, 'text.usetex': True}
    plt.rcParams.update(params)
    plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    plt.rcParams['text.usetex'] = True
    plt.rcParams['text.latex.unicode'] = True    

    # Regular
    #'''
    #plt.errorbar(RVc,Fc,yerr=Ec,color='black',alpha=0.5,label='Combined')
    #plt.scatter(RVc,Fc,color='black',label='Combined')
    #plt.scatter(RV0,F0,color='#FF281C',alpha=0.5,label='2014')
    #plt.scatter(RV1,F1,color='#FF9303',alpha=0.5,label='2015v1')
    #plt.scatter(RV2,F2,color='#0386FF',alpha=0.5,label='2015v2')  
    #plt.scatter(RV3,F3,color='#00B233',alpha=0.5,label='2016v3')
    #'''  

    norm = 1e-14
    ax1.errorbar(RV0b,F0b/norm,yerr=E0b/norm,color='black')
    ax1.scatter(RV0b,F0b/norm,color='#FF281C',s=25,edgecolor='k',label='2014',zorder=3)
    ax1.text(450,4,'2014',ha='left')

    ax2.errorbar(RV1b,F1b/norm,yerr=E1b/norm,color='black')
    ax2.scatter(RV1b,F1b/norm,color='#FF9303',s=25,edgecolor='k',label='2015v1',zorder=3)
    ax2.text(450,4,'2015v1',ha='left')


    ax3.errorbar(RV2b,F2b/norm,yerr=E2b/norm,color='black')
    ax3.scatter(RV2b,F2b/norm,color='#0386FF',s=25,edgecolor='k',label='2015v2',zorder=3)
    ax3.text(450,4,'2015v2',ha='left')


    ax4.errorbar(RV3b,F3b/norm,yerr=E3b/norm,color='black')
    ax4.scatter(RV3b,F3b/norm,color='#00B233',s=25,edgecolor='k',label='2016',zorder=3)
    ax4.text(450,4,'2016',ha='left')

    fig.text(0.03, 0.5, r'Flux$/1\times10^{-14}$ (erg/s/cm$^2$/\AA)', ha='center', va='center', rotation='vertical')
    plt.xlabel(r'RV [km/s]')
    plt.xlim(0,580)
    plt.ylim(0,5.5)

    #plt.legend(loc='upper right', numpoints=1)
    #plt.subplots_adjust(hspace=0.43)
    #fig.tight_layout()
    plt.savefig('../plots/Ly_only_cut.pdf', bbox_inches='tight', pad_inches=0.1,dpi=300)
    #plt.show()

if __name__ == '__main__':
    main()
