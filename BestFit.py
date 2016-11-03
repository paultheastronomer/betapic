import numpy as np
import matplotlib.pyplot as plt

def wave2RV(Wave,rest_wavelength,RV_BP):
    c = 299792458
    rest_wavelength = rest_wavelength*(RV_BP*1.e3)/c + rest_wavelength # Convert to beta pic reference frame
    delta_wavelength = Wave-rest_wavelength
    RV = ((delta_wavelength/rest_wavelength)*c)/1.e3	# km/s
    return RV

def Bin_data(x,y1,e1,bin_pnts):
    bin_size    = int(len(x)/bin_pnts)
    bins        = np.linspace(x[0], x[-1], bin_size)
    digitized   = np.digitize(x, bins)
    bin_y       = np.array([y1[digitized == i].mean() for i in range(0, len(bins))])
    bin_e       = np.array([e1[digitized == i].mean() for i in range(0, len(bins))])
    return bins, bin_y, bin_e/np.sqrt(bin_pnts)

def main():  

    dat_directory = "/home/paw/science/betapic/data/HST/dat/" 

    RVa,Pa                                  = np.genfromtxt(dat_directory+'PolyFit.dat',unpack=True)
    v,f_star,f_abs_ism,f_abs_bp,f_after_fit = np.genfromtxt(dat_directory+'Ly_Fit.dat',unpack=True)
    #v,f_star,f_abs_ism,f_abs_bp,f_abs_X,f_after_fit = np.genfromtxt(dat_directory+'Ly_Fit_2mod.dat',unpack=True)
    LyFit18                                 = np.genfromtxt(dat_directory+'Ly_Fit_18.dat',unpack=True)
    LyFit19                                 = np.genfromtxt(dat_directory+'Ly_Fit_19.dat',unpack=True)
    W_cut, F_cut, E_cut = np.genfromtxt(dat_directory+'Ly-alpha_no_AG_2016_06_23.txt',skip_header=500,unpack=True)      
    
    Wo, Fo, Eo      = np.genfromtxt(dat_directory+'Ly_sky_subtracted_no_central_data_2016_06_21.txt',unpack=True)
    W, F, E         = np.genfromtxt(dat_directory+'Ly_sky_subtracted_no_central_data_2016_06_21.txt',unpack=True,skip_header=9027,skip_footer= 7155)

    x1  = 920
    x2  = 1200
    x3  = 2050
    x4  = -1250

    LyA         = 1215.6702
    RefRV       = 0     # Set to 0 km/s and not 20.5 km/s because the variable v is already shifted by 20.5 km/s from LyFit.py
    bin_pnts    = 3
    
    RV      = wave2RV(W,LyA,RefRV)
    RVo     = wave2RV(Wo,LyA,RefRV)
    RV_cut  = wave2RV(W_cut,LyA,RefRV)
    
    fig = plt.figure(figsize=(6.5,4.5))
    #fig = plt.figure(figsize=(11,5))
    fontlabel_size  = 18
    tick_size       = 18
    params = {'backend': 'wxAgg', 'lines.markersize' : 2, 'axes.labelsize': fontlabel_size, 'font.size': fontlabel_size, 'legend.fontsize': 15, 'xtick.labelsize': tick_size, 'ytick.labelsize': tick_size, 'text.usetex': True}
    plt.rcParams.update(params)
    plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    plt.rcParams['text.usetex'] = True
    plt.rcParams['text.latex.unicode'] = True    


    RVb, Fb, Eb             =   Bin_data(RV,F,E,bin_pnts)
    RVob, Fob, Eob          =   Bin_data(RVo,Fo,Eo,bin_pnts)
    RVb_cut, Fb_cut, Eb_cut =   Bin_data(RV_cut,F_cut,E_cut,bin_pnts)

    # Different log(N)
    '''
    plt.errorbar(RVb,Fb,yerr=Eb,fmt=None,ecolor='black',zorder=3)
    plt.scatter(RVb,Fb, marker='o', color='k',zorder=3,label=r'Method 2')
    plt.plot(v,LyFit18,lw=3,color='#12B84B',label=r'Best fit')
    plt.plot(v,f_after_fit,lw=3,color='#EE0600',label=r'Best fit')
    plt.plot(v,LyFit19,lw=3,color='#1695A3',label=r'Best fit')
    '''
    
    # AG subtracted
    #'''
    plt.errorbar(RVb_cut,Fb_cut,yerr=Eb_cut,fmt=None,ecolor='black',zorder=3)
    plt.scatter(RVb_cut,Fb_cut, marker='o', edgecolor="black",color='#F433FF',zorder=3,label=r'Method 1')
    plt.plot(v,f_star,lw=3,color='gray',label=r'$\beta$ Pictoris')
    plt.plot(v,f_abs_ism,lw=1.2,color='#FFA500',label=r'ISM')
    plt.plot(v,f_abs_bp,lw=1.2,color='#0386ff',label=r'Gas disk')
    #plt.plot(v,f_abs_X,lw=3.0,color='purple',label=r'Component X')
    plt.plot(v,f_after_fit,lw=3,color='#FF281C',label=r'Best fit')
    plt.errorbar(RVb,Fb,yerr=Eb,fmt=None,ecolor='black',zorder=3)
    plt.scatter(RVb,Fb, marker='o', color='k',zorder=3,label=r'Method 2')
    #'''
    
    # AG FREE
    '''
    plt.errorbar(RVb_cut,Fb_cut,yerr=Eb_cut,fmt=None,ecolor='black',zorder=3)
    plt.scatter(RVb_cut,Fb_cut, marker='o', edgecolor="black",color='#F433FF',zorder=3)
    plt.text(720,3.1e-14,r'$\mathrm{O\,V}$',ha='center')
    plt.text(40,4.7e-14,r'$\mathrm{Ly}\alpha$',ha='center')
    
    plt.plot(RVa[x2:x3],Pa[x2:x3],color="red",lw=1.0,zorder=4)
    plt.plot(RVa[x1:x2],Pa[x1:x2],color="black",lw=5.0,zorder=4)
    plt.plot(RVa[x1:x2],Pa[x1:x2],color="#FFB23E",lw=3.0,zorder=4)
    plt.plot(RVa[x3:x4],Pa[x3:x4],color="black",lw=5.0,zorder=4)
    plt.plot(RVa[x3:x4],Pa[x3:x4],color="#FFB23E",lw=3.0,zorder=4)
     
    plt.xlim(-2000,4000)
    plt.ylim(0,5.4e-14)
    '''

    plt.xlabel(r'Radial Velocity [km/s]')
    plt.ylabel('Flux (erg/s/cm$^2$/\AA)')
    plt.xlim(-250,270) 
    plt.ylim(-2.8e-14,8.0e-14)

    #plt.legend(loc='upper left', numpoints=1)
    plt.minorticks_on()
    fig.tight_layout()
    #plt.savefig('../plots/AG_corrected.pdf', bbox_inches='tight', pad_inches=0.1,dpi=300)
    #plt.savefig('../plots/DiffColDens3.pdf', bbox_inches='tight', pad_inches=0.1,dpi=300)
    plt.show()



if __name__ == '__main__':
    main()

