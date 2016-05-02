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


def CreateRight(RV,F):
    RVnew   = []
    Fnew    = []
    for i in range(len(RV)):
        if RV[i]*-1 > 0:
            RVnew.append(RV[i]*-1)
            Fnew.append(F[i])
    if len(RVnew) < 228:
        add = 228 - len(RVnew)
        Fnew = np.concatenate((Fnew,np.zeros(add)))
        RVnew = np.concatenate((RVnew,np.zeros(add)))      
    return np.array(RVnew), np.array(Fnew)

def CreateLeft(RV,F):
    RVnew   = []
    Fnew    = []
    for i in range(len(RV)):
        if RV[i] > 0:
            RVnew.append(RV[i])
            Fnew.append(F[i])
    if len(RVnew) < 228:
        add = 228 - len(RVnew)
        Fnew = np.concatenate((np.zeros(add),Fnew))
        RVnew = np.concatenate((np.zeros(add),RVnew))      
    return np.array(RVnew), np.array(Fnew)

def main():    

    dat_directory = "/home/paw/science/betapic/data/HST/dat/" 

    #Wc, Fc, Ec  = np.genfromtxt(dat_directory+'Ly-alpha.dat',skiprows=830,skip_footer=0,unpack=True)
    W0, F0, E0  = np.genfromtxt(dat_directory+'Ly-alpha_B_2014.dat',skiprows=830,skip_footer=0,unpack=True)
    W1, F1, E1  = np.genfromtxt(dat_directory+'Ly-alpha_B_10Dec.dat',skiprows=830,skip_footer=0,unpack=True)
    W2, F2, E2  = np.genfromtxt(dat_directory+'Ly-alpha_B_24Dec.dat',skiprows=830,skip_footer=0,unpack=True)
    W3, F3, E3  = np.genfromtxt(dat_directory+'Ly-alpha_B_30Jan.dat',skiprows=830,skip_footer=0,unpack=True)
        
    rest_wavelength = 1215.6702 
    RV_BP           = 20.5
    
    # Convert to RV with beta Pic as reference frame
    #RVc         = wave2RV(Wc,rest_wavelength,RV_BP)
    
    RV0         = wave2RV(W0,rest_wavelength,RV_BP)
    RV1         = wave2RV(W1,rest_wavelength,RV_BP)
    RV2         = wave2RV(W2,rest_wavelength,RV_BP)
    RV3         = wave2RV(W3,rest_wavelength,RV_BP)

    bin_pnts    = 3

    RV0b, F0b, E0b   =   Bin_data(RV0,F0,E0,bin_pnts)
    RV1b, F1b, E1b   =   Bin_data(RV1,F1,E1,bin_pnts)
    RV2b, F2b, E2b   =   Bin_data(RV2,F2,E2,bin_pnts)
    RV3b, F3b, E3b   =   Bin_data(RV3,F3,E3,bin_pnts)

    # Plot the results
    fig = plt.figure(figsize=(6,4.5))
    #fig = plt.figure(figsize=(14,5))
    fontlabel_size  = 18
    tick_size       = 18
    params = {'backend': 'wxAgg', 'lines.markersize' : 2, 'axes.labelsize': fontlabel_size, 'font.size': fontlabel_size, 'legend.fontsize': fontlabel_size, 'xtick.labelsize': tick_size, 'ytick.labelsize': tick_size, 'text.usetex': True}
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

    # Binned
    #'''
    plt.errorbar(RV0b,F0b,yerr=E0b,color='#FF281C')
    plt.scatter(RV0b,F0b,color='#FF281C',s=40,edgecolor='k',label='2014') 

    plt.errorbar(RV1b,F1b,yerr=E1b,color='#FF9303')
    plt.scatter(RV1b,F1b,color='#FF9303',s=40,edgecolor='k',label='2015v1') 

    plt.errorbar(RV2b,F2b,yerr=E2b,color='#0386FF')
    plt.scatter(RV2b,F2b,color='#0386FF',s=40,edgecolor='k',label='2015v2') 

    plt.errorbar(RV3b,F3b,yerr=E3b,color='#00B233')
    plt.scatter(RV3b,F3b,color='#00B233',s=40,edgecolor='k',label='2016')

    plt.xlabel(r'RV [km/s]')
    plt.ylabel('Flux (erg/s/cm$^2$/\AA)')
    plt.xlim(-500,615)
    plt.ylim(0,3.5e-14)
    #'''

    plt.legend(loc='upper right', numpoints=1)
    #plt.savefig('Ly_red_wing.pdf', bbox_inches='tight', pad_inches=0.1,dpi=300)
    plt.show()

if __name__ == '__main__':
    main()
