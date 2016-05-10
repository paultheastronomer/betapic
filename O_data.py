import numpy as np
import matplotlib.pyplot as plt
from math import factorial
import sys

def Bin_data(x,y1,e1,bin_pnts):
    bin_size    = int(len(x)/bin_pnts)
    bins        = np.linspace(x[0], x[-1], bin_size)
    digitized   = np.digitize(x, bins)
    bin_y       = np.array([y1[digitized == i].mean() for i in range(0, len(bins))])
    bin_e       = np.array([e1[digitized == i].mean() for i in range(0, len(bins))])
    return bins, bin_y, bin_e/np.sqrt(bin_pnts)

def weighted_avg_and_errorbars(Flux, Err):
    """
    Return the weighted average and Error bars.
    """
    weights=1./(Err**2)
    average = np.average(Flux, axis=0, weights=weights)
    errorbars_2 = np.sum(weights*(Err**2), axis=0) / np.sum(weights, axis=0)
    return average, np.sqrt(errorbars_2)

def CF(flux,flux_err,ref,ref_err,n1,n2):
    flux        = replace_with_median(flux)
    flux_err    = replace_with_median(flux_err)
    ratio = np.average(flux[n1:n2], axis=0, weights=1./(flux_err[n1:n2]**2))/ \
            np.average(ref[n1:n2],  axis=0, weights=1./(ref_err[n1:n2]**2 ))                       
    return 1./ratio

def ShiftAG(AG,units):
    zeros   = np.zeros(abs(units))
    if units > 0.:
        AG      = np.concatenate((zeros,AG), axis=1)[:-units]
    else:
        AG      = np.concatenate((AG,zeros), axis=1)[abs(units):]
    return AG

def replace_with_median(X):
    X[np.isnan(X)] = 0
    m = np.median(X[X > 0])
    X[X == 0] = m
    return X

def PlotCut():
    return 1

def main():    
    #############################################################################################
    #
    # Load shifted spectra from files
    #
    # W = wavelength, RV = radial velocity, F = flux, E = error, AG = airglow, AGerr = airglow error
    #
    # FX_Y      => X = position, Y= visit number (starting from 0)
    #
    # i.e. F1_2 => Flux measurement during second position during the third visit)
    #
    ############################################################################################# 
    W, RV, F0_0, E0_0, AG0, AG0err = np.genfromtxt('/home/paw/science/betapic/data/HST/dat/A_2014.dat',unpack=True,skiprows=500,skip_footer=6000)

    W, RV, F0_1, E0_1, F1_1, E1_1, F2_1, E2_1, AG1, AG1err, F_ave_w_1 = np.genfromtxt('/home/paw/science/betapic/data/HST/dat/A_10Dec.dat',unpack=True,skiprows=500,skip_footer=6000)

    W, RV, F0_2, E0_2, F1_2, E1_2, F2_2, E2_2, F3_2, E3_2, AG2, AG2err, F_ave_w_2 = np.genfromtxt('/home/paw/science/betapic/data/HST/dat/A_24Dec.dat',unpack=True,skiprows=500,skip_footer=6000)

    W, RV, F0_3, E0_3, F1_3, E1_3, F2_3, E2_3, F3_3, E3_3, AG3, AG3err, F_ave_w_3 = np.genfromtxt('/home/paw/science/betapic/data/HST/dat/A_30Jan.dat',unpack=True,skiprows=500,skip_footer=6000)
    ############################################################################################# 

    dat_directory = "/home/paw/science/betapic/data/HST/dat/"

    # Choose a region to normalise the spectra
    # units refer to array elements.
    n1 = 0
    n2 = 500
    
    # Uncomment lines below to see region used to correct the spectra
    #plt.plot(W,F0_1)
    #plt.plot(W[n1:n2],F0_1[n1:n2])
    #plt.show()
    
    # Calculate the Correction Factor 
    C   = [CF(F0_1,E0_1,F0_0,E0_0,n1,n2),CF(F1_1,E1_1,F0_0,E0_0,n1,n2),CF(F2_1,E2_1,F0_0,E0_0,n1,n2),\
    CF(F0_2,E0_2,F0_0,E0_0,n1,n2),CF(F1_2,E1_2,F0_0,E0_0,n1,n2),CF(F2_2,E2_2,F0_0,E0_0,n1,n2),CF(F3_2,E3_2,F0_0,E0_0,n1,n2),\
    CF(F0_3,E0_3,F0_0,E0_0,n1,n2),CF(F1_3,E1_3,F0_0,E0_0,n1,n2),CF(F2_3,E2_3,F0_0,E0_0,n1,n2),CF(F3_3,E3_3,F0_0,E0_0,n1,n2)]
    
    np.savetxt(dat_directory+"rescaling_factors_O.txt",C)
    
    F  = [F0_1,F1_1,F2_1,F0_2,F1_2,F2_2,F3_2,F0_3,F1_3,F2_3,F3_3]
    E  = [E0_1,E1_1,E2_1,E0_2,E1_2,E2_2,E3_2,E0_3,E1_3,E2_3,E3_3]
    

    Fc = [[] for _ in range(len(C))]
    Ec = [[] for _ in range(len(C))]

    for i in range(len(C)):
        Fc[i] = F[i]*C[i]   # Correct for lower efficiency
        Ec[i] = E[i]*C[i]   # accordingly correct the tabulated error bars


    Flux_10Dec = np.array([Fc[0],Fc[1],Fc[2]])
    Err_10Dec  = np.array([Ec[0],Ec[1],Ec[2]])

    Flux_26Dec = np.array([Fc[3],Fc[4],Fc[5],Fc[6]])
    Err_26Dec  = np.array([Ec[3],Ec[4],Ec[5],Ec[6]])

    Flux_30Jan = np.array([Fc[7],Fc[8],Fc[9],Fc[10]])
    Err_30Jan  = np.array([Ec[7],Ec[8],Ec[9],Ec[10]])
    
    
    Flux_w_10Dec, Err_w_10Dec    =  weighted_avg_and_errorbars(Flux_10Dec,Err_10Dec)
    Flux_w_26Dec, Err_w_26Dec    =  weighted_avg_and_errorbars(Flux_26Dec,Err_26Dec)
    Flux_w_30Jan, Err_w_30Jan    =  weighted_avg_and_errorbars(Flux_30Jan,Err_30Jan)


    # Combining AG measurements. Not including AG2 due to problem with data.
    AirG                = np.array([AG0,AG1,AG2,AG3])
    AirG_err            = np.array([AG0err,AG1err,AG2err,AG3err])
    AirG_W, AirG_W_err  = weighted_avg_and_errorbars(AirG,AirG_err) 

    '''    
    AG1    = ShiftAG(AirG_W,-28)    #2015v1 +0.8" AG
    AG2    = ShiftAG(AirG_W,-32)    #2015v2 +0.8" AG
    AG3    = ShiftAG(AirG_W,-25)    #2016 +0.8" AG

    AG4    = ShiftAG(AirG_W,-42)    #2015v2 +1.1" AG
    AG5    = ShiftAG(AirG_W,-43)    #2016 +1.1" AG

    AG6    = ShiftAG(AirG_W,32)     #2015v1 -0.8" AG
    AG7    = ShiftAG(AirG_W,35)     #2015v2 -0.8" AG
    AG8    = ShiftAG(AirG_W,33)     #2016 -0.8" AG
    '''
    

    bin_pnts = 3
    Wb, AirG_W_b, AirG_W_err_b          = Bin_data(W,AirG_W, AirG_W_err,bin_pnts)
    Wb, Flux_b_2014, Err_b_2014         = Bin_data(W,F0_0,E0_0,bin_pnts)
    Wb, Flux_w_b_10Dec, Err_w_b_10Dec   = Bin_data(W,Flux_w_10Dec,Err_w_10Dec,bin_pnts)
    Wb, Flux_w_b_26Dec, Err_w_b_26Dec   = Bin_data(W,Flux_w_26Dec,Err_w_26Dec,bin_pnts)
    Wb, Flux_w_b_30Jan, Err_w_b_30Jan   = Bin_data(W,Flux_w_30Jan,Err_w_30Jan,bin_pnts)
    

    # Plot the results
    fig = plt.figure(figsize=(8,6))
    #fig = plt.figure(figsize=(14,5))
    fontlabel_size  = 18
    tick_size       = 18
    params = {'backend': 'wxAgg', 'lines.markersize' : 2, 'axes.labelsize': fontlabel_size, 'font.size': fontlabel_size, 'legend.fontsize': 15, 'xtick.labelsize': tick_size, 'ytick.labelsize': tick_size, 'text.usetex': True}
    plt.rcParams.update(params)
    plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    plt.rcParams['text.usetex'] = True
    plt.rcParams['text.latex.unicode'] = True   
    
    plt.plot(Wb,AirG_W_b,color="blue",alpha=0.5)
    #plt.plot(Wb,Flux_b_2014,color="blue")
    plt.plot(Wb,Flux_b_2014-8*AirG_W_b,color="#FF281C",label='2014')

    plt.plot(Wb,Flux_w_b_10Dec,color="#FF9303",label='2015v1')

    #plt.plot(Wb,Flux_w_b_26Dec,color="blue")
    plt.plot(Wb,Flux_w_b_26Dec-AirG_W_b,color="#0386FF",label='2015v2')
    
    #plt.plot(Wb,Flux_w_b_30Jan,color="blue")
    plt.plot(Wb,Flux_w_b_30Jan-2*AirG_W_b,color="#00B233",label='2016')
    

    plt.xlabel(r'Wavelength (\AA)')
    plt.ylabel('Flux (erg/s/cm$^2$/\AA)')
    plt.xlim(1300,1308)
    plt.ylim(0.0,1.6e-13)
    plt.legend(loc='lower left', numpoints=1)
    fig.tight_layout()
    plt.savefig('OI.pdf', bbox_inches='tight', pad_inches=0.1,dpi=300)
    plt.show()

if __name__ == '__main__':
    main()
