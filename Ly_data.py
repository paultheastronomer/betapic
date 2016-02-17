import numpy as np
import matplotlib.pyplot as plt

def ND(flux,ref,n1,n2):
    return 1. - np.mean(flux[n1:n2])/np.mean(ref[n1:n2])

def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, axis=0, weights=weights)
    variance = np.average((values-average)**2, axis=0, weights=weights)  # Fast and numerically precise
    return average, np.sqrt(variance)

def main():    

    # Load shifted spectra from files
    #
    # W = wavelength, RV = radial velocity, F = flux, E = error, AG = airglow, AGerr = airglow error
    #
    # FX_Y      => X = position, Y= visit number (starting from 0)
    #
    # i.e. F1_2 => Flux measurement during second position during the third visit)
    #
    ############################################################################################# 
    W, RV, F0_0, E0_0, AG0, AG0err = np.genfromtxt('/home/paw/science/betapic/data/HST/dat/B_2014.dat',unpack=True,skiprows=7000,skip_footer=6080)

    W, RV, F0_1, E0_1, F1_1, E1_1, F2_1, E2_1, AG1, AG1err, F_ave_w_1 = np.genfromtxt('/home/paw/science/betapic/data/HST/dat/B_10Dec.dat',unpack=True,skiprows=7000,skip_footer=6080)

    W, RV, F0_2, E0_2, F1_2, E1_2, F2_2, E2_2, F3_2, E3_2, AG2, AG2err, F_ave_w_2 = np.genfromtxt('/home/paw/science/betapic/data/HST/dat/B_24Dec.dat',unpack=True,skiprows=7000,skip_footer=6080)

    W, RV, F0_3, E0_3, F1_3, E1_3, F2_3, E2_3, F3_3, E3_3, AG3, AG3err, F_ave_w_3 = np.genfromtxt('/home/paw/science/betapic/data/HST/dat/B_30Jan.dat',unpack=True,skiprows=7000,skip_footer=6080)
    ############################################################################################# 

    # Choose a region to normalise the spectra
    # units refer to array elements. 450 = start at the 451st element 
    n1 = 450
    n2 = 800
    
    # Calculate the factor of missing flux (N ~ 0 not much flux missing, N~1 lots of flux missing)
    N   = [ND(F0_1,F0_0,n1,n2),ND(F1_1,F0_0,n1,n2),ND(F2_1,F0_0,n1,n2),\
    ND(F0_2,F0_0,n1,n2),ND(F1_2,F0_0,n1,n2),ND(F2_2,F0_0,n1,n2),ND(F3_2,F0_0,n1,n2),\
    ND(F0_3,F0_0,n1,n2),ND(F1_3,F0_0,n1,n2),ND(F2_3,F0_0,n1,n2),ND(F3_3,F0_0,n1,n2)]
    
    F  = [F0_1,F1_1,F2_1,F0_2,F1_2,F2_2,F3_2,F0_3,F1_3,F2_3,F3_3]
    E  = [E0_1,E1_1,E2_1,E0_2,E1_2,E2_2,E3_2,E0_3,E1_3,E2_3,E3_3]
    Fc = [[] for _ in range(len(N))]
    Ec = [[] for _ in range(len(N))]
    for i in range(len(N)):
        Fc[i] = F[i]+N[i]*np.mean(F0_0[n1:n2])      # Correct for missing flux
        Ec[i] = np.sqrt(E[i]**2+(N[i]*E0_0)**2)     # Increase uncertainty if flux difference is large

    # -0.8" Ly-alpha wing
    #############################################################################################    
    F1_errs                 =  np.array([Ec[1],Ec[4],Ec[8]])
    F1_ave_w, F1_ave_err    =  weighted_avg_and_std([Fc[1],Fc[4],Fc[8]], 1./F1_errs**2)
    #############################################################################################    

    # 0.8" Ly-alpha wing
    #############################################################################################
    F2_errs                 =  np.array([Ec[2],Ec[5],Ec[9]])
    F2_ave_w, F2_ave_err    =  weighted_avg_and_std([Fc[2],Fc[5],Fc[9]], 1./F2_errs**2)
    #############################################################################################

    # 1.1" Ly-alpha wing
    #############################################################################################
    F3_errs                 =  np.array([E3_2,E3_3])
    F3_ave_w, F3_ave_err    =  weighted_avg_and_std([Fc[6],Fc[10]], 1./F3_errs**2)
    #############################################################################################


    # Combining 0.8" and 1.1" data of the Ly-alpha wing
    F_errs                  = np.array([F2_ave_err,F3_ave_err])
    F_ave_r_w, F_ave_r_err  = weighted_avg_and_std([F2_ave_w,F3_ave_w], 1./F_errs**2)

    # Combine all data regardless of shift
    E_tot               = np.array([E0_0,Ec[0],Ec[1],Ec[2],Ec[3],Ec[4],Ec[5],Ec[6],Ec[7],Ec[8],Ec[9],Ec[10]])
    F_tot, F_tot_err    = weighted_avg_and_std([F0_0,Fc[0],Fc[1],Fc[2],Fc[3],Fc[4],Fc[5],Fc[6],Fc[7],Fc[8],Fc[9],Fc[10]], 1./E_tot**2)
    
    # Decide at which RV airglow affects the different measurements
    # Units in km/s
    shift_0_l       = -337.5
    shift_0_r       =  271.7
    shift_08_l      = -183.3
    shift_08_r      =  142.5
    shift_11_r      =  120

    # Plot the spectra using the cuts above and write to .dat file.
    f = open('Ha.dat', 'w+')
    for j in range(len(RV)):
        if RV[j] < shift_0_l or RV[j] > shift_0_r:
            plt.errorbar(RV[j],F_tot[j],yerr=F_tot_err[j],color='black')
            print >> f, " ","{: 1.10e}".format(W[j])," "+"{: 1.10e}".format(F_tot[j])," "+"{: 1.10e}".format(F_tot_err[j])
        if  shift_0_l < RV[j] < shift_08_l:
            plt.errorbar(RV[j],F1_ave_w[j],yerr=F1_ave_err[j],color='blue')
            print >> f, " ","{: 1.10e}".format(W[j])," "+"{: 1.10e}".format(F1_ave_w[j])," "+"{: 1.10e}".format(F1_ave_err[j])
        if  shift_08_r < RV[j] < shift_0_r:
            plt.errorbar(RV[j],F2_ave_w[j],yerr=F2_ave_err[j],color='blue')
            print >> f, " ","{: 1.10e}".format(W[j])," "+"{: 1.10e}".format(F2_ave_w[j])," "+"{: 1.10e}".format(F2_ave_err[j])
        if  shift_11_r < RV[j] < shift_08_r:
            plt.errorbar(RV[j],F3_ave_w[j],yerr=F3_ave_err[j],color='red')
            print >> f, " ","{: 1.10e}".format(W[j])," "+"{: 1.10e}".format(F3_ave_w[j])," "+"{: 1.10e}".format(F3_ave_err[j])
    f.close()
    
    # Plot the region used for normalisation
    plt.step(RV[n1:n2],F_tot[n1:n2],color='green')    

    plt.xlabel(r'RV (km/s)')
    plt.ylabel('Flux')  
    plt.xlim(-2000,800)
    plt.ylim(-0.3e-14,3.5e-14)
    plt.show()
	
if __name__ == '__main__':
    main()
