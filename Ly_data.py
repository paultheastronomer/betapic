import numpy as np
import matplotlib.pyplot as plt

def CF(flux,flux_err,ref,ref_err,n1,n2):
    ratio = np.average(flux[n1:n2], axis=0, weights=1./(flux_err[n1:n2]**2))/ \
            np.average(ref[n1:n2],  axis=0, weights=1./(ref_err[n1:n2]**2 ))                       
    return 1./ratio

def weighted_avg_and_errorbars(Flux, Err):
    """
    Return the weighted average and Error bars.
    """
    weights=1./(Err**2)
    average = np.average(Flux, axis=0, weights=weights)
    errorbars_2 = np.sum((weights*Err)**2, axis=0)/ ((np.sum(weights, axis=0))**2)
    return average, np.sqrt(errorbars_2)

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
    W, RV, F0_0, E0_0, AG0, AG0err = np.genfromtxt('/home/paw/science/betapic/data/HST/dat/B_2014.dat',unpack=True,skiprows=7000,skip_footer=6000)

    W, RV, F0_1, E0_1, F1_1, E1_1, F2_1, E2_1, AG1, AG1err, F_ave_w_1 = np.genfromtxt('/home/paw/science/betapic/data/HST/dat/B_10Dec.dat',unpack=True,skiprows=7000,skip_footer=6000)

    W, RV, F0_2, E0_2, F1_2, E1_2, F2_2, E2_2, F3_2, E3_2, AG2, AG2err, F_ave_w_2 = np.genfromtxt('/home/paw/science/betapic/data/HST/dat/B_24Dec.dat',unpack=True,skiprows=7000,skip_footer=6000)

    W, RV, F0_3, E0_3, F1_3, E1_3, F2_3, E2_3, F3_3, E3_3, AG3, AG3err, F_ave_w_3 = np.genfromtxt('/home/paw/science/betapic/data/HST/dat/B_30Jan.dat',unpack=True,skiprows=7000,skip_footer=6000)
    ############################################################################################# 

    # Choose a region to normalise the spectra
    # units refer to array elements. 450 = start at the 451st element 
    n1 = 450
    n2 = 800
    
    # Calculate the Correction Factor 
    C   = [CF(F0_1,E0_1,F0_0,E0_0,n1,n2),CF(F1_1,E1_1,F0_0,E0_0,n1,n2),CF(F2_1,E2_1,F0_0,E0_0,n1,n2),\
    CF(F0_2,E0_2,F0_0,E0_0,n1,n2),CF(F1_2,E1_2,F0_0,E0_0,n1,n2),CF(F2_2,E2_2,F0_0,E0_0,n1,n2),CF(F3_2,E3_2,F0_0,E0_0,n1,n2),\
    CF(F0_3,E0_3,F0_0,E0_0,n1,n2),CF(F1_3,E1_3,F0_0,E0_0,n1,n2),CF(F2_3,E2_3,F0_0,E0_0,n1,n2),CF(F3_3,E3_3,F0_0,E0_0,n1,n2)]
    
    F  = [F0_1,F1_1,F2_1,F0_2,F1_2,F2_2,F3_2,F0_3,F1_3,F2_3,F3_3]
    E  = [E0_1,E1_1,E2_1,E0_2,E1_2,E2_2,E3_2,E0_3,E1_3,E2_3,E3_3]
    Fc = [[] for _ in range(len(C))]
    Ec = [[] for _ in range(len(C))]
    
    for i in range(len(C)):
        Fc[i] = F[i]*C[i]   # Correct for lower efficiency
        Ec[i] = E[i]*C[i]   # accordingly correct the tabulated error bars

    # -0.8" Ly-alpha wing
    #############################################################################################    
    Flux = np.array([Fc[1],Fc[4],Fc[8]])
    Err  = np.array([Ec[1],Ec[4],Ec[8]])
    F1, F1_err    =  weighted_avg_and_errorbars(Flux,Err)         
    #############################################################################################    

    # 0.8" Ly-alpha wing
    #############################################################################################
    Flux = np.array([Fc[2],Fc[5],Fc[9]])
    Err  = np.array([Ec[2],Ec[5],Ec[9]])
    #Flux = np.array([Fc[2],Fc[5]])
    #Err  = np.array([Ec[2],Ec[5]])
    F2, F2_err    =  weighted_avg_and_errorbars(Flux,Err)         
    #############################################################################################

    # 1.1" Ly-alpha wing
    #############################################################################################
    Flux = np.array([Fc[6],Fc[10]])
    Err  = np.array([Ec[6],Ec[10]])
    #Flux = np.array([Fc[6]])
    #Err  = np.array([Ec[6]])
    F3, F3_err    =  weighted_avg_and_errorbars(Flux,Err)         
    #############################################################################################

    # Combining 0.8" and 1.1" data of the Ly-alpha wing
    #############################################################################################
    Flux = np.array([Fc[2],Fc[5],Fc[9],Fc[6],Fc[10]])
    Err  = np.array([Ec[2],Ec[5],Ec[9],Ec[6],Ec[10]])
    #Flux = np.array([Fc[2],Fc[5],Fc[6]])
    #Err  = np.array([Ec[2],Ec[5],Ec[6]])
    F4, F4_err    =  weighted_avg_and_errorbars(Flux,Err)         
    #############################################################################################

    # Combine all data regardless of shift
    #############################################################################################
    Flux = np.array([F0_0,Fc[0],Fc[1],Fc[2],Fc[3],Fc[4],Fc[5],Fc[6],Fc[7],Fc[8],Fc[9],Fc[10]])
    Err  = np.array([E0_0,Ec[0],Ec[1],Ec[2],Ec[3],Ec[4],Ec[5],Ec[6],Ec[7],Ec[8],Ec[9],Ec[10]])
    #Flux = np.array([Fc[0],Fc[1],Fc[2],Fc[3],Fc[4],Fc[5],Fc[6],Fc[7]])
    #Err  = np.array([Ec[0],Ec[1],Ec[2],Ec[3],Ec[4],Ec[5],Ec[6],Ec[7]])
    F_tot, F_tot_err    =  weighted_avg_and_errorbars(Flux,Err)
    #############################################################################################
    
    # Decide at which RV airglow affects the different measurements
    # Units in km/s
    shift_0_l       = -235#337.5
    shift_0_r       =  220#271.7
    shift_08_l      = shift_0_l+68#-183.3
    shift_08_r      = shift_0_r-68#157#142.5
    shift_11_r      = shift_0_r-94# 106


    #plt.plot(RV,AG0)
    plt.errorbar(RV,F_tot,yerr=F_tot_err,fmt='',ecolor='black')
    plt.plot(RV,F_tot, marker='o', color='k')
    # Plot the spectra using the cuts above and write to .dat file.
    f = open('Ly-alpha.dat', 'w+')
    for j in range(len(RV)):
        # Save 0.0" shift data
        if RV[j] < shift_0_l:
            print >> f, " ","{: 1.10e}".format(W[j])," "+"{: 1.10e}".format(F_tot[j])," "+"{: 1.10e}".format(F_tot_err[j])       
        if RV[j] > shift_0_r:
            print >> f, " ","{: 1.10e}".format(W[j])," "+"{: 1.10e}".format(F_tot[j])," "+"{: 1.10e}".format(F_tot_err[j])
        
        # Save 0.8" shift data
        if  shift_0_l < RV[j] < shift_08_l:
            plt.errorbar(RV[j],F1[j],yerr=F1_err[j],color='blue')
            print >> f, " ","{: 1.10e}".format(W[j])," "+"{: 1.10e}".format(F1[j])," "+"{: 1.10e}".format(F1_err[j])
        if  shift_08_r < RV[j] < shift_0_r:
            #plt.errorbar(RV[j],F2[j],yerr=F2_err[j],fmt='',color='blue')
            plt.plot(RV[j],F2[j], marker='o', color='b')
            print >> f, " ","{: 1.10e}".format(W[j])," "+"{: 1.10e}".format(F2[j])," "+"{: 1.10e}".format(F2_err[j])
        # Save 1.1" shift data
        if  shift_11_r < RV[j] < shift_08_r:
            #plt.errorbar(RV[j],F3[j],yerr=F3_err[j],fmt='',color='red')
            plt.plot(RV[j],F3[j], marker='o', color='r')
            print >> f, " ","{: 1.10e}".format(W[j])," "+"{: 1.10e}".format(F3[j])," "+"{: 1.10e}".format(F3_err[j])
    f.close()
    
    # Plot the region used for normalisation
    plt.step(RV[n1:n2],F_tot[n1:n2],color='green')

    plt.xlabel(r'RV (km/s)')
    plt.ylabel('Flux')
    plt.xlim(-500,500)  
    #plt.xlim(-2000,800)
    plt.ylim(-0.3e-14,5.5e-14)
    plt.show()

if __name__ == '__main__':
    main()
