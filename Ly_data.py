import numpy as np
import matplotlib.pyplot as plt
import sys

def CF(flux,flux_err,ref,ref_err,n1,n2):
    ratio = np.average(flux[n1:n2], axis=0, weights=1./(flux_err[n1:n2]**2))/ \
            np.average(ref[n1:n2],  axis=0, weights=1./(ref_err[n1:n2]**2 ))                       
    return 1./ratio

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

def weighted_avg_and_errorbars(Flux, Err):
    """
    Return the weighted average and Error bars.
    """
    weights=1./(Err**2)
    average = np.average(Flux, axis=0, weights=weights)
    errorbars_2 = np.sum((weights*Err)**2, axis=0)/ ((np.sum(weights, axis=0))**2)
    return average, np.sqrt(errorbars_2)

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
    W, RV, F0_0, E0_0, AG0, AG0err = np.genfromtxt('C:\Users\Alain Lecavelier\Desktop\HST_BetaPic\Data\dat\B_2014.dat',unpack=True,skiprows=7000,skip_footer=6080)

    W, RV, F0_1, E0_1, F1_1, E1_1, F2_1, E2_1, AG1, AG1err, F_ave_w_1 = np.genfromtxt('C:\Users\Alain Lecavelier\Desktop\HST_BetaPic\Data\dat\B_10Dec.dat',unpack=True,skiprows=7000,skip_footer=6080)

    W, RV, F0_2, E0_2, F1_2, E1_2, F2_2, E2_2, F3_2, E3_2, AG2, AG2err, F_ave_w_2 = np.genfromtxt('C:\Users\Alain Lecavelier\Desktop\HST_BetaPic\Data\dat\B_24Dec.dat',unpack=True,skiprows=7000,skip_footer=6080)

    W, RV, F0_3, E0_3, F1_3, E1_3, F2_3, E2_3, F3_3, E3_3, AG3, AG3err, F_ave_w_3 = np.genfromtxt('C:\Users\Alain Lecavelier\Desktop\HST_BetaPic\Data\dat\B_30Jan.dat',unpack=True,skiprows=7000,skip_footer=6080)
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
        Fc[i] = F[i]+N[i]*np.mean(F0_0[n1:n2])      # Correct for missing flux                  ### ALAIN COMMENT ::  ADD A CONSTANT ???? Why "mean" ??
        Ec[i] = np.sqrt(E[i]**2+(N[i]*E0_0)**2)     # Increase uncertainty if flux difference is large

    # -0.8" Ly-alpha wing
    #############################################################################################    
    F1_errs                 =  np.array([Ec[1],Ec[4],Ec[8]])
    F1_ave_w, F1_ave_err    =  weighted_avg_and_std([Fc[1],Fc[4],Fc[8]], 1./F1_errs**2)         #### ERROR BARS ARE FROM DISPERSION NOT FROM HTS TABULATED ERRORS ??
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
    shift_0_l       = -450#-337.5
    shift_0_r       =  271.7
    shift_08_l      = -183.3
    shift_08_r      =  142.5
    shift_11_r      =  120

    # Plot the spectra using the cuts above and write to .dat file.
    f = open('C:\Users\Alain Lecavelier\Desktop\HST_BetaPic\Data\dat\La.dat', 'w+')
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

    ############################################
    ########################  Code Alain #######
    ############################################

    # Calculate the Correction Factor 
    C   = [CF(F0_1,E0_1,F0_0,E0_0,n1,n2),CF(F1_1,E1_1,F0_0,E0_0,n1,n2),CF(F2_1,E2_1,F0_0,E0_0,n1,n2),\
    CF(F0_2,E0_2,F0_0,E0_0,n1,n2),CF(F1_2,E1_2,F0_0,E0_0,n1,n2),CF(F2_2,E2_2,F0_0,E0_0,n1,n2),CF(F3_2,E3_2,F0_0,E0_0,n1,n2),\
    CF(F0_3,E0_3,F0_0,E0_0,n1,n2),CF(F1_3,E1_3,F0_0,E0_0,n1,n2),CF(F2_3,E2_3,F0_0,E0_0,n1,n2),CF(F3_3,E3_3,F0_0,E0_0,n1,n2)]
    
    F  = [F0_1,F1_1,F2_1,F0_2,F1_2,F2_2,F3_2,F0_3,F1_3,F2_3,F3_3]
    E  = [E0_1,E1_1,E2_1,E0_2,E1_2,E2_2,E3_2,E0_3,E1_3,E2_3,E3_3]
    Fc = [[] for _ in range(len(C))]
    Ec = [[] for _ in range(len(C))]
    for i in range(len(N)):
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
    F2, F2_err    =  weighted_avg_and_errorbars(Flux,Err)         
    #############################################################################################

    # 1.1" Ly-alpha wing
    #############################################################################################
    Flux = np.array([Fc[6],Fc[10]])
    Err  = np.array([Ec[6],Ec[10]])
    F3, F3_err    =  weighted_avg_and_errorbars(Flux,Err)         
    #############################################################################################

    # Combining 0.8" and 1.1" data of the Ly-alpha wing
    #############################################################################################
    Flux = np.array([Fc[2],Fc[5],Fc[9],Fc[6],Fc[10]])
    Err  = np.array([Ec[2],Ec[5],Ec[9],Ec[6],Ec[10]])
    F4, F4_err    =  weighted_avg_and_errorbars(Flux,Err)         
    #############################################################################################

    # Combine all data regardless of shift
    Flux = np.array([F0_0,Fc[0],Fc[1],Fc[2],Fc[3],Fc[4],Fc[5],Fc[6],Fc[7],Fc[8],Fc[9],Fc[10]])
    Err  = np.array([E0_0,Ec[0],Ec[1],Ec[2],Ec[3],Ec[4],Ec[5],Ec[6],Ec[7],Ec[8],Ec[9],Ec[10]])
    F_tot, F_tot_err    =  weighted_avg_and_errorbars(Flux,Err)         
    
    # Decide at which RV airglow affects the different measurements
    # Units in km/s
    shift_0_l       = -450#-337.5
    shift_0_r       =  271.7
    shift_08_l      = -183.3
    shift_08_r      =  142.5
    shift_11_r      =  120

    # Plot the spectra using the cuts above and write to .dat file.
    f = open('C:\Users\Alain Lecavelier\Desktop\HST_BetaPic\Data\dat\La_Alain.dat', 'w+')
    i11=0
    i12=0
    i21=0
    i22=0
    i31=0
    i32=0
    il1=0
    il2=0
    ir1=0
    ir2=0

    for j in range(len(RV)):
        if RV[j] < shift_0_l:
            if il1==0 : il1=j
            il2=j
            plt.errorbar(RV[j],F_tot[j],yerr=F_tot_err[j],color='black')
#            plt.errorbar(RV[j],F0_0[j],yerr=E0_0[j],color='black')
            print >> f, " ","{: 1.10e}".format(W[j])," "+"{: 1.10e}".format(F_tot[j])," "+"{: 1.10e}".format(F_tot_err[j])
        if RV[j] > shift_0_r:
            if ir1==0 : ir1=j
            ir2=j
            plt.errorbar(RV[j],F_tot[j],yerr=F_tot_err[j],color='black')
#            plt.errorbar(RV[j],F0_0[j],yerr=E0_0[j],color='black')
            print >> f, " ","{: 1.10e}".format(W[j])," "+"{: 1.10e}".format(F_tot[j])," "+"{: 1.10e}".format(F_tot_err[j])
        if  shift_0_l < RV[j] < shift_08_l:
            if i11==0 : i11=j
            i12=j
            plt.errorbar(RV[j],F1[j],yerr=F1_err[j],color='blue')
            print >> f, " ","{: 1.10e}".format(W[j])," "+"{: 1.10e}".format(F1[j])," "+"{: 1.10e}".format(F1_err[j])
        if  shift_08_r < RV[j] < shift_0_r:
            if i21==0 : i21=j
            i22=j
            plt.errorbar(RV[j],F2[j],yerr=F2_err[j],color='blue')
            print >> f, " ","{: 1.10e}".format(W[j])," "+"{: 1.10e}".format(F2[j])," "+"{: 1.10e}".format(F2_err[j])
        if  shift_11_r < RV[j] < shift_08_r:
            if i31==0 : i31=j
            i32=j
            plt.errorbar(RV[j],F3[j],yerr=F3_err[j],color='red')
            print >> f, " ","{: 1.10e}".format(W[j])," "+"{: 1.10e}".format(F3[j])," "+"{: 1.10e}".format(F3_err[j])
    f.close()

    print il1,il2,i11,i12,i31,i32,i21,i22,ir1,ir2    
    # Plot the region used for normalisation
    plt.step(RV[il1:il2],F_tot[il1:il2],color='green')
    plt.step(RV[ir1:ir2],F_tot[ir1:ir2],color='green')
    plt.step(RV[i11:i12],F1[i11:i12],color='green')
    plt.step(RV[i21:i22],F2[i21:i22],color='green')
    plt.step(RV[i31:i32],F3[i31:i32],color='green')

    plt.xlabel(r'RV (km/s)')
    plt.ylabel('Flux')  
    plt.xlim(-2000,800)
    plt.ylim(-0.3e-14,3.5e-14)
    plt.show()

	
if __name__ == '__main__':
    main()
