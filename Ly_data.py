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
    Flux        = replace_with_median(Flux)
    Err         = replace_with_median(Err)
    weights     =1./(Err**2)
    average     = np.average(Flux, axis=0, weights=weights)
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
        AG      = np.concatenate((zeros,AG))[:-units]
    else:
        AG      = np.concatenate((AG,zeros))[abs(units):]
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
    W, RV, F0_0, E0_0, AG0, AG0err = np.genfromtxt('/home/paw/science/betapic/data/HST/dat/B_2014.dat',unpack=True)

    W, RV, F0_1, E0_1, F1_1, E1_1, F2_1, E2_1, AG1, AG1err, F_ave_w_1 = np.genfromtxt('/home/paw/science/betapic/data/HST/dat/B_10Dec.dat',unpack=True)

    W, RV, F0_2, E0_2, F1_2, E1_2, F2_2, E2_2, F3_2, E3_2, AG2, AG2err, F_ave_w_2 = np.genfromtxt('/home/paw/science/betapic/data/HST/dat/B_24Dec.dat',unpack=True)

    W, RV, F0_3, E0_3, F1_3, E1_3, F2_3, E2_3, F3_3, E3_3, AG3, AG3err, F_ave_w_3 = np.genfromtxt('/home/paw/science/betapic/data/HST/dat/B_30Jan.dat',unpack=True)
    ############################################################################################# 

    dat_directory = "/home/paw/science/betapic/data/HST/dat/"


    # Choose a region to normalise the spectra
    # units refer to array elements.
    n1 = 8400
    n2 = 8900
    
    # Uncomment to see region
    plt.plot(W,F0_0)
    plt.plot(W,F0_1)
    plt.plot(W[n1:n2],F0_1[n1:n2])
    plt.show()
    sys.exit()
    
    # Calculate the Correction Factor 
    C   = [CF(F0_1,E0_1,F0_0,E0_0,n1,n2),CF(F1_1,E1_1,F0_0,E0_0,n1,n2),CF(F2_1,E2_1,F0_0,E0_0,n1,n2),\
    CF(F0_2,E0_2,F0_0,E0_0,n1,n2),CF(F1_2,E1_2,F0_0,E0_0,n1,n2),CF(F2_2,E2_2,F0_0,E0_0,n1,n2),CF(F3_2,E3_2,F0_0,E0_0,n1,n2),\
    CF(F0_3,E0_3,F0_0,E0_0,n1,n2),CF(F1_3,E1_3,F0_0,E0_0,n1,n2),CF(F2_3,E2_3,F0_0,E0_0,n1,n2),CF(F3_3,E3_3,F0_0,E0_0,n1,n2)]
    
    np.savetxt(dat_directory+"rescaling_factors.txt",C)
    
    F  = [F0_1,F1_1,F2_1,F0_2,F1_2,F2_2,F3_2,F0_3,F1_3,F2_3,F3_3]
    E  = [E0_1,E1_1,E2_1,E0_2,E1_2,E2_2,E3_2,E0_3,E1_3,E2_3,E3_3]
    

    Fc = [[] for _ in range(len(C))]
    Ec = [[] for _ in range(len(C))]

    for i in range(len(C)):
        Fc[i] = F[i]*C[i]   # Correct for lower efficiency
        Ec[i] = E[i]*C[i]   # accordingly correct the tabulated error bars

    
    # -0.8" Ly-alpha wing
    #############################################################################################    
    # For all data uncomment the two lines below
    Flux = np.array([Fc[1],Fc[4],Fc[8]])
    Err  = np.array([Ec[1],Ec[4],Ec[8]])
    
    # For 10 Dec data uncomment the two lines below
    #Flux = np.array([Fc[1]])
    #Err  = np.array([Ec[1]])

    # For 24 Dec data uncomment the two lines below
    #Flux = np.array([Fc[4]])
    #Err  = np.array([Ec[4]])

    # For 30 Jan data uncomment the two lines below
    #Flux = np.array([Fc[8]])
    #Err  = np.array([Ec[8]])
    
    F1, F1_err    =  weighted_avg_and_errorbars(Flux,Err)         

    #############################################################################################    

    # 0.8" Ly-alpha wing
    #############################################################################################
    # For all data uncomment the two lines below
    Flux = np.array([Fc[2],Fc[9]])
    Err  = np.array([Ec[2],Ec[9]])
    
    # Fc[5] shows strange "emission feature" not consistent with the 1.1" offset data
    # and has thus been removed. To include it uncomment the two lines below.
    #Flux = np.array([Fc[2],Fc[5],Fc[9]])
    #Err  = np.array([Ec[2],Ec[5],Ec[9]])

    # For 10 Dec data uncomment the two lines below
    #Flux = np.array([Fc[2]])
    #Err  = np.array([Ec[2]])

    # For 24 Dec data uncomment the two lines below
    #Flux = np.array([Fc[5]])
    #Err  = np.array([Ec[5]])

    # For 30 Jan data uncomment the two lines below
    #Flux = np.array([Fc[9]])
    #Err  = np.array([Ec[9]])

    F2, F2_err    =  weighted_avg_and_errorbars(Flux,Err)         
    #############################################################################################

    # 1.1" Ly-alpha wing
    #############################################################################################
    # For all data uncomment the two lines below
    Flux = np.array([Fc[6],Fc[10]])
    Err  = np.array([Ec[6],Ec[10]])

    # For 24 Dec data uncomment the two lines below    
    #Flux = np.array([Fc[6]])
    #Err  = np.array([Ec[6]])

    # For 30 Jan data uncomment the two lines below    
    #Flux = np.array([Fc[10]])
    #Err  = np.array([Ec[10]])
    
    F3, F3_err    =  weighted_avg_and_errorbars(Flux,Err)         
    #############################################################################################

    Flux = np.array([F0_0,Fc[0],Fc[1],Fc[2],Fc[3],Fc[4],Fc[5],Fc[6],Fc[7],Fc[8],Fc[9],Fc[10]])
    Err  = np.array([E0_0,Ec[0],Ec[1],Ec[2],Ec[3],Ec[4],Ec[5],Ec[6],Ec[7],Ec[8],Ec[9],Ec[10]])

    # For 2014 data see line straight after "F_tot, F_tot_err"
    
    # For 10 Dec data uncomment the two lines below
    #Flux = np.array([Fc[0],Fc[1],Fc[2]])
    #Err  = np.array([Ec[0],Ec[1],Ec[2]])

    # For 24 Dec data uncomment the two lines below
    #Flux = np.array([Fc[3],Fc[4],Fc[5],Fc[6]])
    #Err  = np.array([Ec[3],Ec[4],Ec[5],Ec[6]])

    # For 30 Jan data uncomment the two lines below
    #Flux = np.array([Fc[7],Fc[8],Fc[9],Fc[10]])
    #Err  = np.array([Ec[7],Ec[8],Ec[9],Ec[10]])
    
    F_tot, F_tot_err    =  weighted_avg_and_errorbars(Flux,Err)
    #F_tot, F_tot_err    =  F0_0, E0_0
    #############################################################################################


    # Combining AG measurements. Not including AG2 due to problem with data.
    AirG                = np.array([AG0,AG1,AG3])
    AirG_err            = np.array([AG0err,AG1err,AG3err])
    AirG_W, AirG_W_err  = weighted_avg_and_errorbars(AirG,AirG_err) 
    
    AG1    = ShiftAG(AirG_W,-28)    #2015v1 +0.8" AG
    AG2    = ShiftAG(AirG_W,-32)    #2015v2 +0.8" AG
    AG3    = ShiftAG(AirG_W,-25)    #2016 +0.8" AG

    AG4    = ShiftAG(AirG_W,-42)    #2015v2 +1.1" AG
    AG5    = ShiftAG(AirG_W,-43)    #2016 +1.1" AG

    AG6    = ShiftAG(AirG_W,32)     #2015v1 -0.8" AG
    AG7    = ShiftAG(AirG_W,35)     #2015v2 -0.8" AG
    AG8    = ShiftAG(AirG_W,33)     #2016 -0.8" AG


    # 0.0"
    #plt.plot(RV,AirG_W)
    #plt.plot(RV,E0_0/2)
    #plt.plot(RV,E0_1/2)
    #plt.plot(RV,E0_2/2)
    #plt.plot(RV,E0_3/2)

    # -0.8"
    #plt.plot(RV,AG6)
    #plt.plot(RV,AG7)
    #plt.plot(RV,AG8)
    #plt.plot(RV,F1_err/2)    

    # +0.8"    
    #plt.plot(RV,AG1)
    #plt.plot(RV,AG2)
    #plt.plot(RV,AG3)
    #plt.plot(RV,F2_err/2)    


    #plt.plot(RV,Fc[5],color="blue")
    #plt.plot(RV,Fc[6],color="green")

    #plt.plot(RV,Fc[9],color="blue")
    #plt.plot(RV,Fc[10],color="green")
    
    # +1.1"    
    #plt.plot(RV,AG4)
    #plt.plot(RV,AG5)
    #plt.plot(RV,AG6)
    #plt.plot(RV,F3_err/2)  
        
    #plt.plot(RV,AG1,color="blue")
    #plt.plot(RV,AG2,color="red")
    #plt.plot(RV,AG3,color="green")
    
    #plt.plot(RV,AG4,color="orange")
    #plt.plot(RV,AG5,color="purple")

    #plt.plot(RV,AG6,color="blue")
    #plt.plot(RV,AG7,color="red")
    #plt.plot(RV,AG8,color="green")
    
    #plt.plot(RV,AirG_W,color="purple")
    
    #plt.errorbar(W[start:stop],F0_0[start:stop],yerr=E0_0[start:stop],color="red")
    #plt.xlim(50,500)
    #plt.xlim(-300,400)
    #plt.ylim(0,2e-14)
    #plt.show()
    #sys.exit()


    # Decide at which RV airglow affects the different measurements
    # Units in km/s
    shift_0_l       = -240#-365#337.5 350
    shift_0_r       =  230#295#315#271.7 295
    shift_08_l      = -145#-164#shift_0_l+68#-183.3  ok
    shift_08_r      = 150#135#143#shift_0_r-68#157#142.5 140!
    shift_11_r      = 115#115#113#shift_0_r-94# 108 ok
    
    f = open(dat_directory+'Ly-alpha_no_AG.txt', 'w+')
    for j in range(len(RV)):
        # Save 0.0" shift data
        if RV[j] < shift_0_l:
            plt.plot(RV[j],F_tot[j], marker='o', color='black')
            print >> f, " ","{: 1.10e}".format(W[j])," "+"{: 1.10e}".format(F_tot[j])," "+"{: 1.10e}".format(F_tot_err[j])       
        if RV[j] > shift_0_r:
            plt.plot(RV[j],F_tot[j], marker='o', color='black')
            print >> f, " ","{: 1.10e}".format(W[j])," "+"{: 1.10e}".format(F_tot[j])," "+"{: 1.10e}".format(F_tot_err[j])
        # Save 0.8" shift data
        if  shift_0_l < RV[j] < shift_08_l:
            #plt.errorbar(RV[j],F1[j],yerr=F1_err[j],color='green')
            plt.plot(RV[j],F1[j], marker='o', color='b')
            print >> f, " ","{: 1.10e}".format(W[j])," "+"{: 1.10e}".format(F1[j])," "+"{: 1.10e}".format(F1_err[j])
        if  shift_08_r < RV[j] < shift_0_r:
            #plt.errorbar(RV[j],F2[j],yerr=F2_err[j],fmt='',color='green')
            plt.plot(RV[j],F2[j], marker='o', color='b')
            print >> f, " ","{: 1.10e}".format(W[j])," "+"{: 1.10e}".format(F2[j])," "+"{: 1.10e}".format(F2_err[j])
        #'''
        # Save 1.1" shift data
        if  shift_11_r < RV[j] < shift_08_r:
            #plt.errorbar(RV[j],F3[j],yerr=F3_err[j],fmt='',color='red')
            plt.plot(RV[j],F3[j], marker='o', color='r')
            print >> f, " ","{: 1.10e}".format(W[j])," "+"{: 1.10e}".format(F3[j])," "+"{: 1.10e}".format(F3_err[j])
        #'''
    f.close()

    plt.xlim(-1000,1000)
    plt.xlabel(r'RV (km/s)')
    plt.ylabel('Flux')
    plt.show()

if __name__ == '__main__':
    main()
