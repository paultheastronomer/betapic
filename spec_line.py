import matplotlib.pyplot as plt
import numpy as np

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
    ref_err     = replace_with_median(ref_err)
    ratio = np.average(flux[n1:n2], axis=0, weights=1./(flux_err[n1:n2]**2))/ \
            np.average(ref[n1:n2],  axis=0, weights=1./(ref_err[n1:n2]**2 ))                       
    return 1./ratio

def wave2RV(Wave,rest_wavelength,RV_BP):
    c = 299792458
    rest_wavelength = rest_wavelength*(RV_BP*1.e3)/c + rest_wavelength # Convert to beta pic reference frame
    delta_wavelength = Wave-rest_wavelength
    RV = ((delta_wavelength/rest_wavelength)*c)/1.e3	# km/s
    return RV

def replace_with_median(X):
    X[np.isnan(X)] = 0
    m = np.median(X[X > 0])
    X[X == 0] = m
    return X

def findCenter(w,l):
    for i in range(len(w)):
      if w[i] > l:
        if abs(l - w[i-1]) < abs(l - w[i]):
          ans = i-1
          break
        else:
          ans = i
          break
    return ans

def main():

    # Parameters you can change
    #====================================================
    species             = 'OV'      # Name of species
    part                = 'B'       # Part of the spectrum
    line_of_interest    = 1218.3440 # Wavelength of line
    RV_BP               = 20.5      # RV of Beta Pic
    width               = 450       # [-2*width:2*width]
    bin_pnts            = 7         # Number of points to bin
    n1                  = 620#300       # Start norm region #CII 150
    n2                  = 800#360       # End norm region   #CII 320
    #====================================================
    
    dat_directory = "/home/paw/science/betapic/data/HST/dat/"
    W, RV, F0_0, E0_0, AG0, AG0err                                                  = np.genfromtxt(dat_directory+part+"_2014.dat",unpack=True)
    W, RV, F0_1, E0_1, F1_1, E1_1, F2_1, E2_1, AG1, AG1err, F_ave_w_1               = np.genfromtxt(dat_directory+part+"_10Dec.dat",unpack=True)
    W, RV, F0_2, E0_2, F1_2, E1_2, F2_2, E2_2, F3_2, E3_2, AG2, AG2err, F_ave_w_2   = np.genfromtxt(dat_directory+part+"_24Dec.dat",unpack=True)
    W, RV, F0_3, E0_3, F1_3, E1_3, F2_3, E2_3, F3_3, E3_3, AG3, AG3err, F_ave_w_3   = np.genfromtxt(dat_directory+part+"_30Jan.dat",unpack=True)

    mid_pnt = findCenter(W,line_of_interest)

    # Only work with region of interest
    W       = W[mid_pnt-width:mid_pnt+width]
    RV      = wave2RV(W,line_of_interest,RV_BP)
    
    F0_0    = F0_0[mid_pnt-width:mid_pnt+width]
    F0_1    = F0_1[mid_pnt-width:mid_pnt+width]
    F1_1    = F1_1[mid_pnt-width:mid_pnt+width]
    F2_1    = F2_1[mid_pnt-width:mid_pnt+width]
    F0_2    = F0_2[mid_pnt-width:mid_pnt+width]
    F1_2    = F1_2[mid_pnt-width:mid_pnt+width]
    F2_2    = F2_2[mid_pnt-width:mid_pnt+width]
    F3_2    = F3_2[mid_pnt-width:mid_pnt+width]
    F0_3    = F0_3[mid_pnt-width:mid_pnt+width]
    F1_3    = F1_3[mid_pnt-width:mid_pnt+width]
    F2_3    = F2_3[mid_pnt-width:mid_pnt+width]
    F3_3    = F3_3[mid_pnt-width:mid_pnt+width]

    AG0     = AG0[mid_pnt-width:mid_pnt+width]
    AG1     = AG1[mid_pnt-width:mid_pnt+width]
    AG2     = AG2[mid_pnt-width:mid_pnt+width]
    AG3     = AG3[mid_pnt-width:mid_pnt+width]    

    E0_0    = E0_0[mid_pnt-width:mid_pnt+width]
    E0_1    = E0_1[mid_pnt-width:mid_pnt+width]
    E1_1    = E1_1[mid_pnt-width:mid_pnt+width]
    E2_1    = E2_1[mid_pnt-width:mid_pnt+width]
    E0_2    = E0_2[mid_pnt-width:mid_pnt+width]
    E1_2    = E1_2[mid_pnt-width:mid_pnt+width]
    E2_2    = E2_2[mid_pnt-width:mid_pnt+width]
    E3_2    = E3_2[mid_pnt-width:mid_pnt+width]
    E0_3    = E0_3[mid_pnt-width:mid_pnt+width]
    E1_3    = E1_3[mid_pnt-width:mid_pnt+width]
    E2_3    = E2_3[mid_pnt-width:mid_pnt+width]
    E3_3    = E3_3[mid_pnt-width:mid_pnt+width]

    AG0err  = AG0err[mid_pnt-width:mid_pnt+width]
    AG1err  = AG1err[mid_pnt-width:mid_pnt+width]
    AG2err  = AG2err[mid_pnt-width:mid_pnt+width]
    AG3err  = AG3err[mid_pnt-width:mid_pnt+width]

    # Combining AG measurements.
    AirG                = np.array([AG0,AG1,AG2,AG3])
    AirG_err            = np.array([AG0err,AG1err,AG2err,AG3err])
    AirG_W, AirG_W_err  = weighted_avg_and_errorbars(AirG,AirG_err) 

    # Calculate the Correction Factor 
    C   = [CF(F0_1,E0_1,F0_0,E0_0,n1,n2),CF(F1_1,E1_1,F0_0,E0_0,n1,n2),CF(F2_1,E2_1,F0_0,E0_0,n1,n2),\
    CF(F0_2,E0_2,F0_0,E0_0,n1,n2),CF(F1_2,E1_2,F0_0,E0_0,n1,n2),CF(F2_2,E2_2,F0_0,E0_0,n1,n2),CF(F3_2,E3_2,F0_0,E0_0,n1,n2),\
    CF(F0_3,E0_3,F0_0,E0_0,n1,n2),CF(F1_3,E1_3,F0_0,E0_0,n1,n2),CF(F2_3,E2_3,F0_0,E0_0,n1,n2),CF(F3_3,E3_3,F0_0,E0_0,n1,n2)]
    
    # Creating a list with observations which need correcting (flux offset)
    F  = [F0_1,F1_1,F2_1,F0_2,F1_2,F2_2,F3_2,F0_3,F1_3,F2_3,F3_3]
    E  = [E0_1,E1_1,E2_1,E0_2,E1_2,E2_2,E3_2,E0_3,E1_3,E2_3,E3_3]   


    Fc = [[] for _ in range(len(C))]
    Ec = [[] for _ in range(len(C))]

    for i in range(len(C)):
        Fc[i] = F[i]*C[i]   # Correct for lower efficiency
        Ec[i] = E[i]*C[i]   # accordingly correct the tabulated error bars

    Flux_10Dec  = np.array([Fc[0],Fc[1],Fc[2]])
    Err_10Dec   = np.array([Ec[0],Ec[1],Ec[2]])

    Flux_26Dec  = np.array([Fc[3],Fc[4],Fc[5],Fc[6]])
    Err_26Dec   = np.array([Ec[3],Ec[4],Ec[5],Ec[6]])

    Flux_30Jan  = np.array([Fc[7],Fc[8],Fc[9],Fc[10]])
    Err_30Jan   = np.array([Ec[7],Ec[8],Ec[9],Ec[10]])
    
    Flux_tot    = np.array([Fc[0],Fc[1],Fc[2],Fc[3],Fc[4],Fc[5],Fc[6],Fc[7],Fc[8],Fc[9],Fc[10]])
    Err_tot     = np.array([Ec[0],Ec[1],Ec[2],Ec[3],Ec[4],Ec[5],Ec[6],Ec[7],Ec[8],Ec[9],Ec[10]])    

    Flux_tot    = np.array([Fc[1],Fc[2],Fc[3],Fc[4],Fc[5],Fc[6],Fc[7],Fc[8],Fc[9],Fc[10]])
    Err_tot     = np.array([Ec[1],Ec[2],Ec[3],Ec[4],Ec[5],Ec[6],Ec[7],Ec[8],Ec[9],Ec[10]])        
    
    Flux_w_10Dec, Err_w_10Dec   =  weighted_avg_and_errorbars(Flux_10Dec,Err_10Dec)
    Flux_w_26Dec, Err_w_26Dec   =  weighted_avg_and_errorbars(Flux_26Dec,Err_26Dec)
    Flux_w_30Jan, Err_w_30Jan   =  weighted_avg_and_errorbars(Flux_30Jan,Err_30Jan)
    Flux_w_tot, Err_w_tot       =  weighted_avg_and_errorbars(Flux_tot,Err_tot)

    fig = plt.figure(figsize=(7,5))
    
    # Fancy customisation to make the plot look nice
    #================================================
    fontlabel_size  = 18
    tick_size       = 18
    params = {'backend': 'wxAgg', 'lines.markersize' : 2, 'axes.labelsize': fontlabel_size, 'font.size': fontlabel_size, 'legend.fontsize': 15, 'xtick.labelsize': tick_size, 'ytick.labelsize': tick_size, 'text.usetex': True}
    plt.rcParams.update(params)
    plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    plt.rcParams['text.usetex'] = True
    plt.rcParams['text.latex.unicode'] = True    
    #================================================

    # Plot region used to normalise
    #plt.plot([RV[n1],RV[n1]],[F0_0.min(),F0_0.max()],'--k')
    #plt.plot([RV[n2],RV[n2]],[F0_0.min(),F0_0.max()],'--k')
    
    
    plt.plot([0,0],[0,F0_0.max()],'--k')
    
    if bin_pnts > 1:
        RVb, F0_0b, E0_0b	                =	Bin_data(RV,F0_0,E0_0,bin_pnts)
        RVb,Flux_w_b_10Dec,Err_w_b_10Dec	=	Bin_data(RV,Flux_w_10Dec,Err_w_10Dec,bin_pnts)
        RVb,Flux_w_b_26Dec,Err_w_b_26Dec	=	Bin_data(RV,Flux_w_26Dec,Err_w_26Dec,bin_pnts)
        RVb,Flux_w_b_30Jan,Err_w_b_30Jan	=	Bin_data(RV,Flux_w_30Jan,Err_w_30Jan,bin_pnts)
        RVb,Flux_w_b_tot,Err_w_b_tot	    =	Bin_data(RV,Flux_w_tot,Err_w_tot,bin_pnts)
        
        RVb,AG0b,AG0errb                    =	Bin_data(RV,AG0,AG0err,bin_pnts)
        RVb,AG1b,AG1errb                    =	Bin_data(RV,AG1,AG1err,bin_pnts)
        RVb,AG2b,AG2errb                    =	Bin_data(RV,AG2,AG2err,bin_pnts)
        RVb,AG3b,AG3errb                    =	Bin_data(RV,AG3,AG3err,bin_pnts)
        RVb,AirG_W_b, AirG_W_b_err                    =	Bin_data(RV,AirG_W, AirG_W_err,bin_pnts)
        
        # Used to find the AG factors
        #plt.step(RVb,F0_0b-Flux_w_b_10Dec,color="#FF281C",label='2014')
        #plt.step(RVb,13*AirG_W_b,color="purple")
        #'''
        
        plt.text(95,4.7e-14,r'$\mathrm{O\,V}$',va='center')
        plt.text(-440,3e-15,r'$\mathrm{Airglow}$',va='center',rotation=-10)
        #plt.plot([450,550],[4e-14,3e-14],color="black")
 
        # Plot the airglow
        plt.step(RVb,AirG_W_b,color="#87CEEB",lw=2)
        
        plt.errorbar(RVb,F0_0b,yerr=E0_0b,fmt=None,ecolor='black',zorder=2)
        plt.scatter(RVb,F0_0b, marker='o',s=50, edgecolor="black",color='#FF281C',zorder=3,label=r'2014')
        
        plt.errorbar(RVb,Flux_w_b_10Dec,yerr=Err_w_b_10Dec,fmt=None,ecolor='black',zorder=2)
        plt.scatter(RVb,Flux_w_b_10Dec, marker='o',s=50, edgecolor="black",color='#FF9303',zorder=3,label=r'2015v1')

        plt.errorbar(RVb,Flux_w_b_26Dec,yerr=Err_w_b_26Dec,fmt=None,ecolor='black',zorder=2)
        plt.scatter(RVb,Flux_w_b_26Dec, marker='o',s=50, edgecolor="black",color='#0386FF',zorder=3,label=r'2015v2')

        plt.errorbar(RVb,Flux_w_b_30Jan,yerr=Err_w_b_30Jan,fmt=None,ecolor='black',zorder=2)
        plt.scatter(RVb,Flux_w_b_30Jan, marker='o',s=50, edgecolor="black",color='#00B233',zorder=3,label=r'2016')
        
        #plt.step(RVb,Flux_w_b_tot,color="black",lw=1.2,label='All data combined')
        
    else:
        plt.step(RV,F0_0,color="#FF281C",label='2014')
        plt.step(RV,Flux_w_10Dec,color="#FF9303",label='2015v1')
        plt.step(RV,Flux_w_26Dec,color="#0386FF",label='2015v2')
        plt.step(RV,Flux_w_30Jan,color="#00B233",label='2016')
        plt.step(RV,Flux_w_tot,color="black",lw=1.2,label='All data combined')
        


    # Plot the airglow
    #plt.step(RV,AirG_W,color="#87CEEB")
    
    # Place a legend in the lower right
    plt.legend(loc='upper right', numpoints=1)
    
    # Add labels to the axis
    plt.xlabel('RV [km/s]')
    plt.ylabel('Flux (erg/s/cm$^2$/\AA)')
    
    plt.ylim(0,5.5e-14)
    plt.xlim(-550,550)
    plt.minorticks_on()

    fig.tight_layout()
    # Produce a .pdf
    plt.savefig(species+'_'+str(line_of_interest)+'.pdf', bbox_inches='tight', pad_inches=0.1,dpi=300) 
    plt.show()

    # Uncomment below to save to text file
    #np.savetxt(dat_directory+"NI_20160512_v1.txt",np.column_stack((W,Flux_w_tot, Err_w_tot)))
        
    # Uncomment below to make a datafile compatible with owens.f
    '''
    f = open(dat_directory+'CII_26d.dat', 'w+')
    for i in range(len(W)):
        print >> f, " ","{: 1.10e}".format(W[i])," "+"{: 1.10e}".format(Flux_w_tot[i])," "+"{: 1.10e}".format(Err_w_tot[i])
    f.close()
    '''     

if __name__ == '__main__':
    main()
