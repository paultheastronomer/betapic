import matplotlib.pyplot as plt
import numpy as np

from src.statistics import Stats
from src.calculations import Calc

s   = Stats()
c   = Calc()


def main():

    # Parameters you can change
    #===============================================================
    species             = 'N'       # Name of species
    part                = 'B'       # Part of the spectrum
    line_of_interest    = 1199.5496#1134.1653 # Wavelength of line
    RV_BP               = 20.5      # RV of Beta Pic
    width               = 450       # [-2*width:2*width]
    bin_pnts            = 5         # Number of points to bin
    n1                  = 650       # Start of noramilsation region
    n2                  = 850       # End of noramilsation region
    #===============================================================
    
    dat_directory = "/home/paw/science/betapic/data/HST/dat/"

    W, RV, F0_0, E0_0, AG0, AG0err                                                  = np.genfromtxt(dat_directory+'B_2014.dat',unpack=True)
    W, RV, F0_1, E0_1, F1_1, E1_1, F2_1, E2_1, AG1, AG1err, F_ave_w_1               = np.genfromtxt(dat_directory+'B_10Dec.dat',unpack=True)
    W, RV, F0_2, E0_2, F1_2, E1_2, F2_2, E2_2, F3_2, E3_2, AG2, AG2err, F_ave_w_2   = np.genfromtxt(dat_directory+'B_24Dec.dat',unpack=True)
    W, RV, F0_3, E0_3, F1_3, E1_3, F2_3, E2_3, F3_3, E3_3, AG3, AG3err, F_ave_w_3   = np.genfromtxt(dat_directory+'B_30Jan.dat',unpack=True)


    mid_pnt = c.FindCenter(W,line_of_interest)

    # Only work with region of interest
    W       = W[mid_pnt-width:mid_pnt+width]
    RV      = c.wave2RV(W,line_of_interest,RV_BP)
    
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
    AirG_W, AirG_W_err  = c.WeightedAvg(AirG,AirG_err) 

    # Calculate the Correction Factor 
    C   = [c.CF(F0_1,E0_1,F0_0,E0_0,n1,n2),c.CF(F1_1,E1_1,F0_0,E0_0,n1,n2),c.CF(F2_1,E2_1,F0_0,E0_0,n1,n2),\
    c.CF(F0_2,E0_2,F0_0,E0_0,n1,n2),c.CF(F1_2,E1_2,F0_0,E0_0,n1,n2),c.CF(F2_2,E2_2,F0_0,E0_0,n1,n2),c.CF(F3_2,E3_2,F0_0,E0_0,n1,n2),\
    c.CF(F0_3,E0_3,F0_0,E0_0,n1,n2),c.CF(F1_3,E1_3,F0_0,E0_0,n1,n2),c.CF(F2_3,E2_3,F0_0,E0_0,n1,n2),c.CF(F3_3,E3_3,F0_0,E0_0,n1,n2)]
    
    # Creating a list with observations which need correcting (flux offset)
    F  = [F0_1,F1_1,F2_1,F0_2,F1_2,F2_2,F3_2,F0_3,F1_3,F2_3,F3_3]
    E  = [E0_1,E1_1,E2_1,E0_2,E1_2,E2_2,E3_2,E0_3,E1_3,E2_3,E3_3]   



    Fc = [[] for _ in range(len(C))]
    Ec = [[] for _ in range(len(C))]
    
    
    E0_0    = c.ReplaceWithMedian(E0_0)
    for i in range(len(C)):
        Fc[i] = F[i]*C[i]   # Correct for lower efficiency    
        E[i]  = c.ReplaceWithMedian(E[i]) 
        #Ec[i] = np.sqrt((C[i]*E[i])**2 +E0_0**2)  
        Ec[i] = E[i]*C[i]   # accordingly correct the tabulated error bars
        
    Flux_10Dec  = np.array([Fc[0],Fc[1],Fc[2]])
    Err_10Dec   = np.array([Ec[0],Ec[1],Ec[2]])

    Flux_26Dec  = np.array([Fc[3],Fc[4],Fc[5],Fc[6]])
    Err_26Dec   = np.array([Ec[3],Ec[4],Ec[5],Ec[6]])

    Flux_30Jan  = np.array([Fc[7],Fc[8],Fc[9],Fc[10]])
    Err_30Jan   = np.array([Ec[7],Ec[8],Ec[9],Ec[10]])
    
    factor      = 9
    Fcs         = Fc[0] - factor*AirG_W
    Ecs         = np.sqrt(Ec[0]**2+(factor*AirG_W_err)**2)
    
    Flux_tot    = np.array([Fcs,Fc[1],Fc[2],Fc[3],Fc[4],Fc[5],Fc[6],Fc[7],Fc[8],Fc[9],Fc[10]])
    Err_tot     = np.array([Ecs,Ec[1],Ec[2],Ec[3],Ec[4],Ec[5],Ec[6],Ec[7],Ec[8],Ec[9],Ec[10]])

    Flux_tot    = np.array([Fcs,Fc[1],Fc[2],Fc[3],Fc[4],Fc[5],Fc[6],Fc[7],Fc[8],Fc[9],Fc[10]])
    Err_tot     = np.array([Ecs,Ec[1],Ec[2],Ec[3],Ec[4],Ec[5],Ec[6],Ec[7],Ec[8],Ec[9],Ec[10]])


    
    #print Err_tot
    
    #sys.exit()

    #Flux_tot    = np.array([Fc[1],Fc[2],Fc[3],Fc[4],Fc[5],Fc[6],Fc[7],Fc[8],Fc[9],Fc[10]])
    #Err_tot     = np.array([Ec[1],Ec[2],Ec[3],Ec[4],Ec[5],Ec[6],Ec[7],Ec[8],Ec[9],Ec[10]])        
    
    Flux_w_10Dec, Err_w_10Dec   =  c.WeightedAvg(Flux_10Dec,Err_10Dec)
    Flux_w_26Dec, Err_w_26Dec   =  c.WeightedAvg(Flux_26Dec,Err_26Dec)
    Flux_w_30Jan, Err_w_30Jan   =  c.WeightedAvg(Flux_30Jan,Err_30Jan)
    Flux_w_tot, Err_w_tot       =  c.WeightedAvg(Flux_tot,Err_tot)

    fig = plt.figure(figsize=(12,8))
    
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
    plt.plot([W[n1],W[n1]],[F0_0.min(),F0_0.max()],'--k')
    plt.plot([W[n2],W[n2]],[F0_0.min(),F0_0.max()],'--k')
    
    
    #plt.plot([0,0],[0,F0_0.max()],'--k')
    
    if bin_pnts > 1:
        Wb, F0_0b, E0_0b	            =	c.BinData(W,F0_0,E0_0,bin_pnts)
        Wb,Flux_w_b_10Dec,Err_w_b_10Dec =	c.BinData(W,Flux_w_10Dec,Err_w_10Dec,bin_pnts)
        Wb,Flux_w_b_26Dec,Err_w_b_26Dec =	c.BinData(W,Flux_w_26Dec,Err_w_26Dec,bin_pnts)
        Wb,Flux_w_b_30Jan,Err_w_b_30Jan =	c.BinData(W,Flux_w_30Jan,Err_w_30Jan,bin_pnts)
        Wb,Flux_w_b_tot,Err_w_b_tot	    =	c.BinData(W,Flux_w_tot,Err_w_tot,bin_pnts)
        
        Wb,AG0b,AG0errb                 =	c.BinData(W,AG0,AG0err,bin_pnts)
        Wb,AG1b,AG1errb                 =	c.BinData(W,AG1,AG1err,bin_pnts)
        Wb,AG2b,AG2errb                 =	c.BinData(W,AG2,AG2err,bin_pnts)
        Wb,AG3b,AG3errb                 =	c.BinData(W,AG3,AG3err,bin_pnts)
        Wb,AirG_W_b, AirG_W_b_err       =	c.BinData(W,AirG_W, AirG_W_err,bin_pnts)
        
        F0_sky_sub = F0_0b-9*AirG_W_b
        
        
        #plt.errorbar(RVb,F0_0b,yerr=E0_0b,fmt=None,ecolor='black',zorder=2)
        #plt.scatter(RVb,F0_0b, marker='o',s=50, edgecolor="black",color='#FF281C',zorder=3,label=r'2014')
        #plt.step(Wb,F0_0b,lw=1.5,color='#FF281C',zorder=3,label=r'2014')
        plt.step(Wb,AG0b,lw=1.5,color='#FF281C',alpha=0.5,zorder=3,label=r'2014')
        plt.step(Wb,F0_sky_sub,lw=1.5,color='#FF281C',zorder=3,label=r'2014')
        
        #plt.errorbar(RVb,Flux_w_b_10Dec,yerr=Err_w_b_10Dec,fmt=None,ecolor='black',zorder=2)
        #plt.scatter(RVb,Flux_w_b_10Dec, marker='o',s=50, edgecolor="black",color='#FF9303',zorder=3,label=r'2015v1')
        plt.step(Wb,Flux_w_b_10Dec,lw=1.5,color='#FF9303',zorder=3,label=r'2015v1')
        plt.step(Wb,AG1b,lw=1.5,color='#FF9303',alpha=0.5,zorder=3,label=r'2015v1')

        #plt.errorbar(RVb,Flux_w_b_26Dec,yerr=Err_w_b_26Dec,fmt=None,ecolor='black',zorder=2)
        #plt.scatter(RVb,Flux_w_b_26Dec, marker='o',s=50, edgecolor="black",color='#0386FF',zorder=3,label=r'2015v2')
        plt.step(Wb,Flux_w_b_26Dec,lw=1.5,color='#0386FF',zorder=3,label=r'2015v2')
        plt.step(Wb,AG2b,lw=1.5,color='#0386FF',alpha=0.5,zorder=3,label=r'2015v2')
        #plt.step(RVb,Flux_w_b_26Dec-0.8*AirG_W_b,lw=1.5,color='#0386FF',zorder=3,label=r'2015v2')

        #plt.errorbar(RVb,Flux_w_b_30Jan,yerr=Err_w_b_30Jan,fmt=None,ecolor='black',zorder=2)
        #plt.scatter(RVb,Flux_w_b_30Jan, marker='o',s=50, edgecolor="black",color='#00B233',zorder=3,label=r'2016')
        #plt.step(RVb,Flux_w_b_30Jan,lw=1.5,color='#00B233',zorder=3,label=r'2016')
        plt.step(Wb,Flux_w_b_30Jan-AirG_W_b,lw=1.5,color='#00B233',zorder=3,label=r'2016')
        plt.step(Wb,AG3b,lw=1.5,color='#00B233',alpha=0.5,zorder=3,label=r'2016')
        
        plt.step(Wb,Flux_w_b_tot,lw=1.5,color='black',alpha=0.5,zorder=3,label=r'Total')
        
        #plt.step(RVb,Flux_w_b_tot,color="black",lw=1.2,label='All data combined')
        
    else:
        plt.step(RV,F0_0,color="#FF281C",label='2014')
        plt.step(RV,Flux_w_10Dec,color="#FF9303",label='2015v1')
        plt.step(RV,Flux_w_26Dec,color="#0386FF",label='2015v2')
        plt.step(RV,Flux_w_30Jan,color="#00B233",label='2016')
        plt.step(RV,Flux_w_tot,color="black",lw=1.2,label='All data combined')
        
    
    # Add labels to the axis
    plt.xlabel('RV [km/s]')
    plt.ylabel('Flux (erg/s/cm$^2$/\AA)')
    
    plt.ylim(0.0,2.2e-14)
    #plt.xlim(1198.6,1202)  # For NI at 1999
    #plt.xlim(1133,1137)
    plt.minorticks_on()

    fig.tight_layout()
    # Produce a .pdf
    #plt.savefig('../plots/'+species+'_'+str(line_of_interest)+'.png', bbox_inches='tight', pad_inches=0.1,dpi=300) 
    plt.show()

    # Uncomment below to save to text file
    #np.savetxt(dat_directory+"NI_20160512_v1.txt",np.column_stack((W,Flux_w_tot, Err_w_tot)))
        
    # Uncomment below to make a datafile compatible with owens.f
    #'''
    f = open(dat_directory+'N_1999.dat', 'w+')
    for i in range(len(W)):
        print >> f, " ","{: 1.10e}".format(W[i])," "+"{: 1.10e}".format(Flux_w_tot[i])," "+"{: 1.10e}".format(Err_w_tot[i])
    f.close()
    #'''     

if __name__ == '__main__':
    main()
