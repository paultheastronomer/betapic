import numpy as np
import matplotlib.pyplot as plt
import sys

from src.calculations import Calc
c   = Calc()

def main():    

    dat_directory = "/home/paw/science/betapic/data/HST/dat/" 

    rest_wavelength = 1215.6702 
    RV_BP           = 20.5
    bin_pnts        = 3

    W, RV, F0_0, E0_0, AG0, AG0err                                                  = np.genfromtxt(dat_directory+'B_2014.dat',unpack=True)
    W, RV, F0_1, E0_1, F1_1, E1_1, F2_1, E2_1, AG1, AG1err, F_ave_w_1               = np.genfromtxt(dat_directory+'B_10Dec.dat',unpack=True)
    W, RV, F0_2, E0_2, F1_2, E1_2, F2_2, E2_2, F3_2, E3_2, AG2, AG2err, F_ave_w_2   = np.genfromtxt(dat_directory+'B_24Dec.dat',unpack=True)
    W, RV, F0_3, E0_3, F1_3, E1_3, F2_3, E2_3, F3_3, E3_3, AG3, AG3err, F_ave_w_3   = np.genfromtxt(dat_directory+'B_30Jan.dat',unpack=True)

    #Wc, Fc, Ec  = np.genfromtxt(dat_directory+'Ly-alpha.dat',skiprows=830,skip_footer=0,unpack=True)
    W0, F0, E0  = np.genfromtxt(dat_directory+'Ly-alpha_B_2014.dat',skip_header=8500,skip_footer=6500,unpack=True)
    W1, F1, E1  = np.genfromtxt(dat_directory+'Ly-alpha_B_10Dec.dat',skip_header=8500,skip_footer=6500,unpack=True)
    W2, F2, E2  = np.genfromtxt(dat_directory+'Ly-alpha_B_24Dec.dat',skip_header=8500,skip_footer=6500,unpack=True)
    W3, F3, E3  = np.genfromtxt(dat_directory+'Ly-alpha_B_30Jan.dat',skip_header=8500,skip_footer=6500,unpack=True)
    
    Ws, Fs, Es  = np.genfromtxt(dat_directory+'Ly_sky_subtracted_no_central_data_2016_06_21.txt',unpack=True,skip_header=8850,skip_footer= 7110)
    RVs         = c.Wave2RV(Ws,rest_wavelength,RV_BP)
    
    Wo, Fo, Eo  = np.genfromtxt(dat_directory+'Ly-alpha_no_AG_2016_06_23.txt',unpack=True,skip_header=8000,skip_footer= 6000)
    RVo         = c.Wave2RV(Wo,rest_wavelength,RV_BP)      

    AirG                = np.array([AG0,AG1,AG2,AG3])
    AirG_err            = np.array([AG0err,AG1err,AG2err,AG3err])
    AirG_W, AirG_W_err  = c.WeightedAvg(AirG,AirG_err) 
    
    RV0             = c.Wave2RV(W0,rest_wavelength,RV_BP)
    RV1             = c.Wave2RV(W1,rest_wavelength,RV_BP)
    RV2             = c.Wave2RV(W2,rest_wavelength,RV_BP)
    RV3             = c.Wave2RV(W3,rest_wavelength,RV_BP)

    RV0b, F0b, E0b  =   c.BinData(RV0,F0,E0,bin_pnts)
    RV1b, F1b, E1b  =   c.BinData(RV1,F1,E1,bin_pnts)
    RV2b, F2b, E2b  =   c.BinData(RV2,F2,E2,bin_pnts)
    RV3b, F3b, E3b  =   c.BinData(RV3,F3,E3,bin_pnts)

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
    AGs    = c.ShiftAG(AG0,-1)
    ax1.errorbar(RV0b,F0b/norm,yerr=E0b/norm,color='black')
    ax1.scatter(RV0b,F0b/norm,color='#FF281C',s=25,edgecolor='k',label='2014',zorder=3)
    ax1.fill_between(RV, -1, AG0/norm,color="#D3D3D3")
    ax1.plot(RV,AG0/norm,color="#545454")
    #ax1.plot(RVs,Fs/norm,color="black")
    #ax1.plot(RVo,Fo/norm,color="red")
    ax1.text(40,2.5,'Airglow',ha='left')
    ax1.text(435,4,'2014',ha='center')
    ax1.text(435,2.5,'no shift',ha='center')

    AGs    = c.ShiftAG(AG1,-28)
    ax2.errorbar(RV1b,F1b/norm,yerr=E1b/norm,color='black')
    ax2.scatter(RV1b,F1b/norm,color='#FF9303',s=25,edgecolor='k',label='2015v1',zorder=3)
    ax2.fill_between(RV, -1, AGs/norm,color="#D3D3D3")
    ax2.plot(RV,AGs/norm,color="#545454")
    #ax2.plot(RVs,Fs/norm,color="black")
    #ax2.plot(RVo,Fo/norm,color="red")
    ax2.text(7,3,'Airglow',ha='left',rotation='-40')
    ax2.text(435,4,'2015v1',ha='center')
    ax2.text(435,2.8,r'$+0.8$" shift',ha='center')

    AGs    = c.ShiftAG(AG2,-41)
    ax3.errorbar(RV2b,F2b/norm,yerr=E2b/norm,color='black')
    ax3.scatter(RV2b,F2b/norm,color='#0386FF',s=25,edgecolor='k',label='2015v2',zorder=3)
    ax3.fill_between(RV, -1, AGs/norm,color="#D3D3D3")
    ax3.plot(RV,AGs/norm,color="#545454")
    #ax3.plot(RVs,Fs/norm,color="black")
    #ax3.plot(RVo,Fo/norm,color="red")
    ax3.text(4,3,'Airglow',ha='left',rotation='-55')
    ax3.text(435,4,'2015v2',ha='center')
    ax3.text(435,2.8,r'$+1.1$" shift',ha='center')

    AGs    = c.ShiftAG(AG3,-42)
    ax4.errorbar(RV3b,F3b/norm,yerr=E3b/norm,color='black')
    ax4.scatter(RV3b,F3b/norm,color='#00B233',s=25,edgecolor='k',label='2016',zorder=3)
    ax4.fill_between(RV, -1, AGs/norm,color="#D3D3D3")
    ax4.plot(RV,AGs/norm,color="#545454")
    #ax4.plot(RVs,Fs/norm,color="black")
    #ax4.plot(RVo,Fo/norm,color="red")
    ax4.text(4,3,'Airglow',ha='left',rotation='-55')
    ax4.text(435,4,'2016',ha='center')
    ax4.text(435,2.8,r'$+1.1$" shift',ha='center')

    fig.text(0.03, 0.5, r'Flux$/1\times10^{-14}$ (erg/s/cm$^2$/\AA)', ha='center', va='center', rotation='vertical')
    plt.xlabel(r'RV [km/s]')
    plt.xlim(0,580)
    plt.ylim(0,5.5)

    plt.minorticks_on()
    #plt.legend(loc='upper right', numpoints=1)
    
    fig.tight_layout()
    plt.subplots_adjust(left=0.10)
    plt.savefig('../plots/Ly_only_cut.pdf', bbox_inches='tight', pad_inches=0.1,dpi=300)
    plt.show()

if __name__ == '__main__':
    main()
