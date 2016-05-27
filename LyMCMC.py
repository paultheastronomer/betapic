#!/usr/bin/env python
import numpy as np
import sys

from src.statistics import Stats
from src.model import Model
from src.mcmc import MCMC

s   = Stats()
m   = Model()
mc  = MCMC() 

def wave2RV(Wave,rest_wavelength,RV_BP):
    c = 299792458
    rest_wavelength = rest_wavelength*(RV_BP*1.e3)/c + rest_wavelength # Convert to beta pic reference frame
    delta_wavelength = Wave-rest_wavelength
    RV = ((delta_wavelength/rest_wavelength)*c)/1.e3	# km/s
    return RV

def main():    

    dat_directory   = "/nethome/pwilson/betapic/data/HST/dat/"

    Wo, Fo, Eo      = np.genfromtxt(dat_directory+'Ly_sky_subtracted_no_central_data_2016_05_26.txt',unpack=True,skip_header=8300,skip_footer= 6700)
    W, F, E         = np.genfromtxt(dat_directory+'Ly_sky_subtracted_no_central_data_2016_05_26.txt',unpack=True,skip_header=8850,skip_footer= 7110)
    
    # To fit the non-sky subtracted (only cut) data uncomment the two lines below.
    #Wo, Fo, Eo         = np.genfromtxt(dat_directory+'Ly-alpha_no_AG_2016_05_26.txt',unpack=True,skip_header=8000,skip_footer= 6000)
    #W, F, E         = np.genfromtxt(dat_directory+'Ly-alpha_no_AG_2016_05_26.txt',unpack=True,skip_header=8859,skip_footer= 7102)
    

    
    ### Parameters ##############################
    ModelType   = 2         # best model is 2
    mode        = 'mcmc'      # mcmc or lm
    LyA         = 1215.6702 # Heliocentric: 1215.6702
    BetaPicRV   = 20.5

    # ISM parameters
    v_ism       = 10.0      # RV of the ISM (relative to Heliocentric)  
    nh_ism      = 18.4      # Column density ISM
    b_ism       = 7.        # Turbulent velocity
    T_ism       = 7000.     # Temperature of ISM

    # Beta Pic parameters
    v_bp        = 33.5     #20.5# RV of the beta Pic (relative to Heliocentric)
    nh_bp       = 19.4    # Column density beta Pic, Fitting param
    b_bp        = 7.0       # Turbulent velocity
    T_bp        = 1000.     # Temperture of gas in beta Pic disk

    # Extra component parameters
    v_X         = 40.0      # RV of the beta Pic (relative to Heliocentric)
    nh_X        = 19.0      # Column density beta Pic, Fitting param
    b_X         = 6.0       # Turbulent velocity
    T_X         = 1000.     # Temperture of gas in beta Pic disk

    # Stellar emission line parameters
    max_f       = 9.79e-12 # Fitting param 
    dp          = 0.0 
    uf          = 2.94     # Fitting param
    av          = 0.035     # Fitting param

    slope       = -0.0008205

    sigma_kernel= 3.5

    v   = np.arange(-len(Wo),len(Wo),1) # RV values
    l   = LyA*(1.0 + v/3e5)                     # Corresponding wavengths
    vBP= wave2RV(l,LyA,BetaPicRV) 

    RV  = wave2RV(W,LyA,BetaPicRV)     # BetaPic rest frame
    RVo = wave2RV(Wo,LyA,BetaPicRV)
    
    if ModelType == 1:
        Par     = [nh_bp,max_f,uf,av,slope] # Free parameters
        Const   = [W,l,LyA,BetaPicRV,sigma_kernel,dp,v_ism,nh_ism,b_ism,T_ism,v_bp,b_bp,T_bp]
        step= np.array([0.1,2e-12,0.03,.001,1e-4])/3.
    if ModelType == 2:
        Par     = [nh_bp,max_f,uf,av,v_bp] # Free parameters
        Const   = [W,l,LyA,BetaPicRV,sigma_kernel,dp,v_ism,nh_ism,b_ism,T_ism,b_bp,T_bp]
        step= np.array([0.01,1e-12,0.04,.004,0.8])/2.
    if ModelType == 3:
        Par     = [nh_bp,max_f,uf,av,v_X,nh_X] # Free parameters
        Const   = [W,l,LyA,BetaPicRV,sigma_kernel,dp,v_ism,nh_ism,b_ism,T_ism,v_bp,b_bp,T_bp,b_X,T_X]
        step= np.array([0.1,2e-12,0.03,.001,5,0.1])/3.
    if ModelType == 4:
        Par     = [nh_bp,max_f,uf,av] # Free parameters
        Const   = [W,l,LyA,BetaPicRV,sigma_kernel,dp,v_ism,nh_ism,b_ism,T_ism,v_bp,b_bp,T_bp]
        step= np.array([0.1,2e-12,0.03,.001,5,0.1])/3.
    #############################################


    X = F, E, m.LyModel(Par,Const,ModelType)[0]

    chain, moves = mc.McMC(W,X,m.LyModel, ModelType, Par, Const, step,1e5)
    
    outfile = 'chains/chain_s_'+sys.argv[1]
    if ModelType == 2:
        np.savez(outfile, nh_bp = chain[:,0], max_f = chain[:,1], uf = chain[:,2], av = chain[:,3], v_H = chain[:,4])
    
    Pout = chain[moves,:]
    P_plot1 = [0,1]
    P_plot2 = [2,3]
    P_plot3 = [3,4]
    PU1 = mc.Median_and_Uncertainties(P_plot1,step,chain)
    PU2 = mc.Median_and_Uncertainties(P_plot2,step,chain)
    PU3 = mc.Median_and_Uncertainties(P_plot3,step,chain)
    
    print "\nlog(N(H)) =\t" ,PU1[0][0],"\t+",PU1[1][0],"\t-",PU1[2][0]
    print "Fmax =\t\t"      ,PU1[0][1],"\t+",PU1[1][1],"\t-",PU1[2][1]
    print "uf=\t\t"         ,PU2[0][0],"\t+",PU2[1][0],"\t-",PU2[2][0]
    print "av=\t\t"         ,PU2[0][1],"\t+",PU2[1][1],"\t-",PU2[2][1]
    print "V_H=\t\t"       ,PU3[0][1],"\t+",PU3[1][1],"\t-",PU3[2][1]

if __name__ == '__main__':
    main()

