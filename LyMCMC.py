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
    #dat_directory   = "/home/paw/science/betapic/data/HST/dat/"
    Wa, Fa, Ea      = np.genfromtxt(dat_directory+'Ly-alpha_no_AG_2016_08_24.txt',unpack=True,skip_header=7500,skip_footer= 4500)
    Wo, Fo, Eo      = np.genfromtxt(dat_directory+'Ly_sky_subtracted_no_central_data_2016_09_14.txt',unpack=True,skip_header=8300,skip_footer= 6700)
    W, F, E         = np.genfromtxt(dat_directory+'Ly_sky_subtracted_no_central_data_2016_09_14.txt',unpack=True,skip_header=9027,skip_footer= 7155)
    
    x1  = 920
    x2  = 1200
    x3  = 2050
    x4  = -1250
    
    fitting_region_x  = np.concatenate((Wa[x1:x2],Wa[x3:x4]))
    fitting_region_y  = np.concatenate((Fa[x1:x2],Fa[x3:x4]))
    fitting_region_err= np.concatenate((Ea[x1:x2],Ea[x3:x4]))

    
    order = 4
    weights = 1/fitting_region_err**2
    z       = np.polyfit(fitting_region_x, fitting_region_y, 3, rcond=None, full=False, w=weights)
    pn = np.poly1d(z)
    
    '''
    plt.step(Wa,Fa)
    plt.step(fitting_region_x,fitting_region_y)
    plt.plot(Wa,pn(Wa))
    #plt.step(Wa[920:1200],Fa[920:1200])
    #plt.step(Wa[2050:-650],Fa[2050:-650])
    plt.show()
    sys.exit()
    '''
    

    # To fit the non-sky subtracted (only cut) data uncomment the two lines below.
    #Wo, Fo, Eo         = np.genfromtxt(dat_directory+'Ly-alpha_no_AG_2016_06_23.txt',unpack=True,skip_header=8000,skip_footer= 6000)
    #W, F, E         = np.genfromtxt(dat_directory+'Ly-alpha_no_AG_2016_06_23.txt',unpack=True,skip_header=8859,skip_footer= 7102)
    
    ### Parameters ##############################
    ModelType   = 8         # see description in src/model.py
    mode        = 'mcmc'      # mcmc or lm
    LyA         = 1215.6702 # Heliocentric: 1215.6702
    cLight      = 299792458
    BetaPicRV   = 20.5

    # ISM parameters
    v_ism       = 10.0      # RV of the ISM (relative to Heliocentric)  
    nh_ism      = 18.2     # Column density ISM
    b_ism       = 2.9       # Turbulent velocity
    T_ism       = 6000.     # Temperature of ISM

    # Beta Pic parameters
    v_bp        = 64.7     #20.5# RV of the beta Pic (relative to Heliocentric)
    nh_bp       = 18.6    # Column density beta Pic, Fitting param
    b_bp        = 7.0       # Turbulent velocity
    T_bp        = 1000.     # Temperture of gas in beta Pic disk

    # Extra component parameters
    v_X         = 70.0      # RV of the beta Pic (relative to Heliocentric)
    nh_X        = 18.4      # Column density beta Pic, Fitting param
    b_X         = 7.0       # Turbulent velocity
    T_X         = 1000.     # Temperture of gas in beta Pic disk

    # Stellar emission line parameters
    max_f       = 5.00e-10 # Fitting param 
    dp          = 0.0 
    uf          = 313     # Fitting param
    av          = 4.95     # Fitting param

    slope       = -0.0008205

    sigma_kernel= 6.5

    v   = np.arange(-len(Wo),len(Wo),1) # RV values
    l   = LyA*(1.0 + v/3e5)                     # Corresponding wavengths

    continuum_fit = pn(l)
    
    if ModelType == 1:
        Par     = [nh_bp,max_f,uf,av,slope] # Free parameters
        Const   = [W,l,LyA,BetaPicRV,sigma_kernel,dp,v_ism,nh_ism,b_ism,T_ism,v_bp,b_bp,T_bp]
        step= np.array([0.1,2e-12,0.03,.001,1e-4])/3.
    if ModelType == 2:
        Par     = [nh_bp,max_f,uf,av,v_bp] # Free parameters
        Const   = [W,l,LyA,BetaPicRV,sigma_kernel,dp,v_ism,nh_ism,b_ism,T_ism,b_bp,T_bp]
        step= np.array([0.03,2e-12,0.08,.01,1])/3.
    if ModelType == 3:
        Par     = [nh_bp,max_f,uf,av,v_X,nh_X] # Free parameters
        Const   = [W,l,LyA,BetaPicRV,sigma_kernel,dp,v_ism,nh_ism,b_ism,T_ism,v_bp,b_bp,T_bp,b_X,T_X]
        step= np.array([0.1,2e-12,0.03,.001,5,0.1])/3.
    if ModelType == 4:
        Par     = [nh_bp,max_f,uf,av] # Free parameters
        Const   = [W,l,LyA,BetaPicRV,sigma_kernel,dp,v_ism,nh_ism,b_ism,T_ism,v_bp,b_bp,T_bp]
        step= np.array([0.1,2e-12,0.03,.001,5,0.1])/3.
    if ModelType == 5:
        Par     = [nh_bp,max_f,uf,av,v_bp,nh_ism] # Free parameters
        Const   = [W,l,LyA,BetaPicRV,sigma_kernel,dp,v_ism,b_ism,T_ism,b_bp,T_bp,continuum_fit]
        step= np.array([0.05,1e-11,0.01,.002,3.,0.05])
    if ModelType == 6:
        Par     = [nh_bp,max_f,uf,av,v_X,nh_X,nh_ism] # Free parameters
        Const   = [W,l,LyA,BetaPicRV,sigma_kernel,dp,v_ism,b_ism,T_ism,v_bp,b_bp,T_bp,b_X,T_X,continuum_fit]
        step= np.array([0.06,1e-12,0.005,.0005,4.,0.06,0.05])
    if ModelType == 8:
        Par     = [nh_bp,max_f,uf,av,v_bp,nh_ism,b_bp,T_bp] # Free parameters
        Const   = [W,l,LyA,BetaPicRV,sigma_kernel,dp,v_ism,b_ism,T_ism,continuum_fit]
        step= np.array([0.05,1e-11,0.01,.002,3.,0.05,0.5,100])

    #############################################


    X = F, E, m.LyModel(Par,Const,ModelType)[0]

    #chain, moves = mc.McMC(W,X,m.LyModel, ModelType, Par, Const, step,1.67e5)
    chain, moves = mc.McMC(W,X,m.LyModel, ModelType, Par, Const, step,1.67e5)
    
    outfile = 'chains/chain_Q_'+sys.argv[1]
    #np.savez(outfile, nh_bp = chain[:,0], max_f = chain[:,1], uf = chain[:,2], av = chain[:,3], v_X = chain[:,4], nh_X = chain[:,5], nh_ISM = chain[:,6])
    #np.savez(outfile, nh_bp = chain[:,0], max_f = chain[:,1], uf = chain[:,2], av = chain[:,3], v_H = chain[:,4], nh_ISM = chain[:,5])
    np.savez(outfile, nh_bp = chain[:,0], max_f = chain[:,1], uf = chain[:,2], av = chain[:,3], v_H = chain[:,4], nh_ISM = chain[:,5], b_BP = chain[:,6], T_BP = chain[:,7])
    
    #'''
    Pout = chain[moves,:]
    P_plot1 = [0,1]
    P_plot2 = [2,3]
    P_plot3 = [4,5]
    P_plot4 = [6,7]

    PU1 = mc.Median_and_Uncertainties(P_plot1,step,chain)
    PU2 = mc.Median_and_Uncertainties(P_plot2,step,chain)
    PU3 = mc.Median_and_Uncertainties(P_plot3,step,chain)
    PU4 = mc.Median_and_Uncertainties(P_plot4,step,chain)
    
    print "\nlog(N(H)) =\t" ,PU1[0][0],"\t+",PU1[1][0],"\t-",PU1[2][0]
    print "Fmax =\t\t"      ,PU1[0][1],"\t+",PU1[1][1],"\t-",PU1[2][1]
    print "uf=\t\t"         ,PU2[0][0],"\t+",PU2[1][0],"\t-",PU2[2][0]
    print "av=\t\t"         ,PU2[0][1],"\t+",PU2[1][1],"\t-",PU2[2][1]
    print "v_X=\t\t"        ,PU3[0][0],"\t+",PU3[1][0],"\t-",PU3[2][0]
    print "nh_H=\t\t"       ,PU3[0][1],"\t+",PU3[1][1],"\t-",PU3[2][1]
    print "b_BP=\t\t"       ,PU4[0][0],"\t+",PU4[1][0],"\t-",PU4[2][0]
    print "T_BP=\t\t"       ,PU4[0][1],"\t+",PU4[1][1],"\t-",PU4[2][1]



if __name__ == '__main__':
    main()
