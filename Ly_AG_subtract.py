#WHAT IS GOING ON!?

import numpy as np
import matplotlib.pyplot as plt
import sys

from src.calculations import Calc
c   = Calc()

def main():    

    dat_directory = "/home/paw/science/betapic/data/HST/dat/"

    #Wc, Fc, Ec  = np.genfromtxt('Ly-alpha.dat',skip_header=830,skip_footer=0,unpack=True)
    W, RV, F0_0, E0_0, AG0, AG0err                                                  = np.genfromtxt(dat_directory+'B_2014.dat',unpack=True)
    W, RV, F0_1, E0_1, F1_1, E1_1, F2_1, E2_1, AG1, AG1err, F_ave_w_1               = np.genfromtxt(dat_directory+'B_10Dec.dat',unpack=True)
    W, RV, F0_2, E0_2, F1_2, E1_2, F2_2, E2_2, F3_2, E3_2, AG2, AG2err, F_ave_w_2   = np.genfromtxt(dat_directory+'B_24Dec.dat',unpack=True)
    W, RV, F0_3, E0_3, F1_3, E1_3, F2_3, E2_3, F3_3, E3_3, AG3, AG3err, F_ave_w_3   = np.genfromtxt(dat_directory+'B_30Jan.dat',unpack=True)
    
    rescale    = np.genfromtxt(dat_directory+'rescaling_factors.txt',unpack=True)
    rescale    = np.array(np.append(1.,rescale))# added 1. since we scaled to 2014 data.  
    rescale    = np.array([rescale])            # 

    # Rescaling errors
    F = np.array([F0_0,F0_1,F1_1,F2_1,F0_2,F1_2,F2_2,F3_2,F0_3,F1_3,F2_3,F3_3])
    E = np.array([E0_0,E0_1,E1_1,E2_1,E0_2,E1_2,E2_2,E3_2,E0_3,E1_3,E2_3,E3_3])

    F = F*rescale.T
    E = E*rescale.T 
    
    LyA     = 1215.6702
    
    ff      = 20
    l1      = 20.5-ff
    l2      = 20.5+ff

    # Combining AG measurements. Not including AG2 due to problem with data.
    AirG                = np.array([AG0,AG1,AG3])
    AirG_err            = np.array([AG0err,AG1err,AG3err])    
    AirG_W, AirG_W_err  = c.WeightedAvg(AirG,AirG_err)

    # 1 visit 0.0" offset
    AGs    = c.ShiftAG(AirG_W,-1)
    factor, ferr    = c.FindFactor(RV,F[0]/AGs,E[0],AirG_W_err,l1,l2)
    n001 = F[0]-AGs*factor
    en001 = np.sqrt(E[0]**2+(factor*AirG_W_err)**2)

    # 2 visit 0.0" offset
    AGs    = c.ShiftAG(AirG_W,-1)
    
    factor, ferr    = c.FindFactor(RV,F[1]/AGs,E[1],AirG_W_err,l1,l2)
    n002 = F[1]-AGs*factor
    en002 = np.sqrt(E[1]**2+(factor*AirG_W_err)**2)

    # 3 visit 0.0" offset
    AGs    = c.ShiftAG(AirG_W,2)    
    factor, ferr    = c.FindFactor(RV,F[4]/AGs,E[4],AirG_W_err,l1,l2)
    n003 = F[4]-AGs*factor
    en003 = np.sqrt(E[4]**2+(factor*AirG_W_err)**2)
    
    # 4 visit 0.0" offset
    AGs    = c.ShiftAG(AirG_W,-2)
    factor, ferr    = c.FindFactor(RV,F[8]/AGs,E[8],AirG_W_err,l1,l2)
    n004 = F[8]-AGs*factor
    en004 = np.sqrt(E[8]**2+(factor*AirG_W_err)**2)

    # 2 visit -0.8" offset
    #===============================
    AGs    = c.ShiftAG(AirG_W,33)
    factor, ferr    = c.FindFactor(RV,F[2]/AGs,E[2],AirG_W_err,l1,l2)
    m082 = F[2]-AGs*factor
    em082 = np.sqrt(E[2]**2+(factor*AirG_W_err)**2)

    # 3 visit -0.8" offset
    #===============================
    AGs    = c.ShiftAG(AirG_W,37)
    factor, ferr    = c.FindFactor(RV,F[5]/AGs,E[5],AirG_W_err,l1,l2)
    m083 = F[5]-AGs*factor
    em083 = np.sqrt(E[5]**2+(factor*AirG_W_err)**2)  
    
    # 4 visit -0.8" offset
    #===============================
    AGs    = c.ShiftAG(AirG_W,33)
    factor, ferr    = c.FindFactor(RV,F[9]/AGs,E[9],AirG_W_err,l1,l2)
    m084 = F[9]-AGs*factor
    em084 = np.sqrt(E[9]**2+(factor*AirG_W_err)**2)

    # 2 visit +0.8" offset
    #===============================
    AGs    = c.ShiftAG(AirG_W,-28)
    factor, ferr    = c.FindFactor(RV,F[3]/AGs,E[3],AirG_W_err,l1,l2)    
    p082 = F[3]-AGs*factor
    ep082 = np.sqrt(E[3]**2+(factor*AirG_W_err)**2)

    # 3 visit +0.8" offset
    #===============================
    AGs    = c.ShiftAG(AirG_W,-32)        # or -33
    factor, ferr    = c.FindFactor(RV,F[6]/AGs,E2_2,AirG_W_err,l1,l2)
    p083 = F[6]-AGs*factor
    ep083 = np.sqrt(E[6]**2+(factor*AirG_W_err)**2)

    # 4 visit +0.8" offset
    #===============================
    AGs    = c.ShiftAG(AirG_W,-28)
    factor, ferr    = c.FindFactor(RV,F[10]/AGs,E[10],AirG_W_err,l1,l2)
    p084 = F[10]-AGs*factor
    ep084 = np.sqrt(E[10]**2+(factor*AirG_W_err)**2)

    # 3 visit +1.1" offset
    #===============================
    AGs    = c.ShiftAG(AirG_W,-41)
    factor, ferr    = c.FindFactor(RV,F[7]/AGs,E[7],AirG_W_err,l1,l2)
    p113 = F[7]-AGs*factor
    ep113 = np.sqrt(E[7]**2+(factor*AirG_W_err)**2)
    #===============================


    # 4 visit +1.1" offset
    #===============================
    AGs    = c.ShiftAG(AirG_W,-42)
    factor, ferr    = c.FindFactor(RV,F[11]/AGs,E[11],AirG_W_err,l1,l2)
    p114 = F[11]-AGs*factor
    ep114 = np.sqrt(E[11]**2+(factor*AirG_W_err)**2)
    #===============================

    Wo, Fo, Eo  = np.genfromtxt(dat_directory+'Ly-alpha_no_AG_2016_06_21.txt',unpack=True)

    RVo         = c.Wave2RV(Wo,LyA,0.)

    fig = plt.figure(figsize=(7,5))
    fontlabel_size  = 18
    tick_size       = 18
    params = {'backend': 'wxAgg', 'lines.markersize' : 2, 'axes.labelsize': fontlabel_size, 'font.size': fontlabel_size, 'legend.fontsize': 15, 'xtick.labelsize': tick_size, 'ytick.labelsize': tick_size, 'text.usetex': True}
    plt.rcParams.update(params)
    plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    plt.rcParams['text.usetex'] = True
    plt.rcParams['text.latex.unicode'] = True    

    # Combining first then dividing method:
    # =========================================
    # =========================================

    blue                = np.array([m082,m083,m084])
    blue_err            = np.array([em082,em083,em084])

    #blue                = np.array([n002,n003,n004,m082,m083,m084])
    #blue_err            = np.array([en002,en003,en004,em082,em083,em084])

    red                 = np.array([p082,p083,p084,p113,p114])
    red_err             = np.array([ep082,ep083,ep084,ep113,ep114])    

    #red                 = np.array([n002,n003,n004,p082,p083,p084,p113,p114])
    #red_err             = np.array([en002,en003,en004,ep082,ep083,ep084,ep113,ep114]) 
    
    F_blue, F_err_blue  = c.WeightedAvg(blue,blue_err)
    F_red, F_err_red    = c.WeightedAvg(red,red_err)

    plt.plot(RVo,Fo,marker='o',color="black")

    f = open(dat_directory+'N_2016v1_2016_07_27.txt','w+')
    for i in range(len(W)):
        if W[i] > LyA:
            print >> f, W[i], F_red[i], F_err_red[i]
            plt.plot(RV[i],F_red[i],marker='o',color="red")
        else:
            print >> f, W[i], F_blue[i], F_err_blue[i]
            plt.plot(RV[i],F_blue[i],marker='o',color="blue")
    f.close()

    plt.ylim(-3e-13,3.0e-13)
    plt.xlim(-610,610)
    plt.show()
    

if __name__ == '__main__':
    main()
