import numpy as np
import matplotlib.pyplot as plt
import sys

def wave2RV(Wave,rest_wavelength,RV_BP):
    c = 299792458
    rest_wavelength = rest_wavelength*(RV_BP*1.e3)/c + rest_wavelength # Convert to beta pic reference frame
    delta_wavelength = Wave-rest_wavelength
    RV = ((delta_wavelength/rest_wavelength)*c)/1.e3	# km/s
    return RV

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

def Bin_data(x,y1,e1,bin_pnts):
    bin_size    = int(len(x)/bin_pnts)
    bins        = np.linspace(x[0], x[-1], bin_size)
    digitized   = np.digitize(x, bins)
    bin_y       = np.array([y1[digitized == i].mean() for i in range(0, len(bins))])
    bin_e       = np.array([e1[digitized == i].mean() for i in range(0, len(bins))])
    return bins, bin_y ,bin_e/np.sqrt(bin_pnts)

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

def FindFactor(RV,D,E,A,l1,l2):
    region  = []
    err     = []
    E         = replace_with_median(E)
    for i in range(len(RV)):
        if l1 < RV[i] < l2:
            region.append(D[i])
            err.append(np.sqrt(E[i]**2+A[i]**2))
    region  = np.array(region)
    err     = np.array(err)
    weights = 1./(err**2)
    wm      = np.average(region, axis=0, weights=weights)
    errorbars_2 = np.sum(weights*(err**2), axis=0) / np.sum(weights, axis=0)
    return wm, np.sqrt(errorbars_2)

def main():    

    dat_directory = "/home/paw/science/betapic/data/HST/dat/"

    #Wc, Fc, Ec  = np.genfromtxt('Ly-alpha.dat',skip_header=830,skip_footer=0,unpack=True)
    W, RV, F0_0, E0_0, AG0, AG0err                                                  = np.genfromtxt(dat_directory+'B_2014.dat',unpack=True,skip_header=8800,skip_footer= 6500)
    W, RV, F0_1, E0_1, F1_1, E1_1, F2_1, E2_1, AG1, AG1err, F_ave_w_1               = np.genfromtxt(dat_directory+'B_10Dec.dat',unpack=True,skip_header=8800,skip_footer= 6500)
    W, RV, F0_2, E0_2, F1_2, E1_2, F2_2, E2_2, F3_2, E3_2, AG2, AG2err, F_ave_w_2   = np.genfromtxt(dat_directory+'B_24Dec.dat',unpack=True,skip_header=8800,skip_footer= 6500)
    W, RV, F0_3, E0_3, F1_3, E1_3, F2_3, E2_3, F3_3, E3_3, AG3, AG3err, F_ave_w_3   = np.genfromtxt(dat_directory+'B_30Jan.dat',unpack=True,skip_header=8800,skip_footer= 6500)
    
    rescale    = np.genfromtxt(dat_directory+'rescaling_factors.txt',unpack=True)
    rescale    = np.array(np.append(1.,rescale))# added 1. since we scaled to 2014 data.  
    rescale    = np.array([rescale])            # 

    # Rescaling errors
    F = np.array([F0_0,F0_1,F1_1,F2_1,F0_2,F1_2,F2_2,F3_2,F0_3,F1_3,F2_3,F3_3])
    E = np.array([E0_0,E0_1,E1_1,E2_1,E0_2,E1_2,E2_2,E3_2,E0_3,E1_3,E2_3,E3_3])


    F = F*rescale.T
    #E = E*rescale.T        # Ask Alain about this
    
    LyA             = 1215.6702
    
    ff = 20
    l1  = 20.5-ff
    l2  = 20.5+ff


    # Combining AG measurements. Not including AG2 due to problem with data.
    AirG                = np.array([AG0,AG1,AG3])
    AirG_err            = np.array([AG0err,AG1err,AG3err])    
    AirG_W, AirG_W_err  = weighted_avg_and_errorbars(AirG,AirG_err)

    # 1 visit 0.0" offset
    AGs    = ShiftAG(AirG_W,-1)
    factor, ferr    = FindFactor(RV,F[0]/AGs,E[0],AirG_W_err,l1,l2)
    n001 = F[0]-AGs*factor
    en001 = np.sqrt(E[0]**2+(factor*AirG_W_err)**2)

    # 2 visit 0.0" offset
    AGs    = ShiftAG(AirG_W,-1)
    
    factor, ferr    = FindFactor(RV,F[1]/AGs,E[1],AirG_W_err,l1,l2)
    n002 = F[1]-AGs*factor
    en002 = np.sqrt(E[1]**2+(factor*AirG_W_err)**2)

    # 3 visit 0.0" offset
    AGs    = ShiftAG(AirG_W,2)    
    factor, ferr    = FindFactor(RV,F[4]/AGs,E[4],AirG_W_err,l1,l2)
    n003 = F[4]-AGs*factor
    en003 = np.sqrt(E[4]**2+(factor*AirG_W_err)**2)

    
    # 4 visit 0.0" offset
    AGs    = ShiftAG(AirG_W,-2)
    factor, ferr    = FindFactor(RV,F[8]/AGs,E[8],AirG_W_err,l1,l2)
    n004 = F[8]-AGs*factor
    en004 = np.sqrt(E[8]**2+(factor*AirG_W_err)**2)

    # 2 visit -0.8" offset
    #===============================
    AGs    = ShiftAG(AirG_W,33)
    factor, ferr    = FindFactor(RV,F[2]/AGs,E[2],AirG_W_err,l1,l2)
    m082 = F[2]-AGs*factor
    em082 = np.sqrt(E[2]**2+(factor*AirG_W_err)**2)

    # 3 visit -0.8" offset
    #===============================
    AGs    = ShiftAG(AirG_W,37)
    factor, ferr    = FindFactor(RV,F[5]/AGs,E[5],AirG_W_err,l1,l2)
    m083 = F[5]-AGs*factor
    em083 = np.sqrt(E[5]**2+(factor*AirG_W_err)**2)  
    
    # 4 visit -0.8" offset
    #===============================
    AGs    = ShiftAG(AirG_W,33)
    factor, ferr    = FindFactor(RV,F[9]/AGs,E[9],AirG_W_err,l1,l2)
    m084 = F[9]-AGs*factor
    em084 = np.sqrt(E[9]**2+(factor*AirG_W_err)**2)
    

    # 2 visit +0.8" offset
    #===============================
    AGs    = ShiftAG(AirG_W,-28)
    factor, ferr    = FindFactor(RV,F[3]/AGs,E[3],AirG_W_err,l1,l2)    
    p082 = F[3]-AGs*factor
    ep082 = np.sqrt(E[3]**2+(factor*AirG_W_err)**2)

    # 3 visit +0.8" offset
    #===============================
    AGs    = ShiftAG(AirG_W,-32)        # or -33
    factor, ferr    = FindFactor(RV,F[6]/AGs,E2_2,AirG_W_err,l1,l2)
    p083 = F[6]-AGs*factor
    ep083 = np.sqrt(E[6]**2+(factor*AirG_W_err)**2)
 

    # 4 visit +0.8" offset
    #===============================
    AGs    = ShiftAG(AirG_W,-28)
    factor, ferr    = FindFactor(RV,F[10]/AGs,E[10],AirG_W_err,l1,l2)
    p084 = F[10]-AGs*factor
    ep084 = np.sqrt(E[10]**2+(factor*AirG_W_err)**2)


    # 3 visit +1.1" offset
    #===============================
    AGs    = ShiftAG(AirG_W,-41)
    factor, ferr    = FindFactor(RV,F[7]/AGs,E[7],AirG_W_err,l1,l2)
    p113 = F[7]-AGs*factor
    ep113 = np.sqrt(E[7]**2+(factor*AirG_W_err)**2)
    #===============================



    # 4 visit +1.1" offset
    #===============================
    AGs    = ShiftAG(AirG_W,-42)
    factor, ferr    = FindFactor(RV,F[11]/AGs,E[11],AirG_W_err,l1,l2)
    p114 = F[11]-AGs*factor
    ep114 = np.sqrt(E[11]**2+(factor*AirG_W_err)**2)
    #===============================


    '''
    yy1=8
    yy2=13
    plt.plot([-59.5,-59.5],[yy1,yy2],'k')
    plt.plot([80,80],[yy1,yy2],'k')
    plt.plot(RV,F[11]/AGs)
    #plt.plot(RV,n001)
    plt.xlim(-400,400)
    plt.ylim(yy1,yy2)
    plt.show()
    sys.exit() 
    '''

    # Uncomment lines below for plotting
    #plt.xlim(-520,520)
    #plt.savefig('Ly_red_wing.pdf', bbox_inches='tight', pad_inches=0.1,dpi=300)
    #plt.show()
    
    

    Wo, Fo, Eo          = np.genfromtxt(dat_directory+'Ly-alpha_no_AG.txt',unpack=True)
    Wold, Fold, Eold    = np.genfromtxt(dat_directory+'Ly-alpha.dat',unpack=True)

    RVo = wave2RV(Wo,LyA,0.)
    #RV  = wave2RV(W,LyA,0.)
    
    RVold = wave2RV(Wold,LyA,0.)

    # Plot the results
    fig = plt.figure(figsize=(22,16))
    #fig = plt.figure(figsize=(14,5))
    fontlabel_size  = 18
    tick_size       = 18
    params = {'backend': 'wxAgg', 'lines.markersize' : 2, 'axes.labelsize': fontlabel_size, 'font.size': fontlabel_size, 'legend.fontsize': fontlabel_size, 'xtick.labelsize': tick_size, 'ytick.labelsize': tick_size, 'text.usetex': True}
    plt.rcParams.update(params)
    plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    plt.rcParams['text.usetex'] = True
    plt.rcParams['text.latex.unicode'] = True    


    # Combining first then dividing method:
    # =========================================
    # =========================================
    blue        = np.array([n001,n002,n003,n004,m082,m083,m084])
    blue_err    = np.array([en001,en002,en003,en004,em082,em083,em084])
    
    red         = np.array([n001,n002,n003,n004,p082,p084,p113,p114])
    red_err     = np.array([en001,en002,en003,en004,ep082,ep084,ep113,ep114])
    #red         = np.array([p082,p083,p084,p113,p114])
    #red_err     = np.array([ep082,ep083,ep084,ep113,ep114])
    
    
    F_blue, F_err_blue = weighted_avg_and_errorbars(blue,blue_err)
    F_red, F_err_red = weighted_avg_and_errorbars(red,red_err)

    tot     = np.array([n001,n002,n003,n004,m082,m083,m084,p082,p084,p113,p114])
    tot_err = np.array([en001,en002,en003,en004,em083,em082,em084,ep082,ep084,ep113,ep114])
    #tot     = np.array([n001,n002,n003,n004,m082,m083,m084,p082,p083,p084,p113,p114])
    #tot_err = np.array([en001,en002,en003,en004,em082,em083,em084,ep082,ep083,ep084,ep113,ep114])

    tot     = np.array([n001,n002,n003,n004,m082,m083,m084,p082,p084,p113,p114])
    tot_err = np.array([en001,en002,en003,en004,em083,em082,em084,ep082,ep084,ep113,ep114])

    '''
    cc = ["red", "green", "blue","purple"]
    for i in range(len(blue)):

        for j in range(len(RV)):
            if RV[j] < 0:               
                plt.scatter(RV[j], blue[i][j],color=cc[i])

    for i in range(len(red)):
        for j in range(len(RV)):
            if RV[j] > 0:           
                plt.scatter(RV[j], red[i][j],color=cc[i])

    plt.ylim(-6e-13,6.0e-13)
    plt.xlim(-610,610)
    plt.legend(loc='upper left', numpoints=1)
    plt.show()
    sys.exit() 
    '''


    F, F_err    = weighted_avg_and_errorbars(tot,tot_err)       

    #np.savetxt(dat_directory+"Ly_sky_subtracted.txt",np.column_stack((W,F,F_err)))

    bin_pnts = 1
    RVb, Fb_blue, Fb_err_blue   = Bin_data(RV,F_blue,F_err_blue,bin_pnts)
    RVb, Fb_red, Fb_err_red     = Bin_data(RV,F_red,F_err_red,bin_pnts)

    #plt.plot(RV,F_red,color="red")

    for i in range(len(RV)):
        if RV[i] > 0:
            #plt.scatter(RV[i],F_red[i],color="red")
            plt.scatter(RV[i], F_red[i],color="red")
        else:
            #plt.scatter(RV[i],F_blue[i],color="blue")
            plt.scatter(RV[i], F_blue[i],color="blue")
    #'''
    f = open(dat_directory+'Ly_sky_subtracted_new.txt','w+')
    for i in range(len(W)):
        if W[i] > LyA:
            print >> f, W[i], F_red[i], F_err_red[i]
        else:
            print >> f, W[i], F_blue[i], F_err_blue[i]
    f.close()
    #'''

    plt.ylim(-4e-14,8.0e-14)
    plt.xlim(-610,610)

    #plt.errorbar(RVo,Fo,yerr=Eo,color="black")
    plt.plot(RVo,Fo,marker='o',color="black")
    #plt.errorbar(RVold,Fold,yerr=Eold,color="orange")
    #plt.xlim(-610,630)
    #plt.savefig('divided.pdf', bbox_inches='tight', pad_inches=0.1)
    plt.show()
    
    
    
    
    
    

if __name__ == '__main__':
    main()
