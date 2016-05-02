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
        AG      = np.concatenate((zeros,AG), axis=1)[:-units]
    else:
        AG      = np.concatenate((AG,zeros), axis=1)[abs(units):]
    return AG


def weighted_avg_and_errorbars(Flux, Err):
    """
    Return the weighted average and Error bars.
    """
    weights=1./(Err**2)
    average = np.average(Flux, axis=0, weights=weights)
    errorbars_2 = np.sum(weights*(Err**2), axis=0) / np.sum(weights, axis=0)
    return average, np.sqrt(errorbars_2)

def FindFactor(RV,D,E,A,l1,l2):
    region  = []
    err     = []
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

    #Wc, Fc, Ec  = np.genfromtxt('Ly-alpha.dat',skiprows=830,skip_footer=0,unpack=True)
    W, RV, F0_0, E0_0, AG0, AG0err = np.genfromtxt(dat_directory+'B_2014.dat',skiprows=830,skip_footer=0,unpack=True)
    W, RV, F0_1, E0_1, F1_1, E1_1, F2_1, E2_1, AG1, AG1err, F_ave_w_1 = np.genfromtxt(dat_directory+'B_10Dec.dat',skiprows=830,skip_footer=0,unpack=True)
    W, RV, F0_2, E0_2, F1_2, E1_2, F2_2, E2_2, F3_2, E3_2, AG2, AG2err, F_ave_w_2 = np.genfromtxt(dat_directory+'B_24Dec.dat',skiprows=830,skip_footer=0,unpack=True)
    W, RV, F0_3, E0_3, F1_3, E1_3, F2_3, E2_3, F3_3, E3_3, AG3, AG3err, F_ave_w_3  = np.genfromtxt(dat_directory+'B_30Jan.dat',skiprows=830,skip_footer=0,unpack=True)
    
    rescale    = np.genfromtxt(dat_directory+'rescaling_factors2.txt',unpack=True)
    rescale    = np.array([rescale])

    # Rescaling errors
    F = np.array([F0_0,F0_1,F1_1,F2_1,F0_2,F1_2,F2_2,F3_2,F0_3,F1_3,F2_3,F3_3])
    E = np.array([E0_0,E0_1,E1_1,E2_1,E0_2,E1_2,E2_2,E3_2,E0_3,E1_3,E2_3,E3_3])
    
    #print type(F)
    F = F*rescale.T
    #sys.exit()
    #E = E/rescaled_err.T
    
    
    LyA             = 1215.6702
    RV_BP           = 0.0#20.5

     

    # Code for plotting should it be required:
    # =========================================
    fig = plt.figure(figsize=(6,4.5))
    fontlabel_size  = 18
    tick_size       = 18
    params = {'backend': 'wxAgg', 'lines.markersize' : 2, 'axes.labelsize': fontlabel_size, 'font.size': fontlabel_size, 'legend.fontsize': fontlabel_size, 'xtick.labelsize': tick_size, 'ytick.labelsize': tick_size, 'text.usetex': True}
    plt.rcParams.update(params)
    plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    plt.rcParams['text.usetex'] = True
    plt.rcParams['text.latex.unicode'] = True    
    # =========================================

    # Combining AG measurements. Not including AG2 due to problem with data.
    AirG                = np.array([AG0,AG1,AG3])
    AirG_err            = np.array([AG0err,AG1err,AG3err])
    AirG_W, AirG_W_err  = weighted_avg_and_errorbars(AirG,AirG_err)

    #AirG_W_err = AirG_W_err/10.  


    # 1 visit 0.0" offset
    AGs    = ShiftAG(AirG_W,-1)
    factor, ferr    = FindFactor(RV,F[0]/AGs,E[0],AirG_W_err,-59.5,80)
    n000 = F[0]-AGs*factor
    e000 = np.sqrt(E[0]**2+(factor*AirG_W_err)**2)


    # 2 visit 0.0" offset
    AGs    = ShiftAG(AirG_W,-1)
    #plt.plot(RV,F0_1/AirG_W)
    #plt.plot(RV,F0_1/AGs)
    
    factor, ferr    = FindFactor(RV,F[1]/AGs,E[1],AirG_W_err,-59.5,80)
    n001 = F[1]-AGs*factor
    e001 = np.sqrt(E[1]**2+(factor*AirG_W_err)**2)


    # 3 visit 0.0" offset
    AGs    = ShiftAG(AirG_W,1)    
    factor, ferr    = FindFactor(RV,F[4]/AGs,E[4],AirG_W_err,-59.5,80)
    n002 = F[4]-AGs*factor
    e002 = np.sqrt(E[4]**2+(factor*AirG_W_err)**2)
    
    # 4 visit 0.0" offset
    AGs    = ShiftAG(AirG_W,1)
    factor, ferr    = FindFactor(RV,F[8]/AGs,E[8],AirG_W_err,-59.5,80)
    n003 = F[8]-AGs*factor
    e003 = np.sqrt(E[8]**2+(factor*AirG_W_err)**2)
    #plt.plot(RV,F0_1/AirG_W)
    #plt.plot(RV,F0_1/AGs)
    #plt.plot(RV,F0_1)
    #plt.plot(RV,AGs*factor)
    #plt.plot([-59.5,-59.5],[0,10],"--k")
    #plt.plot([80,80],[0,10],"--k")
    #plt.ylim(0.4,1.0)
    #plt.xlim(-300,300)
    #plt.show()

    np.savetxt(dat_directory+"neutral_00_visit1.dat",np.column_stack((W,n000,e000)))
    np.savetxt(dat_directory+"neutral_00_visit2.dat",np.column_stack((W,n001,e001)))
    np.savetxt(dat_directory+"neutral_00_visit3.dat",np.column_stack((W,n002,e002)))
    np.savetxt(dat_directory+"neutral_00_visit4.dat",np.column_stack((W,n003,e003)))
    
    #sys.exit()








    # 2 visit -0.8" offset
    #===============================
    AGs    = ShiftAG(AirG_W,32)
    factor, ferr    = FindFactor(RV,F[2]/AGs,E[2],AirG_W_err,-59.5,80)
    m081 = F[2]-AGs*factor
    e081 = np.sqrt(E[2]**2+(factor*AirG_W_err)**2)

    # 3 visit -0.8" offset
    #===============================
    AGs    = ShiftAG(AirG_W,35)
    factor, ferr    = FindFactor(RV,F[5]/AGs,E[5],AirG_W_err,-59.5,80)
    m082 = F[5]-AGs*factor
    e082 = np.sqrt(E[5]**2+(factor*AirG_W_err)**2)
    
    # 4 visit -0.8" offset
    #===============================
    AGs    = ShiftAG(AirG_W,33)
    factor, ferr    = FindFactor(RV,F[9]/AGs,E[9],AirG_W_err,-59.5,80)
    m083 = F[9]-AGs*factor
    e083 = np.sqrt(E[9]**2+(factor*AirG_W_err)**2)

    np.savetxt(dat_directory+"minus_08_visit2.dat",np.column_stack((W,m081,e081)))
    np.savetxt(dat_directory+"minus_08_visit3.dat",np.column_stack((W,m082,e082)))
    np.savetxt(dat_directory+"minus_08_visit4.dat",np.column_stack((W,m083,e083)))



    # 2 visit +0.8" offset
    #===============================
    AGs    = ShiftAG(AirG_W,-28)
    factor, ferr    = FindFactor(RV,F[3]/AGs,E[3],AirG_W_err,-59.5,80)    
    p082 = F[3]-AGs*factor
    e082 = np.sqrt(E[3]**2+(factor*AirG_W_err)**2)


    # 3 visit +0.8" offset
    #===============================
    AGs    = ShiftAG(AirG_W,-32)        # or -33
    factor, ferr    = FindFactor(RV,F[6]/AGs,E2_2,AirG_W_err,-59.5,80)
    p083 = F[6]-AGs*factor
    e083 = np.sqrt(E[6]**2+(factor*AirG_W_err)**2)


    # 4 visit +0.8" offset
    #===============================
    AGs    = ShiftAG(AirG_W,-25)
    factor, ferr    = FindFactor(RV,F[10]/AGs,E[10],AirG_W_err,-59.5,80)
    p084 = F[10]-AGs*factor
    e084 = np.sqrt(E[10]**2+(factor*AirG_W_err)**2)

    np.savetxt(dat_directory+"plus_08_visit2.dat",np.column_stack((W,p082,e082)))
    np.savetxt(dat_directory+"plus_08_visit3.dat",np.column_stack((W,p083,e083)))
    np.savetxt(dat_directory+"plus_08_visit4.dat",np.column_stack((W,p084,e084)))



    # 3 visit +1.1" offset
    #===============================
    AGs    = ShiftAG(AirG_W,-42)
    factor, ferr    = FindFactor(RV,F[7]/AGs,E[7],AirG_W_err,-59.5,80)
    p113 = F[7]-AGs*factor
    e113 = np.sqrt(E[7]**2+(factor*AirG_W_err)**2)
    #===============================

    # 4 visit +1.1" offset
    #===============================
    AGs    = ShiftAG(AirG_W,-43)
    factor, ferr    = FindFactor(RV,F[11]/AGs,E[11],AirG_W_err,-59.5,80)
    p114 = F[11]-AGs*factor
    e114 = np.sqrt(E[11]**2+(factor*AirG_W_err)**2)
    #===============================

    np.savetxt(dat_directory+"plus_11_visit3.dat",np.column_stack((W,p113,e113)))
    np.savetxt(dat_directory+"plus_11_visit4.dat",np.column_stack((W,p114,e114)))

    # Uncomment lines below for plotting
    #plt.xlim(-520,520)
    #plt.savefig('Ly_red_wing.pdf', bbox_inches='tight', pad_inches=0.1,dpi=300)
    #plt.show()

if __name__ == '__main__':
    main()
