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
    errorbars_2 = np.sum((weights*Err)**2, axis=0)/ ((np.sum(weights, axis=0))**2)
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
    weights=1./(err**2)
    wm      = np.average(region, axis=0, weights=weights)
    errorb  = np.sum((weights*err)**2, axis=0)/ ((np.sum(weights, axis=0))**2)
    return wm, np.sqrt(errorb)

def main():    

    #Wc, Fc, Ec  = np.genfromtxt('Ly-alpha.dat',skiprows=830,skip_footer=0,unpack=True)
    W, RV, f0_0, e0_0, f_AG_0, e_AG_0 = np.genfromtxt('../data/HST/dat/B_2014.dat',skiprows=830,skip_footer=0,unpack=True)
    W, RV, F0_1, E0_1, F1_1, E1_1, F2_1, E2_1, AG1, AG1err, F_ave_w_1 = np.genfromtxt('../data/HST/dat/B_10Dec.dat',skiprows=830,skip_footer=0,unpack=True)
    W, RV, F0_2, E0_2, F1_2, E1_2, F2_2, E2_2, F3_2, E3_2, AG2, AG2err, F_ave_w_2 = np.genfromtxt('../data/HST/dat/B_24Dec.dat',skiprows=830,skip_footer=0,unpack=True)
    W, RV, F0_3, E0_3, F1_3, E1_3, F2_3, E2_3, F3_3, E3_3, AG3, AG3err, F_ave_w_3  = np.genfromtxt('../data/HST/dat/B_30Jan.dat',skiprows=830,skip_footer=0,unpack=True)
    
    LyA             = 1215.6702
    RV_BP           = 0.#20.5

    dat_directory = "/home/paw/science/betapic/data/HST/dat/" 

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
    AirG = np.array([f_AG_0,AG1,AG3])
    AirG_err  = np.array([e_AG_0,AG1err,AG3err])
    AirG_W, AirG_W_err = weighted_avg_and_errorbars(AirG,AirG_err)

      

    # 2 visit -0.8" offset
    #===============================
    AGs    = ShiftAG(AirG_W,32)
    factor, ferr    = FindFactor(RV,F1_1/AGs,E1_1,AirG_W_err,-59.5,80)
    m081 = F1_1-AGs*factor
    e081 = E1_1 + np.sqrt(AirG_W_err**2+ ferr**2)

    # 3 visit -0.8" offset
    #===============================
    AGs    = ShiftAG(AirG_W,35)
    factor, ferr    = FindFactor(RV,F1_2/AGs,E1_2,AirG_W_err,-59.5,80)
    m082 = F1_2-AGs*factor
    e082 = E1_2 + np.sqrt(AirG_W_err**2+ ferr**2)
    
    # 4 visit -0.8" offset
    #===============================
    AGs    = ShiftAG(AirG_W,33)
    factor, ferr    = FindFactor(RV,F1_3/AGs,E1_3,AirG_W_err,-59.5,80)
    m083 = F1_3-AGs*factor
    e083 = E1_3 + np.sqrt(AirG_W_err**2+ ferr**2)

    #np.savetxt(dat_directory+"minus_08_visit2.dat",np.column_stack((W,m081,e081)))
    #np.savetxt(dat_directory+"minus_08_visit3.dat",np.column_stack((W,m082,e082)))
    #np.savetxt(dat_directory+"minus_08_visit4.dat",np.column_stack((W,m083,e083)))



    # 2 visit +0.8" offset
    #===============================
    AGs    = ShiftAG(AirG_W,-28)
    factor, ferr    = FindFactor(RV,F2_1/AGs,E2_1,AirG_W_err,-59.5,80)    
    p082 = F2_1-AGs*factor
    e082 = E2_1 + np.sqrt(AirG_W_err**2+ ferr**2)


    # 3 visit +0.8" offset
    #===============================
    AGs    = ShiftAG(AirG_W,-32)        # or -33
    factor, ferr    = FindFactor(RV,F2_2/AGs,E2_2,AirG_W_err,-59.5,80)
    p083 = F2_2-AGs*factor
    e083 = E2_2 + np.sqrt(AirG_W_err**2+ ferr**2)


    # 4 visit +0.8" offset
    #===============================
    AGs    = ShiftAG(AirG_W,-25)
    factor, ferr    = FindFactor(RV,F2_3/AGs,E2_3,AirG_W_err,-59.5,80)
    p084 = F2_3-AGs*factor
    e084 = E2_3 + np.sqrt(AirG_W_err**2+ ferr**2)

    #np.savetxt(dat_directory+"plus_08_visit2.dat",np.column_stack((W,p082,e082)))
    #np.savetxt(dat_directory+"plus_08_visit3.dat",np.column_stack((W,p083,e083)))
    #np.savetxt(dat_directory+"plus_08_visit4.dat",np.column_stack((W,p084,e084)))



    # 3 visit +1.1" offset
    #===============================
    AGs    = ShiftAG(AirG_W,-42)
    factor, ferr    = FindFactor(RV,F3_2/AGs,E3_2,AirG_W_err,-59.5,80)
    p113 = F3_2-AGs*factor
    e113 = E3_2 + np.sqrt(AirG_W_err**2+ ferr**2)
    #===============================

    # 4 visit +1.1" offset
    #===============================
    AGs    = ShiftAG(AirG_W,-43)
    factor, ferr    = FindFactor(RV,F3_3/AGs,E3_3,AirG_W_err,-59.5,80)
    p114 = F3_3-AGs*factor
    e114 = E3_3 + np.sqrt(AirG_W_err**2+ ferr**2)
    #===============================

    #np.savetxt(dat_directory+"plus_11_visit3.dat",np.column_stack((W,p113,e113)))
    #np.savetxt(dat_directory+"plus_11_visit4.dat",np.column_stack((W,p114,e114)))

    # Uncomment lines below for plotting
    #plt.xlim(-520,520)
    #plt.savefig('Ly_red_wing.pdf', bbox_inches='tight', pad_inches=0.1,dpi=300)
    #plt.show()

if __name__ == '__main__':
    main()
