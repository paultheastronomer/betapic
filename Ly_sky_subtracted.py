import numpy as np
import matplotlib.pyplot as plt
import sys

def Bin_data(x,y1,e1,bin_pnts):
    bin_size    = int(len(x)/bin_pnts)
    bins        = np.linspace(x[0], x[-1], bin_size)
    digitized   = np.digitize(x, bins)
    bin_y       = np.array([y1[digitized == i].mean() for i in range(0, len(bins))])
    bin_e       = np.array([e1[digitized == i].mean() for i in range(0, len(bins))])
    return bins, bin_y ,bin_e/np.sqrt(bin_pnts)

def wave2RV(Wave,rest_wavelength,RV_BP):
    c = 299792458
    rest_wavelength = rest_wavelength*(RV_BP*1.e3)/c + rest_wavelength # Convert to beta pic reference frame
    delta_wavelength = Wave-rest_wavelength
    RV = ((delta_wavelength/rest_wavelength)*c)/1.e3	# km/s
    return RV

def weighted_avg_and_errorbars(Flux, Err):
    """
    Return the weighted average and Error bars.
    """
    weights     =1./(Err**2)
    average     = np.average(Flux, axis=0, weights=weights)
    errorbars_2 = np.sum(weights*(Err**2), axis=0) / np.sum(weights, axis=0)
    return average, np.sqrt(errorbars_2)

def main():    
    dat_directory = "/home/paw/science/betapic/data/HST/dat/" 
    #Wc, Fc, Ec  = np.genfromtxt('Ly-alpha.dat',skiprows=830,skip_footer=0,unpack=True)
    W,n000,en000 = np.genfromtxt(dat_directory+'neutral_00_visit1.dat',unpack=True,skiprows=7000,skip_footer=6000)
    W,n001,en001 = np.genfromtxt(dat_directory+'neutral_00_visit2.dat',unpack=True,skiprows=7000,skip_footer=6000)
    W,n002,en002 = np.genfromtxt(dat_directory+'neutral_00_visit3.dat',unpack=True,skiprows=7000,skip_footer=6000)
    W,n003,en003 = np.genfromtxt(dat_directory+'neutral_00_visit4.dat',unpack=True,skiprows=7000,skip_footer=6000)

    W,m081,em081 = np.genfromtxt(dat_directory+'minus_08_visit2.dat',unpack=True,skiprows=7000,skip_footer=6000)
    W,m082,em082 = np.genfromtxt(dat_directory+'minus_08_visit3.dat',unpack=True,skiprows=7000,skip_footer=6000)
    W,m083,em083 = np.genfromtxt(dat_directory+'minus_08_visit4.dat',unpack=True,skiprows=7000,skip_footer=6000)

    W,p081,ep081 = np.genfromtxt(dat_directory+'plus_08_visit2.dat',unpack=True,skiprows=7000,skip_footer=6000)
    #W,p082,ep082 = np.genfromtxt(dat_directory+'plus_08_visit3.dat',unpack=True,skiprows=7000,skip_footer=6000)
    W,p083,ep083 = np.genfromtxt(dat_directory+'plus_08_visit4.dat',unpack=True,skiprows=7000,skip_footer=6000)

    W,p112,ep112 = np.genfromtxt(dat_directory+'plus_11_visit3.dat',unpack=True,skiprows=7000,skip_footer=6000)
    W,p113,ep113 = np.genfromtxt(dat_directory+'plus_11_visit4.dat',unpack=True,skiprows=7000,skip_footer=6000)
        
    LyA             = 1215.6702
    RV_BP           = 0.0
    Wo, Fo, Eo                          = np.genfromtxt(dat_directory+'Ly-alpha_no_AG.txt',unpack=True)
    Wold, Fold, Eold                    = np.genfromtxt(dat_directory+'Ly-alpha.dat',unpack=True)

    RVo = wave2RV(Wo,LyA,0.)
    RV = wave2RV(W,LyA,0.)
    
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
    blue     = np.array([m081,m082,m083])
    blue_err  = np.array([em081,em082,em083])
    
    red     = np.array([p081,p083,p112,p113])
    red_err     = np.array([ep081,ep083,ep112,ep113])
    #red     = np.array([p081,p082,p083,p112,p113])
    #red_err     = np.array([ep081,ep082,ep083,ep112,ep113])


    
    F_blue, F_err_blue = weighted_avg_and_errorbars(blue,blue_err)
    
    
    F_red, F_err_red = weighted_avg_and_errorbars(red,red_err)

    tot     = np.array([n000,n001,n002,n003,m081,m082,m083,p081,p083,p112,p113])
    tot_err = np.array([en000,en001,en002,en003,em081,em082,em083,ep081p,ep083,ep112,ep113])

    #tot     = np.array([n000,n001,n002,n003,m081,m082,m083,p081,p082,p083,p112,p113])
    #tot_err = np.array([en000,en001,en002,en003,em081,em082,em083,ep081,ep082,ep083,ep112,ep113])

    F, F_err    = weighted_avg_and_errorbars(tot,tot_err)       

    #np.savetxt(dat_directory+"Ly_sky_subtracted.txt",np.column_stack((W,F,F_err)))

    bin_pnts = 6
    RVb, Fb_blue, Fb_err_blue   = Bin_data(RV,F_blue,F_err_blue,bin_pnts)
    RVb, Fb_red, Fb_err_red     = Bin_data(RV,F_red,F_err_red,bin_pnts)

    #plt.plot(RV,F_red,color="red")

    for i in range(len(RVb)):
        if RVb[i] > 0:
            #plt.scatter(RV[i],F_red[i],color="red")
            plt.scatter(RVb[i], Fb_red[i],color="red")
        else:
            #plt.scatter(RV[i],F_blue[i],color="blue")
            plt.scatter(RVb[i], Fb_blue[i],color="blue")
    #'''
    f = open(dat_directory+'Ly_sky_subtracted.txt','w+')
    for i in range(len(W)):
        if W[i] > LyA:
            print >> f, W[i], F_red[i], F_err_red[i]
        else:
            print >> f, W[i], F_blue[i], F_err_blue[i]
    f.close()
    #'''

    plt.ylim(-2e-14,6.0e-14)
    plt.xlim(-610,610)

    #plt.errorbar(RVo,Fo,yerr=Eo,color="black")
    plt.plot(RVo,Fo,marker='o',color="black")
    #plt.errorbar(RVold,Fold,yerr=Eold,color="orange")
    #plt.xlim(-610,630)
    #plt.savefig('divided.pdf', bbox_inches='tight', pad_inches=0.1)
    plt.show()

if __name__ == '__main__':
    main()
