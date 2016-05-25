import numpy as np
import matplotlib.pyplot as plt
import sys

from scipy.optimize import leastsq

from src.statistics import Stats
from src.model import Model
from src.mcmc import MCMC

s   = Stats()
m   = Model()
mc  = MCMC() 


def FindBestParams(params,F,E,Const,ModelType):
    best_P, success = leastsq(s.chi2_lm, params, args=(F,E,Const,ModelType), maxfev=1000)
    return best_P

def wave2RV(Wave,rest_wavelength,RV_BP):
    c = 299792458
    rest_wavelength = rest_wavelength*(RV_BP*1.e3)/c + rest_wavelength # Convert to beta pic reference frame
    delta_wavelength = Wave-rest_wavelength
    RV = ((delta_wavelength/rest_wavelength)*c)/1.e3	# km/s
    return RV

def Bin_data(x,y1,e1,bin_pnts):
    bin_size    = int(len(x)/bin_pnts)
    bins        = np.linspace(x[0], x[-1], bin_size)
    digitized   = np.digitize(x, bins)
    bin_y       = np.array([y1[digitized == i].mean() for i in range(0, len(bins))])
    bin_e       = np.array([e1[digitized == i].mean() for i in range(0, len(bins))])
    return bins, bin_y ,bin_e/np.sqrt(bin_pnts)

def main():    

    dat_directory   = "/home/paw/science/betapic/data/HST/dat/"

    Wo, Fo, Eo      = np.genfromtxt(dat_directory+'Ly_sky_subtracted.txt',unpack=True)
    W, F, E         = np.genfromtxt(dat_directory+'Ly_sky_subtracted.txt',unpack=True,skip_header=100,skip_footer= 630)
    

    #np.savetxt(dat_directory+"Ly_a_20160511_v2.txt",np.column_stack((Wo, Fo, Eo)))
    
    # To fit the non-sky subtracted (only cut) data uncomment the two lines below.
    #Wo, Fo, Eo         = np.genfromtxt(dat_directory+'Ly-alpha_no_AG.txt',unpack=True)
    #W, F, E         = np.genfromtxt(dat_directory+'Ly-alpha_no_AG.txt',unpack=True,skiprows=900,skip_footer= 145)
    
    
    ### Parameters ##############################
    ModelType   = 2  
    mode        = 'mcmc'  # mcmc or lm
    LyA         = 1215.6702 # Heliocentric: 1215.6702
    BetaPicRV   = 0

    # ISM parameters
    v_ism       = 10.0# RV of the ISM (relative to Heliocentric)  
    nh_ism      = 18.4 # Column density ISM
    b_ism       = 7.  # Turbulent velocity
    T_ism       = 7000.   # Temperature of ISM

    # Beta Pic parameters
    v_bp        = 9.0#20.5# RV of the beta Pic (relative to Heliocentric)
    nh_bp       = 19.41 # Column density beta Pic, Fitting param
    b_bp        = 4.0# Turbulent velocity
    T_bp        = 1000.   # Temperture of gas in beta Pic disk

    # Extra component parameters
    v_X         = 40.0  # RV of the beta Pic (relative to Heliocentric)
    nh_X        = 19.0   # Column density beta Pic, Fitting param
    b_X         = 6.0   # Turbulent velocity
    T_X         = 1000.   # Temperture of gas in beta Pic disk

    # Stellar emission line parameters
    max_f       = 7.0e-12 # Fitting param 
    dp          = 0.0 
    uf          = 2.84#3.60# Fitting param
    av          = 0.04#0.1# Fitting param

    slope       = -0.0008205

    sigma_kernel= 3.5

    v   = np.arange(-len(Wo)/1.3,len(Wo)/1.3,1) # RV values
    l   = LyA*(1.0 + v/3e5) # Corresponding wavengths

    
    if ModelType == 1:
        Par     = [nh_bp,max_f,uf,av,slope] # Free parameters
        Const   = [W,l,LyA,BetaPicRV,sigma_kernel,dp,v_ism,nh_ism,b_ism,T_ism,v_bp,b_bp,T_bp]
        step= np.array([0.1,2e-12,0.03,.001,1e-4])/3.
    if ModelType == 2:
        Par     = [nh_bp,max_f,uf,av,v_bp] # Free parameters
        Const   = [W,l,LyA,BetaPicRV,sigma_kernel,dp,v_ism,nh_ism,b_ism,T_ism,b_bp,T_bp]
        step= np.array([0.03,2e-12,0.1,.01,1])/3.
    if ModelType == 3:
        Par     = [nh_bp,max_f,uf,av,v_X,nh_X] # Free parameters
        Const   = [W,l,LyA,BetaPicRV,sigma_kernel,dp,v_ism,nh_ism,b_ism,T_ism,v_bp,b_bp,T_bp,b_X,T_X]
        step= np.array([0.1,2e-12,0.03,.001,5,0.1])/3.
    if ModelType == 4:
        Par     = [nh_bp,max_f,uf,av,nh_X] # Free parameters
        Const   = [W,l,LyA,BetaPicRV,sigma_kernel,dp,v_ism,nh_ism,b_ism,T_ism,v_bp,b_bp,T_bp,b_X,T_X,v_X]
        step= np.array([0.1,2e-12,0.03,.001,5,0.1])/3.
    #############################################

    if mode == 'lm':
        print "Calculating the best parameters..."
        X = F, E, m.LyModel(Par, Const, ModelType)[0]
        print "Chi2 before fit:\t",s.chi2(X)
        print "Chi2red after fit:\t",s.chi2(X)/(len(F)-len(Par))

        Const[0] = l    # Since we want to plot the region where there is no data.

        if ModelType in [3,4]:
            f_before_fit, f_star, f_abs_ism, f_abs_bp, f_abs_X  = m.LyModel(Par,Const,ModelType)
        else:
            f_before_fit, f_star, f_abs_ism, f_abs_bp           = m.LyModel(Par,Const,ModelType)
        
        RV  = wave2RV(W,LyA,BetaPicRV)     # BetaPic rest frame
        RVo = wave2RV(Wo,LyA,BetaPicRV)

        # Plot starting point
        fig = plt.figure(figsize=(7,5))
        fontlabel_size  = 18
        tick_size       = 18
        params = {'backend': 'wxAgg', 'lines.markersize' : 2, 'axes.labelsize': fontlabel_size, 'font.size': fontlabel_size, 'legend.fontsize': fontlabel_size, 'xtick.labelsize': tick_size, 'ytick.labelsize': tick_size, 'text.usetex': True}
        plt.rcParams.update(params)
        plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
        plt.rcParams['text.usetex'] = True
        plt.rcParams['text.latex.unicode'] = True    

        plt.scatter(RVo,Fo,color='black',alpha=0.25,label='Data not used for fit')
        plt.scatter(RV,F,color='black',label='Data used for fit')

        plt.plot(v,f_star,lw=3,color='gray',label=r'$\beta$ Pictoris')
        plt.plot(v,f_abs_ism,lw=1.2,color='#FF9303',label=r'ISM')
        plt.plot(v,f_abs_bp,lw=1.2,color='#0386ff',label=r'Gas disk')
        if ModelType in [3,4]:
            plt.plot(v,f_abs_X,lw=1.2,color='purple',label=r'Component X')
        plt.plot(v,f_before_fit,lw=3,color='#FF281C',label=r'Best fit')

        # Data used to make a plot
        #np.savetxt("nh_1825_fit.dat",np.column_stack((v,f_before_fit)))
   
        plt.xlabel(r'Radial Velocity [km/s]')
        plt.ylabel('Flux (erg/s/cm$^2$/\AA)')
        
        plt.xlim(-700,600)
        plt.ylim(-2.0e-14,0.8e-13)
    
        fig.tight_layout()
        plt.show()
        #sys.exit()            
        
        Const[0] = W
        
        print "\nBest fit paramters:"
        P =  FindBestParams(Par, F, E, Const, ModelType)

        print "\nlog(N(H)) =\t" ,P[0]
        print "Fmax =\t\t"      ,P[1]
        print "uf=\t\t"         ,P[2]
        print "av=\t\t"         ,P[3]
        if ModelType == 1:
            print "Slope\t\t"   ,P[4]
        if ModelType == 2:
            print "V_H\t\t"   ,P[4]
        if ModelType == 3:
            print "v_X\t\t"     ,P[4]
            print "nh_X\t\t"    ,P[5]
        if ModelType == 4:
            print "N_H\t\t"   ,P[4]

        X = F,E,m.LyModel(P,Const, ModelType)[0]
        print "Chi2:\t\t",s.chi2(X)
        print "Chi2 reduced:\t",s.chi2(X)/(len(RV)-len(P)),"\n"

        Const[0] = l    # Since we want to plot the region where there is no data.
        if ModelType in [3,4]:
            f_after_fit, f_star, f_abs_ism, f_abs_bp, f_abs_X   = m.LyModel(P,Const,ModelType)
        else:
            f_after_fit, f_star, f_abs_ism, f_abs_bp            = m.LyModel(P,Const,ModelType)

        bin_pnts = 7
        RVb, Fb, Eb     = Bin_data(RV,F,E,bin_pnts)
        RVob, Fob, Eob  = Bin_data(RVo,Fo,Eo,bin_pnts)

        plt.scatter(RVob,Fob,color='black',alpha=0.25,label='Data not used for fit')

        plt.errorbar(RVb,Fb,yerr=Eb,fmt=None,ecolor='black',zorder=3)
        plt.scatter(RVb,Fb, marker='o', color='k',zorder=3,label='Data used for fit')
        
        '''
        for i in range(len(Wo)):
            if l[0] < Wo[i] < l[-1]:
                plt.scatter(Wo[i],Fo[i],color='black') 
        '''
        plt.plot(v,f_star,lw=3,color='gray',label=r'$\beta$ Pictoris')
        plt.plot(v,f_abs_ism,lw=1.2,color='#FF9303',label=r'ISM')
        plt.plot(v,f_abs_bp,lw=1.2,color='#0386ff',label=r'Gas disk')
        if ModelType in [3,4]:
            plt.plot(v,f_abs_X,lw=1.2,color='purple',label=r'Component X')
        plt.plot(v,f_after_fit,lw=3,color='#FF281C',label=r'Best fit')
   
        plt.xlabel(r'Radial Velocity [km/s]')
        plt.ylabel('Flux (erg/s/cm$^2$/\AA)')

        plt.xlim(-710,600)
        plt.ylim(-2.2e-14,7.3e-14)
        fig.tight_layout()
        plt.show()

        # Saving the data for plotting
        #np.savetxt(dat_directory+"Ly_Fit.dat",np.column_stack((v,f_star,f_abs_ism,f_abs_bp,f_after_fit)))

    elif mode == 'mcmc':
        #X = (F - f_before_fit),E,np.zeros(len(F)),F                                                         # Check this in relation to the Chi2 function!
        X = F, E, m.LyModel(Par,Const,ModelType)[0]
   
        chain, moves = mc.McMC(W,X,m.LyModel, ModelType, Par, Const, step,1e5)
        
        outfile = '../chains/chain_4'
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
        #print "v_X=\t\t"        ,PU3[0][0],"\t+",PU3[1][0],"\t-",PU3[2][0]
        print "V_H=\t\t"       ,PU3[0][1],"\t+",PU3[1][1],"\t-",PU3[2][1]

if __name__ == '__main__':
    main()

