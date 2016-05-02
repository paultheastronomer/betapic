import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.optimize import leastsq
from matplotlib.ticker import LinearLocator
from scipy import stats

from tempfile import TemporaryFile

def voigt_wofz(a, u):
    ''' Compute the Voigt function using Scipy's wofz().

    # Code from https://github.com/nhmc/Barak/blob/\
    087602fb372812c0603441ca1ce6820e96963b88/barak/absorb/voigt.py

    Parameters
    ----------
    a: float
      Ratio of Lorentzian to Gaussian linewidths.
    u: array of floats
      The frequency or velocity offsets from the line centre, in units
      of the Gaussian broadening linewidth.

    See the notes for `voigt` for more details.
    '''
    try:
         from scipy.special import wofz
    except ImportError:
         s = ("Can't find scipy.special.wofz(), can only calculate Voigt "
              " function for 0 < a < 0.1 (a=%g)" % a)  
         print(s)
    else:
         return wofz(u + 1j * a).real

def K(W,l,sigma_kernel):
    ''' LSF
    Dispersion of the theoretical wavelength range
    np.roll is equivalent to the IDL shift function
    '''
    dl     =   np.mean((l-np.roll(l,1))[1:])
    dwave  =   np.median((W-np.roll(W,1))[1:])  
    kernel = np.arange(-len(W)/2.,len(W)/2.,1)
    kernel = np.exp(-kernel**2/2./((sigma_kernel*dwave/dl)**2))
    kernel = kernel/np.sum(kernel)
    return kernel

def flux_star(LyA,v_bp,l,kernel,max_f,dp,uf,av):
    
    # Double Voigt profile
    delta_lambda=   LyA*(v_bp/3e5)
     
    lambda0 =   LyA                     # Lyman alpha center
    lambda1 =   LyA -dp + delta_lambda  # blue peak center
    lambda2 =   LyA +dp + delta_lambda  # red peak center
    u1      =   uf*(l-lambda1)          # blue peak wavelengths
    u2      =   uf*(l-lambda2)          # red peak wavelengths

    f       =   max_f*(voigt_wofz(av,u1)+voigt_wofz(av,u2))
    f_star  =   np.convolve(f,kernel,mode='same')
    
    return f, f_star

def absorption(l,v_RV,nh,vturb,T,LyA):
    
    # [Hydrogen, Deuterium]   
    w       = [LyA,1215.3394]
    mass    = [1.,2.]
    fosc    = [0.416,0.416]
    delta   = np.array([0.627e9,0.627e9]) /(4.*np.pi)
    N_col   = np.array([1.,1.5e-5])*10**nh
    c       = 2.99793e14

    abs_ism = np.ones(len(l))

    for i in range(len(w)):
        b_wid   = np.sqrt((T/mass[i]) + ((vturb/0.129)**2))
        b       = 4.30136955e-3*b_wid
        dnud    = b*c/w[i]
        xc      = l/(1.+v_RV*1.e9/c)
        v       = 1.e4*abs(((c/xc)-(c/w[i]))/dnud)
        tv      = 1.16117705e-14*N_col[i]*w[i]*fosc[i]/b_wid
        a       = delta[i]/dnud
        hav     = tv*voigt_wofz(a,v)
        
        # I am uncertain about the translation from IDL to python here
        # To avoid underflow which occurs when you have exp(small negative number)
        for j in range(len(hav)):
            if hav[j] < 20.:      
                abs_ism[j]  =   abs_ism[j]*np.exp(-hav[j])       
            else:
                abs_ism[j]  =   0.
                
    return abs_ism

def chi2_lm(params,F,E,Const):
    c = LyModel(params,Const)[0]
    # Uncomment to show the Chi^2 value
    #print "Chi^2:\t",np.sum((c - F)**2 / E**2)
    return (c - F)**2 / E**2

def LyModel(params,Const):
    
    # Uncomment to show the changing parameters
    #print "\nN\t= ",params[0],"\t","uf = ",params[2],"\t","slope\t= ",params[4],"\n",\
    #"Fmax\t= ",params[1],"\t","av = ",params[3],"\t","offset\t= ",params[5]

    # Free parameters
    #nh_bp, b_bp, max_f, uf, av, slope = params
    nh_bp, max_f, uf, av, slope = params
    
    offset  = 1.0
    
    # Fixed parameters
    W,l,sigma_kernel,dp,v_ism,v_bp,nh_ism,b_ism,T_ism,T_bp,LyA,b_bp = Const

    kernel      =   K(W,l,sigma_kernel)

    # Calculates the ISM absorption
    abs_ism     =   absorption(l,v_ism,nh_ism,b_ism,T_ism,LyA)
    abs_bp      =   absorption(l,v_bp,nh_bp,b_bp,T_bp,LyA)

    # Stellar Ly-alpha line
    f, f_star   =   flux_star(LyA,v_bp,l,kernel,max_f,dp,uf,av)

    # Stellar spectral profile, as seen from Earth
    # after absorption by the ISM and BP CS disk.
    # Profile has been convolved with HST LSF
    #    -  in (erg cm-2 s-1 A-1)
    f_abs_con   =   np.convolve(f*abs_ism*abs_bp,kernel,mode='same')
    
    # Individual components

    # Absorption by the ISM
    f_abs_ism   =   np.convolve(f*abs_ism,kernel,mode='same')*(l*slope+offset)

    # Absorption by beta Pictoris  
    f_abs_bp    =   np.convolve(f*abs_bp,kernel,mode='same')*(l*slope+offset)

    # Interpolation on COS wavelengths, relative to the star
    f_abs_int   =   np.interp(W,l,f_abs_con)*(W*slope+offset)
    
    f_star      =   f_star*(l*slope+offset)
    
    return f_abs_int, f_star, f_abs_ism, f_abs_bp

def FindBestParams(params,F,E,Const):
    
    best_P, success = leastsq(chi2_lm, params, args=(F,E,Const), maxfev=1000)

    return best_P

def chi2(X):
  '''
  Calculated the Chi2 value
  X[0] => Obs
  X[1] => Err
  X[2] => Model
  '''
  return np.sum(((X[0] - X[2]) / X[1])**2.)

def Merit(X):
  ''' Given a Chi2 value
  we calculate a likelihood as our merit function
  '''
  return np.exp(-chi2(X)/2.)

def Distrib(x):
   '''Finds median and 68% interval of array x.'''
   y    = sorted(x)
   up   = y[int(0.8413*len(y))]
   down = y[int(0.1587*len(y))]
   med  = y[int(0.5*len(y))]
   
   #mean, sigma = np.mean(y), np.std(y)
   #print stats.norm.interval(0.68, loc=mean, scale=sigma)
   #print med,up,down 
   #sys.exit()
   return med,up,down   

def Median_and_Uncertainties(P,S,chain):
  ''' Prints the median parameter values
  with their associated uncertainties '''
  param_ans = []
  param_u   = []
  param_l   = []

  for i in range(len(P)):
    up_err      = Distrib(chain[:,P[i]])[1]-Distrib(chain[:,P[i]])[0]
    median_val  = Distrib(chain[:,P[i]])[0]
    low_err     = Distrib(chain[:,P[i]])[0]-Distrib(chain[:,P[i]])[2]

    param_ans.append(median_val)
    param_u.append(up_err)
    param_l.append(low_err)

  return param_ans,param_u,param_l

def MCMC(x,X,F,P,Const,S,C):
  '''
  x => x-axis values (In this case wavelength)
  X => Data (y,yerr,model)
  F => Function used
  P => Parameters
  S => Scale
  C => Chain length
  '''
  L         = Merit(X)
  moves     = 0
  chain     = np.zeros(shape=(C,len(P)))
  L_chain   = np.zeros(shape=(C,1))
  for i in range(int(C)):
    if i%100 == 0.:
      print (i/C)*100.," % done"
    jump        = np.random.normal(0.,1.,len(S)) * S
    P           = P + jump
    new_fit     = LyModel(P,Const)[0]
    X           = X[0],X[1],new_fit
    L_new       = Merit(X)
    L_chain[i]  = L_new
    ratio       = L_new/L

    if (np.random.random() > ratio):
      P = P - jump
      moved = 0
    else:
      L=L_new
      moved = 1
    moves += moved
    chain[i,:] = np.array(P)
  print "\nAccepted steps: ",round(100.*(moves/C),2),"%"
  return chain, moves

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
    W, F, E         = np.genfromtxt(dat_directory+'Ly_sky_subtracted.txt',unpack=True,skip_footer= 150) #7070 6145
    
    # To fit the non-sky subtracted (only cut) data uncomment the two lines below.
    #Wo, Fo, Eo         = np.genfromtxt(dat_directory+'Ly-alpha_no_CF.dat',unpack=True)
    #W, F, E         = np.genfromtxt(dat_directory+'Ly-alpha_no_CF.dat',unpack=True,skiprows=900,skip_footer= 145)
    
    
    ### Parameters ##############################      
    mode            = 'lm'        # mcmc or lm    
    LyA             = 1215.6702# Heliocentric: 1215.6702

    # ISM parameters
    v_ism           = 10.0        # RV of the ISM (relative to Heliocentric)      
    nh_ism          = 18.4         # Column density ISM
    b_ism           = 7.          # Turbulent velocity
    T_ism           = 7000.       # Temperature of ISM

    # Beta Pic parameters
    v_bp            = 20.5        # RV of the beta Pic (relative to Heliocentric)
    nh_bp           = 18.8       # Column density beta Pic, Fitting param
    b_bp            = 4.0        # Turbulent velocity
    T_bp            = 1000.       # Temperture of gas in beta Pic disk

    # Stellar emission line parameters
    max_f           = 1.7877e-10     # Fitting param                 
    dp              = 0.0 
    uf              = 7.374#3.60        # Fitting param
    av              = 1.5136#0.1        # Fitting param
    
    slope           = -8.18747e-4
    offset          = 1.0

    sigma_kernel    = 7.

    v               = np.arange(-800,500,1)         # RV values
    l               = LyA*(1.0 + v/3e5)             # Corresponding wavengths

    Par             = [nh_bp,max_f,uf,av,slope] # Free parameters
    Const           = [W,l,sigma_kernel,dp,v_ism,v_bp,nh_ism,b_ism,T_ism,T_bp,LyA,b_bp] 
    #step            = np.array([0.01,0.1,6.2e-13,0.1,0.1,0.0001])  # MCMC step size
    #step            = np.array([0.03,2.e-12,5.0,.2,1e-7])  # MCMC step size 0.3
    step            = np.array([0.02,5.e-12,0.03,.002,1e-7])  # MCMC step size 0.3
    #############################################


    if mode == 'lm':
        print "Calculating the best parameters..."
        #Par             = [nh_bp,max_f,uf,av,0]
        #Const           = [W,l,sigma_kernel,dp,v_ism,v_bp,nh_ism,b_ism,T_ism,T_bp,LyA,b_bp]
        X = F,E,LyModel(Par,Const)[0]
        print "Chi2 before fit:\t",chi2(X)

        Const[0]        = l    # Since we want to plot the region where there is no data.
        f_before_fit, f_star, f_abs_ism, f_abs_bp         = LyModel(Par,Const)

        RV = wave2RV(W,LyA,20.5)     # Heliocentric rest frame
        RVo = wave2RV(Wo,LyA,20.5)



        # Plot starting point
        fig = plt.figure(figsize=(8,6))
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
        plt.plot(v,f_before_fit,lw=3,color='#FF281C',label=r'Best fit')


        
        #np.savetxt("nh_1825_fit.dat",np.column_stack((v,f_before_fit)))
   
        plt.xlabel(r'Radial Velocity [km/s]')
        plt.ylabel('Flux (erg/s/cm$^2$/\AA)')
   
        plt.xlabel(r'Wavelength \AA')
        plt.ylabel('Flux')

        
        plt.xlim(-700,600)
        plt.ylim(-2.0e-14,0.4e-13)
    
        fig.tight_layout()
        #plt.savefig('Ly_alpha_b5.pdf', bbox_inches='tight', pad_inches=0.1,dpi=300)
        plt.show()
        #sys.exit()
        
        
        
        Const           = [W,l,sigma_kernel,dp,v_ism,v_bp,nh_ism,b_ism,T_ism,T_bp,LyA,b_bp]
        print "Best fit paramters:"
        P =  FindBestParams(Par,F,E,Const)

        print Par
        print P

        print "\nlog(N(H)) =\t" ,P[0]
        #print "b_bp=\t\t"       ,P[1]
        print "Fmax =\t\t"      ,P[1]
        print "uf=\t\t"         ,P[2]
        print "av=\t\t"         ,P[3]
        print "slope\t\t"       ,P[4]
        #print "offset\t\t"      ,P[5]

        X = F,E,LyModel(P,Const)[0]
        print "Chi2 after fit:\t",chi2(X)

        Const[0] = l    # Since we want to plot the region where there is no data.
        f_after_fit, f_star, f_abs_ism, f_abs_bp         = LyModel(P,Const)

        bin_pnts = 3
        RVb, Fb, Eb     = Bin_data(RV,F,E,bin_pnts)
        RVob, Fob, Eob     = Bin_data(RVo,Fo,Eo,bin_pnts)

        #remove this later
        #Eb = Eb/5.
    
        #plt.plot(W1,AG0)
        plt.scatter(RVob,Fob,color='black',alpha=0.25,label='Data not used for fit')
        plt.errorbar(RVb,Fb,yerr=Eb,color='black',label='Data used for fit')
        
        '''
        for i in range(len(Wo)):
            if l[0] < Wo[i] < l[-1]:
                plt.scatter(Wo[i],Fo[i],color='black') 
        '''
        plt.plot(v,f_star,lw=3,color='gray',label=r'$\beta$ Pictoris')
        plt.plot(v,f_abs_ism,lw=1.2,color='#FF9303',label=r'ISM')
        plt.plot(v,f_abs_bp,lw=1.2,color='#0386ff',label=r'Gas disk')
        #plt.plot(l,f_before_fit,lw=2,color='yellow')
        plt.plot(v,f_after_fit,lw=3,color='#FF281C',label=r'Best fit')
   
        plt.xlabel(r'Radial Velocity [km/s]')
        plt.ylabel('Flux (erg/s/cm$^2$/\AA)')

        #plt.xlim(1212.5,1217.5)
        plt.xlim(-700,600)
        plt.ylim(-2.0e-14,0.4e-13)
        #locs = [1213.,  1214.,  1215.,  1216.,  1217.]
        #plt.xticks(locs, map(lambda x: "%g" % x, locs))
        #plt.legend(loc='upper left', numpoints=1)
        fig.tight_layout()
        #plt.savefig('Ly_alpha.pdf', bbox_inches='tight', pad_inches=0.1,dpi=300)
        plt.show()

        # Saving the data for plotting
        np.savetxt(dat_directory+"Ly_Fit.dat",np.column_stack((v,f_star,f_abs_ism,f_abs_bp,f_after_fit)))


    elif mode == 'mcmc':
        #X = (F - f_before_fit),E,np.zeros(len(F)),F                                                         # Check this in relation to the Chi2 function!
        X = F,E,LyModel(Par,Const)[0]

        
        chain, moves = MCMC(W,X,LyModel,Par,Const,step,1e5)
        
        outfile = 'chain4'#TemporaryFile() #Par             = [nh_bp,b_bp,max_f,uf,av,0] # Free parameters
        np.savez(outfile, nh_bp = chain[:,0], max_f = chain[:,1], uf = chain[:,2], av = chain[:,3], slope = chain[:,4])
        
        Pout = chain[moves,:]
        #print Pout
        P_plot1 = [0,1]
        P_plot2 = [2,3]
        P_plot3 = [4,4]
        PU1 = Median_and_Uncertainties(P_plot1,step,chain)
        PU2 = Median_and_Uncertainties(P_plot2,step,chain)
        PU3 = Median_and_Uncertainties(P_plot3,step,chain)
        
        print "\nlog(N(H)) =\t" ,PU1[0][0],"\t+",PU1[1][0],"\t-",PU1[2][0]
        print "Fmax =\t\t"      ,PU1[0][1],"\t+",PU1[1][1],"\t-",PU1[2][1]
        print "uf=\t\t"       ,PU2[0][0],"\t+",PU2[1][0],"\t-",PU2[2][0]
        print "av=\t\t"         ,PU2[0][1],"\t+",PU2[1][1],"\t-",PU2[2][1]
        print "slope=\t\t"       ,PU3[0][0],"\t+",PU3[1][0],"\t-",PU3[2][0]
        #print "slope\t\t"      ,PU3[0][1],"\t+",PU3[1][1],"\t-",PU3[2][1]

if __name__ == '__main__':
    main()
