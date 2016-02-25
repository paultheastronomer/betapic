import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.optimize import leastsq

def voigt_wofz(a, u):
    """ Compute the Voigt function using Scipy's wofz().

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
    """
    try:
         from scipy.special import wofz
    except ImportError:
         s = ("Can't find scipy.special.wofz(), can only calculate Voigt "
              " function for 0 < a < 0.1 (a=%g)" % a)  
         print(s)
    else:
         return wofz(u + 1j * a).real

def K(W,l,sigma_kernel):
    # LSF
    # Dispersion of the theoretical wavelength range
    # np.roll is equivalent to the IDL shift function
    dl          =   np.mean((l-np.roll(l,1))[1:])
    dwave       =   np.median((W-np.roll(W,1))[1:])  
    kernel = np.arange(-len(W)/2.,len(W)/2.,1)
    kernel = np.exp(-kernel**2/2./((sigma_kernel*dwave/dl)**2))
    kernel = kernel/np.sum(kernel)
    return kernel

def flux_star(LyA,v_bp,l,kernel,max_f,dp,uf,av):
    
    # Double Voigt profile
    delta_lambda=   LyA*(v_bp/3e5)
     
    lambda0     =   LyA                     # Lyman alpha center
    lambda1     =   LyA -dp + delta_lambda  # blue peak center
    lambda2     =   LyA +dp + delta_lambda  # red peak center
    u1          =   uf*(l-lambda1)          # blue peak wavelengths
    u2          =   uf*(l-lambda2)          # red peak wavelengths

    f           =   max_f*(voigt_wofz(av,u1)+voigt_wofz(av,u2))
    f_star      =   np.convolve(f,kernel,mode='same')
    
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

def chi2(params,F,E,W,l,sigma_kernel,dp,v_ism,v_bp,nh_ism,b_ism,b_bp,T_ism,T_bp,LyA):
    c = LyModel(params,W,l,sigma_kernel,dp,v_ism,v_bp,nh_ism,b_ism,b_bp,T_ism,T_bp,LyA)[0]
    print "Chi^2:\t",np.sum((c - F)**2 / E**2)
    return (c - F)**2 / E**2

def LyModel(params,W,l,sigma_kernel,dp,v_ism,v_bp,nh_ism,b_ism,b_bp,T_ism,T_bp,LyA):

    print "N = ",params[0],"\t","Fmax = ",params[1],"\t",\
    "uf = ",params[2],"\t","av = ",params[3]
    print "\n"
    nh_bp       = params[0]
    max_f       = params[1]
    uf          = params[2]
    av          = params[3]
    
    kernel      =   K(W,l,sigma_kernel)

    # Calculates the ISM absorption
    # see IDL function 'calculate_abs_ism'
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
    f_abs_ism   =   np.convolve(f*abs_ism,kernel,mode='same')

    # Absorption by beta Pictoris  
    f_abs_bp    =   np.convolve(f*abs_bp,kernel,mode='same')

    # Interpolation on COS wavelengths, relative to the star
    f_abs_int   =   np.interp(W,l,f_abs_con)
    
    return f_abs_int, f_star, f_abs_ism, f_abs_bp

def FindBestParams(params,F,E,W,l,sigma_kernel,dp,v_ism,v_bp,nh_ism,b_ism,b_bp,T_ism,T_bp,LyA):

    best_P, success = leastsq(chi2, params, args=(F,E,W,l,sigma_kernel,dp,v_ism,v_bp,nh_ism,b_ism,b_bp,T_ism,T_bp,LyA), maxfev=1000)

    return best_P
    
def main():    

    # skiprows=1051 remvoes left hand side data
    Wo, Fo, Eo = np.genfromtxt('Ly-alpha.dat',unpack=True)
    W, F, E = np.genfromtxt('Ly-alpha.dat',skiprows=900,skip_footer=170,unpack=True)

    E = np.ones(len(F))*1e-15
    
    ### Parameters ##############################      
    LyA     =   1215.6702

    # ISM parameters
    v_ism   =   10.0        # RV of the ISM (relative to Heliocentric)      
    nh_ism  =   18.         # Column density ISM
    b_ism   =   7.          # Turbulent velocity
    T_ism   =   7000.       # Temperature of ISM

    # Beta Pic parameters
    v_bp    =   20.5        # RV of the beta Pic (relative to Heliocentric)
    nh_bp   =   18.45       # Column density beta Pic, Fitting param
    b_bp    =   2.          # Turbulent velocity
    T_bp    =   1000.         # Temperture of gas in beta Pic disk

    max_f   =   4.395e-13   # Fitting param                 
    dp      =   0.0 
    uf      =   10.         # Fitting param
    av      =   8.          # Fitting param

    sigma_kernel    =   7.
    #############################################

    v           =   np.arange(-700,700,1)   # RV values
    l           =   LyA*(1.0 + v/3e5)       # Corresponding wavengths

    #Free parameters
    Par         =   [nh_bp,max_f,uf,av]
    P           =  FindBestParams(Par,F,E,W,l,sigma_kernel,dp,v_ism,v_bp,nh_ism,b_ism,b_bp,T_ism,T_bp,LyA)

    
    print "\n======================================="
    print "Starting paramters:"
    f_before_fit        = LyModel(Par,l,l,sigma_kernel,dp,v_ism,v_bp,nh_ism,b_ism,b_bp,T_ism,T_bp,LyA)[0]

    print "Best fit paramters:"
    f_after_fit, f_star, f_abs_ism, f_abs_bp         = LyModel(P,  l,l,sigma_kernel,dp,v_ism,v_bp,nh_ism,b_ism,b_bp,T_ism,T_bp,LyA)

    # Plot the results
    plt.scatter(Wo,Fo,color='black',alpha=0.25)
    plt.scatter(W,F,color='black')
    plt.plot(l,f_star,lw=1,color='gray')
    plt.plot(l,f_abs_ism,lw=1,color='green')
    plt.plot(l,f_abs_bp,lw=1,color='blue')
    #plt.plot(l,f_before_fit,lw=2,color='yellow')
    plt.plot(l,f_after_fit,lw=2,color='red')
    

    #W, F, E = np.genfromtxt('Ly-alpha.dat',skiprows=900,skip_footer=40,unpack=True)
    #plt.scatter(W,F,color='red')    

    plt.xlabel(r'Wavelength $\AA$')
    plt.ylabel('Flux')

    plt.xlim(1213,1217.5)
    plt.ylim(-0.3e-14,7e-14)
    plt.show()

if __name__ == '__main__':
    main()
