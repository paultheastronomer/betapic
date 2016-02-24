import numpy as np
import matplotlib.pyplot as plt
from scipy import special
import sys

def voigt(x, y):
	# The Voigt function is also the real part of 
	# w(z) = exp(-z^2) erfc(iz), the complex probability function,
	# which is also known as the Faddeeva function. Scipy has 
	# implemented this function under the name wofz()
	z = x + 1j*y
	I = special.wofz(z).real
	return I


def absorption(l,v,nh,T,LyA):
    
    # [Hydrogen, Deuterium]
    
    w       = [LyA,1215.3394]
    mass    = [1.,2.]
    fosc    = [0.416,0.416]
    delta   = np.array([0.627e9,0.627e9]) /(4.*np.pi)
    N_col   = np.array([1.,1.5e-5])*10**nh
    c       = 2.99793e14

    for i in range(len(w)):
        b_wid   = np.sqrt((T/mass[i]) + ((v/0.129)**2))
        b       = 4.30136955e-3*b_wid
        dnud    = b*c/w[i]

        xc      = l/(1.+v*1.e9/c)
        v       = 1.e4*abs(((c/xc)-(c/w[i]))/dnud)
        tv      = 1.16117705e-14*N_col[i]*w[i]*fosc[i]/b_wid
        a       = delta[i]/dnud
        hav     = tv*voigt(a,v)
              
        abs_ism = np.ones(len(hav))
        
        # I am uncertain about the translation from IDL to python here
        # The original IDL code is below
        for j in range(len(hav)):
            if hav[j] < 20.:      
                abs_ism[j]  =   abs_ism[j]*np.exp(-hav[j])       
            else:
                abs_ism[j]  =   0.

        '''
        ind=where(hav lt 20.,count)
         if  count gt 0 then $
           abs_ism[ind]=abs_ism[ind]*exp(-hav[ind])  
        ind=where(hav ge 20.,count)
         if  count gt 0 then $
           abs_ism[ind]=0. 
        endfor ; line
        '''
        
    return abs_ism

def main():    

    W, F, E = np.genfromtxt('Ly-alpha.dat',unpack=True)
    
    ### Parameters ##############################
    LyA     =   1215.6737

    # ISM parameters
    v_ism   =   10.0          
    nh_ism  =   18.
    b_ism   =   7.
    T_ism   =   7000.

    # Beta Pic parameters
    v_bp    =   20.5          
    nh_bp   =   18.85
    b_bp    =   2.
    T_bp    =   1000.

    max_f   =   6e-13                  
    dp      =   0.0 
    uf      =   3.
    av      =   7.      

    sigma_kernel    =   7.
    #############################################

    v           =   np.arange(-len(W)/2.,len(W)/2.,1)   # RV values
    l           =   LyA*(1.0 + v/3e5)       # wavelength as emitted from the star
    
    # Calculates the ISM absorption
    # see IDL function 'calculate_abs_ism'
    abs_ism =   absorption(l,v_ism,nh_ism,T_ism,LyA)
    abs_bp  =   absorption(l,v_bp,nh_bp,T_bp,LyA)
    
    # LSF
    # Dispersion of the theoretical wavelength range
    # Dispersion de la plage de longueurs d'onde theorique
    # np.roll is equivalent to the IDL shift function

    dl          =   np.mean((l-np.roll(l,1))[1:])
    dwave       =   np.median((W-np.roll(W,1))[1:])     
    kernel      =   v
    kernel      =   np.exp(-kernel**2/2./((sigma_kernel*dwave/dl)**2))
    kernel      =   kernel/np.sum(kernel)
    
    # Double Voigt profile
    delta_lambda=   LyA*(v_bp/3e5)
     
    lambda0     =   LyA      #lyman alpha center
    lambda1     =   LyA -dp + delta_lambda  #blue peak center
    lambda2     =   LyA +dp + delta_lambda  #red peak center

    u1          =   uf*(l-lambda1)   #blue peak wavelengths
    u2          =   uf*(l-lambda2)   #red peak wavelengths

    # I don't understand the above part. u1 == u2, no?

    f           =   max_f*(voigt(av,u1)+voigt(av,u2))

    # Stellar spectral profile, as seen from Earth after absorption by the ISM and BP CS disk   
    #    -  in (erg cm-2 s-1 A-1)
    f_abs       =   f*abs_ism*abs_bp

    #Stellar spectral profile, after convolution by Hubble LSF 
    #    -  in (erg cm-2 s-1 A-1)
    #print len(f_abs),len(kernel)
    f_abs_con   =   np.convolve(f_abs,kernel,mode='same')
    
    # Interpolation on COS wavelengths, relative to the star
    print len(f_abs_con),len(l),len(W)

    f_abs_int   =   np.interp(f_abs_con,l,W)

    f_abs_bp_0  =   f*abs_bp
    f_abs_bp    =   np.convolve(f_abs_bp_0,kernel,mode='same')

    f_abs_ism_0 =   f*abs_ism
    f_abs_ism   =   np.convolve(f_abs_ism_0,kernel,mode='same')

    f_star      =   np.convolve(f,kernel,mode='same')

    # Plot the results
    plt.plot(l,f_star,color='red')
    plt.plot(W,F,color='black')

    plt.xlabel(r'Wavelength \AA')
    plt.ylabel('Flux')

    plt.ylim(-0.3e-14,5.5e-14)
    plt.show()

if __name__ == '__main__':
    main()
