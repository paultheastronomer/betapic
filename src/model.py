import numpy as np

class Model:
    '''
    A collection of functions for modeling the line absorption.
    '''
        
    def voigt_wofz(self, a, u):

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


    def K(self, W, l, sigma_kernel):
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

        
    def flux_star(self, LyA,BetaPicRV,l,kernel,max_f,dp,uf,av):  

        # Double Voigt profile
        delta_lambda =   LyA*(BetaPicRV/3e5)
         
        lambda0 =   LyA                     # Lyman alpha center
        lambda1 =   LyA -dp + delta_lambda  # blue peak center
        lambda2 =   LyA +dp + delta_lambda  # red peak center
        u1      =   uf*(l-lambda1)          # blue peak wavelengths
        u2      =   uf*(l-lambda2)          # red peak wavelengths

        f       =   max_f*(self.voigt_wofz(av,u1)+self.voigt_wofz(av,u2))
        f_star  =   np.convolve(f,kernel,mode='same')
        
        return f, f_star
        

    def absorption(self, l,v_bp,nh,vturb,T,LyA):
        
        # [Hydrogen, Deuterium]   
        w       = [LyA,1215.3394]
        mass    = [1.,2.]
        fosc    = [0.416,0.416]
        delta   = np.array([0.627e9,0.627e9]) /(4.*np.pi)
        N_col   = np.array([1.,1.5e-5])*10**nh
        c       = 2.99793e14
        k       = 1.38064852e-23    # Boltzmann constant in J/K = m^2*kg/(s^2*K) in SI base units
        u       = 1.660539040e-27   # Atomic mass unit (Dalton) in kg

        abs_ism = np.ones(len(l))

        for i in range(len(w)):
            b_wid   = np.sqrt((T/mass[i]) + ((vturb/np.sqrt(2*k/u)/1e3)**2)) # non-thermal + thermal broadening
            b       = 4.30136955e-3*b_wid
            dnud    = b*c/w[i]
            xc      = l/(1.+v_bp*1.e9/c)
            v       = 1.e4*abs(((c/xc)-(c/w[i]))/dnud)
            tv      = 1.16117705e-14*N_col[i]*w[i]*fosc[i]/b_wid
            a       = delta[i]/dnud
            hav     = tv*self.voigt_wofz(a,v)
            
            # To avoid underflow which occurs when you have exp(small negative number)
            for j in range(len(hav)):
                if hav[j] < 20.:      
                    abs_ism[j]  =   abs_ism[j]*np.exp(-hav[j])       
                else:
                    abs_ism[j]  =   0.
                    
        return abs_ism


    def LyModel(self, params, Const, ModelType):
        
        '''
        ModelType refers to the kind of model you are interested in.

        ModelType = 1
        ========================================================================
        This model includes a slope component
        
            slope   = The slope of model. i.e. model*(slope + 1.)
        ========================================================================    

        ModelType = 2
        ========================================================================
        No extra components but the H absorption is not fixed to the beta pic
        reference frame, but is free to vary.
        ========================================================================            
                
        ModelType = 3
        ========================================================================
        This model includes an additional component X decribed by the free
        paramteres:
        
            v_X     = the velocity of the additional component
            nh_X    = the column density of the additional component
        ========================================================================    

        ModelType = 4
        ========================================================================
        No extra components. NEEDS CHECKING
        ========================================================================  
        
        '''
        
        # Free parameters
        if ModelType == 1:
            nh_bp, max_f, uf, av, slope     = params

        if ModelType == 2:
            nh_bp, max_f, uf, av, v_bp      = params
            
        if ModelType == 3:
            nh_bp, max_f, uf, av, v_X, nh_X = params

        if ModelType == 4:
            nh_bp, max_f, uf, av = params
        
        # Fixed parameters
        if ModelType == 1:
            W,l,LyA,BetaPicRV,sigma_kernel,dp,v_ism,nh_ism,b_ism,T_ism,v_bp,b_bp,T_bp           = Const

        if ModelType == 2:
            W,l,LyA,BetaPicRV,sigma_kernel,dp,v_ism,nh_ism,b_ism,T_ism,b_bp,T_bp                = Const
            
        if ModelType == 3:
            W,l,LyA,BetaPicRV,sigma_kernel,dp,v_ism,nh_ism,b_ism,T_ism,v_bp,b_bp,T_bp,b_X,T_X   = Const

        if ModelType == 4:
            W,l,LyA,BetaPicRV,sigma_kernel,dp,v_ism,nh_ism,b_ism,T_ism,v_bp,b_bp,T_bp           = Const

        kernel      =   self.K(W,l,sigma_kernel)

        # Calculates the ISM absorption
        abs_ism     =   self.absorption(l,v_ism,nh_ism,b_ism,T_ism,LyA)
        abs_bp      =   self.absorption(l,v_bp,nh_bp,b_bp,T_bp,LyA)
        
        if ModelType == 3:
            abs_X       =   self.absorption(l,v_X,nh_X,b_X,T_X,LyA)

        # Stellar Ly-alpha line
        f, f_star   =   self.flux_star(LyA,BetaPicRV,l,kernel,max_f,dp,uf,av)

        # Stellar spectral profile, as seen from Earth
        # after absorption by the ISM and BP CS disk.
        # Profile has been convolved with HST LSF
        #    -  in (erg cm-2 s-1 A-1)

        if ModelType == 3:
            f_abs_con   =   np.convolve(f*abs_ism*abs_bp*abs_X, kernel, mode='same')
        else:
            f_abs_con   =   np.convolve(f*abs_ism*abs_bp, kernel, mode='same')
        
        # Absorption by the ISM
        if ModelType == 1:
            f_abs_ism   =   np.convolve(f*abs_ism, kernel, mode='same')*(l*slope+1.0)
        else:
            f_abs_ism   =   np.convolve(f*abs_ism, kernel, mode='same')
        
        # Absorption by beta Pictoris  
        if ModelType == 1:
            f_abs_bp    =   np.convolve(f*abs_bp, kernel, mode='same')*(l*slope+1.0)
        else:
            f_abs_bp    =   np.convolve(f*abs_bp, kernel, mode='same')

        # Absorption by component X  
        if ModelType == 3:
            f_abs_X    =   np.convolve(f*abs_X, kernel, mode='same')
        
        # Interpolation on COS wavelengths, relative to the star
        if ModelType == 1:
            f_abs_int   =   np.interp(W,l,f_abs_con)*(W*slope+1.0)
            f_star      =   f_star*(l*slope+1.0)
        else:
            f_abs_int   =   np.interp(W,l,f_abs_con)
                    
        if ModelType == 3:
            return f_abs_int, f_star, f_abs_ism, f_abs_bp, f_abs_X
        else:
            return f_abs_int, f_star, f_abs_ism, f_abs_bp
