
import pyfits, os
import matplotlib.pyplot as plt
import numpy as np
import sys
from scipy.integrate import quad


def integrand_theta(theta, rho,R,sigma):	# Integration with respect to theta
    return np.exp(2*rho*R*np.cos(theta) / 2*sigma**2)

def integrand_R(R,rho,sigma):			# Integration with respect to R  
    return R*np.exp(-R**2 / 2*sigma**2)*quad(integrand_theta, 0, 2*np.pi, args=(rho,R,sigma))[0]*np.exp(-rho**2 / 2*sigma**2)

def iround(x):
    """iround(number) -> integer
    Round a number to the nearest integer."""
    return int(round(x) - .5) + (x > 0)

def main():

    dat_directory = "/home/paw/science/betapic/data/HST/dat/" 

    rest_wavelength = 1215.6702 
    RV_BP           = 20.5
    
    
    
    r1              = 6000#9027#8350#14250 # High flux region 9027#
    r2              = 15200#15100#9228#11000#15100   9228#

    W, RV, F0_0, E0_0, AG0, AG0err                                                  = np.genfromtxt(dat_directory+'A_2014.dat',unpack=True)
    W, RV, F0_1, E0_1, F1_1, E1_1, F2_1, E2_1, AG1, AG1err, F_ave_w_1               = np.genfromtxt(dat_directory+'A_10Dec.dat',unpack=True)
    W, RV, F0_2, E0_2, F1_2, E1_2, F2_2, E2_2, F3_2, E3_2, AG2, AG2err, F_ave_w_2   = np.genfromtxt(dat_directory+'A_24Dec.dat',unpack=True)
    W, RV, F0_3, E0_3, F1_3, E1_3, F2_3, E2_3, F3_3, E3_3, AG3, AG3err, F_ave_w_3   = np.genfromtxt(dat_directory+'A_30Jan.dat',unpack=True)

    #print W[9027]
    #print W[9228]
    #sys.exit()
    
    plt.plot(W,F0_1)
    plt.plot(W[r1:r2],F2_1[r1:r2])
    plt.show()
    #sys.exit()    

    fig = plt.figure(figsize=(8,6))
    fontlabel_size = 18
    tick_size = 18
    params = {'backend': 'wxAgg', 'lines.markersize' : 2, 'axes.labelsize': fontlabel_size, 'font.size': fontlabel_size, 'legend.fontsize': fontlabel_size, 'xtick.labelsize': tick_size, 'ytick.labelsize': tick_size, 'text.usetex': True}
    plt.rcParams.update(params)
    plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    plt.rcParams['text.usetex'] = True
    plt.rcParams['text.latex.unicode'] = True   


    ax1 = plt.subplot('111')
    flux_0_arcsec   = (np.median(F0_2[r1:r2])+np.median(F0_3[r1:r2]))/2.
    flux_08_arcsec  = (np.median(F1_2[r1:r2])+np.median(F2_2[r1:r2])+np.median(F1_3[r1:r2])+np.median(F2_3[r1:r2]))/4.
    flux_11_arcsec  = (np.median(F3_2[r1:r2])+np.median(F3_3[r1:r2]))/2.
    line1           = 0.9
    line2           = 1.2

    FWHM		= 2.25	# FWHM in Cross dispersion direction p.56 COS manual
    R       	= 1.25	# Aperture size. R = Diameter of 2.5"/2 = 1.25"
	
    sigma   	= FWHM/ 2*np.sqrt(2*np.log(2))	# Convert from FWHM --> sigma
    rho     	= np.arange(0,4.0,0.1)			# [arcsec]

    flux	= []
    shift	= []

    for i in range(len(rho)):
        flux.append(quad(integrand_R, 0, R, args=(rho[i],sigma))[0])
        shift.append(rho[i])
	
    flux = np.array(flux)
    shift = np.array(shift)
    
    flux_N = flux_0_arcsec*(flux/flux[0])

    S   = [0.,0.8,1.1]

    plt.scatter(S[0],np.median(F0_2[r1:r2]),marker='o',s=70,facecolor='none', edgecolor='k')
    plt.scatter(S[0],np.median(F0_3[r1:r2]),marker='o',s=70,facecolor='none', edgecolor='k')
    
    plt.scatter(S[1],np.median(F2_2[r1:r2]),marker='o',s=70,facecolor='none', edgecolor='k')
    plt.scatter(S[1],np.median(F2_3[r1:r2]),marker='o',s=70,facecolor='none', edgecolor='k')
    
    plt.scatter(S[2],np.median(F3_2[r1:r2]),marker='o',s=70,facecolor='none', edgecolor='k')
    plt.scatter(S[2],np.median(F3_3[r1:r2]),marker='o',s=70,facecolor='none', edgecolor='k')

    plt.plot(shift,flux_N,'-r',lw=1.2)


    



    # These are dummy arrays used for the 10 Dec 2015 observations
    f_empty = []
    e_empty = []
        
    
    decrease1       = 100*(flux_0_arcsec-flux_08_arcsec)/flux_0_arcsec
    decrease2       = 100*(flux_0_arcsec-flux_11_arcsec)/flux_0_arcsec
    
    print decrease1
    print decrease2
    
    plt.plot([line1-0.05,line1],[flux_0_arcsec,flux_0_arcsec],lw=1.2,color='black')
    plt.plot([line1-0.05,line1],[flux_08_arcsec,flux_08_arcsec],lw=1.2,color='black')
    plt.plot([line1,line1],[flux_08_arcsec,flux_0_arcsec],lw=1.2,color='black')
    plt.text(line1+0.01,(flux_0_arcsec+flux_08_arcsec)/2.,str(int(round(decrease1,0)))+'\,\% Decrease',va='center',rotation='-90')
    
    plt.plot([line2-0.05,line2],[flux_0_arcsec,flux_0_arcsec],lw=1.2,color='black')
    plt.plot([line2-0.05,line2],[flux_11_arcsec,flux_11_arcsec],lw=1.2,color='black')
    plt.plot([line2,line2],[flux_11_arcsec,flux_0_arcsec],lw=1.2,color='black')
    plt.text(line2+0.01,(flux_0_arcsec+flux_08_arcsec)/2.,str(int(round(decrease2,0)))+'\,\% Decrease',va='center',rotation='-90')
    
    x1 = [0.0,0.5,0.8,1.1,1.5,2.0,2.5]
    plt.xticks(x1,x1)      
    
    plt.ylim(0.,3.2e-13)
    plt.xlim(-0.6,2.6)
    plt.xlabel('Shift [arcsec]')
    plt.ylabel('Flux (erg/s/cm$^2$/\AA)')


  
    
    ax2 = ax1.twiny()
    plt.xlim(-0.6,2.6)
    plt.xlabel('Shift in RV (km/s)')
    
    x2 = np.round(0.00986246127217*np.array(x1)/0.0285,3)
    
    rest_wavelength = 1215.6702
    RV_BP=20.5
    c = 299792458
    rest_wavelength = rest_wavelength*(RV_BP*1.e3)/c + rest_wavelength # Convert to beta pic reference frame
    delta_wavelength = x2
    RV = ((delta_wavelength/rest_wavelength)*c)/1.e3	# km/s
    
    print type(RV)
    print RV
    RVr = [0,43,68,94,128,171,213]#round(RV,2)

    plt.xticks(x1,RVr)  

    y1 = [flux_0_arcsec,flux_08_arcsec,flux_11_arcsec,1e-13,0.5e-13,0.0]
    percent = np.zeros(len(y1))
    for k in range(len(y1)):
        percent[k] = round(100*((flux_N[0] - y1[k])/flux_N[0]),1)
        print y1[k],percent[k]


    y3 = percent
    ax3 = ax1.twinx()
    plt.ylim(0.,3.2e-13)
    
    y4 = [100,83,65,43,22,0][::-1]#manual
    plt.yticks(y1,y4)
    plt.ylabel(r'Flux loss (\%)')
    #for j in range(len(shift)):
    #  print shift[j], 100*(flux_N[0] - flux_N[j])/flux_N[0]#[0.8]
    #print np.argmax(flux_N[0.8])
    #sys.exit()

    fig.tight_layout()
    plt.savefig('../plots/theoretical_flux_loss.pdf', bbox_inches='tight', pad_inches=0.1)
    
    #for i in range(len(F)):
    #    plt.step(W,F[i])
    #    plt.step(W[start:stop],F[i][start:stop])
    #plt.legend(loc='upper left', numpoints=1)
    
    #fig.tight_layout()
    #plt.savefig('FEB_quiet_regions.pdf', bbox_inches='tight', pad_inches=0.1,dpi=300)
    plt.show()

if __name__ == '__main__':
    main()
