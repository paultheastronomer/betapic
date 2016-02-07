from scipy.integrate import quad
import numpy as np
import matplotlib.pyplot as plt

def integrand_theta(x, rho,R,sigma):    
    return np.exp(2*rho*R*np.cos(x) / 2*sigma**2)

def integrand_R(y,rho,sigma):    
    return y*np.exp(-y**2 / 2*sigma**2)*quad(integrand_theta, 0, 2*np.pi, args=(rho,y,sigma))[0]*np.exp(-rho**2 / 2*sigma**2)

def main():

	R       = 1.25      # size of aperture. R = Diameter of 2.5"/2 = 1.25"
	FWHM    = 0.715     # FWHM in arcseconds: 6.5 pix * 0.11"/pix = 0.715"
	sigma   = 2*np.sqrt(2*np.log(2))
	rho     = np.arange(0,2.6,0.1)

	flux	= []
	shift	= []

	for i in range(len(rho)):
		flux.append(quad(integrand_R, 0, R, args=(rho[i],sigma))[0])
		shift.append(rho[i])

	
	flux = np.array(flux)
	
	plt.plot(shift,flux/flux[0],'-k')
	plt.ylabel('Flux loss')
	plt.xlabel('Arsecond shift away from central position')
	plt.show()

if __name__ == '__main__':
    main()
