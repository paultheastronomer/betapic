from scipy.integrate import quad
import numpy as np
import matplotlib.pyplot as plt

def integrand_theta(theta, rho,R,sigma):	# Integration with respect to theta
    return np.exp(2*rho*R*np.cos(theta) / 2*sigma**2)

def integrand_R(R,rho,sigma):			# Integration with respect to R  
    return R*np.exp(-R**2 / 2*sigma**2)*quad(integrand_theta, 0, 2*np.pi, args=(rho,R,sigma))[0]*np.exp(-rho**2 / 2*sigma**2)

def main():

	FWHM		= 1.7	# FWHM in Cross dispersion direction p.56 COS manual
	R       	= 1.25	# Aperture size. R = Diameter of 2.5"/2 = 1.25"
	
	sigma   	= FWHM/ 2*np.sqrt(2*np.log(2))	# Convert from FWHM --> sigma
	rho     	= np.arange(0,2.5,0.1)			# [arcsec]

	flux	= []
	shift	= []

	for i in range(len(rho)):
		flux.append(quad(integrand_R, 0, R, args=(rho[i],sigma))[0])
		shift.append(rho[i])
	
	flux = np.array(flux)
	shift = np.array(shift)
	
	plt.plot(shift,flux/flux[0],'-k')
	plt.ylabel('Flux loss')
	plt.xlabel('Arsecond shift away from central position')
	plt.show()

if __name__ == '__main__':
    main()
