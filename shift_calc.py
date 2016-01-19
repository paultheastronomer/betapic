#
# Shift spectra
#
# Written by: Paul A. Wilson
# pwilson@iap.fr
#

import pyfits, os
import matplotlib.pyplot as plt
import numpy as np
import sys


def Extract(fits_file):
    f = pyfits.open(fits_file)
    tbdata = f[1].data
    wavelength = tbdata['WAVELENGTH']
    flux = tbdata['FLUX']
    err = tbdata['ERROR']
    return wavelength[1], flux[1], err[1]

#https://scipy.github.io/old-wiki/pages/Cookbook/SavitzkyGolay
def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    import numpy as np
    from math import factorial
    
    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError, msg:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')

def shift_spec(ref,spec,error,wave,start,stop):
	#ref		= savitzky_golay(ref, 41, 3)[start:stop]
	#spec	= savitzky_golay(spec, 41, 3)[start:stop]
	ref		= ref[start:stop]
	spec	= spec[start:stop]	
	error	= error[start:stop]
	wave	= wave[start:stop]
	
	ref = ref - np.mean(ref)
	spec = spec - np.mean(spec)

	c = np.correlate(spec,ref,mode='full')

	x = np.arange(c.size)
	c_max = np.argmax(c)

    
	if ref.size-1 > c_max:
	  shift = wave[ref.size-1]-wave[np.argmax(c)]
	  units = (ref.size-1)-np.argmax(c)
	  print "Shift:",shift,"Angstrom or ",units,"pixel elements" #flux.size-1 because python starts index at 0
	  zeros = np.zeros(units)
	  spec	= np.concatenate((zeros, spec), axis=1)[:-units]
	else:
	  c = np.correlate(ref,spec,mode='full')
	  shift = wave[np.argmax(c)]-wave[ref.size-1]
	  units	= abs(np.argmax(c)-(ref.size-1))
	  print "Shift:",shift,"Angstrom or",-1*units,"pixel elements"
	  zeros = np.zeros(units)
	  spec	= np.concatenate((spec, zeros), axis=1)[units:]
	
	return wave,spec,error

def main():
    fits_location = '/home/paw/science/betapic/data/HST/2016/visit_2/'
    dir_contents = os.listdir(fits_location)
    fits = sorted([fn for fn in dir_contents if fn.startswith('l') and fn.endswith('sum.fits')])

    wavelength, flux, err 		= Extract(fits_location+fits[0]) # Ref spec
    wavelength1, flux1, err1 	= Extract(fits_location+fits[1]) # Comparison spec
    wavelength2, flux2, err2 	= Extract(fits_location+fits[2])
    wavelength3, flux3, err3 	= Extract(fits_location+fits[3])

    start	= 10000
    stop	= 16000
    
    F = [[] for _ in range(len(fits)-1)]
    W = [[] for _ in range(len(fits)-1)]
    E = [[] for _ in range(len(fits)-1)]
    
    W[0],F[0],E[0]	=	shift_spec(flux,flux,err,wavelength,start,stop)
    W[1],F[1],E[1]	=	shift_spec(flux,flux1,err1,wavelength,start,stop)
    W[2],F[2],E[2]	=	shift_spec(flux,flux2,err2,wavelength,start,stop)
    W[3],F[3],E[3]	=	shift_spec(flux,flux3,err3,wavelength,start,stop)

    F = np.array(F)
    E = np.array(E)
    print E[1]
    print E[1][-1]

    F_median = np.median(F, axis=0)
    F_ave =  np.average(F, axis=0)
    F_ave_w =  np.average(F, axis=0,weights=1/E**2)

    #plt.errorbar(W[0],F[3],yerr=E[3])
    plt.step(W[0],F[0])
    #plt.plot(W[0],F_median)
    #plt.plot(W[0],F_ave)
    plt.step(W[0],F_ave_w)
    #plt.plot(W[0],F_ave-F_ave_w)
    
    #plt.plot((x-(flux.size-1)),c)
    plt.show()
    
if __name__ == '__main__':
    main()
