#
# Shift spectra
#
# Written by: Paul A. Wilson
# pwilson@iap.fr
#

import pyfits, os
import matplotlib.pyplot as plt
import numpy as np


def Extract(fits_file):
    f = pyfits.open(fits_file)
    tbdata = f[1].data
    wavelength = tbdata['WAVELENGTH']
    flux = tbdata['FLUX']
    err = tbdata['ERROR']
    return wavelength[1], flux[1], err[1]

def main():
    fits_location = '/home/paw/science/betapic/data/HST/2016/visit_2/'
    dir_contents = os.listdir(fits_location)
    fits = sorted([fn for fn in dir_contents if fn.startswith('l') and fn.endswith('sum.fits')])

    wavelength, flux, err 		= Extract(fits_location+fits[0])
    wavelength1, flux1, err1 	= Extract(fits_location+fits[1])
    
    start	= 10000
    stop	= 14000
    
    wavelength	= wavelength[start:stop]
    flux 		= flux[start:stop]
    flux1 		= flux1[start:stop]
    
    c = np.correlate(flux1,flux,mode='full')
    x = np.arange(c.size)
    
    c_max = np.argmax(c)
    
    if flux.size-1 > c_max:
      print "Shift:",wavelength[np.argmax(c)]-wavelength[flux.size-1] #flux.size-1 because python starts index at 0
    else:
      c = np.correlate(flux,flux1,mode='full')
      print "Shift:",wavelength[flux.size-1]-wavelength[np.argmax(c)]
    print len(wavelength),len(c)
    
    plt.plot((x-(flux.size-1)),c)
    plt.show()
    
if __name__ == '__main__':
    main()
