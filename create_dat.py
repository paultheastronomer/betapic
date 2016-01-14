#
# Create .txt files from
# x1d COS images
#
# Written by: Paul A. Wilson
# pwilson@iap.fr
#

import pyfits
import os
import numpy as np


def Extract(fits_file):
    f = pyfits.open(fits_file)
    # To see the columns uncommment the next line
    # print f[1].data.columns
    tbdata = f[1].data
    net = tbdata['NET']
    wavelength = tbdata['WAVELENGTH']
    flux = tbdata['FLUX']
    err = tbdata['ERROR']
    return wavelength[0], flux[0], err[0], net[0]

def main():
    fits_location = '/home/paw/science/betapic/data/new_data/'
    dir_contents = os.listdir(fits_location)
    fits = sorted([fn for fn in dir_contents if fn.startswith('l') and fn.endswith('_x1dsum.fits')])
        
    for i in range(len(fits)):
      wavelength, flux, err, net = Extract(fits_location+fits[i])
      np.savetxt("data/spectrum.dat",np.column_stack((wavelength,flux,err)))

if __name__ == '__main__':
    main()
