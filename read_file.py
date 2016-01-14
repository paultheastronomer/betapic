#
# Read x1d COS images
#
# Written by: Paul A. Wilson
# pwilson@iap.fr
#

import pyfits, os
import matplotlib.pyplot as plt
import numpy as np


def Extract_Time(fits_file):
    f = pyfits.open(fits_file)
    start = float(f[1].header['EXPSTART'])
    end = float(f[1].header['EXPEND'])
    img_time = (start+end)/2.
    return img_time

def Extract(fits_file):
    f = pyfits.open(fits_file)
    tbdata = f[1].data
    net = tbdata['NET'].mean()
    wavelength = tbdata['WAVELENGTH']
    flux = tbdata['FLUX']
    err = tbdata['ERROR']
    return wavelength[1], flux[1], err[1], net

def main():
    fits_location = '/home/paw/science/betapic/data/HST/2016/visit_2/'
    dir_contents = os.listdir(fits_location)
    fits = sorted([fn for fn in dir_contents if fn.startswith('l') and fn.endswith('sum.fits')])

    for i in range(len(fits)):
        if i != -1:

          wavelength, flux, err, net = Extract(fits_location+fits[i])

          plt.step(wavelength,flux,lw=1.,label=str(fits[i]))
          plt.plot()

    plt.ylim(0,1.5e-13)
    plt.xlabel('Wavelength ($\AA$)')
    plt.ylabel('Flux')
    plt.legend(loc='upper right', numpoints=1)
    #plt.savefig('AG.pdf', bbox_inches='tight', pad_inches=0.1)
    plt.show()

if __name__ == '__main__':
    main()
