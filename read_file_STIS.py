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

def Extract(fits_file,j):
    f = pyfits.open(fits_file)
    tbdata = f[1].data
    net = tbdata['NET'].mean()
    wavelength = tbdata['WAVELENGTH']
    flux = tbdata['FLUX']
    err = tbdata['ERROR']
    return wavelength[j], flux[j], err[j]#, net

def main():
    fits_location = '/home/paw/science/betapic/data/HST/2016/visit_3/'
    dir_contents = os.listdir(fits_location)
    fits = sorted([fn for fn in dir_contents if fn.startswith('o') and fn.endswith('x1d.fits')])

    f = open('STIS.dat','a+')
    for i in range(len(fits)):
        if i != -1:
          for j in range(38):
            wavelength, flux, err = Extract(fits_location+fits[i],37-j)
            plt.step(wavelength,flux,lw=1.,color="black")
            plt.plot()
            for k in range(len(flux)):
                print >> f, wavelength[k], flux[k], err[k]

    f.close()

    plt.xlabel('Wavelength ($\AA$)')
    plt.ylabel('Flux')
    plt.show()

if __name__ == '__main__':
    main()
    
    # PS. Sort with: sort -k1 STIS.dat >> STIS_sorted.dat
    # in terminal later
