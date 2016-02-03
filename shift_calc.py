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


def Extract(fits_file, part,start,stop):
    f = pyfits.open(fits_file)
    tbdata = f[1].data
    wavelength = tbdata['WAVELENGTH']
    flux = tbdata['FLUX']
    err = tbdata['ERROR']
    if part == 'A':
        return wavelength[0][start:-stop], flux[0][start:-stop], err[0][start:-stop]
    else:
        return wavelength[1][start:-stop], flux[1][start:-stop], err[1][start:-stop]

def shift_spec(ref,spec,error,wave,start,stop):
    # This routine correlates the spectrum: spec
    # with the reference spectrum: ref
	ref_c	= ref[start:stop]   # Reference spectrum
	spec_c	= spec[start:stop]	# Spectrum to be shifted
	error_c	= error[start:stop] # Error on spec to be shifted
	wave_c	= wave[start:stop]  # Wavelength of spec to be shifted
	
	ref_c       = ref_c - np.mean(ref_c)
	spec_c      = spec_c - np.mean(spec_c)

	c           = np.correlate(spec_c,ref_c,mode='full')

	x           = np.arange(c.size)
	c_max       = np.argmax(c)      # Maximum correlation

	print "=================================="
	if ref_c.size-1 > c_max:        # If spectrum is redshifted
	  #ref_c.size-1 because python starts index at 0
	  shift = wave_c[ref_c.size-1]-wave_c[np.argmax(c)]
	  units = (ref_c.size-1)-np.argmax(c)
	  print "Pixel Shift:\t",units
	  print "Angstrom Shift:\t",shift
	  zeros = np.zeros(units)
	  spec	= np.concatenate((zeros, spec), axis=1)[:-units]
	else:                           # If spectrum is blueshifted
	  c = np.correlate(ref_c,spec_c,mode='full')
	  shift = wave_c[np.argmax(c)]-wave_c[ref_c.size-1]
	  units	= abs(np.argmax(c)-(ref_c.size-1))
	  print "Pixel Shift:\t",units
	  print "Angstrom Shift:\t",shift
	  zeros     = np.zeros(units)
	  spec	= np.concatenate((spec, zeros), axis=1)[units:]
	print "=================================="

	return wave,spec,error

def getData(fits_location,part,start,stop):
    dir_contents = os.listdir(fits_location)
    fits = sorted([fn for fn in dir_contents if fn.startswith('l') and fn.endswith('sum.fits')])
    NumFits = len(fits)
    # Extracting data from fits files
    wavelength0, flux0, err0 	= Extract(fits_location+fits[0],part,start,stop)

    if fits_location[-13:] == '2014/visit_1/':
        return wavelength0, flux0 ,err0, NumFits
    
    elif fits_location[-13:] == '2015/visit_1/':
        wavelength1, flux1, err1 	= Extract(fits_location+fits[1],part,start,stop)
        wavelength2, flux2, err2 	= Extract(fits_location+fits[2],part,start,stop)
        return wavelength0, wavelength1, wavelength2, flux0, flux1, flux2, err0, err1, err2, NumFits
    
    else:
        wavelength1, flux1, err1 	= Extract(fits_location+fits[1],part,start,stop)
        wavelength2, flux2, err2 	= Extract(fits_location+fits[2],part,start,stop)
        wavelength3, flux3, err3 	= Extract(fits_location+fits[3],part,start,stop)
        return wavelength0, wavelength1, wavelength2, wavelength3, flux0, flux1, flux2, flux3, err0, err1, err2, err3,NumFits

def Bin_data(x,y0,y1,y2,y3,bin_pnts):
    bin_size = int(len(x)/bin_pnts)
    bins = np.linspace(x[0], x[-1], bin_size)
    digitized = np.digitize(x, bins)
    bin_y0 = np.array([y0[digitized == i].mean() for i in range(0, len(bins))])
    bin_y1 = np.array([y1[digitized == i].mean() for i in range(0, len(bins))])
    bin_y2 = np.array([y2[digitized == i].mean() for i in range(0, len(bins))])
    bin_y3 = np.array([y3[digitized == i].mean() for i in range(0, len(bins))])
    return bins, bin_y0, bin_y1, bin_y2, bin_y3        

def replace_with_median(X):
    X[np.isnan(X)] = 0
    m = np.median(X[X > 0])
    X[X == 0] = m
    return X

def ExportShitedSpectra(w0,f0,f1,f2,f3,f4,e0,e1,e2,e3,e4,NumFits,start,stop):
   
    # Creating empty arrays to be filled with
    # shifted spectra
    F = [[] for _ in range(NumFits-1)]    # -1 because we want to avoid the airglow observation
    W = [[] for _ in range(NumFits-1)]
    E = [[] for _ in range(NumFits-1)]    

    W[0],F[0],E[0]	=	shift_spec(f0,f1,e1,w0,start,stop)
    W[1],F[1],E[1]	=	shift_spec(f0,f2,e2,w0,start,stop)
    W[2],F[2],E[2]	=	shift_spec(f0,f3,e3,w0,start,stop)

    if NumFits > 4:
        W[3],F[3],E[3]	=	shift_spec(f0,f4,e4,w0,start,stop)

    F = np.array(F)
    E = np.array(E)
    
    F_ave =  np.average(F, axis=0)
    F_ave_w =  np.average(F, axis=0,weights=1./E**2)
    
    return W[0], F_ave_w


def main():
    
    # Configure these paramters before running
    ##########################################
    start       = 1000  # Wavelength element
    stop	    = 800   # start/stop point.
    part        = 'A'   # A = red, B = blue
    bin_pnts    = 10.
    x_lim1      = 1288
    x_lim2      = 1433
    dat_directory = "/home/paw/science/betapic/data/HST/dat/"   
    ##########################################

    fits_location = '/home/paw/science/betapic/data/HST/2015/'

    # Load data visit 1 2014
    fits_location = '/home/paw/science/betapic/data/HST/2014/visit_1/'
    w0_0,f0_0,e0_0,NumFits_0               = getData(fits_location,part,start,stop)
    
    # Load data visit 1 2015
    fits_location = '/home/paw/science/betapic/data/HST/2015/visit_1/'
    w0_1,w1_1,w2_1,f0_1,f1_1,f2_1,e0_1,e1_1,e2_1,NumFits_1                = getData(fits_location,part,start,stop)
    
    # Load data visit 2 2015
    fits_location = '/home/paw/science/betapic/data/HST/2015/visit_2/'
    w0_2,w1_2,w2_2,w3_2,f0_2,f1_2,f2_2,f3_2,e0_2,e1_2,e2_2,e3_2,NumFits_2 = getData(fits_location,part,start,stop)

    # Load data visit 3 2016
    fits_location = '/home/paw/science/betapic/data/HST/2016/visit_3/'
    w0_3,w1_3,w2_3,w3_3,f0_3,f1_3,f2_3,f3_3,e0_3,e1_3,e2_3,e3_3,NumFits_3 = getData(fits_location,part,start,stop)    
    
    # Bin the data by bin_pnts (defined above)
    w_bin, y0_bin, y1_bin, y2_bin, y3_bin = Bin_data(w0_0,f0_0,f0_1,f0_2,f0_3,bin_pnts)
    
    y0_bin = replace_with_median(y0_bin)
    y1_bin = replace_with_median(y1_bin)
    y2_bin = replace_with_median(y2_bin)
    y3_bin = replace_with_median(y3_bin)

    ratio1 = y0_bin/y1_bin
    ratio2 = y0_bin/y2_bin
    ratio3 = y0_bin/y3_bin

    m1 = np.median(ratio1)
    m2 = np.median(ratio2)
    m3 = np.median(ratio3)
    
    s1 = np.std(ratio1)
    s2 = np.std(ratio2)
    s3 = np.std(ratio3)


    fig = plt.figure(figsize=(10,14))
    fontlabel_size = 18
    tick_size = 18
    params = {'backend': 'wxAgg', 'lines.markersize' : 2, 'axes.labelsize': fontlabel_size, 'text.fontsize': fontlabel_size, 'legend.fontsize': fontlabel_size, 'xtick.labelsize': tick_size, 'ytick.labelsize': tick_size, 'text.usetex': True}
    plt.rcParams.update(params)
    plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    plt.rcParams['text.usetex'] = True
    plt.rcParams['text.latex.unicode'] = True   


    ax = plt.subplot(211)
    # Plot the ratios of the specra
    plt.step(w_bin,ratio1+0.5,label='2014/2015v1',color="#FF32B1") 
    plt.step(w_bin,ratio2,label='2014/2015v2',color="#13C1CC")
    plt.step(w_bin,ratio3-0.5,label='2014/2016v3',color="#BDAC42")
    plt.xlim(x_lim1,x_lim2)
    plt.ylim(0.25,2.3)
    plt.legend(loc='upper left', numpoints=1)

    # Define new start and stop values
    # which determine the region to be
    # cross correlated.
    start = 5300
    stop  = 10300
    
    ax2 = plt.subplot(212)
    # Plot of the individual spectra
    plt.step(w_bin,y0_bin+6e-13,lw=1.2,color="#FF281C",label='2014')
    plt.step(w_bin,y1_bin+4e-13,lw=1.2,color="#FF9303",label='2015v1')
    plt.step(w_bin,y2_bin+2e-13,lw=1.2,color="#0386FF",label='2015v2')
    plt.step(w_bin,y3_bin,lw=1.2,color="#00B233",label='2016v3') 
    plt.xlim(x_lim1,x_lim2)
    plt.ylim(0.,1.2e-12)   

    plt.step(w0_0[start:stop],f0_0[start:stop])
    plt.legend(loc='upper left', numpoints=1)

    fig.tight_layout()
    #plt.savefig('FEB_quiet_regions.pdf', bbox_inches='tight', pad_inches=0.1,dpi=300)
    plt.show()

    # These are dummy arrays used for the 10 Dec 2015 observations
    f_empty = []
    e_empty = []
    
    print "\n\nShifting 10 Dec 2015 observations:"
    W, F_ave_w_1 = ExportShitedSpectra(w0_0,f0_0,f0_1,f1_1,f2_1,f_empty,e0_0,e0_1,e1_1,e2_1,e_empty,NumFits_1,start,stop)
    
    print "\n\nShifting 24 Dec 2015 observations:"
    W, F_ave_w_2 = ExportShitedSpectra(w0_0,f0_0,f0_2,f1_2,f2_2,f3_2,e0_0,e0_2,e1_2,e2_2,e3_2,NumFits_2,start,stop)
    
    print "\n\nShifting 30 Jan 2016 observations:"
    W, F_ave_w_3 = ExportShitedSpectra(w0_0,f0_0,f0_3,f1_3,f2_3,f3_3,e0_0,e0_3,e1_3,e2_3,e3_3,NumFits_3,start,stop)

    fig = plt.figure(figsize=(14,10))
    fontlabel_size = 18
    tick_size = 18
    params = {'backend': 'wxAgg', 'lines.markersize' : 2, 'axes.labelsize': fontlabel_size, 'text.fontsize': fontlabel_size, 'legend.fontsize': fontlabel_size, 'xtick.labelsize': tick_size, 'ytick.labelsize': tick_size, 'text.usetex': True}
    plt.rcParams.update(params)
    plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    plt.rcParams['text.usetex'] = True
    plt.rcParams['text.latex.unicode'] = True  
    
    plt.step(w0_0,f0_0,color='#FF281C',lw=1.2,label='2014')
    plt.step(W,F_ave_w_1,color='#FF9303',lw=1.2,label='2015v1')
    plt.step(W,F_ave_w_2,color='#0386FF',lw=1.2,label='2015v2')
    plt.step(W,F_ave_w_3,color='#00B233',lw=1.2,label='2016v3')
    plt.legend(loc='upper left', numpoints=1)
    plt.xlabel('Wavelength \AA')
    plt.ylabel('Flux (erg/s/cm$^2$/\AA)')
    fig.tight_layout()
    #plt.savefig('all_data_A.pdf', bbox_inches='tight', pad_inches=0.1)
    plt.show()
    
    F_ave_w_1   = replace_with_median(F_ave_w_1)
    F_ave_w_2   = replace_with_median(F_ave_w_2)
    F_ave_w_3   = replace_with_median(F_ave_w_3)
    
    #np.savetxt(dat_directory+"A.dat",np.column_stack((W,f0_0,F_ave_w_1,F_ave_w_2,F_ave_w_3)))
    
if __name__ == '__main__':
    main()
