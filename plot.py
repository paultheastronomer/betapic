#
# Read x1d COS images
#
# Written by: Paul A. Wilson
# pwilson@iap.fr
#

import pyfits, os
import matplotlib.pyplot as plt
import numpy as np
import sys

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

def Bin_data(x,y1,y2,y3,bin_pnts):
    bin_size = int(len(x)/bin_pnts)
    bins = np.linspace(x[0], x[-1], bin_size)
    digitized = np.digitize(x, bins)
    bin_y1 = np.array([y1[digitized == i].mean() for i in range(0, len(bins))])
    bin_y2 = np.array([y2[digitized == i].mean() for i in range(0, len(bins))])
    bin_y3 = np.array([y3[digitized == i].mean() for i in range(0, len(bins))])
    
    #binned_pts = len(x)/bin_size
    #bin_error = np.array(bin_error/np.sqrt(binned_pts))
    #print "Binned points: ",binned_pts
    return bins, bin_y1, bin_y2, bin_y3#bin_y4

def main():

    w0,f0, err0 = np.loadtxt('/home/paw/science/betapic/owens/BP_AllS_B.dat',unpack=True)
    
    w0 = w0[7203:7903]
    f0 = f0[7203:7903]
    

    w1,f1 = np.loadtxt('/home/paw/science/betapic/data/HST/2016/visit_1/NI_weighted_mean_no_err_visit1.dat',unpack=True)
   
    w2,f2 = np.loadtxt('/home/paw/science/betapic/data/HST/2016/visit_2/NI_weighted_mean_no_err_visit2.dat',unpack=True)

    bin_pnts = 3
    #x0_bin, f0_bin, f1_bin, f2_bin	=	Bin_data(w0,f0,f1,f2,bin_pnts)
    w1_bin, f0_bin, f1_bin, f2_bin	=	Bin_data(w1,f0,f1,f2,bin_pnts)
  
    #plt.step(w0,f0,label='Visit 2014')
    #plt.step(w2,f2,label='Visit 2, 2015')
    #plt.step(w1,f1,label='Visit 1, 2015')
    
    diff = np.median(f1_bin[182:]) - np.median(f2_bin[182:])
    f2_bin_corr = f2_bin+diff
    
    #'''
    plt.step(w1_bin,f0_bin,label='Visit 2014')
    plt.step(w1_bin,f2_bin_corr,label='Visit 2, 2015')
    #plt.step(w1_bin[182:],f2_bin[182:],label='Visit 2, 2015')
    plt.step(w1_bin,f1_bin,label='Visit 1, 2015')    
    #'''
    #comets = savitzky_golay(f1_bin-f2_bin_corr, 11, 3)
    
    #plt.step(w1_bin,f1_bin-f2_bin_corr,color='black')
    #plt.step(w1_bin,comets,color='red')
    
    
    plt.xlabel('Wavelength ($\AA$)')
    plt.ylabel('Flux')
    plt.legend(loc='upper right', numpoints=1)
    plt.xlim(1198,1202)
    #plt.ylim(0,2e-14)
    #plt.savefig('AG.pdf', bbox_inches='tight', pad_inches=0.1)
    plt.show()

if __name__ == '__main__':
    main()
