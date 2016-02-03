import pyfits, os
import matplotlib.pyplot as plt
import numpy as np
import sys


def Bin_data(x,y1,y2,y3,bin_pnts):
    bin_size = int(len(x)/bin_pnts)
    bins = np.linspace(x[0], x[-1], bin_size)
    digitized = np.digitize(x, bins)
    bin_y1 = np.array([y1[digitized == i].mean() for i in range(0, len(bins))])
    bin_y2 = np.array([y2[digitized == i].mean() for i in range(0, len(bins))])
    bin_y3 = np.array([y3[digitized == i].mean() for i in range(0, len(bins))])
    return bins, bin_y1, bin_y2, bin_y3

def main():

    #w0,f0, err0 = np.loadtxt('/home/paw/science/betapic/python/sky_background_model/BP_AllS_A.dat',unpack=True)
    w0,f0, err0 = np.loadtxt('/home/paw/science/betapic/data/HST/2014/dat/2014_beta_sum_A.dat',unpack=True)
    
    ws0,fs0, errs0 = np.loadtxt('/home/paw/science/betapic/data/HST/2014/dat/AG_2014_A.dat',unpack=True)
    #ws0,fs0, errs0 = np.loadtxt('/home/paw/science/betapic/data/HST/2014/dat/2014_sky_sum.dat',unpack=True)

    w0 = w0[12000:13000]
    f0 = f0[12000:13000]
    
    ws0 = ws0[12000:13000]
    fs0 = fs0[12000:13000]  

    w1,f1 = np.loadtxt('/home/paw/science/betapic/data/HST/2015/visit_1/SiV_weighted_mean_no_err_visit1.dat',unpack=True) 
    w2,f2 = np.loadtxt('/home/paw/science/betapic/data/HST/2015/visit_2/SiV_weighted_mean_no_err_visit2.dat',unpack=True)

    k = 1.5
    #f0 = f0 - k*fs0
    
    bin_pnts = 3
    w1_bin, f0_bin, f1_bin, f2_bin	=	Bin_data(w1,f0,f1,f2,bin_pnts)
    
    diff = np.median(f1_bin[:120]) - np.median(f2_bin[:120])
    diff2 = np.median(f1_bin[:120]) - np.median(f0_bin[:120])
    f2_bin_corr = f2_bin+diff
    f0_bin_corr = f0_bin+diff2


    fig = plt.figure(figsize=(10,6))
    fontlabel_size = 18
    tick_size = 18
    params = {'backend': 'wxAgg', 'lines.markersize' : 2, 'axes.labelsize': fontlabel_size, 'text.fontsize': fontlabel_size, 'legend.fontsize': fontlabel_size, 'xtick.labelsize': tick_size, 'ytick.labelsize': tick_size, 'text.usetex': True}
    plt.rcParams.update(params)
    plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    
    plt.step(w1_bin,f0_bin_corr,color="black",lw=2.5)
    plt.step(w1_bin,f0_bin_corr,color="#FF281C",lw=1.2,label='Visit 2014')
    
    plt.step(w1_bin,f2_bin_corr,color="black",lw=2.5)
    plt.step(w1_bin,f2_bin_corr,color="#0386ff",lw=1.2,label='Visit 2, 2015')
    
    plt.step(w1_bin,f1_bin,color="black",lw=2.5)
    plt.step(w1_bin,f1_bin,color="#FF9303",lw=1.2,label='Visit 1, 2015')

    
    plt.xlabel('Wavelength (\AA)')
    plt.ylabel('Flux')
    plt.legend(loc='lower left', numpoints=1)
    
    locs,labels = plt.xticks()
    plt.xticks(locs, map(lambda x: "%g" % x, locs))
    #plt.ylim(0,2e-14)
    plt.xlim(1399,1407)
    #plt.savefig('SiV.pdf', bbox_inches='tight', pad_inches=0.1)
    plt.show()

if __name__ == '__main__':
    main()
