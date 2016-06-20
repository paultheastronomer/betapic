import numpy as np

class Calc:
    '''
    A collection of calculations.
    '''

    def BinData(self,x,y1,e1,bin_pnts):
        bin_size    = int(len(x)/bin_pnts)
        bins        = np.linspace(x[0], x[-1], bin_size)
        digitized   = np.digitize(x, bins)
        bin_y       = np.array([y1[digitized == i].mean() for i in range(0, len(bins))])
        bin_e       = np.array([e1[digitized == i].mean() for i in range(0, len(bins))])
        return bins, bin_y, bin_e/np.sqrt(bin_pnts)

    def WeightedAvg(self,Flux, Err):
        """
        Return the weighted average and Error bars.
        """
        weights=1./(Err**2)
        average = np.average(Flux, axis=0, weights=weights)
        errorbars_2 = np.sum(weights*(Err**2), axis=0) / np.sum(weights, axis=0)
        return average, np.sqrt(errorbars_2)

    def CF(self,flux,flux_err,ref,ref_err,n1,n2):
        flux        = self.ReplaceWithMedian(flux)
        flux_err    = self.ReplaceWithMedian(flux_err)
        ref_err     = self.ReplaceWithMedian(ref_err)
        ratio = np.average(flux[n1:n2], axis=0, weights=1./(flux_err[n1:n2]**2))/ \
                np.average(ref[n1:n2],  axis=0, weights=1./(ref_err[n1:n2]**2 ))                       
        return 1./ratio
        
    def wave2RV(self,Wave,rest_wavelength,RV_BP):
        c = 299792458
        rest_wavelength = rest_wavelength*(RV_BP*1.e3)/c + rest_wavelength # Convert to beta pic reference frame
        delta_wavelength = Wave-rest_wavelength
        RV = ((delta_wavelength/rest_wavelength)*c)/1.e3	# km/s
        return RV

    def FindCenter(self,w,l):
        for i in range(len(w)):
          if w[i] > l:
            if abs(l - w[i-1]) < abs(l - w[i]):
              ans = i-1
              break
            else:
              ans = i
              break
        return ans

    def ReplaceWithMedian(self, X):
        X[np.isnan(X)] = 0
        m = np.median(X[X > 0])
        X[X == 0] = m
        return X
