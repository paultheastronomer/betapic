# Betapic

[![GPL](https://img.shields.io/badge/license-GNU%20GPLv3-brightgreen.svg)](http://choosealicense.com/licenses/gpl-3.0/)

This is a collection of python programs used to work with _HST_/COS data. It is still very much in progress. The results of much of this code will be published in Wilson et al. (2016).


**shift_calc.py** - Shifts the offset COS spectra and outputs aligned spectra using Numpy's correlate package.

**Ly_data.py** - Creates the data file containing obsevations centered on the Ly-alpha line. Airglow contaminated regions have been cut from the data.

**Ly_plot.py** - Produces a plot of the cut data (not sky subtracted). The input is from Ly_data.py. Used as Fig 1. in Wilson et al. (2016).

**Ly_AG_subtract.py** - Subtracts the airglow contamination of the Ly-alpha line and combines all obtained data into a final spectrum.

**LyFit.py** - Fits the Ly-alpha line using voigt profiles and estimates column densities. Can also run a basic Metropolis-Hastings MCMC which outputs the random walk chains.

**BestFit.py** - Produces a plot of the best fit to the Ly-alpha line. Both with and without sky subtraction. Used to generate Fig. 2, 3 and 4.

**posterior.py** - Plots the posterior distributions of the MCMC chains calculated using LyFit.py. Used to generate Fig. 5.

**FluxLoss.py** - Plots the theoretical shape of flux loss as a function of shift along the dispersion axis. The program is used to generate the figure in the appendix.


Additional code
-----------

**create_dat.py** - Converts a .fits file into a .dat file with columns: wavelength, flux, err, net

**read_file.py** - Plots calibrated *sum.fits files. Used as a quick look function.

**O_data.py** - Plots the weighted average of each visit centered on the 1304 lines

**spec_line.py** - Plots all COS data centered on a particular line. Useful for exploring FEB absorption signatures.

**create_STIS_dat.py** - Converts a STIS .fits file into a .dat file with columns: wavelength, flux, err

**ColDens.py** - Plots the posterior distributions of H column density estimate from two components and Vx, the velocity of the additional component.

**theoretical_flux_loss.py** - Plots the theoretical flux loss curve using the derivation in the appendix in Wilson et al. (2016)
