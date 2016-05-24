# Betapic

[![GPL](https://img.shields.io/badge/license-GNU%20GPLv3-brightgreen.svg)](http://choosealicense.com/licenses/gpl-3.0/)

This is a collection of python programs used to work with _HST_/COS data. It is still very much in progress.

**create_dat.py** - Converts a .fits file into a .dat file with columns: wavelength, flux, err, net

**shift_calc.py** - Shifts the offset COS spectra and outputs aligned spectra using Numpy's correlate package.

**Ly_data.py** - Creates the data file containing obsevations centered on the Ly-alpha line.

**O_data.py** - Plots the weighted average of each visit centered on the 1304 lines

**Ly_plot.py** - Produces a plot of the cut data (not sky subtracted). Used as Fig 1. in Wilson et al. (2016).

**Ly_AG_subtract.py** - Subtracts the airglow contamination of the Ly-alpha line.

**Ly_fit.py** - Fits the Ly-alpha line using voigt profiles and estimates column densities. Can also run a basic Metropolis-Hastings MCMC which outputs the random walk chains.

**BestFit.py** - Produces a plot showing the best fit to the Ly-alpha line. Both with and without sky subtraction.

**posterior.py** - Plots the posterior distributions of the MCMC chains calculated using Ly_fit.py.

**ColDens.py** - Plots the posterior distributions of H column density estimate from two components.


Additional code
-----------

**read_file.py** - Plots calibrated *sum.fits files. Used as a quick look function.

**spec_line.py** - Plots all COS data centered on a particular line. Useful for exploring FEB absorption signatures.

**create_STIS_dat.py** - Converts a STIS .fits file into a .dat file with columns: wavelength, flux, err
