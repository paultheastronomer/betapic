# Betapic

[![GPL](https://img.shields.io/badge/license-GNU%20GPLv3-brightgreen.svg)](http://choosealicense.com/licenses/gpl-3.0/)

This is a collection of python programs used to work with _HST_/COS data.

**create_dat.py** - Converts a .fits file into a .dat file with columns: wavelength, flux, err, net

**shift_calc.py** - Shifts the offset COS spectra and outputs aligned spectra using Numpy's correlate package.

**Ly_data.py** - Creates the data file containing obsevations centered on the Ly-alpha line.

**Ly_fit.py** - Fits the Ly-alpha line using voigt profiles and estimates column densities. Can also run a basic Metropolis-Hastings MCMC which outputs the random walk chains.

**posterior.py** - Plots the posterior distributions of the MCMC chains calculated using Ly_fit.py.


Additional code
-----------

**read_file.py** - Plots calibrated *sum.fits files. Used as a quick look function.

**spec_line.py** - Plots all COS data centered on a particular line. Useful for exploring FEB absorption signatures.

**create_STIS_dat.py** - Converts a STIS .fits file into a .dat file with columns: wavelength, flux, err
