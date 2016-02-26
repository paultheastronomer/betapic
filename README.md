# Betapic

This is a collection of python programs used to work with _HST_/COS data.

**create_dat.py** - Converts a .fits file into a .dat file with columns: wavelength, flux, err, net

**read_file.py** - Plots calibrated *sum.fits files. Used as a quick look function.

**shift_calc.py** - Shift the offset COS spectra and output aligned spectra using Numpy's correlate package.

**spec_line.py** - Plots all COS data centered on a particular line. Useful for exploring FEB absorption signatures.

**Ly_data.py** - Creates the data file containing obsevations around the Ly-alpha line.

**Ly_fit.py** - Fits the Ly-alpha line using voigt profiles and estimates column densities.


Additional useful code
-----------

**create_STIS_dat.py** - Converts a STIS .fits file into a .dat file with columns: wavelength, flux, err
