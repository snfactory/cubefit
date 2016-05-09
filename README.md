CubeFit
=======

Fit supernova + galaxy model on a series of spectral data cubes from
the Nearby Supernova Factory.


Installation
------------

To install a release version:

```
pip install http://github.com/snfactory/cubefit/archive/v0.4.2.tar.gz
```

Release versions are listed
[here](http://github.com/snfactory/cubefit/releases). CubeFit has the
following dependencies:

- Python 2.7 or 3.3+
- numpy
- scipy (for optimization)
- [fitsio](https://github.com/esheldon/fitsio) (for FITS file I/O)
- [pyfftw](http://hgomersall.github.io/pyFFTW) (for fast FFTs)
- cython


Usage
-----

Fit the model, write output to given file:

```
cubefit config.json output.fits
```

Producing subtracted data cubes is a separate step:

```
cubefit-subtract config.json output.fits
```

This reads the input and output filenames listed in the configuration file, and
the results of the fit saved in `output.fits`.

With either command, you can run with `-h` or `--help` options to see all the
optional arguments: `cubefit --help`.

Input format
------------

CubeFit requires the following keys in the input JSON file. Keys are
case-sensitive. Additional keys are simply ignored.

| Parameter        | Type   | Description [units]                   |
| ---------------- | ------ | ------------------------------------- |
| `"filenames"`    | *list* | FITS data cube files
| `"xcenters"`     | *list* | x position of MLA center relative to SN **[spaxels]**
| `"ycenters"`     | *list* | y position of MLA center relative to SN **[spaxels]**
| `"psf_params"`   | *list of lists* | PSF parameters for each epoch
| `"refs"`         | *list* | index of final refs in lists **[0-based indexing]**
| `"master_ref"`   |        | index of "master" final ref **[0-based indexing]**
| `"outnames"`     | *list* | Output subtracted cube filenames for `cubefit-subtract`
| `"sn_outnames"`  | *list* | Output SN spectrum filenames for `cubefit-subtract`

In each input FITS file, the following header keywords are used:

| Keyword    | Description [units]                           |
| ---------- | --------------------------------------------- |
| `AIRMASS`  | Airmass of observation                        |
| `PRESSURE` | Atmospheric pressure **[mmHg]**               |
| `TEMP`     | Atmospheric temperature **[degrees Celcius]** |
| `PARANG`   | Parallactic angle **[degrees]**               |
| `CHANNEL`  | `B` or `R`                                    |

Output format
-------------

CubeFit outputs a FITS file with two extensions.

- **HDU 1:** (image) Full galaxy model. E.g., a 32 x 32 x 779 array.

- **HDU 2:** (binary table) Per-epoch results. Columns:
  - `yctr` Fitted y position of data in model frame
  - `xctr` fitted x position of data in model frame
  - `sn` (1-d array) fitted SN spectrum in this epoch
  - `sky` (1-d array) fitted sky spectrum in this epoch
  - `galeval` (3-d array) Galaxy model evaluated on this epoch
    (e.g., 15 x 15 x 779)
  - `sneval` (3-d array) SN model evaluated on this epoch (e.g., 15 x 15 x 779)


- The **HDU 1 header** contains the wavelength solution in the "CRPIX3",
  "CRVAL3", "CDELT3" keywords. These are simply propagated from input
  data cubes.

- The **HDU 2 header** contains the fitted SN position in the model frame in the
  "SNX" and "SNY" kewords.

Here is an example demonstrating how to reproduce the full model of the scene
for the first epoch:

```python
>>> import fitsio
>>> f = fitsio.FITS("PTF09fox_B.fits")
>>> f[1]

  file: PTF09fox_B.fits
  extension: 1
  type: BINARY_TBL
  extname: epochs
  rows: 17
  column info:
    yctr                f8  
    xctr                f8  
    sn                  f4  array[779]
    sky                 f4  array[779]
    galeval             f4  array[15,15,779]
    sneval              f4  array[15,15,779]

>>> epochs = f[1].read()
>>> i = 0
>>> scene = epochs["sky"][i, :, None, None] + epochs["galeval"][i] + epochs["sneval"][i]
```

What it does
------------

A description of the original implemetation of the algorithm can be found in [Bongard et al (2011)](http://adsabs.harvard.edu/abs/2011MNRAS.418..258B).

**The data**

The data are N 3-d cubes, where N are observations at different times
(typically N ~ 10-30).  Each cube has two spatial directions and one
wavelength direction. For SNFactory, the dimensions are 15 x 15 x M,
where M=779 for the blue channel M=1572 for the red channel. Each cube
has both a data array and a corresponding weight (error) array.

Some of the data cubes are designated as "final refs", meaning that there is
assumed to be no SN light in these epochs.

**The model**

The model parameters consist of:

- 3-d galaxy model ("unconvolved"). Default size is 32 x
  32 x M, where the pixel size of the model matches that of the data
  (both spatially and in wavelength). The model extends past the
  data spatially, and the spectral grid matches the data.
- N 1-d sky spectra
- N 1-d supernova spectra
- position of each of the N data cubes: 2N parameters
- position of the SN: 2 parameters

Some of these parameters are fixed. For example, the SN spectrum in
each of the final refs is fixed to zero. The position of one of the
final refs (the "master" final ref) is fixed to the input values
because only the relative offset between data cube positions is
discernable.

Additionally there are inputs that describe the point spread function
(PSF) in each epoch. These are necessary for propagating the model to
the data. They are derived externally and are not varied in the
fit. The PSF is "3-d": it is defined on a grid of wavelengths. The
spatial shape can vary between wavelengths, as can the position of the
center of the PSF. The change in the center position with wavelength
encodes the amount of atmospheric differential refraction (ADR).

Finally, there are two regularization parameters that determine the
penalty on the galaxy model for being "rougher" (having a high
pixel-to-pixel variation).

**Fitting Procedure**

These steps are carried out in ``cubefit.main.cubefit()`` (which is called from
the command-line script).

1. **Initialization**

   - Read configuration file
   - Read in datacubes
   - Initialize the PSF model for each epoch, based on values in the configuration
     file and image header.
   - Initialize all model parameters to zero, except sky. For the sky, we
     make an initial heuristic guess based on only the data.
   - Initialize the regularization, based on the input regularization
     parameters and a rough guess at the average galaxy spectrum.

2. **Fit the galaxy model to just the master final ref**

   The position of the master final ref is fixed with respect to the model,
   so there is no position fit here. Also, the sky of the master final ref is
   fixed to the initial guess (as the galaxy model and sky level are
   degenerate).

3. **Fit the position and sky of the remaining final refs**

4. **Re-fit galaxy model using all final refs**

5. **Fit position, sky and SN position of all the non-final refs**


Internal API documentation
--------------------------

This is intended as a rough guide to the major components of the code.

| Console scripts                   |                                           |
| --------------------------------- | ----------------------------------------- |
| `main.cubefit()`                  | Entry point for `cubefit` script          |
| `main.cubefit_subtract()`         | Entry point for `cubefit-subtract` script |
| `main.cubefit_plot()`             | Entry point for `cubefit-plot` script     |


| Data structure and I/O  |                                          |
| ----------------------- | ---------------------------------------- |
| `io.DataCube`           | Container for data and weight arrays     |
| `io.read_datacube`      | Read a two-HDU FITS file into a DataCube |


| PSF model                        |                                          |
| -------------------------------- | ---------------------------------------- |
| `psf.GaussianMoffatPSF`          | A 3-d PSF model made up of a Gaussian + Mofffat profile. |
| `psf.TabularPSF`                 | A 3-d PSF model defined by a 3-d array. |
| `main.snfpsf()`                  | Instatiate a 3-d PSF model based on SNFactory-specific parameterization. |
| `psffuncs.gaussian_moffat_psf()` | Evaluate a 3-d Gaussian + Moffat profile on a 3-d array. |

Note: The `ADR` class from the SNfactory Toolbox package and the
 `Hyper_PSF3D_PL` class from libExtractStar are used in
 `cubefit.main.snfpsf()`


| Fitting                               |                                          |
| ------------------------------------- | ---------------------------------------- |
| `fitting.RegularizationPenalty`       | Callable that returns the penalty and gradient on it. |
| `fitting.guess_sky()`                 | Guess sky based on lower signal spaxels compatible with variance |
| `fitting.fit_galaxy_single()`         | Fit the galaxy model to a single epoch of data. |
| `fitting.fit_galaxy_sky_multi()`      | Fit the galaxy model to multiple data cubes. |
| `fitting.fit_position_sky()`          | Fit data position and sky for a single epoch (fixed galaxy model). |
| `fitting.fit_position_sky_sn_multi()` | Fit data pointing (nepochs), SN position (in model frame), SN amplitude (nepochs), and sky level (nepochs). |


| Plotting                     |                                                        |
| ---------------------------- | ------------------------------------------------------ |
| `plotting.plot_timeseries()` | Return a figure showing data and model for all epochs. |
| `plotting.plot_epoch()`      | Make a more detailed figure for a single epoch.        |

These functions are called by `cubefit-plot`.


Developer Documentation
-----------------------

**Running Tests:**

If you've clone the repository (rather than using pip to install), you
can run tests with `setup.py test`. Requires the `pytest` package
(available via pip or conda).


License
-------

All code in this repository is released under the MIT
license. However, since CubeFit uses FFTW, which is GPL-licensed, the
software as a whole is bound by the terms of the GPLv2.

Practically, this means that one can copy the code here, remove the
dependence on FFTW and release the code under the MIT license (or
other permissive license).
