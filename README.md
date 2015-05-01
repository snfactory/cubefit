DDTPy
=====

Python implementation of the spectral data cube modeling code DDT

Installation
------------

**Dependencies:**

- Python 2.7
- numpy
- scipy (for optimization)
- [fitsio](https://github.com/esheldon/fitsio) (for FITS file I/O)
- [pyfftw](http://hgomersall.github.io/pyFFTW) (for fast FFTs)

**Install command:** `./setup.py install`

**Running Tests:** Requires the `pytest` package (available via pip or
conda).  Execute `py.test` in the root of the source code repository
or `test` directory.

Usage
-----

On the command line

```
ddt-fit conf.json /path/to/data/dir outputfile.pkl
```

Data & Model
------------

**The data**

The data are N 3-d cubes, where N are observations at different times
(typically N ~ 10-30).  Each cube has two spatial directions and one
wavelength direction (typical dimensions 15 x 15 x 800). Each cube has
both a data array and a corresponding weight (error) array.

**The model**

The model parameters consist of:

- 3-d galaxy model ("unconvolved"). Default size is approximately 32 x
  32 x 800, where the pixel size of the model matches that of the data
  (both spatially and in wavelength) but the model extends past the
  data spatially.
- N 1-d sky spectra
- N 1-d supernova spectra
- position of each of the N data cubes relative to a "master final
  ref": 2 x (N-1) parameters
- position of the SN in the "master final ref" frame of reference: 2
  parameters

Additionally there are inputs that describe "atmospheric
conditions". These are necessary for propagating the model to the
data. They are derived externally and are not varied in the fit:

- Amount and direction of atmospheric differential refraction in each
  observation.
- wavelength-dependent PSF model for each observation.

Finally, there are two regularization parameters that determine the
penalty on the galaxy model for being "rougher" (having a high
pixel-to-pixel variation).

Procedure
---------

These steps are carried out in ``ddtpy.main`` (which is called from
the command-line script).

1. **Initialization**

   - Read in datacubes
   - Read in parameters that determine the PSF and ADR and initialize the
     "atmospheric model" that contains these for each epoch.
   - Initialize all model parameters to zero, except sky. For the sky, we
     make an initial heuristic guess based on only the data and a
     sigma-clipping process.
   - Initialize the regularization, based on the input regularization
     parameters and a rough guess at the average galaxy spectrum.

2. **Fit the galaxy model to just the master final ref**

   The model is defined in the frame of the master final ref, so there
   is no position fit here. Also, the sky of the master final ref is
   fixed to the initial guess (as the galaxy model and sky level are
   degenerate).

3. **Fit the position and sky of the remaining final refs**

4. **Re-fit galaxy model using all final refs**

5. **Fit position, sky and SN position of all the non-final refs**


API
---

|              |               |
| ------------ | ------------- |
| `ddtpy.main` | Do everything |


**Data structure and I/O**

|                       |                                          |
| --------------------- | ---------------------------------------- |
| `ddtpy.DataCube`      | Container for data and weight arrays     |
| `ddtpy.read_datacube` | Read a two-HDU FITS file into a DataCube |


**ADR and PSF model**

|                                  |                                          |
| -------------------------------- | ---------------------------------------- |
| `ddtpy.paralactic_angle`         | Return paralactic angle in radians, including MLA tilt |
| `ddtpy.gaussian_plus_moffat_psf` | Evaluate a gaussian+moffat function on a 2-d grid |
| `ddtpy.psf_3d_from_params`       | Create a wavelength-dependent Gaussian+Moffat PSF from given parameters |
| `ddtpy.AtmModel`                 |  Atmospheric conditions (PSF and ADR) model for a single observation |

*Additionally, the ADR class from the SNfactory Toolbox package is used.*

**Fitting**

|                                   |                                          |
| --------------------------------- | ---------------------------------------- |
| `ddtpy.RegularizationPenalty`     | Callable that returns the penalty and gradient on it. |
| `ddtpy.guess_sky`                 | Guess sky based on lower signal spaxels compatible with variance |
| `ddtpy.fit_galaxy_single`         | Fit the galaxy model to a single epoch of data. |
| `ddtpy.fit_galaxy_sky_multi`      | Fit the galaxy model to multiple data cubes. |
| `ddtpy.fit_position_sky`          | Fit data position and sky for a single epoch (fixed galaxy model). |
| `ddtpy.fit_position_sky_sn_multi` | Fit data pointing (nepochs), SN position (in model frame), SN amplitude (nepochs), and sky level (nepochs). |



**Utilities**

|                             |                                          |
| --------------------------- | ---------------------------------------- |
| `ddtpy.fft_shift_phasor_2d` | phasor array used to shift an array (in real space) by multiplication in fourier space. |
| `ddtpy.plot_timeseries`     | Return a figure showing data and model. |


License
-------

MIT
