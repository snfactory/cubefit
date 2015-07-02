CubeFit
=======

Fit supernova + galaxy model on a Nearby Supernova Factory spectral data cube.


Installation
------------

**Dependencies:**

- Python 2.7
- numpy
- scipy (for optimization)
- [fitsio](https://github.com/esheldon/fitsio) (for FITS file I/O)
- [pyfftw](http://hgomersall.github.io/pyFFTW) (for fast FFTs)

**Install command:** `./setup.py install`

**Running Tests:** Run `./test.py`. Requires the `pytest` package
(available via pip or conda).


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

CubeFit currently expects to find the following keys in the input JSON file:

| Parameter                  | Type   | Description                           |
| -------------------------- | ------ | ------------------------------------- |
| `"IN_CUBE"`                | *list* | file names
| `"PARAM_AIRMASS"`          | *list* | airmass for each epoch
| `"PARAM_P"`                | *list* | pressure in mmHg for each epoch
| `"PARAM_T"`                | *list* | temperature in degrees Celcius for each epoch
| `"PARAM_IS_FINAL_REF"`     | *list* | Integers indicating whether epoch is final ref (0 or 1)
| `"PARAM_FINAL_REF"`        |        | index of "master" final ref (1-based indexing)
| `"PARAM_TARGET_XP"`        | *list* | *See NOTE below*
| `"PARAM_TARGET_YP"`        | *list* | *See NOTE below*
| `"PARAM_PSF_TYPE"`         |        | must be `"GS-PSF"`
| `"PARAM_PSF_ES"`           | *list of lists* | PSF parameters for each epoch
| `"PARAM_SPAXEL_SIZE"`      |        | instrument spaxel size in arcseconds
| `"PARAM_LAMBDA_REF"`       |        | reference wavelength in Angstroms
| `"PARAM_HA"`               |        | position of the target (RA) in degrees
| `"PARAM_DEC"`              |        | position of the target (DEC) in degrees
| `"PARAM_MLA_TILT"`         |        | MLA tilt in radians
| `"OUT_DATACUBE_SUBTRACTION_FILE"` | *list* | Output filenames to write galaxy-subtracted cubes to (only used in cubefit-subtract)

*NOTE: We're currently unsure of what the `XP` and `YP` parameters are
in the current DDT input files. We think they give the location of
the center of the master final ref relative to the center of each
exposure in spaxels. That is, these parameters are always 0 for the
master final ref. For an exposure offset by 1 spaxel in each direction
to the upper right (northwest) the parameters would be each be -1.*

The above parameter names and conventions are chosen to comply with the existing
DDT input files. If we are creating a new script that produces these config
files, it would be nice to choose some new names without the baggage of DDT
conventions. Here are the preferred names for the future. *Note that these do
not currently work!*

| Parameter            | Type   | Description                           |
| -------------------- | ------ | ------------------------------------- |
| `"FILENAMES"`        | *list* | file names
| `"AIRMASS"`          | *list* | airmass for each epoch
| `"PRESSURE"`         | *list* | pressure in mmHg for each epoch
| `"TEMPERATURE"`      | *list* | temperature in degrees Celcius for each epoch
| `"IS_FINAL_REF"`     | *list* | Boolean (true/false) indicating whether epoch is final ref
| `"MASTER_FINAL_REF"` |        | index of "master" final ref (1-based indexing)
| `"XCTR"`             | *list* | x position of MLA center relative to SN
| `"YCTR"`             | *list* | y position of MLA center relative to SN
| `"PSF_PARAMS"`       | *list of lists* | PSF parameters for each epoch
| `"SPAXEL_SIZE"`      |        | instrument spaxel size in arcseconds
| `"WAVE_REF"`         |        | reference wavelength in Angstroms
| `"RA"`               |        | position of the target (RA) in degrees
| `"DEC"`              |        | position of the target (DEC) in degrees
| `"MLA_TILT"`         |        | MLA tilt in radians (?)

*NOTE: the meaning of `X` and `Y` parameters is different than in the previous
table!*

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

These steps are carried out in ``cubefit.main`` (which is called from
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


Internal API documentation
--------------------------

|                |                  |
| -------------- | ---------------- |
| `cubefit.main` | Main entry point |


**Data structure and I/O**

|                         |                                          |
| ----------------------- | ---------------------------------------- |
| `cubefit.DataCube`      | Container for data and weight arrays     |
| `cubefit.read_datacube` | Read a two-HDU FITS file into a DataCube |


**ADR and PSF model**

|                                    |                                          |
| ---------------------------------- | ---------------------------------------- |
| `cubefit.paralactic_angle`         | Return paralactic angle in radians, including MLA tilt |
| `cubefit.gaussian_plus_moffat_psf` | Evaluate a gaussian+moffat function on a 2-d grid |
| `cubefit.psf_3d_from_params`       | Create a wavelength-dependent Gaussian+Moffat PSF from given parameters |
| `cubefit.AtmModel`                 |  Atmospheric conditions (PSF and ADR) model for a single observation |

*Additionally, the ADR class from the SNfactory Toolbox package is used.*

**Fitting**

|                                     |                                          |
| ----------------------------------- | ---------------------------------------- |
| `cubefit.RegularizationPenalty`     | Callable that returns the penalty and gradient on it. |
| `cubefit.guess_sky`                 | Guess sky based on lower signal spaxels compatible with variance |
| `cubefit.fit_galaxy_single`         | Fit the galaxy model to a single epoch of data. |
| `cubefit.fit_galaxy_sky_multi`      | Fit the galaxy model to multiple data cubes. |
| `cubefit.fit_position_sky`          | Fit data position and sky for a single epoch (fixed galaxy model). |
| `cubefit.fit_position_sky_sn_multi` | Fit data pointing (nepochs), SN position (in model frame), SN amplitude (nepochs), and sky level (nepochs). |



**Utilities**

|                               |                                          |
| ----------------------------- | ---------------------------------------- |
| `cubefit.fft_shift_phasor_2d` | phasor array used to shift an array (in real space) by multiplication in fourier space. |
| `cubefit.plot_timeseries`     | Return a figure showing data and model. |


License
-------

CubeFit is released under the MIT license. See LICENSE.md.
