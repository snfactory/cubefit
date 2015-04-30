DDTPy
=====

Usage
-----

On the command line::

    ddt-fit conf.json /path/to/data/dir outputfile.pkl


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

**Initialization**

- Read in datacubes
- Read in parameters that determine the PSF and ADR and initialize the
  "atmospheric model" that contains these for each epoch.
- Initialize all model parameters to zero, except sky. For the sky, we
  make an initial heuristic guess based on only the data and a
  sigma-clipping process.
- Initialize the regularization, based on the input regularization
  parameters and a rough guess at the average galaxy spectrum.

**Fit the galaxy model to just the master final ref**

The model is defined in the frame of the master final ref, so there is
no position fit here. Also, the sky of the master final ref is fixed
to the initial guess (as the galaxy model and sky level are
degenerate).

**Fit the position and sky of the remaining final refs**



**Re-fit galaxy model using all final refs**

**Fit position, sky and SN position of all the non-final refs**


Reference/API
-------------

.. autosummary::
   :toctree: api

   ddtpy.main

**Data structure and I/O**

.. autosummary::
   :toctree: api

   ddtpy.DataCube
   ddtpy.read_datacube


**ADR and PSF model**

.. autosummary::
   :toctree: api

   ddtpy.paralactic_angle
   ddtpy.gaussian_plus_moffat_psf
   ddtpy.psf_3d_from_params
   ddtpy.AtmModel

*Additionally, the ADR class from the SNfactory Toolbox package is used*

**Fitting**

.. autosummary::
   :toctree: api

   ddtpy.RegularizationPenalty
   ddtpy.guess_sky
   ddtpy.fit_galaxy_single
   ddtpy.fit_galaxy_sky_multi
   ddtpy.fit_position_sky
   ddtpy.fit_position_sky_sn_multi


**Utilities**

.. autosummary::
   :toctree: api

   ddtpy.fft_shift_phasor_2d


**Plotting**

.. autosummary::
   :toctree: api

   ddtpy.plot_timeseries


Appendix
--------

.. toctree::
   :maxdepth: 1

   gradient
