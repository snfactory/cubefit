v0.4.2 (2015-12-27)
===================

- Add checks for reasonable temperature and pressure values in FITS headers.
- Revert from using setuptools' console_scripts to generate scripts.
- Enable `setup.py test` to run tests and move tests inside package.

v0.4.1 (2015-12-21)
===================

Fix install bug in setup.py.

v0.4.0 (2015-12-10)
===================

- Code organization: Refactor `AtmModel` class to `PSF` classes.
- Tweak: Add PSF model `GaussianMoffatPSF` which samples point source
  directly on output grid for the final position fitting step. This is
  now the default behavior. Also, uses subpixel sampling when creating
  FFT convolution kernel (for convolving galaxy).
- Tweak: Loosen position bounds from 2 spaxels to 3 spaxels
- Tweak: Do not multiply regularization penalty by nepochs in multi-ref fit
- Tweak: Get metadata such as pressure, temperature and paralactic angle
  from image header instead of config file
- Results: Add Axis 1 and 2 WCS keywords to output header
- Results: Add chisq, position bound information to output (2nd HDU)
- Plotting: Improved plots, remove ADR and waveslice plots
- Testing: Integration tests

v0.3.0 (2015-10-19)
===================

- Switch from `fmin` to `fmin_l_bfgs_b` with analytic gradient in first
  position fitting step.
- Analytic gradient calculation in last position fitting step.
- Fix a ~4% effect in scaling of PSF.

v0.2.1 (2015-09-10)
===================

- Added cubefit version to FITS header in output files written by
  `cubefit` and `cubefit-subtract`.
- `cubefit-subtract` now writes SN spectra from cubefit to separate files.
  Output file names are taken from the "sn_outnames" configuration parameter.

v0.2.0 (2015-08-14)
===================

- Improved sky guess algorithm with minimal bias towards negative values.
- In `cubefit-subtract`, add fitted SN position to output FITS header.
- In `cubefit-plot` output all PNG images rather than EPS.
- In `cubefit-plot` output one plot per epoch, when `--plotepochs` is used.
- In `cubefit-plot` output ADR plot.
- In `cubefit-plot` indicate scale of residuals in image plots.
- Fix bug in diagnostic output

v0.1.0 (2015-07-11)
===================

First release.
