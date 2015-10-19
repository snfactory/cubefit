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
