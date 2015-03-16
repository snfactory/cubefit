Fitting Algorithm
-----------------

The following steps are done in `ddtpy.main`:

**Loading:**

- Load data
- Load PSF model and use it to initialze the full model.

**Initial heuristic "guessing":**

- Make a heuristic guess at the sky spectrum for all the epochs.
- calculate a rough "average" galaxy model spectrum for just the final refs
  by subtracting off the sky guess and averaging over spatial coordinates and
  epochs.

**Fit model to final refs:**

- Fit the 3-d galaxy model to just the "master" final ref, keeping the
  sky fixed. The model is defined in the frame of the master final ref.
- Fit all final ref positions with respect to the "master" final ref.
- Recalculate the sky in all final refs
- Fit the 3-d galaxy model to all final refs simultaneously, keeping the
  relative positions fixed.
- TODO: loop over this procedure

**Next part:**

- Fit all the things.

