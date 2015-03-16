from __future__ import print_function, division

import copy

import numpy as np
from scipy.optimize import leastsq, fmin_l_bfgs_b, fmin_bfgs


def guess_sky(cube, clip, maxiter=10):
    """Guess sky based on lower signal spaxels compatible with variance

    Parameters
    ----------
    cube : DataCube
    clip : float
        Number of standard deviations (not variances) to use as
        the clipping limit (on individual pixels).
    maxiter : int
        Maximum number of sigma-clipping interations. Default is 10.

    Returns
    -------
    sky : np.ndarray (2-d)        
        Sky level for each epoch and wavelength. Shape is (nt, nw).
    """

    nspaxels = cube.ny * cube.nx

    weight = np.copy(cube.weight)
    var = 1.0 / weight

    # Loop until mask stops changing size or until a maximum
    # number of iterations.
    avg = None
    oldmask = None
    mask = None
    for j in range(maxiter):
        oldmask = mask

        # weighted average spectrum (masked array).
        # We use a masked array because some of the wavelengths 
        # may have all-zero weights for every pixel.
        # The masked array gets propagated so that `mask` is a
        # masked array of booleans!
        avg = np.ma.average(cube.data, weights=weight, axis=(1, 2))
        deviation = cube.data - avg[:, None, None]
        mask = deviation**2 > clip**2 * var

        # Break if the mask didn't change.
        if (oldmask is not None and
            (mask.data == oldmask.data).all() and
            (mask.mask == oldmask.mask).all()):
            break

        # set weights of masked pixels to zero. masked elements
        # of the mask are *not* changed.
        weight[mask] = 0.0
        var[mask] = 0.0

    # convert to normal (non-masked) array. Masked wavelengths are 
    # set to zero in this process.
    return np.asarray(avg)


def fit_galaxy_single(galaxy, sky, cube, ctr, atm, regpenalty):
    """
    Fit *just* the galaxy model to a single data cube.
    """

    # subtract sky from data
    data_minus_sky = cube.data - sky[:, None, None]

    # parameters for fitter need to be 1-d
    galparams0 = np.ravel(galaxy)

    # Define objective function to minimize. This adjusts SN and Sky
    # and returns the regularized chi squared and its gradient.
    def objective_func(galparams):

        # galparams is 1-d (raveled version of galaxy); reshape to 3-d.
        tmp = galparams.reshape(galaxy.shape)

        m = atm.evaluate_galaxy(tmp, (cube.ny, cube.nx), ctr)
        diff = data_minus_sky - m
        wdiff = cube.weight * diff
        chisq_val = np.sum(wdiff * diff)
        chisq_grad = atm.gradient_helper(-2. * wdiff, (cube.ny, cube.nx), ctr)
        rval, rgrad = regpenalty(galaxy)

        # Reshape gradient to 1-d when returning.
        return (chisq_val + rval), np.ravel(chisq_grad + rgrad)

    # run minimizer
    galparams, f, d = fmin_l_bfgs_b(objective_func, galparams0)

    print("optimization finished\n"
          "function minimum: {:f}".format(f))
    print("info dict: ")
    for k, v in d.iteritems():
        print(k, " : ", v)

    return galparams.reshape(galaxy.shape)

def fit_galaxy_multi(galaxy, skys, cubes, ctrs, atms, regpenalty):
    """
    Fit the galaxy model to a single data cube.
    """

    # subtract sky from data
    data_minus_sky = cube.data - sky[:, None, None]

    # parameters for fitter need to be 1-d
    galparams0 = np.ravel(galaxy)

    # Define objective function to minimize. This adjusts SN and Sky
    # and returns the regularized chi squared and its gradient.
    def objective_func(galparams):

        # galparams is 1-d (raveled version of galaxy); reshape to 3-d.
        tmp = galparams.reshape(galaxy.shape)

        m = atm.evaluate_galaxy(tmp, (cube.ny, cube.nx), ctr)
        diff = data_minus_sky - m
        wdiff = cube.weight * diff
        chisq_val = np.sum(wdiff * diff)
        chisq_grad = atm.gradient_helper(-2. * wdiff, (cube.ny, cube.nx), ctr)
        rval, rgrad = regpenalty(galaxy)

        # Reshape gradient to 1-d when returning.
        return (chisq_val + rval), np.ravel(chisq_grad + rgrad)

    # run minimizer
    galparams, f, d = fmin_l_bfgs_b(objective_func, galparams0)

    print("optimization finished\n"
          "function minimum: {:f}".format(f))
    print("info dict: ")
    for k, v in d.iteritems():
        print(k, " : ", v)

    return galparams.reshape(galaxy.shape)

# TODO: should we change this to use a general-purpose optimizer rather 
# than leastsq? Leastsq seems like a strange choice for this problem
# from what I can tell.
def fit_position(galaxy, sky, cube, ctr0, atm, maxiter):
    """Fit data position for epoch i_t, keeping galaxy model fixed.

    Parameters
    ----------
    model : DDTModel
    data : DDTData
    i_t : int
        Epoch number.

    Returns
    -------
    y, x : float, float
    """

    # subtract sky from data
    data_minus_sky = cube.data - sky[:, None, None]

    # Define a function that returns the sqrt(weight) * (data-model)
    # for the given epoch i_t, given the data position.
    # scipy.optimize.leastsq will minimize the sum of the squares of this
    # function's return value, so we're minimizing
    # sum(weight * residual^2), which seems reasonable.
    def objective_func(ctr):
        m = atm.evaluate_galaxy(galaxy, (cube.ny, cube.nx), ctr)
        out = np.sqrt(cube.weight) * (data_minus_sky - m)
        return np.ravel(out)

    ctr, info = leastsq(objective_func, ctr0, maxiter=maxiter)
    if info not in [1, 2, 3, 4]:
        raise RuntimeError("leastsq didn't converge properly")

    return pos[0], pos[1]
