from __future__ import print_function, division

import copy

import numpy as np
from scipy.optimize import leastsq, fmin_l_bfgs_b

__all__ = ["guess_sky", "fit_sky", "fit_sky_and_sn", "fit_model",
           "fit_position"]


def guess_sky(ddtdata, sig, maxiter=10):
    """Guess sky based on lower signal spaxels compatible with variance

    Parameters
    ----------
    sig : float
        Number of standard deviations (not variances) to use as
        the clipping limit (on individual pixels).
    maxiter : int
        Maximum number of sigma-clipping interations. Default is 10.

    Returns
    -------
    sky : np.ndarray (2-d)        
        Sky level for each epoch and wavelength. Shape is (nt, nw).
    """

    nspaxels = ddtdata.data.shape[2] * ddtdata.data.shape[3]
    sky = np.zeros((ddtdata.nt, ddtdata.nw), dtype=np.float64)

    for i in range(ddtdata.nt):
        data = ddtdata.data[i]
        weight = np.copy(ddtdata.weight[i])
        var = 1.0 / weight

        # Loop until ind stops changing size or until a maximum
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
            avg = np.ma.average(data, weights=weight, axis=(1, 2))
            deviation = data - avg[:, None, None]
            mask = deviation**2 > sig**2 * var

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
        sky[i] = np.asarray(avg)

    return sky


def fit_sky(model, data, i_t):
    """Estimate the sky level for a single epoch, assuming no SN flux.

    Given a fixed galaxy in the model, the (assumed spatially flat) sky
    background is estimated from the difference between the data and model.
    This is inteded for use only on final ref epochs, where we "know" the
    SN flux is zero.

    Parameters
    ----------
    model : DDTModel
        The model.
    data : DDTData
        The data.
    i_t : int
        The index of the epoch of interest.

    Returns
    -------
    sky : ndarray
        1-D sky spectrum.
    """

    d = data.data[i_t, :, :, :]
    w = data.weight[i_t, :, :, :]
    m = model.evaluate(i_t, data.xctr[i_t], data.yctr[i_t],
                       (data.ny, data.nx), which='galaxy')

    return np.average(d - m, weights=w, axis=(1, 2))


def fit_sky_and_sn(model, data, i_t):
    """Estimate the sky and SN level for a single epoch.

    Given a fixed galaxy and fixed SN PSF shape in the model, the
    (assumed spatially flat) sky background and SN flux are estimated.

    Parameters
    ----------
    model : DDTModel
        The model.
    data : DDTData
        The data.
    i_t : int
        The index of the epoch of interest.

    Returns
    -------
    sky : ndarray
        1-d sky spectrum for given epoch.
    sn : ndarray
        1-d SN spectrum for given epoch.
    """

    d = data.data[i_t, :, :, :]
    w = data.weight[i_t, :, :, :]

    galmodel = model.evaluate(i_t, data.xctr[i_t], data.yctr[i_t],
                              (data.ny, data.nx), which='galaxy')

    snmodel = model.evaluate(i_t, data.xctr[i_t], data.yctr[i_t],
                             (data.ny, data.nx), which='snscaled')

    A11 = (w * snmodel**2).sum(axis=(1, 2))
    A12 = (-w * snmodel).sum(axis=(1, 2))
    A21 = A12
    A22 = w.sum(axis=(1, 2))

    denom = A11*A22 - A12*A21

    # There are some cases where we have slices with only 0
    # values and weights. Since we don't mix wavelengths in
    # this calculation, we put a dummy value for denom and
    # then put the sky and sn values to 0 at the end.
    mask = denom == 0.0
    if not np.all(A22[mask] == 0.0):
        raise ValueError("found null denom for slices with non null "
                         "weight")
    denom[mask] = 1.0

    # w2d, w2dy w2dz are used to calculate the variance using 
    # var(alpha x) = alpha^2 var(x)*/
    tmp = w * d
    wd = tmp.sum(axis=(1, 2))
    wdsn = (tmp * snmodel).sum(axis=(1, 2))
    wdgal = (tmp * galmodel).sum(axis=(1, 2))

    tmp = w * galmodel
    wgal = tmp.sum(axis=(1, 2))
    wgalsn = (tmp * snmodel).sum(axis=(1, 2))
    wgal2 = (tmp * galmodel).sum(axis=(1, 2))

    b_sky = (wd * A11 + wdsn * A12) / denom
    c_sky = (wgal * A11 + wgalsn * A12) / denom        
    b_sn = (wd * A21 + wdsn * A22) / denom
    c_sn = (wgal * A21 + wgalsn * A22) / denom

    sky = b_sky - c_sky
    sn = b_sn - c_sn

    sky[mask] = 0.0
    sn[mask] = 0.0

    return sky, sn


def chisq(model, data, i_t):
    """Return chi squared value and gradent for single data epoch.
    
    Parameters
    ----------
    model : DDTModel 
    data : DDTData
    i_t : int
        Epoch number.
    
    Returns
    -------
    chisq : float
    chisq_gradient : np.ndarray (3-d)
        Gradient with respect to model galaxy.
    """

    m = model.evaluate(i_t, data.xctr[i_t], data.yctr[i_t],
                       (data.ny, data.nx), which='all')
    r = data.data[i_t] - m
    wr = data.weight[i_t] * r
    grad = model.gradient_helper(i_t, -2.*wr, data.xctr[i_t],
                                 data.yctr[i_t], (data.ny, data.nx))

    return np.sum(wr * r), grad


def regularization_penalty(model, data):
    """computes regularization penalty and gradient for a given galaxy model
    
    Parameters
    ----------
    model : DDTModel 
    data : DDTData
    
    Returns
    -------
    penalty : float
    penalty_gradient : np.ndarray
        Gradient with respect to model galaxy (l
    """

    galdiff = model.gal - model.galprior
    galdiff /= model.mean_gal_spec[:, None, None]
    dw = galdiff[1:, :, :] - galdiff[:-1, :, :]
    dy = galdiff[:, 1:, :] - galdiff[:, :-1, :]
    dx = galdiff[:, :, 1:] - galdiff[:, :, :-1]

    # Regularlization penalty term
    val = (model.mu_xy * np.sum(dx**2) +
           model.mu_xy * np.sum(dy**2) +
           model.mu_wave * np.sum(dw**2))
    
    # Gradient in regularization penalty term
    #
    # This is clearer when the loops are explicitly written out.
    # For a loop that goes over all adjacent elements in a given dimension,
    # one would do (pseudocode):
    # for i in ...:
    #     d = arr[i+1] - arr[i]
    #     penalty += hyper * d^2
    #     gradient[i+1] += 2 * hyper * d
    #     gradient[i]   -= 2 * hyper * d

    grad = np.zeros_like(model.gal)
    grad[:, :, 1:] += 2. * model.mu_xy * dx
    grad[:, :, :-1] -= 2. * model.mu_xy * dx
    grad[:, 1:, :] += 2. * model.mu_xy * dy
    grad[:, :-1,:] -= 2. * model.mu_xy * dy
    grad[1:, :, :] += 2. * model.mu_wave * dw
    grad[:-1, :, :] -= 2. * model.mu_wave * dw
    
    return val, grad


def fit_model(model, data, epochs):
    """Fit galaxy, SN and sky part of model, keeping data positions fixed.

    Parameters
    ----------
    model : DDTModel
        Model.
    data : DDTData
        Data.
    epochs : list of int
        List of epoch indicies to use in fit.

    """

    # Define objective function to minimize. This adjusts SN and Sky
    # and returns the regularized chi squared and its gradient.
    def objective_func(galparams):

        # Set galaxy in model.
        model.gal = galparams.reshape(model.gal.shape)

        # Change model's SN and sky to optimal values given this galaxy
        for i_t in epochs:
            if i_t == data.master_final_ref:
                continue
            if data.is_final_ref[i_t]:
                sky = fit_sky(model, data, i_t)
                model.sky[i_t, :] = sky
            else:
                sky, sn = fit_sky_and_sn(model, data, i_t)
                model.sky[i_t, :] = sky
                model.sn[i_t, :] = sn

        # Add up chisq for each epoch and add regularization gradient.
        chisq_tot = 0.
        chisq_grad = np.zeros_like(model.gal)
        for i_t in epochs:
            val, grad = chisq(model, data, i_t)
            chisq_tot += val
            chisq_grad += grad
        rval, rgrad = regularization_penalty(model, data)

        return (chisq_tot + rval), np.ravel(chisq_grad + rgrad)

    galparams0 = np.ravel(model.gal)
    galparams, f, d = fmin_l_bfgs_b(objective_func, galparams0)
    model.gal = galparams.reshape(model.gal.shape)

    print("optimization finished\n"
          "function minimum: {:f}".format(f))
    print("info dict: ")
    for k, v in d.iteritems():
        print(k, " : ", v)


# TODO: should we change this to use a general-purpose optimizer rather 
# than leastsq? Leastsq seems like a strange choice for this problem
# from what I can tell.
def fit_position(model, data, i_t, maxiter=100):
    """Fit data position for epoch i_t, keeping galaxy model
    fixed. Doesn't modify model or data.

    Parameters
    ----------
    model : DDTModel
    data : DDTData
    i_t : int
        Epoch number.

    Returns
    -------
    x, y : float, float
        x and y position.
    """

    # Define a function that returns the sqrt(weight) * (data-model)
    # for the given epoch i_t, given the data position.
    # scipy.optimize.leastsq will minimize the sum of the squares of this
    # function's return value, so we're minimizing
    # sum(weight * residual^2), which seems reasonable.
    def objective_func(pos):
        m = model.evaluate(i_t, pos[0], pos[1], (data.ny, data.nx),
                           which='all')
        out = np.sqrt(data.weight[i_t]) * (data.data[i_t] - m)
        return np.ravel(out)

    pos0 = [data.xctr[i_t], data.yctr[i_t]]  # initial position

    pos, info = leastsq(objective_func, pos0)
    if info not in [1, 2, 3, 4]:
        raise RuntimeError("leastsq didn't converge properly")

    return pos[0], pos[1]


def fit_position_sn_sky(model, data, epochs):
    """Fit data pointing (nepochs), SN position (in model frame),
    SN amplitude (nepochs), and sky level (nepochs). This is meant to be
    used only on epochs with SN light.

    In practice, given the data pointing and SN position, determining
    the sky level and SN amplitude is a linear problem. Therefore, we
    have only the data pointing and sn position as parameters in the
    (nonlinear) optimization and determine the sky and sn amplitude in
    each iteration.
    """

    # In the objective function, pos is a 1-d array:
    #
    # [sn_x, sn_y, x_ctr[0], y_ctr[0], x_ctr[1], y_ctr[1], ...]
    #
    # length is 2 + 2*nepochs
    def objective_func(pos):
        totchisq = 0.

        # set model parameters
        model.sn_x = pos[0]
        model.sn_y = pos[1]

        for n, i_t in enumerate(epochs):
            data.xctr[i_t] = pos[2+2*n]
            data.yctr[i_t] = pos[3+2*n]
            sky, sn = fit_sky_and_sn(model, data, i_t)
            model.sky[i_t, :] = sky
            model.sn[i_t, :] = sn
            
            m = model.evaluate(i_t, data.xctr[i_t], data.yctr[i_t],
                               (data.ny, data.nx), which='all')
            r = data.data[i_t] - m
            totchisq += np.sum(data.weight[i_t] * r * r)

        return totchisq

    # initial positions
    pos0 = np.hstack((model.sn_x, model.sn_y,
                      np.ravel(zip(data.xctr[epochs], data.yctr[epochs]))))

    pos, info = leastsq(objective_func, pos0)
    if info not in [1, 2, 3, 4]:
        raise RuntimeError("leastsq didn't converge properly")

    return pos
