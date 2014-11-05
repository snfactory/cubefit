from __future__ import print_function, division

import copy

import numpy as np
from scipy.optimize import leastsq, fmin_l_bfgs_b

__all__ = ["guess_sky", "fit_sky", "fit_sky_and_sn", "fit_model_all_epoch",
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


def penalty_g_all_epoch(x, model, data):
    """computes likelihood and regularization penalty for a given galaxy model
    
    Parameters
    ----------
    x : np.ndarray
        1-d array, flattened 3-d galaxy model
    model : DDTModel 
    data : DDTData
    
    Returns
    -------
    penalty : float
    gradient : np.ndarray
        1-d array of length x.size giving the gradient on penalty.

    Notes
    -----
    Used in op_mnb (DDT/OptimPack-1.3.2/yorick/OptimPack1.i)
    * Compute likelihood term and gradient on NORMALIZED x
    *       1. compute 4-D model: g(x)
    *       2. apply convolution: H.g(x)
    *       3. apply resampling: R.H.g(x)
    *       4. compute residuals and penalty
    *       5. compute gradient by transposing steps 3, 2, and 1
    """

    model.gal = x.reshape(model.gal.shape)

    # Change model's SN and sky to optimal values given this galaxy
    for i_t in range(data.nt):
        if i_t == data.master_final_ref:
            continue
        if data.is_final_ref[i_t]:
            sky = fit_sky(model, data, i_t)
            model.sky[i_t, :] = sky
        else:
            sky, sn = fit_sky_and_sn(model, data, i_t)
            model.sky[i_t, :] = sky
            model.sn[i_t, :] = sn

    lkl_penalty, lkl_grad = likelihood_penalty(model, data)
    rgl_penalty, rgl_grad = regularization_penalty(model, data)

    tot_penalty = lkl_penalty + rgl_penalty
    tot_grad = lkl_grad + rgl_grad
    
    return tot_penalty, tot_grad

def likelihood_penalty(model, data):
    """computes likelihood and likelihood gradient for galaxy model
    
    Parameters
    ----------
    x : np.ndarray
        1-d array, flattened 3-d galaxy model
    model : DDTModel 
    data : DDTData
    
    Returns
    -------
    penalty : float
    gradient : np.ndarray
        1-d array of length x.size giving the gradient on penalty.
    """

    lkl_err = 0.0
    grad = np.empty_like(model.gal)
    for i_t in range(data.nt):
        m = model.evaluate(i_t, data.xctr[i], data.yctr[i],
                           (data.ny, data.nx), which='all')
        r = data.data[i_t] - m
        wr = data.weight[i_t] * r
        lkl_err += np.sum(wr * r)

        # gradient
        grad += model.gradient_helper(i_t, wr, data.xctr[i], data.yctr[i],
                                      (data.ny, data.nx))
        
    return lkl_err, grad.reshape(model.gal.size)


def regularization_penalty(model, data):
    """computes regularization penalty and gradient for a given galaxy model
    
    Parameters
    ----------
    x : np.ndarray
        1-d array, flattened 3-d galaxy model
    model : DDTModel 
    data : DDTData
    
    Returns
    -------
    penalty : float
    gradient : np.ndarray
        1-d array of length x.size giving the gradient on penalty.
    """

    galdiff = model.gal - model.galprior
    galdiff /= data.data_avg[:, None, None]
    dw = galdiff[1:, :, :] - galdiff[:-1, :, :]
    dy = galdiff[:, 1:, :] - galdiff[:, :-1, :]
    dx = galdiff[:, :, 1:] - galdiff[:, :, :-1]

    # Regularlization penalty term
    rgl_err = (model.mu_xy * np.sum(dx**2) +
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

    rgl_grad = np.zeros(galdiff.shape, dtype=np.float64)
    rgl_grad[:, :, 1:] += 2. * model.mu_xy * dx
    rgl_grad[:, :, :-1] -= 2. * model.mu_xy * dx
    rgl_grad[:, 1:, :] += 2. * model.mu_xy * dy
    rgl_grad[:, :-1,:] -= 2. * model.mu_xy * dy
    rgl_grad[1:, :, :] += 2. * model.mu_wave * dw
    rgl_grad[:-1, :, :] -= 2. * model.mu_wave * dw
    
    return rgl_err, rgl_grad.reshape(model.gal.size)


def fit_model_all_epoch(model, data, maxiter=1000):
    """fit galaxy, SN and sky and update the model accordingly, keeping
    registration fixed.
    
    Parameters
    ----------
    model : DDTModel
    data : DDTData

    """
    
    x0 = model.gal.reshape((model.gal.size))

    #bounds = zip(np.ones(model.gal.size)*10e-6,
    #             np.ones(model.gal.size)*data.data.max())
    x, f, d = fmin_l_bfgs_b(penalty_g_all_epoch, x0, args=(model, data)) 
    
    model.gal = x.reshape(model.gal.shape)

    print("optimization finished\n"
          "function minimum: {:f}".format(f))
    print("info dict: ", d)


# TODO: should we change this to use a general-purpose optimizer rather 
# than leastsq? Leastsq seems like a strange choice for this problem
# from what I can tell.
def fit_position(data, model, i_t, maxiter=100):
    """Fit data position for epoch i_t, keeping galaxy model
    fixed. Doesn't modify model or data.

    Parameters
    ----------
    data : DDTData
    model : DDTModel
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
