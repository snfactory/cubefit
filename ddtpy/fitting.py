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


def determine_sky_and_sn(galmodel, snmodel, data, weight):
    """Estimate the sky and SN level for a single epoch.

    Given a fixed galaxy and fixed SN PSF shape in the model, the
    (assumed spatially flat) sky background and SN flux are estimated.

    Parameters
    ----------
    galmodel : ndarray (3-d)
        The model, evaluated on the data grid.
    snmodel : ndarray (3-d)
        The PSF, evaluated on the data grid at the SN position.
    data : ndarray (3-d)
    weight : ndarray (3-d)

    Returns
    -------
    sky : ndarray
        1-d sky spectrum for given epoch.
    sn : ndarray
        1-d SN spectrum for given epoch.
    """

    A11 = (weight * snmodel**2).sum(axis=(1, 2))
    A12 = (-weight * snmodel).sum(axis=(1, 2))
    A21 = A12
    A22 = weight.sum(axis=(1, 2))

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
    tmp = weight * data
    wd = tmp.sum(axis=(1, 2))
    wdsn = (tmp * snmodel).sum(axis=(1, 2))
    wdgal = (tmp * galmodel).sum(axis=(1, 2))

    tmp = weight * galmodel
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



def fit_galaxy_single(galaxy0, data, weight, ctr, atm, regpenalty):
    """Fit the galaxy model to a single epoch of data.

    Parameters
    ----------
    galaxy0 : ndarray (3-d)
        Initial galaxy model.
    data : ndarray (3-d)
        Sky-subtracted data.
    weight : ndarray (3-d)
    ctr : tuple
        Length 2 tuple giving y, x position of data in model coordinates.
    """

    # parameters for fitter need to be 1-d
    galparams0 = np.ravel(galaxy0)
    dshape = data.shape[1:3]

    # Define objective function to minimize. This adjusts SN and Sky
    # and returns the regularized chi squared and its gradient.
    def objective_func(galparams):

        # galparams is 1-d (raveled version of galaxy); reshape to 3-d.
        gal3d = galparams.reshape(galaxy0.shape)
        m = atm.evaluate_galaxy(gal3d, dshape, ctr)
        diff = data - m
        wdiff = weight * diff
        chisq_val = np.sum(wdiff * diff)
        chisq_grad = atm.gradient_helper(-2. * wdiff, dshape, ctr)
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


def fit_galaxy_multi(galaxy0, datas, weights, ctrs, atms, regpenalty):
    """Fit the galaxy model to multiple data cubes.

    Parameters
    ----------
    galaxy0 : ndarray (3-d)

    datas : list of ndarray
        Sky-subtracted data for each epoch to fit.
    """

    # parameters for fitter need to be 1-d
    galparams0 = np.ravel(galaxy0)
    dshape = datas[0].shape[1:3]

    # Define objective function to minimize. This adjusts SN and Sky
    # and returns the regularized chi squared and its gradient.
    def objective_func(galparams):

        # galparams is 1-d (raveled version of galaxy); reshape to 3-d.
        gal3d = galparams.reshape(galaxy0.shape)

        val = 0.
        grad = np.zeros_like(gal3d)
        for data, weight, ctr, atm in zip(datas, weights, ctrs, atms):
            m = atm.evaluate_galaxy(gal3d, dshape, ctr)
            diff = data - m
            wdiff = weight * diff
            val += np.sum(wdiff * diff)
            grad += atm.gradient_helper(-2. * wdiff, dshape, ctr)
        rval, rgrad = regpenalty(galaxy)

        # Reshape gradient to 1-d when returning.
        return (val + rval), np.ravel(grad + rgrad)

    # run minimizer
    galparams, f, d = fmin_l_bfgs_b(objective_func, galparams0)

    print("optimization finished\n"
          "function minimum: {:f}".format(f))
    print("info dict: ")
    for k, v in d.iteritems():
        print(k, " : ", v)

    return galparams.reshape(galaxy0.shape)


# TODO: should we change this to use a general-purpose optimizer rather 
# than leastsq? Leastsq seems like a strange choice for this problem
# from what I can tell.
def fit_position_sky(galaxy, data, weight, ctr0, atm, maxiter):
    """Fit data position and sky for a single epoch (fixed galaxy model).

    Parameters
    ----------
    galaxy : ndarray (3-d)
    data : ndarray (3-d)
    weight : ndarray(3-d)
    ctr0 : (float, float)
        Initial center.

    Returns
    -------
    ctr : (float, float)
        (y, x) center position.
    sky : ndarray (1-d)
        Fitted sky.
    """

    spatial_shape = data.shape[1:3]
    sqrtweight = np.sqrt(weight)

    # Define a function that returns the sqrt(weight) * (data-model)
    # for the given epoch i_t, given the data position.
    # scipy.optimize.leastsq will minimize the sum of the squares of this
    # function's return value, so we're minimizing
    # sum(weight * residual^2), which seems reasonable.
    def objective_func(ctr):
        gal = atm.evaluate_galaxy(galaxy, spatial_shape, ctr)

        # determine sky (linear problem)
        resid = data - gal
        sky = np.average(resid, weights=weight, axis=(1, 2))

        out = sqrtweight * (resid - sky[:, None, None])
        return np.ravel(out)

    ctr, info = leastsq(objective_func, ctr0, maxiter=maxiter)
    if info not in [1, 2, 3, 4]:
        raise RuntimeError("leastsq didn't converge properly")

    # get last-calculated sky.
    gal = atm.evaluate_galaxy(galaxy, spatial_shape, ctr)
    sky = np.average(data - gal, weights=weight, axis=(1, 2))

    return tuple(ctr), sky


def fit_position_sn_sky(galaxy, datas, weights, ctrs0, snctr0, atms, maxiter):
    """Fit data pointing (nepochs), SN position (in model frame),
    SN amplitude (nepochs), and sky level (nepochs). This is meant to be
    used only on epochs with SN light.

    Parameters
    ----------


    Returns
    -------
    

    Given the data pointing and SN position, determining
    the sky level and SN amplitude is a linear problem. Therefore, we
    have only the data pointing and sn position as parameters in the
    (nonlinear) optimization and determine the sky and sn amplitude in
    each iteration.
    """

    # In the objective function, pos is a 1-d array:
    #
    # [sn_y, sn_x, y_ctr[0], x_ctr[0], y_ctr[1], x_ctr[1], ...]
    #
    # length is 2 + 2*nepochs
    def objective_func(allctrs):

        # pull sn ctr and data ctrs out of input array.
        snctr = (allctrs[0], allctrs[1])
        ctrs = [(c[i], c[i+1]) for i in range(2, len(allctrs), 2)]

        # initialize return values
        grad = np.zeros_like(allctrs)
        chisq = 0.

        for data, weight, ctr, atm in zip(datas, weights, ctrs, atms):

            # calculate chisq for this epoch
            gal = atm.evaluate_galaxy(galaxy, data.shape[1:3], ctr)
            psf = atm.evaluate_point_source(snctr, data.shape[1:3], ctr)
            sky, sn = determine_sky_and_sn(gal, psf, data, weight)
            scene = sky[:, None, None] + gal + sn[:, None, None] * psf
            chisq = np.sum(weight * (data - scene)**2)

        # -------------------------------------------------------------------
        # old
        dx = 0.01
        grad_helper_chi_x = 0.
        grad_helper_chi_y = 0.
        sn_grad_helper = np.zeros(len(pos)-2)

        # Do x-direction gradient on sn position in model
        model.sn_x = pos[0]+dx
        model.sn_y = pos[1]

        for n, i_t in enumerate(epochs):
            data.xctr[i_t] = pos[2+2*n]
            data.yctr[i_t] = pos[3+2*n]
            sky, sn = determine_sky_and_sn(model, data, i_t)
            model.sky[i_t, :] = sky
            model.sn[i_t, :] = sn
            
            m = model.evaluate(i_t, data.xctr[i_t], data.yctr[i_t],
                               (data.ny, data.nx), which='all')
            r = data.data[i_t] - m
            grad_helper_chi_x += np.sum(data.weight[i_t] * r * r)

        # Do y-direction gradient on sn position in model
        model.sn_x = pos[0]
        model.sn_y = pos[1]+dx

        for n, i_t in enumerate(epochs):
            data.xctr[i_t] = pos[2+2*n]
            data.yctr[i_t] = pos[3+2*n]
            sky, sn = fit_sky_and_sn(model, data, i_t)
            model.sky[i_t, :] = sky
            model.sn[i_t, :] = sn
            
            m = model.evaluate(i_t, data.xctr[i_t], data.yctr[i_t],
                               (data.ny, data.nx), which='all')
            r = data.data[i_t] - m
            grad_helper_chi_y += np.sum(data.weight[i_t] * r * r)


        # set model parameters
        model.sn_x = pos[0]
        model.sn_y = pos[1]

        for n, i_t in enumerate(epochs):
            # Do x-direction gradient on data position wrt model
            data.xctr[i_t] = pos[2+2*n]+dx
            data.yctr[i_t] = pos[3+2*n]
            sky, sn = fit_sky_and_sn(model, data, i_t)
            model.sky[i_t, :] = sky
            model.sn[i_t, :] = sn
            
            m = model.evaluate(i_t, data.xctr[i_t], data.yctr[i_t],
                               (data.ny, data.nx), which='all')
            r = data.data[i_t] - m
            chi_dx = np.sum(data.weight[i_t] * r * r)

            # Do y-direction gradient on data position wrt model
            data.xctr[i_t] = pos[2+2*n]
            data.yctr[i_t] = pos[3+2*n]+dx
            sky, sn = fit_sky_and_sn(model, data, i_t)
            model.sky[i_t, :] = sky
            model.sn[i_t, :] = sn
            
            m = model.evaluate(i_t, data.xctr[i_t], data.yctr[i_t],
                               (data.ny, data.nx), which='all')
            r = data.data[i_t] - m
            chi_dy = np.sum(data.weight[i_t] * r * r)

            # Get chisq for pos
            data.xctr[i_t] = pos[2+2*n]
            data.yctr[i_t] = pos[3+2*n]
            sky, sn = fit_sky_and_sn(model, data, i_t)
            model.sky[i_t, :] = sky
            model.sn[i_t, :] = sn
            
            m = model.evaluate(i_t, data.xctr[i_t], data.yctr[i_t],
                               (data.ny, data.nx), which='all')
            r = data.data[i_t] - m
            chisq = np.sum(data.weight[i_t] * r * r)
            totchisq += chisq

            old_grad_x = (chi_dx - chisq)/dx
            old_grad_y = (chi_dy - chisq)/dx
            
            # Get the analytic solution for the gradient:
            m, m_grad_y, m_grad_x = model.evaluate(i_t, data.xctr[i_t],
                        data.yctr[i_t], (data.ny, data.nx), which='galaxy+der')

            r = data.data[i_t] - m
            chisq2 = np.sum(data.weight[i_t] * r * r)
            grad_y = -np.sum(2*data.weight[i_t] * r * m_grad_y)
            grad_x = -np.sum(2*data.weight[i_t] * r * m_grad_x)
            grad[2+2*n] = grad_x
            grad[3+2*n] = grad_y
            print('Chisq', i_t, chisq, chisq2)
            print('Grad', old_grad_x, old_grad_y, grad_x, grad_y)
        grad[0] = (grad_helper_chi_x - totchisq)/dx
        grad[1] = (grad_helper_chi_y - totchisq)/dx
        print(totchisq)
        return totchisq, grad

    # initial positions
    pos0 = np.hstack((model.sn_x, model.sn_y,
                      np.ravel(zip(data.xctr[epochs], data.yctr[epochs]))))
    bounds = zip(-2*np.ones(len(pos0)), 2*np.ones(len(pos0)))
    #pos, info
    pos_fit = fmin_l_bfgs_b(objective_func, pos0, #approx_grad=True,
                            iprint=0, callback=call, bounds=bounds)
    #if info not in [1, 2, 3, 4]:
    #    raise RuntimeError("leastsq didn't converge properly")
    print(pos_fit)
    return pos_fit[0]
