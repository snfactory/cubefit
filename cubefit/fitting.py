from __future__ import print_function, division

import copy
import logging

import numpy as np
from scipy.optimize import fmin_l_bfgs_b

__all__ = ["guess_sky", "fit_galaxy_single", "fit_galaxy_sky_multi",
           "fit_position_sky", "fit_position_sky_sn_multi",
           "RegularizationPenalty"]


def _check_result(warnflag, msg):
    """Check result of fmin_l_bfgs_b()"""
    if warnflag == 0:
        return
    if warnflag == 1:
        raise RuntimeError("too many function calls or iterations "
                           "in fmin_l_bfgs_b()")
    if warnflag == 2:
        raise RuntimeError("fmin_l_bfgs_b() exited with warnflag=2: %s" % msg)
    raise RuntimeError("unknown warnflag: %s" % warnflag)


def _log_result(fn, fval, niter, ncall):
    """Write the supplementary results from optimizer to the log"""
    logging.info("        success: %3d iterations, %3d calls, val=%8.2f",
                 niter, ncall, fval)


def guess_sky(cube, npix=10):
    """Guess sky based on lowest signal pixels.

    With the small field of fiew of an IFU, we have no guarantee of
    getting an accurate measurement of the real sky level; the galaxy
    might extend far past the edges of the IFU.

    Here we simply take a weighted average of the lowest `npix` pixels
    at each wavelength. This estimate will be higher than the real sky
    value (which would be lower than the lowest pixel value in the
    absence of noise), but its about the best we can do.

    Parameters
    ----------
    cube : DataCube
    npix : int

    Returns
    -------
    sky : np.ndarray (1-d)
        Sky level at each wavelength.
    """

    # reshape data to (nw, nspaxels)
    flatshape = (cube.nw, cube.ny * cube.nx)
    flatdata = cube.data.reshape(flatshape)
    flatweight = cube.weight.reshape(flatshape)

    # get rid of spaxels that are *all* zero weight
    mask = ~np.all(flatweight == 0.0, axis=0)
    flatdata = flatdata[:, mask]
    flatweight = flatweight[:, mask]

    # average over wavelengths: 1-d array of (nspaxels,)
    avg = np.average(flatdata, weights=flatweight, axis=0)

    # get indicies of lowest `npix` spaxels in flattened data
    idx = np.argsort(avg)[0:npix]

    # get average spectrum of those spaxels
    sky = np.average(flatdata[:, idx], weights=flatweight[:, idx], axis=1)

    return sky


def determine_sky(data, weight, g, ggrad=None):
    """Determine optimal sky given data and galaxy model"""

    num = np.sum((data - g) * weight, axis=(1, 2))
    denom = np.sum(weight, axis=(1, 2))

    # avoid divide-by-zero errors
    mask = (denom == 0.)
    denom[mask] = 1.

    sky = num / denom
    sky[mask] = 0.

    if ggrad is None:
        return sky

    else:
        skygrad = -np.sum(ggrad * weight, axis=(2, 3)) / denom
        skygrad[:, mask] = 0.
        return sky, skygrad


def sky_and_sn(data, weight, g, s, ggrad=None, sgrad=None):
    """Estimate the sky and SN level for a single epoch.

    Given a fixed galaxy and fixed SN PSF shape in the model, the
    (assumed spatially flat) sky background and SN flux are estimated.

    Parameters
    ----------
    data : ndarray (3-d)
        Data array.
    weight : ndarray (3-d)
        Weight array.
    g : ndarray (3-d)
        The galaxy model, evaluated on the data grid.
    s : ndarray (3-d)
        The PSF, evaluated on the data grid at the SN position.
    galgrad : ndarray (4-d)
        Gradient in g with respect to data ctr y, x and sn position, y, x)
    sngrad : ndarray (4-d)
        Gradient in s with repsect to data ctr y, x and
        sn position y, x.

    Returns
    -------
    sky : ndarray
        1-d sky spectrum for given epoch.
    sn : ndarray
        1-d SN spectrum for given epoch.
    """

    A = np.sum(weight * s**2, axis=(1, 2))
    B = np.sum(weight * s, axis=(1, 2))
    C = np.sum(weight, axis=(1, 2))
    D = np.sum(weight * data, axis=(1, 2))
    E = np.sum(weight * data * s, axis=(1, 2))
    F = np.sum(weight * g, axis=(1, 2))
    G = np.sum(weight * g * s, axis=(1, 2))

    denom = A * C - B**2

    # There are some cases where we have spaxels with all 0 values and
    # weights. Set denom to 1.0 in these cases to avoid divide-by-zero
    # errors. We double check that the weights are all zero.
    mask = (denom == 0.0)
    denom[mask] = 1.0
    if not np.all(C[mask] == 0.0):
        raise ValueError("found null denom for slices with non null "
                         "weight")

    sky = (D*A - E*B - F*A + G*B) / denom
    sn = (-D*B + E*C + F*B - G*C) / denom

    sky[mask] = 0.0
    sn[mask] = 0.0

    # calculate gradient in sky and SN w.r.t. positions.
    if ggrad is not None and sgrad is not None:
        dA = np.sum(2.* weight * s * sgrad, axis=(2, 3))
        dB = np.sum(weight * sgrad, axis=(2, 3))
        dE = np.sum(weight * data * sgrad, axis=(2, 3))
        dF = np.sum(weight * ggrad, axis=(2, 3))
        dG = np.sum(weight * g * sgrad + weight * s * ggrad,
                    axis=(2, 3))

        skygradnum = D*dA - dE*B - E*dB - dF*A - F*dA + dG*B + G*dB
        sngradnum = -D*dB + dE*C + dF*B + F*dB - dG*C
        ddenom = dA*C - 2.*B*dB

        skygrad = (skygradnum + sky * ddenom) / denom
        sngrad = (sngradnum + sn * ddenom) / denom

        return sky, sn, skygrad, sngrad

    else:
        return sky, sn


def chisq_galaxy_single(galaxy, data, weight, ctr, psf):
    """Chi^2 and gradient (not including regularization term) for a single
    epoch."""

    scene = psf.evaluate_galaxy(galaxy, data.shape[1:3], ctr)
    r = data - scene
    wr = weight * r
    val = np.sum(wr * r)
    grad = psf.gradient_helper(-2. * wr, data.shape[1:3], ctr)

    return val, grad


def chisq_galaxy_sky_single(galaxy, data, weight, ctr, psf):
    """Chi^2 and gradient (not including regularization term) for 
    single epoch, allowing sky to float."""

    g = psf.evaluate_galaxy(galaxy, data.shape[1:3], ctr)
    sky = determine_sky(data, weight, g)
    scene = sky[:, None, None] + g

    r = data - scene
    wr = weight * r
    val = np.sum(wr * r)

    # See note in docs/gradient.tex for the (non-trivial) derivation
    # of this gradient!
    tmp = np.sum(wr, axis=(1, 2)) / np.sum(weight, axis=(1, 2))
    vtwr = weight * tmp[:, None, None]
    grad = psf.gradient_helper(-2. * (wr - vtwr), data.shape[1:3], ctr)

    return val, grad


def chisq_galaxy_sky_multi(galaxy, datas, weights, ctrs, psfs):
    """Chi^2 and gradient (not including regularization term) for 
    multiple epochs, allowing sky to float."""

    val = 0.0
    grad = np.zeros_like(galaxy)
    for data, weight, ctr, psf in zip(datas, weights, ctrs, psfs):
        epochval, epochgrad = chisq_galaxy_sky_single(galaxy, data, weight,
                                                      ctr, psf)
        val += epochval
        grad += epochgrad

    return val, grad


def chisq_position_sky(ctr, galaxy, data, weight, psf):
    """chisq and gradient for fit_position_sky"""

    g, ggrad = psf.evaluate_galaxy(galaxy, data.shape[1:3], ctr, grad=True)

    sky, skygrad = determine_sky(data, weight, g, ggrad=ggrad)

    scene = sky[:, None, None] + g
    dscene = skygrad[:, :, None, None] + ggrad

    diff = data - scene
    chisq = np.sum(weight * diff**2)
    chisqgrad = -2. * np.sum(weight * diff * dscene, axis=(1, 2, 3))

    logging.debug("(%f, %f) chisq=%f", ctr[0], ctr[1], chisq)

    return chisq, chisqgrad


def fit_galaxy_single(galaxy0, data, weight, ctr, psf, regpenalty, factor):
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
    factor : float
        Factor used in fmin_l_bfgs_b to determine fit accuracy.
    """

    # Define objective function to minimize.
    # Returns chi^2 (including regularization term) and its gradient.
    def objective(galparams):

        # galparams is 1-d (raveled version of galaxy); reshape to 3-d.
        galaxy = galparams.reshape(galaxy0.shape)
        cval, cgrad = chisq_galaxy_single(galaxy, data, weight, ctr, psf)
        rval, rgrad = regpenalty(galaxy)
        totval = cval + rval
        logging.debug(u'\u03C7\u00B2 = %8.2f (%8.2f + %8.2f)', totval, cval, rval)

        # ravel gradient to 1-d when returning.
        return totval, np.ravel(cgrad + rgrad)

    # run minimizer
    galparams0 = np.ravel(galaxy0)  # fit parameters must be 1-d
    galparams, f, d = fmin_l_bfgs_b(objective, galparams0, factr=factor)
    _check_result(d['warnflag'], d['task'])
    _log_result("fmin_l_bfgs_b", f, d['nit'], d['funcalls'])

    return galparams.reshape(galaxy0.shape)


def fit_galaxy_sky_multi(galaxy0, datas, weights, ctrs, psfs, regpenalty,
                         factor):
    """Fit the galaxy model to multiple data cubes.

    Parameters
    ----------
    galaxy0 : ndarray (3-d)
        Initial galaxy model.
    datas : list of ndarray
        Sky-subtracted data for each epoch to fit.
    """

    nepochs = len(datas)

    # Get initial chisq values for info output.
    cvals = []
    for data, weight, ctr, psf in zip(datas, weights, ctrs, psfs):
        cval, _ = chisq_galaxy_sky_single(galaxy0, data, weight, ctr, psf)
        cvals.append(cval)

    logging.info(u"        initial \u03C7\u00B2/epoch: [%s]",
                 ", ".join(["%8.2f" % v for v in cvals]))

    # Define objective function to minimize.
    # Returns chi^2 (including regularization term) and its gradient.
    def objective(galparams):

        # galparams is 1-d (raveled version of galaxy); reshape to 3-d.
        galaxy = galparams.reshape(galaxy0.shape)
        cval, cgrad = chisq_galaxy_sky_multi(galaxy, datas, weights,
                                             ctrs, psfs)
        rval, rgrad = regpenalty(galaxy)

        totval = cval + rval
        logging.debug(u'\u03C7\u00B2 = %8.2f (%8.2f + %8.2f)', totval, cval, rval)

        # ravel gradient to 1-d when returning.
        return totval, np.ravel(cgrad + rgrad)

    # run minimizer
    galparams0 = np.ravel(galaxy0)  # fit parameters must be 1-d
    galparams, f, d = fmin_l_bfgs_b(objective, galparams0, factr=factor)
    _check_result(d['warnflag'], d['task'])

    galaxy = galparams.reshape(galaxy0.shape)

    # Get final chisq values.
    cvals = []
    for data, weight, ctr, psf in zip(datas, weights, ctrs, psfs):
        cval, _ = chisq_galaxy_sky_single(galaxy, data, weight, ctr, psf)
        cvals.append(cval)
    logging.info(u"        final   \u03C7\u00B2/epoch: [%s]",
                 ", ".join(["%8.2f" % v for v in cvals]))

    _log_result("fmin_l_bfgs_b", f, d['nit'], d['funcalls'])

    # get last-calculated skys, given galaxy.
    skys = []
    for data, weight, ctr, psf in zip(datas, weights, ctrs, psfs):
        scene = psf.evaluate_galaxy(galaxy, data.shape[1:3], ctr)
        sky = np.average(data - scene, weights=weight, axis=(1, 2))
        skys.append(sky)

    return galaxy, skys


def fit_position_sky(galaxy, data, weight, ctr0, psf, bounds):
    """Fit data position and sky for a single epoch (fixed galaxy model).

    Parameters
    ----------
    galaxy : ndarray (3-d)
    data : ndarray (3-d)
    weight : ndarray(3-d)
    ctr0 : (float, float)
        Initial center.
    bounds : [(float, float), (float, float)]
        Lower and upper bounds on each parameter. Order:
        (lower y, upper y), (lower x, upper x).

    Returns
    -------
    ctr : (float, float)
        (y, x) center position.
    sky : ndarray (1-d)
        Fitted sky.
    """

    ctr, f, d = fmin_l_bfgs_b(chisq_position_sky, ctr0,
                              args=(galaxy, data, weight, psf),
                              iprint=0, callback=None, bounds=bounds)
    _check_result(d['warnflag'], d['task'])
    _log_result("fmin_l_bfgs_b", f, d['nit'], d['funcalls'])

    # get last-calculated sky.
    g = psf.evaluate_galaxy(galaxy, data.shape[1:3], ctr)
    sky = determine_sky(data, weight, g)

    return tuple(ctr), sky


def chisq_position_sky_sn_multi(allctrs, galaxy, datas, weights, psfs):
    """Function to minimize. `allctrs` is a 1-d ndarray:

    [yctr[0], xctr[0], yctr[1], xctr[1], ..., snyctr, snxctr]

    where the indicies are
    """

    nepochs = len(datas)

    snctr_ind = slice(2*nepochs, 2*nepochs+2)
    snctr = tuple(allctrs[snctr_ind])

    # initialize return values
    chisq = 0.
    chisqgrad = np.zeros_like(allctrs)

    for i in range(nepochs):
        data = datas[i]
        weight = weights[i]
        psf = psfs[i]
        ctr_ind = slice(2*i, 2*i+2)
        ctr = tuple(allctrs[ctr_ind])

        g, ggrad = psf.evaluate_galaxy(galaxy, data.shape[1:3], ctr, grad=True)
        s, sgrad = psf.point_source(snctr, data.shape[1:3], ctr, grad=True)

        # add galaxy gradient with SN position
        ggrad = np.vstack((ggrad, np.zeros_like(ggrad)))

        sky, sn, skygrad, sngrad = sky_and_sn(data, weight, g, s,
                                              ggrad=ggrad, sgrad=sgrad)

        scene = sky[:, None, None] + g + sn[:, None, None] * s
        diff = data - scene
        chisq += np.sum(weight * diff**2)

        # gradient on chisq for this epoch with position and sn position
        dscene = (skygrad[:, :, None, None] + ggrad +
                  sngrad[:, :, None, None] * s + sn[:, None, None] * sgrad)
        dchisq = -2. * np.sum(weight * diff * dscene, axis=(1, 2, 3))

        # add gradient to right place in chisqgrad
        chisqgrad[ctr_ind] += dchisq[0:2]
        chisqgrad[snctr_ind] += dchisq[2:4]

    return chisq, chisqgrad


def fit_position_sky_sn_multi(galaxy, datas, weights, yctr0, xctr0, snctr0,
                              psfs, factor, yctrbounds, xctrbounds,
                              snctrbounds):
    """Fit data pointing (nepochs), SN position (in model frame),
    SN amplitude (nepochs), and sky level (nepochs). This is meant to be
    used only on epochs with SN light.

    Parameters
    ----------
    galaxy : ndarray (3-d)
    datas : list of ndarray (3-d)
    weights : list of ndarray (3-d)
    ctrs0 : list of tuples
        Initial data positions (y, x)
    snctr0 : tuple
        Initial SN position.
    psfs : list of PSFModels
    relbound : float
        Bound on positions relative to initial positions. Bounds will
        be ``(intial - relbound, initial + relbound)``.

    Returns
    -------
    fctrs : list of tuples
        Fitted data positions.
    fsnctr : tuple
        Fitted SN position.
    skys : list of ndarray (1-d)
        FItted sky spectra for each epoch.
    sne : list of ndarray (1-d)
        Fitted SN spectra for each epoch.

    Notes
    -----
    Given the data pointing and SN position, determining
    the sky level and SN amplitude is a linear problem. Therefore, we
    have only the data pointing and sn position as parameters in the
    (nonlinear) optimization and determine the sky and sn amplitude in
    each iteration.
    """

    nepochs = len(datas)
    assert len(weights) == len(yctr0) == len(xctr0) == len(psfs) == nepochs

    # Reshape initial positions to [y0, x0, y1, x1, ... , ysn, xsn]
    allctrs0 = np.empty(2*nepochs + 2, dtype=np.float64)
    allctrs0[0:2*nepochs:2] = yctr0
    allctrs0[1:2*nepochs:2] = xctr0
    allctrs0[2*nepochs:2*nepochs+2] = snctr0

    # reshape bounds to [(y0min, y0max), (x0min, x0max), ...]
    bounds = np.empty((2*nepochs + 2, 2), dtype=np.float64)
    bounds[0:2*nepochs:2, :] = yctrbounds
    bounds[1:2*nepochs:2, :] = xctrbounds
    bounds[2*nepochs:2*nepochs+2, :] = snctrbounds

    def callback(params):
        for i in range(len(params)//2-1):
            logging.debug('Epoch %s: %s, %s', i, params[2*i], params[2*i+1])
        logging.debug('SN position %s, %s', params[-2], params[-1])
    logging.debug('Bounds:')
    callback(bounds)
    logging.debug('')

    fallctrs, f, d = fmin_l_bfgs_b(chisq_position_sky_sn_multi, allctrs0,
                                   args=(galaxy, datas, weights, psfs),
                                   iprint=0, callback=callback, bounds=bounds,
                                   factr=factor)
    _check_result(d['warnflag'], d['task'])
    _log_result("fmin_l_bfgs_b", f, d['nit'], d['funcalls'])

    # pull out fitted positions
    fyctr = fallctrs[0:2*nepochs:2].copy()
    fxctr = fallctrs[1:2*nepochs:2].copy()
    fsnctr = fallctrs[2*nepochs:2*nepochs+2].copy()

    # evaluate final sky and sn in each epoch
    skys = []
    sne = []
    for i in range(nepochs):
        g = psfs[i].evaluate_galaxy(galaxy, datas[i].shape[1:3],
                                    (fyctr[i], fxctr[i]))
        s = psfs[i].point_source(fsnctr, datas[i].shape[1:3],
                                 (fyctr[i], fxctr[i]))
        sky, sn = sky_and_sn(datas[i], weights[i], g, s)
        skys.append(sky)
        sne.append(sn)

    return fyctr, fxctr, fsnctr, skys, sne


class RegularizationPenalty(object):
    """Callable that returns the penalty and gradient on it."""
    
    def __init__(self, galprior, mean_gal_spec, mu_xy, mu_wave):
        self.galprior = galprior
        self.mean_gal_spec = mean_gal_spec
        self.mu_xy = mu_xy
        self.mu_wave = mu_wave

    def __call__(self, galmodel):
        """Return regularization penalty and gradient for a given galaxy model.

        Parameters
        ----------
        TODO

        Returns
        -------
        penalty : float
        penalty_gradient : ndarray
            Gradient with respect to model galaxy
        """

        galdiff = galmodel - self.galprior
        galdiff /= self.mean_gal_spec[:, None, None]
        dw = galdiff[1:, :, :] - galdiff[:-1, :, :]
        dy = galdiff[:, 1:, :] - galdiff[:, :-1, :]
        dx = galdiff[:, :, 1:] - galdiff[:, :, :-1]

        # Regularlization penalty term
        val = (self.mu_wave * np.sum(dw**2) +
               self.mu_xy * np.sum(dy**2) +
               self.mu_xy * np.sum(dx**2))

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

        grad = np.zeros_like(galdiff)
        grad[1:, :, :] += 2. * self.mu_wave * dw
        grad[:-1, :, :] -= 2. * self.mu_wave * dw
        grad[:, 1:, :] += 2. * self.mu_xy * dy
        grad[:, :-1, :] -= 2. * self.mu_xy * dy
        grad[:, :, 1:] += 2. * self.mu_xy * dx
        grad[:, :, :-1] -= 2. * self.mu_xy * dx

        grad /= self.mean_gal_spec[:, None, None]  # put back normalization

        return val, grad
