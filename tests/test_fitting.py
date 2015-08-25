#/usr/bin/env py.test

from __future__ import print_function

import os
import sys
import sysconfig

import numpy as np
from numpy.fft import fft2, ifft2
from numpy.testing import assert_allclose
from scipy.optimize import check_grad, approx_fprime

# Use built version of cubefit, because C-extensions
#dirname = "lib.{}-{}.{}".format(sysconfig.get_platform(),
#                                sys.version_info[0],
#                                sys.version_info[1])
#sys.path.insert(0, os.path.join("build", dirname))

import cubefit
from cubefit.fitting import (sky_and_sn,
                             chisq_galaxy_single,
                             chisq_galaxy_sky_multi,
                             chisq_position_sky_sn_multi)

# -----------------------------------------------------------------------------
# Helper functions

def assert_real(x):
    if np.all((x.imag == 0.) & (x.real == 0.)):
        return
    absfrac = np.abs(x.imag / x.real)
    mask = absfrac < 1.e-3 #1.e-4
    if not np.all(mask):
        raise RuntimeError("array not real: max imag/real = {:g}"
                           .format(np.max(absfrac)))


def fftconvolve(x, kernel):
    """convolve 2-d array x with a kernel *centered* in array."""
    
    ny, nx = kernel.shape
    xctr, yctr = (nx-1)/2., (ny-1)/2.
    
    # Phasor that will shift kernel to be centered at (0., 0.)
    fshift = cubefit.fft_shift_phasor_2d(kernel.shape, (-xctr, -yctr))
    
    return ifft2(fft2(kernel) * fft2(x) * fshift).real


def plot_gradient(im, fname, **kwargs):
    """Helper function for debugging only."""
    import matplotlib.pyplot as plt

    plt.imshow(im, cmap="bone", interpolation="nearest", origin="lower",
               **kwargs)
    plt.colorbar()
    plt.savefig(fname)
    plt.clf()

# -----------------------------------------------------------------------------

def test_sky_and_sn():
    truesky = 3. * np.ones((10,))
    truesn = 2. * np.ones((10,))

    gal = np.ones((10, 5, 5))  # fake galaxy, after convolution with PSF
    psf = np.zeros((10, 5, 5))  
    psf[:, 3, 3] = 1.  # psf is a single pixel

    data = gal + truesky[:, None, None] + truesn[:, None, None] * psf
    weight = np.ones_like(data)
    
    sky, sn = cubefit.fitting.sky_and_sn(data, weight, gal, psf)

    assert_allclose(sky, truesky)
    assert_allclose(sn, truesn)

class TestFitting:
    def setup_class(self):
        """Create some dummy data and an AtmModel."""

        # some settings
        MODEL_SHAPE = (32, 32)
        nt = 3
        nw = 3
        ny = 15
        nx = 15
        yoffs = np.array([7,8,9])  # offset between model and data
        xoffs = np.array([8,9,7])

        # True data yctr, xctr given offset
        self.trueyctrs = yoffs + (ny-1)/2. - (MODEL_SHAPE[0]-1)/2.
        self.truexctrs = xoffs + (nx-1)/2. - (MODEL_SHAPE[1]-1)/2.

        # Create a "true" underlying galaxy. This can be anything, but it
        # should not be all zeros or flat. The fourth and fifth parameters
        # are ellipticity and alpha.
        ellip = 4.5 * np.ones(nw)
        alpha = 6.0 * np.ones(nw)
        p = cubefit.GaussMoffatPSF(ellip, alpha)
        truegal = p(MODEL_SHAPE, np.zeros(nw) - 2., np.zeros(nw) - 2.)

        # Create a PSF. The fourth and fifth parameters are ellipticity and
        # alpha.
        ellip = 1.5 * np.ones(nw)
        alpha = 2.0 * np.ones(nw)
        psfmodel = cubefit.GaussMoffatPSF(ellip, alpha)
        psf = psfmodel(MODEL_SHAPE, np.zeros(nw), np.zeros(nw))

        # create the data by convolving the true galaxy model with the psf
        # and taking a slice.
        cubes = []
        for j in range(nt):
            data = np.empty((nw, ny, nx), dtype=np.float32)        
            for i in range(nw):
                data_2d = fftconvolve(truegal[i], psf[i])
                data[i, :, :] = data_2d[yoffs[j]:yoffs[j]+ny, 
                                        xoffs[j]:xoffs[j]+nx]
            cubes.append(cubefit.DataCube(data, np.ones_like(data), 
                                          np.ones(nw)))
        self.cubes = cubes

        # make a fake AtmModel
        adr_refract = np.zeros((2, nw))
        self.atm = cubefit.AtmModel(psf, adr_refract)

        # initialize galaxy model
        self.galaxy = np.zeros((nw, MODEL_SHAPE[0], MODEL_SHAPE[1]))
        self.truegal = truegal


    def test_chisq_galaxy_single_gradient(self):
        """Test that gradient function (used in galaxy fitting) returns value
        close to what you get with a finite differences method.
        """

        EPS = 1.e-7

        data = self.cubes[0].data
        weight = self.cubes[0].weight
        atm = self.atm
        ctr = (0., 0.)

        # analytic gradient is `grad`
        val, grad = chisq_galaxy_single(self.galaxy, data, weight, ctr, atm)

        # save data - model residuals for finite differences chi^2 gradient.
        # need to carry out subtraction in float64 to avoid round-off errors.
        scene = atm.evaluate_galaxy(self.galaxy, data.shape[1:3], ctr)
        r0 = data.astype(np.float64) - scene

        # finite differences gradient: alter each element by EPS one
        # at a time and recalculate chisq.
        fdgrad = np.zeros_like(self.galaxy)
        nk, nj, ni = self.galaxy.shape
        for k in range(nk):
            for j in range(nj):
                for i in range(ni):
                    self.galaxy[k, j, i] += EPS
                    scene = atm.evaluate_galaxy(self.galaxy, data.shape[1:3],
                                                ctr)
                    self.galaxy[k, j, i] -= EPS # reset model value.

                    # NOTE: rather than calculating
                    # chisq1 - chisq0 = sum(w * r1^2) - sum(w * r0^2)
                    # we calculate
                    # sum(w * (r1^2 - r0^2))
                    # which is the same quantity but avoids summing large
                    # numbers.
                    r1 = data.astype(np.float64) - scene
                    chisq_diff = np.sum(weight * (r1**2 - r0**2))
                    fdgrad[k, j, i] = chisq_diff / EPS

        assert_allclose(grad, fdgrad, rtol=0.001, atol=0.)

    def test_chisq_galaxy_sky_multi_gradient(self):
        """Test that gradient function (used in galaxy fitting) returns value
        close to what you get with a finite differences method.
        """

        EPS = 1.e-8

        datas = [self.cubes[0].data]
        weights = [self.cubes[0].weight]
        atms = [self.atm]
        ctrs = [(0., 0.)]

        # analytic gradient is `grad`
        _, grad = chisq_galaxy_sky_multi(self.galaxy, datas, weights, ctrs,
                                         atms)

        # NOTE: Following is specific to only having one cube!
        data = datas[0]
        weight = weights[0]
        atm = atms[0]
        ctr = ctrs[0]

        # save data - model residuals for finite differences chi^2 gradient.
        # need to carry out subtraction in float64 to avoid round-off errors.
        scene = atm.evaluate_galaxy(self.galaxy, data.shape[1:3], ctr)
        r0 = data.astype(np.float64) - scene
        sky = np.average(r0, weights=weight, axis=(1, 2))
        r0 -= sky[:, None, None]

        # finite differences gradient: alter each element by EPS one
        # at a time and recalculate chisq.
        fdgrad = np.zeros_like(self.galaxy)
        nk, nj, ni = self.galaxy.shape
        for k in range(nk):
            for j in range(nj):
                for i in range(ni):
                    self.galaxy[k, j, i] += EPS
                    scene = atm.evaluate_galaxy(self.galaxy, data.shape[1:3],
                                                ctr)
                    self.galaxy[k, j, i] -= EPS # reset model value.

                    # NOTE: rather than calculating
                    # chisq1 - chisq0 = sum(w * r1^2) - sum(w * r0^2)
                    # we calculate
                    # sum(w * (r1^2 - r0^2))
                    # which is the same quantity but avoids summing large
                    # numbers.
                    r1 = data.astype(np.float64) - scene
                    sky = np.average(r1, weights=weight, axis=(1, 2))
                    r1 -= sky[:, None, None]
                    chisq_diff = np.sum(weight * (r1**2 - r0**2))
                    fdgrad[k, j, i] = chisq_diff / EPS

        assert_allclose(grad, fdgrad, rtol=0.005, atol=0.)


    def pixel_regpenalty_diff(self, regpenalty, galmodel, k, j, i, eps):
        """What is the difference in the regpenalty caused by changing
        galmodel[k, j, i] by EPS?"""

        def galnorm(k, j, i, eps=0.0):
            return ((galmodel[k, j, i] + eps - regpenalty.galprior[k, j, i]) /
                    regpenalty.mean_gal_spec[k])

        dchisq = 0.

        if k > 0:
            d0 = galnorm(k, j, i)      - galnorm(k-1, j, i)
            d1 = galnorm(k, j, i, eps) - galnorm(k-1, j, i)
            dchisq += regpenalty.mu_wave * (d1**2 - d0**2)
        if k < galmodel.shape[0] - 1:
            d0 = galnorm(k+1, j, i) - galnorm(k, j, i)
            d1 = galnorm(k+1, j, i) - galnorm(k, j, i + eps)
            dchisq += regpenalty.mu_wave * (d1**2 - d0**2)

        if j > 0:
            d0 = galnorm(k, j, i)      - galnorm(k, j-1, i)
            d1 = galnorm(k, j, i, eps) - galnorm(k, j-1, i)
            dchisq += regpenalty.mu_xy * (d1**2 - d0**2)
        if j < galmodel.shape[1] - 1:
            d0 = galnorm(k, j+1, i) - galnorm(k, j, i)
            d1 = galnorm(k, j+1, i) - galnorm(k, j, i, eps)
            dchisq += regpenalty.mu_xy * (d1**2 - d0**2)

        if i > 0:
            d0 = galnorm(k, j, i)      - galnorm(k, j, i-1)
            d1 = galnorm(k, j, i, eps) - galnorm(k, j, i-1)
            dchisq += regpenalty.mu_xy * (d1**2 - d0**2)
        if i < galmodel.shape[2] - 1:
            d0 = galnorm(k, j, i+1) - galnorm(k, j, i)
            d1 = galnorm(k, j, i+1) - galnorm(k, j, i, eps)
            dchisq += regpenalty.mu_xy * (d1**2 - d0**2)

        return dchisq

    def test_regularization_penalty_gradient(self):
        """Ensure that regularization penalty gradient matches what you
        get with a finite-differences approach."""

        EPS = 1.e-9
        mu_wave = 0.07
        mu_xy = 0.001

        # set galaxy model to best-fit (so that it is not all zeros!)
        self.galaxy[:, :, :] = self.truegal

        mean_gal_spec = np.average(self.cubes[0].data, axis=(1, 2))
        galprior = np.zeros_like(self.galaxy)
        regpenalty = cubefit.RegularizationPenalty(galprior, mean_gal_spec,
                                                   mu_xy, mu_wave)

        _, grad = regpenalty(self.galaxy)
        fdgrad = np.zeros_like(self.galaxy)
        nk, nj, ni = self.galaxy.shape
        for k in range(nk):
            for j in range(nj):
                for i in range(ni):
                    fdgrad[k, j, i] = self.pixel_regpenalty_diff(
                        regpenalty, self.galaxy, k, j, i, EPS) / EPS

        rtol = 0.001
        atol = 1.e-5*np.max(np.abs(fdgrad))
        assert_allclose(grad, fdgrad, rtol=rtol, atol=atol)

    def test_point_source(self):
        """Test that evaluate_point_source returns the expected point source.
        """

        psf = self.atm.evaluate_point_source((0., 0.), (15, 15), (0., 0.))
        
    def test_fit_position_grad(self):
        """Test the gradient of the sn and sky position fitting function
        """
        
        def func_part(ctrs, galaxy, datas, weights, atms):
            chisq, grad = chisq_position_sky_sn_multi(ctrs, galaxy, 
                                                      datas, weights, atms)
            return chisq

        def grad_part(ctrs, galaxy, datas, weights, atms):
            chisq, grad = chisq_position_sky_sn_multi(ctrs, galaxy, 
                                                      datas, weights, atms)
            return grad

        x0s = np.zeros(8)
        datas = [cube.data for cube in self.cubes]
        weights = [cube.weight for cube in self.cubes]
        atms = [self.atm for cube in self.cubes]

        code_grad = grad_part(x0s, self.galaxy, datas, weights, atms)
        test_grad = approx_fprime(x0s, func_part, np.sqrt(np.finfo(float).eps),
                            self.galaxy, datas, weights, atms)
        assert_allclose(code_grad[:-2], test_grad[:-2], rtol=0.005)
