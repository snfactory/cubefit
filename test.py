#!/usr/bin/env py.test

from __future__ import print_function

import numpy as np
from numpy.fft import fft2, ifft2
from numpy.testing import assert_allclose

import cubefit
from cubefit.fitting import (determine_sky_and_sn,
                             chisq_galaxy_single,
                             chisq_galaxy_sky_multi)

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


def plot_gradient(im, fname):
    """Helper function for debugging only."""
    import matplotlib.pyplot as plt

    plt.imshow(im, cmap="bone", interpolation="nearest", origin="lower")
    plt.colorbar()
    plt.savefig(fname)
    plt.clf()

# -----------------------------------------------------------------------------

def test_determine_sky_and_sn():
    truesky = 3. * np.ones((10,))
    truesn = 2. * np.ones((10,))

    gal = np.ones((10, 5, 5))  # fake galaxy, after convolution with PSF
    psf = np.zeros((10, 5, 5))  
    psf[:, 3, 3] = 1.  # psf is a single pixel

    data = gal + truesky[:, None, None] + truesn[:, None, None] * psf
    weight = np.ones_like(data)
    
    sky, sn = cubefit.fitting.determine_sky_and_sn(gal, psf, data, weight)

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
        yoff, xoff = (8, 8)  # offset between model and data

        # True data yctr, xctr given offset
        trueyctr = yoff + (ny-1)/2. - (MODEL_SHAPE[0]-1)/2.
        truexctr = xoff + (nx-1)/2. - (MODEL_SHAPE[1]-1)/2.

        # Create a "true" underlying galaxy. This can be anything, but it
        # should not be all zeros or flat. The fourth and fifth parameters
        # are ellipticity and alpha.
        truegal_2d = cubefit.gaussian_plus_moffat_psf(MODEL_SHAPE, 13.5, 13.5,
                                                      4.5, 6.0, 0.5)
        truegal = np.tile(truegal_2d, (nw, 1, 1))

        # Create a PSF. The fourth and fifth parameters are ellipticity and
        # alpha.
        psf_2d = cubefit.gaussian_plus_moffat_psf(MODEL_SHAPE, 15.5, 15.5,
                                                  1.5, 2.0, 0.)
        psf = np.tile(psf_2d, (nw, 1, 1))

        # create the data by convolving the true galaxy model with the psf
        # and taking a slice.
        data = np.empty((nw, ny, nx), dtype=np.float32)
        for i in range(nw):
            data_2d = fftconvolve(truegal[i], psf[i])
            data[i, :, :] = data_2d[yoff:yoff+ny, xoff:xoff+nx]

        # cube
        self.cube = cubefit.DataCube(data, np.ones_like(data), np.ones(nw))

        # make a fake AtmModel
        adr_refract = np.zeros((2, nw))
        self.atm = cubefit.AtmModel(psf, adr_refract)

        # initialize galaxy model
        self.galaxy = np.zeros((nw, MODEL_SHAPE[0], MODEL_SHAPE[1]))

        # True data yctr, xctr given offset
        self.truegal = truegal
        self.trueyctr = yoff + (ny-1)/2. - (MODEL_SHAPE[0]-1)/2.
        self.truexctr = xoff + (nx-1)/2. - (MODEL_SHAPE[1]-1)/2.


    def test_chisq_galaxy_single_gradient(self):
        """Test that gradient function (used in galaxy fitting) returns value
        close to what you get with a finite differences method.
        """

        EPS = 1.e-7

        data = self.cube.data
        weight = self.cube.weight
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

        datas = [self.cube.data]
        weights = [self.cube.weight]
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

    def test_regularization_penalty_gradient(self):
        """Ensure that regularization penalty gradient matches what you
        get with a finite-differences approach."""

        EPS = 1.e-8
        mu_wave = 0.07
        mu_xy = 0.001

        # set galaxy model to best-fit (so that it is not all zeros!)
        self.galaxy[:, :, :] = self.truegal

        mean_gal_spec = np.average(self.cube.data, axis=(1, 2))
        galprior = np.zeros_like(self.galaxy)
        regpenalty = cubefit.RegularizationPenalty(galprior, mean_gal_spec,
                                                   mu_xy, mu_wave)

        chisq0, grad = regpenalty(self.galaxy)
        fdgrad = np.zeros_like(self.galaxy)
        nk, nj, ni = self.galaxy.shape
        for k in range(nk):
            for j in range(nj):
                for i in range(ni):
                    self.galaxy[k, j, i] += EPS
                    chisq1, _ = regpenalty(self.galaxy)
                    self.galaxy[k, j, i] -= EPS # reset model value.
                    fdgrad[k, j, i] = (chisq1 - chisq0) / EPS

        plot_gradient(grad[0], "grad.png")
        plot_gradient(fdgrad[0], "fdgrad.png")

        assert_allclose(grad, fdgrad, rtol=0.005, atol=0.)

    def test_point_source(self):
        """Test that evaluate_point_source returns the expected point source.
        """

        psf = self.atm.evaluate_point_source((0., 0.), (15, 15), (0., 0.))
        
