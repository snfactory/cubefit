"""PSF tests."""

import numpy as np
from numpy.testing import assert_allclose

import cubefit
from . import psffuncs_pure

def get_gaussian_moffat_psf(subpix):
    """Helper function for constructing a GaussianMoffatPSF for testing"""

    nw = 4
    sigma = np.ones(nw)
    alpha = np.ones(nw)
    beta = 2. * np.ones(nw)
    ellip = np.array([0.5, 1.0, 1.5, 2.0])
    eta = 2. * np.ones(nw)
    yctr = np.array([0., 0.5, 1.0, 1.5])
    xctr = np.array([0., 0.5, 1.0, 1.5])

    return cubefit.GaussianMoffatPSF(sigma, alpha, beta, ellip, eta,
                                     yctr, xctr, (32, 32), subpix=subpix)


def test_gaussian_moffat_norm():
    """Test normalization. of GaussMoffatPSF"""

    psf = get_gaussian_moffat_psf(3)

    # sample onto 100 x 100 grid ~ infinitly large
    A = psf.point_source((0., 0.), (100, 100), (0., 0.))

    sums = A.sum(axis=(1, 2)) # sum at each wavelength.

    assert_allclose(sums, 1., rtol=0.0005)


def test_comparison_to_pure_python():
    psf = get_gaussian_moffat_psf(1)
    A = psf.point_source((0., 0.), (100, 100), (0., 0.))
    B = psffuncs_pure.gaussian_moffat_psf(psf.sigma, psf.alpha, psf.beta,
                                          psf.ellipticity, psf.eta, psf.yctr,
                                          psf.xctr, (100, 100))

    assert not np.all(A == 0.)
    assert not np.all(B == 0.)
    assert_allclose(A, B)


def test_old_version():
    MODEL_SHAPE = (32, 32)
    REFWAVE = 5000.
    psf_params = [1.659338,   # from PTF09fox_B epoch 0
                  1.921927,
                  -0.185608,
                  1.851729,
                  2.121143,
                  -0.148159,
                  0.066841]
    wave = np.linspace(3200., 5600., 800)

    # new version
    relwave = wave / REFWAVE - 1.0
    ellip = psf_params[0] * np.ones_like(wave)
    alpha = (psf_params[1] +
             psf_params[2] * relwave +
             psf_params[3] * relwave**2)
    sigma = 0.545 + 0.215 * alpha
    beta  = 1.685 + 0.345 * alpha
    eta   = 1.040 + 0.0   * alpha 

    psf = cubefit.GaussianMoffatPSF(sigma, alpha, beta, ellip, eta,
                                    np.zeros_like(wave), np.zeros_like(wave),
                                    MODEL_SHAPE, subpix=1)
    A = psf.point_source((0., 0.), MODEL_SHAPE, (0., 0.))

    # old version
    B = psffuncs_pure.psf_3d_from_params(psf_params, wave, REFWAVE,
                                         MODEL_SHAPE)

    assert_allclose(A, B, rtol=1.e-2)
