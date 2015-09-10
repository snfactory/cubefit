"""PSF tests."""

import numpy as np
from numpy.testing import assert_allclose

import cubefit
import psf_alt

class TestGaussMoffatPSF:
    def setup_class(self):
        self.ellip = np.array([0.5, 1.0, 1.5, 2.0])
        self.alpha = np.ones(4)
        self.psf = cubefit.GaussMoffatPSF(self.ellip, self.alpha)

    def test_norm(self):
        """Test normalization. of GaussMoffatPSF"""

        # sample on 100 x 100 grid ~ infinitly large
        xctr = np.zeros_like(self.ellip)
        yctr = np.zeros_like(self.ellip)
        x = self.psf((100, 100), yctr, xctr, subpix=3)

        sums = (np.sum(x, axis=(1,2))) # sum at each wavelength.
        assert_allclose(sums, 1., rtol=0.0005)

    def test_comparison_to_pure_python(self):
        psf2 = psf_alt.GaussMoffatPSF(self.ellip, self.alpha)

        xctr = np.zeros_like(self.ellip)
        yctr = np.zeros_like(self.ellip)
        x = self.psf((100, 100), yctr, xctr, subpix=1)
        y = psf2((100, 100), yctr, xctr)

        assert_allclose(x, y)

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

    # old version
    psfarray = psf_alt.psf_3d_from_params(psf_params, wave, REFWAVE,
                                          MODEL_SHAPE)

    # new version

    relwave = wave / REFWAVE - 1.0
    ellip = psf_params[0] * np.ones_like(wave)
    alpha = (psf_params[1] +
             psf_params[2] * relwave +
             psf_params[3] * relwave**2)

    psf = cubefit.GaussMoffatPSF(ellip, alpha)
    psfarray2 = psf(MODEL_SHAPE, np.zeros_like(wave), np.zeros_like(wave))

    assert_allclose(psfarray2, psfarray)
