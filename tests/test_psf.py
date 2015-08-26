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
