from __future__ import print_function

import numpy as np
from numpy.fft import fft2, ifft2
from numpy.testing import assert_allclose

import ddtpy


def convolve_fft(x, kernel):
    """convolve 2-d array x with a kernel *centered* in array."""
    
    ny, nx = kernel.shape
    xctr, yctr = (nx-1)/2., (ny-1)/2.
    
    # Phasor that will shift kernel to be centered at (0., 0.)
    fshift = ddtpy.fft_shift_phasor_2d(kernel.shape, (-xctr, -yctr))
    
    return ifft2(fft2(kernel) * fft2(x) * fshift).real


def chisq(galaxy, data, weight, ctr, atm):
    m = atm.evaluate_galaxy(galaxy, data.shape[1:3], ctr)
    diff = data - m
    wdiff = weight * diff
    chisq_val = np.sum(wdiff * diff)
    chisq_grad = atm.gradient_helper(-2. * wdiff, data.shape[1:3], ctr)

    return chisq_val, chisq_grad


class TestFitting:
    def setup_class(self):
        """Create some dummy data and an AtmModel."""

        # some settings
        nt = 1
        nw = 1
        ellipticity = 1.5
        alpha = 2.0

        wave = np.linspace(4000., 6000., nw)

        # Create arbitrary (non-zero!) data.
        self.truegal = ddtpy.gaussian_plus_moffat_psf((32, 32), 13.5, 13.5,
                                                      4.5, 6.0, 0.5)
        psf = ddtpy.gaussian_plus_moffat_psf((32, 32), 15.5, 15.5,
                                             ellipticity, alpha, 0.)

        # convolve the true galaxy model with the psf, take a 15x15 subslice
        data_2d = convolve_fft(self.truegal, psf)
        data = np.empty((nw, 15, 15))
        xoff = 8
        yoff = 8
        data[0, :, :] = data_2d[yoff:yoff+15, xoff:xoff+15]

        self.data = data
        self.weight = np.ones_like(data)
        self.xctr_true = 15.5 - (xoff + 7.)
        self.yctr_true = 15.5 - (yoff + 7.)

        # add a wavelength dimension (length 1.... nw must be 1!)
        self.truegal = self.truegal[None, :, :]
        psf = psf[None, :, :]

        # make a fake AtmModel
        adr_refract = np.zeros((2, nw))
        self.atm = ddtpy.AtmModel(psf, adr_refract)
        self.psf = psf

        # model
        self.galaxy = np.zeros_like(self.truegal)


    def test_gradient(self):
        """Test that gradient functions (used in galaxy fitting) return values
        'close' to what you get with a finite differences method."""
        
        EPS = 1.e-10

        # analytic gradient
        chisq_val, chisq_grad = chisq(self.galaxy, self.data, self.weight,
                                      (0., 0.), self.atm)

        # finite differences gradient
        fd_chisq_grad = np.zeros_like(self.galaxy)
        for j in range(32):
            for i in range(32):
                self.galaxy[0,j,i] += EPS
                new_chisq_val, _ = chisq(self.galaxy, self.data, self.weight,
                                      (0., 0.), self.atm)
                self.galaxy[0,j,i] = 0.

                fd_chisq_grad[0,j,i] = (new_chisq_val - chisq_val) / EPS

        assert_allclose(chisq_grad, fd_chisq_grad, rtol=0.02)

    def test_point_source(self):
        """Test that evaluate_point_source returns the expected point source.
        """

        psf = self.atm.evaluate_point_source((0., 0.), (15, 15), (0., 0.))
        
        from matplotlib import pyplot as plt

        plt.imshow(psf[0], origin='lower', cmap='Greys', interpolation='nearest')
        plt.savefig("testpsf.png")


"""Test the likelihood gradient. 

This is a sanity check to see if the gradient function is returning
the naive result for individual elements. The likelihood is given
by

L = sum_i w_i * (d_i - m_i)^2

where i represents pixels, d is the data, and m is the model
*sampled onto the data frame*. We want to know the derivative
with respect to model parameters x_j.

dL/dx_j = sum_i -2 w_i (d_i - m_i) dm_i/dx_j

dm_i/dx_j is the change in the resampled model due to changing model
parameter j. Changing model parameter j is adjusting a single pixel
in the model. The result in the data frame is a PSF at the position
corresponding to model pixel j.

"""
