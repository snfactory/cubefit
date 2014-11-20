from __future__ import print_function

import numpy as np
from numpy.fft import fft2, ifft2
from numpy.testing import assert_allclose

import ddtpy
from ddtpy.fitting import (penalty_g_all_epoch, likelihood_penalty,
                           regularization_penalty)


def convolve_fft(x, kernel):
    """convolve 2-d array x with a kernel *centered* in array."""
    
    ny, nx = kernel.shape
    xctr, yctr = (nx-1)/2., (ny-1)/2.
    
    # Phasor that will shift kernel to be centered at (0., 0.)
    fshift = ddtpy.fft_shift_phasor_2d(kernel.shape, (-xctr, -yctr))
    
    return ifft2(fft2(kernel) * fft2(x) * fshift).real


class TestFitting:
    def setup_class(self):
        """Create an instance of DDTData and DDTModel so that we can fit them"""

        # some settings
        nt = 1
        nw = 1
        ellipticity = 1.5
        alpha = 2.0

        # Create model.
        wave = np.linspace(4000., 6000., nw)
        adr_dx = np.zeros((nt,nw))
        adr_dy = np.zeros((nt,nw))
        mu_xy = 1.0e-3
        mu_wave = 7.0e-2
        sky_guess = np.zeros((nt,nw))
        self.model = ddtpy.DDTModel(nt, wave,
                                    ellipticity * np.ones((nt, nw)),
                                    alpha * np.ones((nt, nw)),
                                    adr_dx, adr_dy, mu_xy, mu_wave,
                                    0., 0., sky_guess)


        # Create arbitrary (non-zero!) data.
        self.truegal = ddtpy.gaussian_plus_moffat_psf((32, 32), 13.5, 13.5,
                                                      4.5, 6.0, 0.5)
        psf = ddtpy.gaussian_plus_moffat_psf((32, 32), 15.5, 15.5,
                                             ellipticity, alpha, 0.)
        data_2d = convolve_fft(self.truegal, psf)
        data = np.empty((nt, nw, 15, 15))
        xoff = 8
        yoff = 8
        data[0, 0, :, :] = data_2d[yoff:yoff+15, xoff:xoff+15]
        self.xctr_true = 15.5 - (xoff + 7.)
        self.yctr_true = 15.5 - (yoff + 7.)

        # Create a DDTData instance from the data.
        weight = np.ones_like(data)
        header = {}
        is_final_ref = np.ones(nt,dtype = bool)
        master_final_ref = 0
        xctr_init = np.zeros(nt)
        yctr_init = np.zeros(nt)
        self.data = ddtpy.DDTData(data, weight, wave, xctr_init, yctr_init,
                                  is_final_ref, master_final_ref, header)

    def test_gradient(self):
        """Test that gradient functions (used in galaxy fitting) return values
        'close' to what you get with a finite differences method."""
        
        x_diff = 1.e-10

        # analytic gradient
        lkl_err, lkl_grad = likelihood_penalty(self.model, self.data)

        # finite differences gradient
        fd_lkl_grad = np.zeros(self.model.gal.size, dtype=np.float)
        for j in range(32):
            for i in range(32):
                self.model.gal[0,j,i] += x_diff
                new_lkl_err, _ = likelihood_penalty(self.model, self.data)
                self.model.gal[0,j,i] = 0.
                fd_lkl_grad[j*32+i] = (new_lkl_err - lkl_err) / x_diff

        assert_allclose(lkl_grad, fd_lkl_grad, rtol=0.016)


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

