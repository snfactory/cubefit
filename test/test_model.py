from __future__ import print_function

import numpy as np

from ddtpy import DDTModel, DDTData
from ddtpy.psf import gaussian_plus_moffat_psf_4d
from ddtpy.fitting import penalty_g_all_epoch, likelihood_penalty
from ddtpy.fitting import regularization_penalty

class TestFitting:

    def setup_class(self):

        nt = 3
        nw = 500
        ellipticity = 1.5*np.ones((nt, nw))
        alpha = 2.0*np.ones((nt,nw))
        adr_dx, adr_dy = np.zeros((nt,nw)), np.zeros((nt,nw))
        spaxel_size = 0.43
        mu_xy = 1.0e-3
        mu_wave = 7.0e-2
        data_xctr_init = np.ones(nt)
        data_yctr_init = np.ones(nt)
        sky_guess = np.zeros((nt,nw))

        self.model = DDTModel((nt, nw), ellipticity, alpha, adr_dx, adr_dy,
                         spaxel_size, mu_xy, mu_wave, data_xctr_init,
                         data_yctr_init, sky_guess)

        data = gaussian_plus_moffat_psf_4d((15,15), 5., 5., ellipticity, alpha)
        weight = np.ones_like(data)
        header = {}
        wave = np.arange(500)
        is_final_ref = np.ones(nt,dtype = bool)
        master_final_ref = 0
        
        self.data = DDTData(data, weight, wave, is_final_ref, master_final_ref,
                            header, spaxel_size)
 

    def test_gradient(self):
        
        x = np.copy(self.model.gal.reshape(self.model.gal.size))
        toterr, grad = penalty_g_all_epoch(x, self.model, self.data)
        lkl_err, lkl_grad = likelihood_penalty(self.model, self.data)
        rgl_err, rgl_grad = regularization_penalty(self.model, self.data)
        
        x_diff = 1.e-5
        x[0] += x_diff
        new_toterr, new_grad = penalty_g_all_epoch(x, self.model, self.data)
        new_lkl_err, new_lkl_grad = likelihood_penalty(self.model, self.data)
        new_rgl_err, new_rgl_grad = regularization_penalty(self.model, self.data)
        
        d_lkl = (new_lkl_err - lkl_err)/x_diff
        d_rgl = (new_rgl_err - rgl_err)/x_diff
        d_tot = (new_toterr - toterr)/x_diff

        #assert(d_rgl == rgl_grad[0])
        assert(d_lkl == lkl_grad[0])
        assert(d_tot == grad[0])
