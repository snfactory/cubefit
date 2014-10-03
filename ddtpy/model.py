from numpy import fft


class DDTModel(object):
    """This class is the equivalent of everything else that isn't data
    in the Yorick version.

    Parameters
    ----------
    shape : 2-tuple of int
        Model dimensions in (time, wave). Time and wave must match
        that of the data.
    psf_ellipticity, psf_alpha : np.ndarray (2-d)
        Parameters characterizing the PSF at each time, wavelength. Shape
        of both must match `shape` parameter.
    adr_dx, adr_dy : np.ndarray (2-d)
        Atmospheric differential refraction in x and y directions, in spaxels,
        relative to reference wavelength.
    spaxel_size : float
        Spaxel size in arcseconds.
    mu_xy : float
    mu_wave : float
    sky : np.ndarray (2-d)

    Notes
    -----
    The spatial coordinate system used to align data and the model is
    an arbitrary grid on the sky where the origin is chosen to be at
    the true location of the SN. That is, in the model the SN is
    located at (0., 0.)  by definition and the location and shape of
    the galaxy is defined relative to the SN location. Similarly, the
    pointings/alignment of the data are defined relative to this
    coordinate system. For example, if a pointing is centered at
    coordinates (3.5, 2.5), this means that we believe the central
    spaxel of the data array to be 3.5 spaxels west and 2.5 spaxels
    north of the true SN position. (The units of the coordinate system
    are in spaxels for convenience.)
    """

    MODEL_SHAPE = (32, 32)

    def __init__(self, shape, psf_ellipticity, psf_alpha, adr_dx, adr_dy,
                 spaxel_size, mu_xy, mu_wave, sky):

        ny, nx = MODEL_SHAPE
        nt, nw = shape

        if psf_ellipticity.shape != shape:
            raise ValueError("psf_ellipticity has wrong shape")
        if psf_alpha.shape != shape:
            raise ValueError("psf_alpha has wrong shape")

        # Model shape
        self.nt = nt
        self.nw = nw
        self.ny = ny
        self.nx = nx

        # Galaxy and sky part of the model
        self.gal = np.zeros((nw, ny, nx))
        self.galprior = np.zeros((nw, ny, nx))
        self.sky = sky
        self.sn = np.zeros((nt, nw))
        self.eta = np.ones(nt)  # eta = transmission
        self.final_ref_sky = np.zeros(nw)

        # PSF part of the model
        self.psf = gaussian_plus_moffat_psf_4d(MODEL_SHAPE, 15.5, 15.5,
                                               psf_ellipticity, psf_alpha)
        # This moves the center of the PSF from array coordinates
        # (model_sn_x, model_sn_y) -> (0, 0) [lower left pixel]
        # We suppose this is done because it is needed for convolution.
  
        # self.psf = roll_psf(self.psf, -self.model_sn_x, -self.model_sn_y)
        
        # Pointing of data. This is the location of the central spaxel of the
        # data with respect to the position of the SN in the model. (The
        # position of the SN in the model defines the coordinate system; see
        # comments in docstring.)
        self.data_xctr = np.zeros(nt, dtype=np.float64)
        self.data_yctr = np.zeros(nt, dtype=np.float64)

        # Make up a coordinate system for the model array
        #offx = int((nx-1) / 2.)
        #offy = int((ny-1) / 2.)
        #xcoords = np.arange(-offx, nx - offx)  # x coordinates on array
        #ycoords = np.arange(-offy, ny - offy)  # y coordinates on array

        # sn is "by definition" at array position where coordinates = (0,0)
        # model_sn_x = offx
        # model_sn_y = offy

    

        self.adr_dx = adr_dx
        self.adr_dy = adr_dy
        self.spaxel_size = spaxel_size
        self.mu_xy = mu_xy
        self.mu_wave = mu_wave
        
    def psf_convolve(self, x, i_t, offset=None):
        """Convolve x with psf
        this is where ADR is treated as a phase shift in Fourier space.
        
        Parameters
        ----------
        x : 3-d array
            Model, shape = (nw, ny, nx)
        i_t : int
            To get psf and sn_offset for this exposure
        offset : 1-d array
        
        Returns
        -------
        3-d array
        """
                
        psf = self.psf[i_t]
        
        #x = x.reshape(self.gal.shape)
        #ptr = np.zeros(psf.shape)

        """
        #TODO: Formerly shift was self.sn_offset_y[i_t,k] and 
        # self.sn_offset_x[i_t,k]. Need to ask Seb why this is so
        for k in range(self.nw):
             
            phase_shift_apodize = fft_shift_phasor_2d(
                                        [self.ny, self.nx],
                                        [self.sn_offset_y[i_t,k],
                                         self.sn_offset_x[i_t,k]],
                                        half=1, apodize=self.apodizer)
            ptr[k] = fft.fft(psf[k,:,:] * phase_shift_apodize)
        """    
        out = np.zeros(x.shape)
        
        if offset == None:
            for k in range(self.nw):
                out[k,:,:] = fft.ifft2(fft.fft2(psf[k,:,:])*fft.fft2(x[k,:,:]))
            return out

        else:
            phase_shift = fft_shift_phasor_2d([self.ny, self.nx], offset)
            for k in range(self.nw):
                out[k,:,:] = fft.ifft2(fft.fft2(psf[k,:,:]) * 
                                       phase_shift *
                                       fft.fft2(x[k,:,:]))    
                
            return out
            
    def extract_eta_sn_sky(self, data, i_t, galaxy=None, sn_offset=None,
                       galaxy_offset=None, no_eta=None, i_t_is_final_ref=None,
                       update_ddt=None,
                       calculate_variance=None):
        """calculates sn and sky
        calculates the optimal SN and Sky in the chi^2 sense for given
        PSF and galaxy, including a possible offset of the supernova
        
        Maybe this can be compressed?

        Sky is sky per spaxel
        Parameters
        ----------
        ddt : DDT object
        i_t : int
        a bunch of optional stuff
        Returns
        -------
        extract_dict : dict
            Includes new sky and sn
        Notes
        -----
        Updates ddt if 'update_ddt' is True
        """
        
        d = data.data[i_t,:,:,:]
        w = data.weight[i_t,:,:,:]
        
        if not isinstance(sn_offset, np.ndarray):
            sn_offset = np.array([0., 0.])
        if not isinstance(galaxy_offset, np.ndarray):
            galaxy_offset = np.array([0., 0.])
        if galaxy is None:
            galaxy = ddt.model_gal

        galaxy_model = model.psf_convolve(galaxy, i_t, offset=galaxy_offset)
        z_jlt = ddt.r(np.array([galaxy_model]))[0]
        
        # FIXME: eta won't be fitted on multiple final refs
        if i_t_is_final_ref:
            wd = (w*d)
            wz = (w*z_jlt)
            if calculate_variance:
                v = 1/w
                w2d = w*w*v
                sky_var = ((w2d.sum(axis=-1).sum(axis=-1))/
                           ((w*w).sum(axis=-1).sum(axis=-1)))
            else:
                sky_var = np.ones(ddt.nw)
        
            #/* We fix the sky in the final ref used for fitting */
            i_final_ref = (ddt.final_ref if type(ddt.final_ref) == int else
                           ddt.final_ref[0])
            if ( i_t == i_final_ref):
                if 0: #ddt.fit_final_ref_sky:
                    print ("<ddt_extract_eta_sn_sky>: "+
                          "calculating sky on final ref %d" % i_t)
                    sky = ((wd-wz).sum(axis=-1).sum(axis=-1)/
                           w.sum(axis=-1).sum(axis=-1))
                else:
                    print ("<ddt_extract_eta_sn_sky>: "+
                           "using final_ref_sky on exp %d" % i_t)
                    sky = ddt.model_final_ref_sky
                    # FIXME: here the variance is wrong,
                    # since the sky has been estimated on the signal 
                    sky_var = np.ones(ddt.nw)
          
            else:
                print "<ddt_extract_eta_sn_sky>: calculating sky on exp %d" % i_t
                sky = (wd-wz).sum(axis=-1).sum(axis=-1)/w.sum(axis=-1).sum(axis=-1)
                
            sn  = np.zeros(ddt.nw)
            sn_var = np.ones(ddt.nw)
            eta = 1.
            
        else:
            #print "<ddt_extract_eta_sn_sky>: working on SN+Galaxy exposure %d" % i_t
            sn_psf = make_sn_model(np.ones(ddt.nw), ddt, i_t, offset=sn_offset)
            y_jlt = ddt.r(np.array([sn_psf]))[0]
        
            A22 = w
            A12 = w * y_jlt
            A11 = A12 * y_jlt
        
            A11 = A11.sum(axis=-1).sum(axis=-1)
            A12 = -1.*A12.sum(axis=-1).sum(axis=-1)
            A21 = A12
            A22 = A22.sum(axis=-1).sum(axis=-1)
        
            denom = A11*A22 - A12*A21;
            """ There are some cases where we have slices with only 0 values and
            weights. Since we don't mix wavelengthes in this calculation, we put a 
            dummy value for denom and then put the sky and sn values to 0
            * FIXME: The weight of the full slice where that happens is supposed to
                  be 0, therefore the value of sky and sn for this wavelength 
                  should not matter
            """
            i_bad = np.where(denom == 0)
            if isinstance(i_bad, np.ndarray):
                print ("<ddt_extract_eta_sn_sky> WARNING: "+
                       "found null denominator in %d slices \n" % i_bad.size)
                denom[i_bad] = 1.
          
                if sum(w.sum(axis=-1).sum(axis=-1)[i_bad]) != 0.0:
                    raise ValueError("<ddt_extract_eta_sn_sky> ERROR: "+
                                     "found null denom for slices with non null "+
                                     "weight")
          
        
            # w2d, w2dy w2dz are used to calculate the variance using 
            # var(alpha x) = alpha^2 var(x)*/
            wd = (w*d)
            wdy = (wd*y_jlt)
            wdz = (wd*z_jlt)

            wd = wd.sum(axis=-1).sum(axis=-1)
            wdy = wdy.sum(axis=-1).sum(axis=-1)
            wdz = wdz.sum(axis=-1).sum(axis=-1)

            wz = (w*z_jlt)
            wzy = wz*y_jlt
            wzz = (wz*z_jlt)
            wz = wz.sum(axis=-1).sum(axis=-1)
            wzy = wzy.sum(axis=-1).sum(axis=-1)
            wzz = wzz.sum(axis=-1).sum(axis=-1)
        
            b_sky = wd*A11 + wdy*A12
            b_sky /= denom
            c_sky = wz*A11 + wzy*A12
            c_sky /= denom
        
            b_sn = wd*A21 + wdy*A22
            b_sn /= denom
            c_sn = wz*A21 + wzy*A22
            c_sn /= denom
        
            if no_eta:
                sky = b_sky - c_sky
                sn = b_sn - c_sn

                if calculate_variance:
                    v = 1/w
                    w2d = w*w*v    
                    w2dy = w2d *  y_jlt * y_jlt 
                    w2dz = w2d * z_jlt * z_jlt
                    w2d = w2d.sum(axis=-1).sum(axis=-1)
                    w2dy = w2dy.sum(axis=-1).sum(axis=-1)
                    w2dz = w2dz.sum(axis=-1).sum(axis=-1)

                    b2_sky = w2d*A11*A11 + w2dy*A12*A12
                    b2_sky /= denom**2

                    b_sn = w2d*(A21**2) + w2dy*(A22**2)
                    b_sn /= denom**2
                    
                    sky_var = b_sky
                    sn_var = b_sn
                else:
                    sky_var = np.ones(sky.shape)
                    sn_var = np.ones(sn.shape)
              
                if isinstance(i_bad,np.ndarray):
                    print ("<ddt_extract_eta_sn_sky> WARNING: "+
                           "due to null denom, putting some sn and sky values to 0")
                    sky[i_bad] = 0.
                    sn[i_bad]  = 0.
                    sky_var[i_bad] = 0.
                    sn_var[i_bad] = 0.
               
              
                eta = 1.
                eta_denom = 1.
            else:
                eta = wdz - wz*b_sky - wzy*b_sn
                eta_denom = wzz - wz*c_sky - wzy * c_sn
                if isinstance(i_bad,np.ndarray):
                    eta_denom[i_bad] = 1.
          
                eta = eta(sum)/eta_denom.sum()
                sky = b_sky - eta*c_sky
                sn = b_sn - eta*c_sn
                if calculate_variance:
                    print ("WARNING: variance calculation with "+
                           "eta calculation not implemented")
                    sky_var = np.ones(sky.shape)
                    sn_var = np.ones(sn.shape)
                else:
                    sky_var = np.ones(sky.shape)
                    sn_var = np.ones(sn.shape)
          
                if isinstance(i_bad, np.ndarray):
                    print ("<ddt_extract_eta_sn_sky> WARNING: due to null denom,"+
                           "putting some eta, sn and sky values to 0")
                    sky[i_bad] = 0.
                    sn[i_bad] = 0.
                    eta[i_bad] = 0.
                    sky_var[i_bad] = 0.
                    sn_var[i_bad] = 0.
        
        
        if update_ddt:
        
            ddt.model_sn[i_t] = sn
            ddt.model_sky[i_t] = sky
            ddt.model_eta[i_t] = eta
            # TODO: Fill in sn_var and sky_var.

                              
        extract_dict = {'sky': sky, 'sn': sn}
        return extract_dict # What do I want here? So far no other attributes used

