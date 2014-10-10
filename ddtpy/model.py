
import numpy as np
from numpy.fft import fft2, ifft2


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
        Hyperparameter in spatial (x, y) coordinates. Used in penalty function
        when fitting model.
    mu_wave : float
        Hyperparameter in wavelength coordinate. Used in penalty function
        when fitting model.
    sky : np.ndarray (2-d)
        Initial guess at sky. Sky is a spatially constant value, so the
        shape is the same as ``shape``.

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

        # Coordinates of model grid
        array_xctr = (nx - 1) / 2.0
        array_yctr = (ny - 1) / 2.0
        self.xcoords = np.arange(nx, dtype=np.float64) - array_xctr
        self.ycoords = np.arange(ny, dtype=np.float64) - array_yctr

        # PSF part of the model
        self.psf = gaussian_plus_moffat_psf_4d(MODEL_SHAPE, array_xctr,
                                               array_yctr,
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
    
    def evaluate(self, i_t, xcoords, ycoords, which='galaxy'):
        """Evalute the model at the given coordinates for a single epoch.

        Parameters
        ----------
        i_t : int
            Epoch index.
        xcoords : np.ndarray (1-d)
        ycoords : np.ndarray (1-d)
        which : {'galaxy', 'snscaled'}
            Which part of the model to evaluate: galaxy-only or SN scaled to
            flux of 1.0?

        Returns
        -------
        x : np.ndarray (3-d)
            Shape is (nw, len(ycoords), len(xcoords)).
        """

        # Currently, by design, the coordinate system and model match
        # the spaxel size of the data. Thus, the data array will have
        # equal spacing with all spacings equal to 1.0. This is a
        # requirement for the Fourier-space shifting used here.
        if not (np.all(np.diff(xcoords) == 1.0) and
                np.all(np.diff(ycoords) == 1.0)):
            raise ValueError("xcoords and y coords must have equal spacing, "
                             "with all spacings equal to 1.0")

        if (xcoords[0] < self.xcoords[0] or xcoords[-1] > self.xcoords[-1] or
            ycoords[0] < self.ycoords[0] or ycoords[-1] > self.ycoords[-1]):
            raise ValueError("requested coordinates out of model bounds")
        
        # Figure out the shift needed to put the model onto the requested
        # coordinates.
        xshift = xcoords[0] - self.xcoords[0]
        yshift = ycoords[0] - self.ycoords[0]
        
        # split shift into integer and sub-integer components
        # This is so that we can first apply a fine-shift in Fourier space
        # and later take a sub-array of the model.
        xshift_int = int(xshift + 0.5)
        xshift_fine = xshift - xshift_int
        yshift_int = int(yshift + 0.5)
        yshift_fine = yshift - yshift_int
        
        shift_phasor = fft_shift_phasor_2d(self.MODEL_SHAPE,
                                           (yshift_fine, xshift_fine))

        # get shifted and convolved galaxy
        psf = self.psf[i_t]
        target_shift_conv = np.empty((self.nw, self.ny, self.nx),
                                     dtype=np.float64)
        for j in range(self.nw):
            if which == 'galaxy':
                target_shift_conv[j, :, :] = ifft2(fft2(psf[j, :, :]) *
                                                   shift_phasor *
                                                   fft2(self.gal[j, :, :]))
            elif which == 'snscaled':
                target_shift_conv[j, :, :] = ifft2(fft2(psf[j, :, :]) *
                                                   shift_phasor)
            elif which == 'all': 
                target_shift_conv[j, :, :] = \
                    ifft2((fft2(self.gal[j, :, :]) + self.sn[j]) *
                          fft2(psf[j, :, :]) * shift_phasor) + self.sky[i_t,j]
        # TODO: add ADR!

        # Return a subarray based on the integer shift
        xslice = slice(xshift_int, xshift_int + len(xcoords))
        yslice = slice(yshift_int, yshift_int + len(ycoords))

        return target_shift_conv[:, yslice, xslice]


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
                out[k,:,:] = fft.ifft2(fft.fft2(psf[k,:,:]) *
                                       fft.fft2(x[k,:,:]))
        else:
            phase_shift = fft_shift_phasor_2d([self.ny, self.nx], offset)
            for k in range(self.nw):
                out[k,:,:] = fft.ifft2(fft.fft2(psf[k,:,:]) * phase_shift *
                                       fft.fft2(x[k,:,:]))
        return out


    # TODO: clean up calc_variance and eta commented-out code.
    def update_sn_and_sky(self, data, i_t):
        """Update the SN level and sky level for a single epoch,
        given the current PSF and galaxy and input data.

        calculates the optimal SN and Sky in the chi^2 sense for given
        PSF and galaxy, including a possible offset of the supernova
        
        Parameters
        ----------
        data : DDTData
            The data.
        i_t : int
            The index of the epoch for which to extract eta, SN and sky
        """
        
        d = data.data[i_t, :, :, :]
        w = data.weight[i_t, :, :, :]

        xcoords = np.arange(data.nx) - (data.nx - 1) / 2. + self.data_xctr[i_t]
        ycoords = np.arange(data.ny) - (data.ny - 1) / 2. + self.data_yctr[i_t]

        gal_conv = self.evaluate(i_t, xcoords, ycoords, which='galaxy')

        if data.is_final_ref[i_t]:

            # sky is just weighted average of data - galaxy model, since 
            # there is no SN in a final ref.
            self.sky[i_t, :] = np.average(d - gal_conv, weights=w, axis=(1, 2))

            #if calc_variance:
            #    sky_var = w.sum(axis=(1, 2)) / (w**2).sum(axis=(1, 2))
            #    sn_var = np.ones(self.nw)

        
        # If the epoch is *not* a final ref, the SN is not zero, so we have
        # to do a lot more work.
        else:
            sn_conv = self.evaluate(i_t, xcoords, ycoords, which='snscaled')

            A11 = (w * sn_conv**2).sum(axis=(1, 2))
            A12 = (-w * sn_conv).sum(axis=(1, 2))
            A21 = A12
            A22 = w.sum(axis=(1, 2))
        
            denom = A11*A22 - A12*A21

            # There are some cases where we have slices with only 0
            # values and weights. Since we don't mix wavelengthes in
            # this calculation, we put a dummy value for denom and
            # then put the sky and sn values to 0 at the end.
            mask = denom == 0.0
            if not np.all(A22[mask] == 0.0):
                raise ValueError("found null denom for slices with non null "
                                 "weight")
            denom[mask] = 1.0
          
            
            # w2d, w2dy w2dz are used to calculate the variance using 
            # var(alpha x) = alpha^2 var(x)*/
            tmp = w * d
            wd = tmp.sum(axis=(1, 2))
            wdsn = (tmp * sn_conv).sum(axis=(1, 2))
            wdgal = (tmp * gal_conv).sum(axis=(1, 2))

            tmp = w * gal_conv
            wgal = tmp.sum(axis=(1, 2))
            wgalsn = (tmp * sn_conv).sum(axis=(1, 2))
            wgal2 = (tmp * gal_conv).sum(axis=(1, 2))
        
            b_sky = (wd * A11 + wdsn * A12) / denom
            c_sky = (wgal * A11 + wgalsn * A12) / denom        
            b_sn = (wd * A21 + wdsn * A22) / denom
            c_sn = (wgal * A21 + wgalsn * A22) / denom

            sky = b_sky - c_sky
            sn = b_sn - c_sn
            
            sky[mask] = 0.0
            sn[mask] = 0.0

            self.sky[i_t, :] = sky
            self.sn[i_t, :] = sn

            # if calc_variance:
            #     v = 1/w
            #     w2d = w*w*v    
            #     w2dy = w2d *  sn_conv * sn_conv 
            #     w2dz = w2d * gal_conv * gal_conv
            #     w2d = w2d.sum(axis=(1, 2))
            #     w2dy = w2dy.sum(axis=(1, 2))
            #     w2dz = w2dz.sum(axis=(1, 2))
            #
            #     b2_sky = w2d*A11*A11 + w2dy*A12*A12
            #     b2_sky /= denom**2
            # 
            #     b_sn = w2d*(A21**2) + w2dy*(A22**2)
            #     b_sn /= denom**2
            # 
            #     sky_var = b_sky
            #     sn_var = b_sn


            # If no_eta = True (*do* calculate eta):
            # ======================================
            #
            # eta = wdgal - wz*b_sky - wzy*b_sn
            # eta_denom = wzz - wz*c_sky - wzy * c_sn
            # if isinstance(i_bad,np.ndarray):
            #     eta_denom[i_bad] = 1.
            # 
            # eta = eta(sum)/eta_denom.sum()
            # sky = b_sky - eta*c_sky
            # sn = b_sn - eta*c_sn
            # if calc_variance:
            #     print ("WARNING: variance calculation with "+
            #            "eta calculation not implemented")
            #     sky_var = np.ones(sky.shape)
            #     sn_var = np.ones(sn.shape)
            # else:
            #     sky_var = np.ones(sky.shape)
            #     sn_var = np.ones(sn.shape)
            # 
            # if isinstance(i_bad, np.ndarray):
            #     print ("<ddt_extract_eta_sn_sky> WARNING: due to null denom,"+
            #            "putting some eta, sn and sky values to 0")
            #     sky[i_bad] = 0.
            #     sn[i_bad] = 0.
            #     eta[i_bad] = 0.
            #     sky_var[i_bad] = 0.
            #     sn_var[i_bad] = 0.
