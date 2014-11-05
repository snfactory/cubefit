
import numpy as np
from numpy.fft import fft2, ifft2

from .psf import gaussian_plus_moffat_psf_4d
from .utils import fft_shift_phasor_2d

__all__ = ["DDTModel"]

class DDTModel(object):
    """This class is the equivalent of everything else that isn't data
    in the Yorick version.

    Parameters
    ----------
    nt : 2-tuple of int
        Model dimension in time.
    wave : np.ndarray
        One-dimensional array of wavelengths in angstroms. Should match
        data.
    psf_ellipticity, psf_alpha : np.ndarray (2-d)
        Parameters characterizing the PSF at each time, wavelength. Shape
        of both must be (nt, len(wave)).
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
    skyguess : np.ndarray (2-d)
        Initial guess at sky. Sky is a spatially constant value, so the
        shape is (nt, len(wave)).

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

    def __init__(self, nt, wave, psf_ellipticity, psf_alpha, adr_dx, adr_dy,
                 mu_xy, mu_wave, spaxel_size, skyguess):

        ny, nx = self.MODEL_SHAPE
        nw, = wave.shape

        # Model shape
        self.nt = nt
        self.nw = nw
        self.ny = ny
        self.nx = nx

        # consistency checks on inputs
        s = (nt, nw)
        if not (psf_ellipticity.shape == s):
            raise ValueError("psf_ellipticity has wrong shape")
        if not (psf_alpha.shape == s):
            raise ValueError("psf_alpha has wrong shape")
        if not (adr_dx.shape == adr_dy.shape == s):
            raise ValueError("adr_dx and adr_dy shape must be (nt, nw)")

        # Coordinates of model grid
        array_xctr = (nx - 1) / 2.0
        array_yctr = (ny - 1) / 2.0
        self.xcoords = np.arange(nx, dtype=np.float64) - array_xctr
        self.ycoords = np.arange(ny, dtype=np.float64) - array_yctr

        # wavelength grid of the model (fixed; should be same as data)
        self.wave = wave

        # ADR part of the model (fixed)
        self.adr_dx = adr_dx
        self.adr_dy = adr_dy

        # PSF part of the model (fixed)
        self.psf = gaussian_plus_moffat_psf_4d(self.MODEL_SHAPE,
                                               array_xctr, array_yctr,
                                               psf_ellipticity, psf_alpha)

        # hyperparameters and spaxel size
        self.mu_xy = mu_xy
        self.mu_wave = mu_wave
        self.spaxel_size = spaxel_size

        # Galaxy and sky part of the model
        self.gal = np.zeros((nw, ny, nx))
        self.galprior = np.zeros((nw, ny, nx))
        self.sky = skyguess
        self.sn = np.zeros((nt, nw))
        self.eta = np.ones(nt)  # eta is transmission
        self.final_ref_sky = np.zeros(nw)

        # This moves the center of the PSF from array coordinates
        # (model_sn_x, model_sn_y) -> (0, 0) [lower left pixel]
        # We suppose this is done because it is needed for convolution.
  
        # self.psf = roll_psf(self.psf, -self.model_sn_x, -self.model_sn_y)
        
        # Make up a coordinate system for the model array
        #offx = int((nx-1) / 2.)
        #offy = int((ny-1) / 2.)
        #xcoords = np.arange(-offx, nx - offx)  # x coordinates on array
        #ycoords = np.arange(-offy, ny - offy)  # y coordinates on array

        # sn is "by definition" at array position where coordinates = (0,0)
        # model_sn_x = offx
        # model_sn_y = offy
        

    def evaluate(self, i_t, xctr, yctr, shape, which='galaxy'):
        """Evalute the model on a grid for a single epoch.

        Parameters
        ----------
        i_t : int
            Epoch index.
        shape : tuple
            Two integers giving size of output array (ny, nx).
        xctr, yctr : float
            Center of output array in model coordinates.
        which : {'galaxy', 'snscaled', 'all'}
            Which part of the model to evaluate: galaxy-only or SN scaled to
            flux of 1.0?

        Returns
        -------
        x : np.ndarray (3-d)
            Shape is (nw, len(ycoords), len(xcoords)).
        """

        # min and max coordinates requested (inclusive)
        ny, nx = shape
        xmin = xctr - (nx - 1) / 2.
        xmax = xctr + (nx - 1) / 2.
        ymin = yctr - (ny - 1) / 2.
        ymax = yctr + (ny - 1) / 2.

        if (xmin < self.xcoords[0] or xmax > self.xcoords[-1] or
            ymin < self.ycoords[0] or ymax > self.ycoords[-1]):
            raise ValueError("requested coordinates out of model bounds")
        
        # Figure out the shift needed to put the model onto the requested
        # coordinates.
        xshift = xmin - self.xcoords[0]
        yshift = ymax - self.ycoords[0]
        
        # split shift into integer and sub-integer components
        # This is so that we can first apply a fine-shift in Fourier space
        # and later take a sub-array of the model.
        xshift_int = int(xshift + 0.5)
        xshift_fine = xshift - xshift_int
        yshift_int = int(yshift + 0.5)
        yshift_fine = yshift - yshift_int

        # get shifted and convolved galaxy
        psf = self.psf[i_t]
        target_shift_conv = np.empty((self.nw, self.ny, self.nx),
                                     dtype=np.float64)
                                     
        for j in range(self.nw):
            shift_phasor = fft_shift_phasor_2d(self.MODEL_SHAPE,
                                           (yshift_fine + self.adr_dy[i_t,j],
                                            xshift_fine + self.adr_dx[i_t,j]))

            if which == 'galaxy':
                target_shift_conv[j, :, :] = ifft2(fft2(psf[j, :, :]) *
                                                   shift_phasor *
                                                   fft2(self.gal[j, :, :]))
            elif which == 'snscaled':
                target_shift_conv[j, :, :] = ifft2(fft2(psf[j, :, :]) *
                                                   shift_phasor)
            elif which == 'all': 
                target_shift_conv[j, :, :] = \
                    ifft2((fft2(self.gal[j, :, :]) + self.sn[i_t,j]) *
                          fft2(psf[j, :, :]) * shift_phasor) + self.sky[i_t,j]

        # Return a subarray based on the integer shift
        xslice = slice(xshift_int, xshift_int + len(xcoords))
        yslice = slice(yshift_int, yshift_int + len(ycoords))

        return target_shift_conv[:, yslice, xslice]

    def gradient_helper(self, i_t, x, xctr, yctr, shape):
        """Not sure exactly what this does yet.

        Parameters
        ----------
        i_t : int
            Epoch index.
        x : np.ndarray (3-d)
            Same shape as *data* for single epoch (nw, ny, nx).
        xcoords : np.ndarray (1-d)
        ycoords : np.ndarray (1-d)

        Returns
        -------
        x : np.ndarray (3-d)
            Shape is (nw, len(ycoords), len(xcoords)).
        """

        # min and max coordinates requested (inclusive)
        ny, nx = shape
        xmin = xctr - (nx - 1) / 2.
        xmax = xctr + (nx - 1) / 2.
        ymin = yctr - (ny - 1) / 2.
        ymax = yctr + (ny - 1) / 2.

        if (xmin < self.xcoords[0] or xmax > self.xcoords[-1] or
            ymin < self.ycoords[0] or ymax > self.ycoords[-1]):
            raise ValueError("requested coordinates out of model bounds")
        
        # Figure out the shift needed to put the model onto the requested
        # coordinates.
        xshift = xmin - self.xcoords[0]
        yshift = ymax - self.ycoords[0]
        
        # create 
        target = np.zeros((self.nw, self.ny, self.nx), dtype=np.float64)
        target[:, :x.shape[1], :x.shape[2]] = x
        
        psf = self.psf[i_t]

        for j in range(self.nw):
            shift_phasor = fft_shift_phasor_2d(self.MODEL_SHAPE,
                                               (yshift + self.adr_dy[i_t,j],
                                                xshift + self.adr_dx[i_t,j]))
            
            target[j, :, :] = ifft2(np.conj(fft2(psf[j, :, :]) *
                                            shift_phasor) *
                                    fft2(target[j, :, :]))

        return target



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

        gal_conv = self.evaluate(i_t, data.xctr[i_t], data.yctr[i_t],
                                 (data.ny, data.nx), which='galaxy')

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
            sn_conv = self.evaluate(i_t, data.xctr[i_t], data.yctr[i_t],
                                    (data.ny, data.nx), which='snscaled')

            A11 = (w * sn_conv**2).sum(axis=(1, 2))
            A12 = (-w * sn_conv).sum(axis=(1, 2))
            A21 = A12
            A22 = w.sum(axis=(1, 2))
        
            denom = A11*A22 - A12*A21

            # There are some cases where we have slices with only 0
            # values and weights. Since we don't mix wavelengths in
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
