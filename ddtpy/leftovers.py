# This is a temporary file holding leftover functions that were converted
# from Yorick but don't seem to be needed anymore.
#
# TODO: These can be removed after we decide that we're sure we don't
# need them.

# was in data.py: header keys are currently not used but we might want
# to use them instead of values from the config file in the future, in which
# case this dictionary should be an attribute of DataCube.
SELECT_KEYS = ["OBJECT", "RA", "DEC", "EXPTIME", "EFFTIME", "AIRMASS",
               "LATITUDE", "HA", "TEMP", "PRESSURE", "CHANNEL", "PARANG",
               "DDTXP", "DDTYP"]
def read_select_header_keys(filename):
    """Read select header entries from a FITS file.

    Parameters
    ----------
    filename : str
        FITS filename.

    Returns
    -------
    d : dict
        Dictionary containing all select keys. Values are None for keys
        not found in the header.
    """

    f = fitsio.FITS(filename, "r")
    fullheader = f[0].read_header()
    f.close()
    
    return {key: fullheader.get(key, None) for key in SELECT_KEYS}



# was in fitting.py:
def make_offset_cube(ddt,i_t, sn_offset=None, galaxy_offset=None,
                     recalculate=None):
    """
    This fn is only used in _sn_galaxy_registration_model in registration.py
    FIXME: this doesn't include the SN position offset, 
           that is applied at the PSF convolution step
    """
    if galaxy_offset is None:
        galaxy_offset = np.array([0.,0.])
    
    if sn_offset is None:
        sn_offset = np.array([0.,0.])
  
    model = model.psf_convolve(ddt.model_gal, i_t, 
                              offset=galaxy_offset)
    # recalculate the best SN
    if recalculate:
        # Below fn in ddt_fit_toolbox.i
        sn_sky = model.update_sn_and_sky(ddt, i_t, 
                                          galaxy_offset=galaxy_offset, 
                                          sn_offset=sn_offset)
        sn = sn_sky['sn']
        sky = sn_sky['sky']
    else:
        sn = ddt.model_sn[i_t,:]
        sky = ddt.model_sky[i_t,:]
  
    model += make_sn_model(sn, ddt, i_t, offset=sn_offset)

    model += sky[:,None,None]

    return ddt.r(model)


# was in fitting.py:
def make_sn_model(sn, ddt, i_t, offset=None):
    """offsets in spaxels
    """
    if not isinstance(offset, np.ndarray):
        offset = np.array([0.,0.])
    sn_model = np.zeros((ddt.nw,ddt.psf_ny, ddt.psf_nx))
    sn_model[:,ddt.model_sn_y, ddt.model_sn_x] = sn
    
    return model.psf_convolve(sn_model, i_t, offset=offset)


# was in Model class, but was moved to fitting. This version 
# has all the commented-out variance and eta stuff in case we want to
# add it back in.
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



# was in registration.py:
def shift_galaxy(ddt, offset, galaxy=None):
    """Galaxy offset in SPAXELS
    Parameters
    ----------
    ddt : DDT object
    offset : 1-d array
    galaxy : 3d array 
    
    Returns
    -------
    galaxy_fix : 3d array
    """

    if not isinstance(galaxy, np.ndarray):
        galaxy = ddt.model_gal
  
    offset = offset
    dim_gal = galaxy.shape
    print offset, dim_gal
    phase_shift = fft_shift_phasor(dim_gal, offset, half=1)

    galaxy_fix = np.zeros(dim_gal)

    for i_l in range(dim_gal[0]):
        #galaxy_fix[i_l,:,:] = ddt.FFT(ddt.FFT(galaxy[i_l,:,:])*phase_shift , 2)
        galaxy_fix[i_l,:,:] = ddt.FFT(ddt.FFT(galaxy[i_l,:,:])*phase_shift)
        
    
    return galaxy_fix
    

# was in registration.py:    
def sn_galaxy_registration(ddt, i_t, verb=None, maxiter=None, 
                           fit_flag = 'galaxy', mask_sn=0, recalculate=None):
    """          
    Warning: Offsets are in SPAXELS
          
    Parameters
    ----------       
    mask_sn : 
        level at which the SN starts being masked
    recalculate: bool
        set it if you want to recalculate sky and SN each time
    fit_flag : 'sn' or 'galaxy'
        register the sn or galaxy, keep the other fixed
    Returns
    -------
    anew : 1-d array
        fit of sn or galaxy position.
    Notes
    -----
    a[0] offset_x
    a[1] offset_y
    """
    
    a = np.array([0.,0.])
       
    sqrt_weight = (ddt.weight[i_t,:,:,:])**0.5
  
    # TODO: Fix this once op_nllsq is sorted out.
    extra = {'ddt':ddt, 'i_t':i_t, 'data':ddt.data[i_t,:,:,:],
             'sqrt_weight':sqrt_weight, 'model':_sn_galaxy_registration_model,
             'fit_flag': fit_flag, 'recalculate':recalculate, 'mask_sn':mask_sn}
  
    # TODO: FIND op_nllsq (optimpac probably)
    # Translation of this will probably depend on how optimpac wrapper works
    # - Using fmin_cg as temporary placeholder for optimpack function:
    #anew = op_nllsq(_registration_worker, a, extra=extra, verb=verb,
    #                maxstep=maxiter)
    
    anew = scipy.optimize.fmin_cg(_registration_worker, a, args=extra, 
                                  maxiter=maxiter)
    
    if fit_flag == 'sn':
        sn_offset = a
        galaxy_offset = np.array([0.,0.])
    elif fit_flag == 'galaxy':
        galaxy_offset = a 
        sn_offset = np.array([0.,0.])
    else:
        raise ValueError("fit_flag must be sn or galaxy")
    model = make_offset_cube(ddt, i_t, galaxy_offset=galaxy_offset, 
                             sn_offset=sn_offset, recalculate=recalculate)
    return anew


# was in registration.py:
# TODO: once op_llnsq above is sorted out, maybe this can be removed.
def _sn_galaxy_registration_model(a, ddt, i_t, fit_flag='galaxy',
                                  recalculate=None):
    """
    Parameters
    ----------
    Returns
    -------
    Notes
    -----
    a(1) = sn_offset_x
    a(2) = sn_offset_y
    a(3) = galaxy_offset_x
    a(4) = galaxy_offset_y
    """
    sn_offset = np.array([0.,0.])
    galaxy_offset = array([0.,0.])
    if fit_flag == 'sn':
        sn_offset = a
    elif fit_flag == 'galaxy':
        galaxy_offset = a 
    else:
        raise ValueError("<_ddt_sn_galaxy_registration_model> "+
                         "fit_flag must be sn or galaxy")
  

    print  "sn_offset [%s, %s], galaxy_offset [%s, %s] \n" % (sn_offset[0],
                                                              sn_offset[1],
                                                              galaxy_offset[0],
                                                              galaxy_offset[1])
                                                              
    cube_offset = make_offset_cube(ddt, i_t, galaxy_offset=galaxy_offset, 
                                   sn_offset=sn_offset, recalculate=recalculate)
    return cube_offset


# was in registration.py:
def _registration_worker(a, extra):
    """
    Parameters
    ----------
    a : 1d array
    extra : hash object - will be dict?
    
    Returns
    -------
    wr : 3-d array
    """
    if extra['mask_sn']:
        # SN PSF convolved by delta function 
        sn_mask_32 = make_sn_model(np.zeros(extra['ddt'].ddt_model.nw), 
                                   extra['ddt'], extra['i_t'])

        i_low = np.where(sn_mask_32 <= (extra['mask_sn'] * max(sn_mask_32)))
        sn_mask_32 *= 0.;
        sn_mask_32[i_low] = 1.;
        
        # TODO: Fix this h_set:
        #h_set, extra.ddt, sn_mask_32 = sn_mask_32

        sqrt_weight = extra['sqrt_weight']*ddt.r(sn_mask_32)
    else:
        sqrt_weight = extra['sqrt_weight']
  
  
    wr = sqrt_weight*(extra['data'] - extra['model'](
                                            a, extra['ddt'], extra['i_t'], 
                                            fit_flag=extra['fit_flag'], 
                                            recalculate=extra['recalculate']))
  
    return wr


# was in model.py in DDTModel.
class DDTModel:

    # was in def __init__():
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


# was in psf.py:
def roll_psf(psf, dx, dy):
    """Rolls the psf by dx, dy in x and y coordinates.

    The equivalent of the Yorick version with job = 0 is

    dx, dy = (1 - sn_x, 1 - sn_y) [if sn_x = (nx+1)//2]
    or
    dx, dy = (-sn_x, -sn_y) [if sn_x = (nx-1)//2]  # python indexed

    job=1 is 

    Parameters
    ----------
    psf : 4-d array
    dx, dy : int
        Shift in x and y coordinates.

    Returns
    -------
    rolled_psf : 4-d array

    """

    assert psf.ndim == 4  # just checkin' (in Yorick this is also used for 3-d)
                          # but we haven't implemented that here.

    tmp = np.roll(psf, dy, 2)  # roll along y (axis = 2)
    return np.roll(tmp, dx, 3)  # roll along x (axis = 3)


# was in adr.py
def calc_airmass(ha, dec):
  cos_z = (np.sin(SNIFS_LATITUDE) * np.sin(dec) +
           np.cos(SNIFS_LATITUDE) * np.cos(dec) * np.cos(ha))
  return 1./cos_z
