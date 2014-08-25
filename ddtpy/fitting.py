import copy

import numpy as np
import scipy.optimize

from .registration import shift_galaxy


def calc_residual(ddt, galaxy=None, sn=None, sky=None, eta=None):
    """Returns residual?
    So far only 'm' part of this is being used.
    Parameters
    ----------
    ddt : DDT object
    galaxy : 
    sn : 
    sky : 
    eta :
    
    Returns
    -------
    resid_dict : dict
        Dictionary including residual, cube, data, other stuff
    """
        
    o = np.zeros_like(ddt.data)
    m = np.zeros_like(ddt.data)
    
    
    if galaxy is None:
        galaxy = ddt.model_gal
        
    for i_t in range(ddt.nt):
        # TODO: In DDT, seems like final_ref is list, in example file, it's int
        i_final_ref = (ddt.final_ref if type(ddt.final_ref) == int 
                       else ddt.final_ref[0])
        tmp_galaxy = shift_galaxy(ddt,
                                  [ddt.target_xp[i_t]-
                                   ddt.target_xp[i_final_ref],
                                   ddt.target_yp[i_t]-
                                   ddt.target_yp[i_final_ref]],
                                  galaxy=galaxy)
        # TODO: Fix this (Hack because r requires 4-d array):
        o[i_t,:,:,:] = ddt.r(np.array([tmp_galaxy]))[0] 
        m[i_t,:,:,:] = make_cube(ddt, i_t, galaxy=galaxy, sn=sn, sky=sky, 
                                 eta=eta)

    resid_dict = {}
    resid_dict['d'] = copy.copy(ddt.data)
    resid_dict['w'] = copy.copy(ddt.weight)
    resid_dict['r'] = ddt.data - m
    resid_dict['wr'] = resid_dict['r'] * ddt.weight
    resid_dict['lkl'] = resid_dict['wr'] * resid_dict['r']
    resid_dict['m'] = m
    resid_dict['o'] = o
    resid_dict['l'] = copy.copy(ddt.wave)
    
    if sky is None:
        resid_dict['s'] = ddt.model_sky
        
    return resid_dict
    
def make_cube(ddt, i_t, galaxy=None, sn=None, sky=None, eta=None, 
              galaxy_offset=None):
              
    """
    Parameters
    ----------
    ddt : DDT object
    i_t : int
    galaxy : 
    sn : 
    sky : 
    eta :
    
    Returns
    -------
    ddt.R(...) : 3d array
        whatever ddt.R(model) returns
    
    """
    
    if not isinstance(galaxy, np.ndarray):
        galaxy = ddt.model_gal
    if galaxy_offset is not None:
        galaxy = shift_galaxy(ddt, galaxy_offset, galaxy=galaxy)
    if not isinstance(sn, np.ndarray):
        sn = ddt.model_sn
    if not isinstance(sky, np.ndarray):
        sky = ddt.model_sky
    if not isinstance(eta, np.ndarray):
        eta = ddt.model_eta
    
    # Get galaxy*eta + sn (in center) + sky
    model = make_g(galaxy, sn[i_t,:], sky[i_t,:], eta[i_t], ddt.model_sn_x,
                   ddt.model_sn_y)
    
    # TODO: Hack below to use ddt.r on 3-d array
    return ddt.r(np.array([ddt.H(model, i_t)]))[0]


def make_all_cube(ddt, galaxy=None, sn=None, sky=None, eta=None):
    """
    Parameters
    ----------
    galaxy : 3d array
    sn, sky : 2d arrays
    eta : 1d array
    Returns
    -------
    cube : 4d array
    * FIXME: make a similar script that does G(psf),
    *  so that we can make sure that they give the same result
    """
    if not isinstance(galaxy, np.ndarray):
        galaxy = ddt.model_gal
  
    if not isinstance(sn, np.ndarray):
        sn = ddt.model_sn
  
    if not isinstance(sky, np.ndarray):
        sky = ddt.model_sky
  
    if not isinstance(eta, np.ndarray):
        eta = ddt.model_eta

    cube = np.zeros_like(ddt.data)
    # TODO i_fit is an option in DDT (could be only some phases) (was ddt.i_fit)
    i_fit = np.arange(ddt.nt)
    n_fit = i_fit.size
    for i_n in range(n_fit):
        i_t = i_fit[i_n]
        model = make_g(galaxy, sn[i_t,:], sky[i_t,:], eta[i_t], 
                       ddt.model_sn_x, ddt.model_sn_y) 
        cube[i_t,:,:,:] = ddt.r(np.array([ddt.H(model, i_t)]))[0]
        del model
    
    return cube
        
        
def make_g(galaxy, sn, sky, eta, sn_x, sn_y):
    """Makes a 3d model from a 3d galaxy, 1d sn, 1d, sky
    
    Parameters
    ----------
    galaxy : 3d array
    sn : 1d array
    sky : 1d array
    sn_x, sn_y : int
    
    Returns
    -------
    model : 3d array
    """
    model = galaxy * eta 
    model[:,sn_y, sn_x] += sn
    model += sky[:,None, None] 
    
    return model
    
        
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
  
    model = make_galaxy_model(ddt.model_gal, ddt, i_t, 
                              offset=galaxy_offset)
    # recalculate the best SN
    if recalculate:
        # Below fn in ddt_fit_toolbox.i
        sn_sky = extract_eta_sn_sky(ddt, i_t, no_eta=TRUE, 
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
    
def make_galaxy_model(galaxy, ddt, i_t, offset=None):
    """offsets in spaxels
    Notes
    -----
    This fn is probably unnecessary since it is now basically only one line.
    """
    if not isinstance(offset, np.ndarray):
        offset = np.array([0., 0.,])

    return ddt.H(galaxy, i_t, 2, offset=offset)

def make_sn_model(sn, ddt, i_t, offset=None):
    """offsets in spaxels
    """
    if not isinstance(offset, np.ndarray):
        offset = np.array([0.,0.])
    sn_model = np.zeros((ddt.nw,ddt.psf_ny, ddt.psf_nx))
    sn_model[:,ddt.model_sn_y, ddt.model_sn_x] = sn
    
    return ddt.H(sn_model, i_t, offset=offset)
    
    
def extract_eta_sn_sky(ddt, i_t, galaxy=None, sn_offset=None,
                       galaxy_offset=None, no_eta=None, i_t_is_final_ref=None,
                       update_ddt=None, ddt_data=None,
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
    
    if ddt_data is None:
        d = ddt.data[i_t,:,:,:]
        w = ddt.weight[i_t,:,:,:]
    else:
        print "<extract_eta_sn_sky> Need to add weight keyword then"
        d = ddt_data[i_t,:,:,:]
        #w = ddt_data.weight[i_t,:,:,:]  
    if not isinstance(sn_offset, np.ndarray):
        sn_offset = np.array([0., 0.])
    if not isinstance(galaxy_offset, np.ndarray):
        galaxy_offset = np.array([0., 0.])
    if not isinstance(galaxy, np.ndarray):
        galaxy = ddt.model_gal

    galaxy_model = make_galaxy_model(galaxy, ddt, i_t, offset=galaxy_offset)
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
    
def penalty_g_all_epoch(x, ddt):
    """if i_t is not set, fits all the datacubes at once, else fits the 
        datacube considered
        
    Notes
    -----
    This function is only called in fit_model_all_epoch
    Used in op_mnb (DDT/OptimPack-1.3.2/yorick/OptimPack1.i)
    * Compute likelihood term and gradient on NORMALIZED x
    *       1. compute 4-D model: g(x)
    *       2. apply convolution: H.g(x)
    *       3. apply resampling: R.H.g(x)
    *       4. compute residuals and penalty
    *       5. compute gradient by transposing steps 3, 2, and 1
    """
    
    print "Fitting simultaneously %d exposures" % (ddt.nt)
    # TODO i_fit is an option in DDT (could be only some phases) (was ddt.i_fit)
    i_fit = np.arange(ddt.nt)
    # Extracts sn and sky 
    n_fit = (i_fit).size
    print np.where(x != 0)
    for i_n in range(n_fit):
        i_t = i_fit[i_n]
        sn_sky = extract_eta_sn_sky(ddt, i_t, no_eta=True,
                                    galaxy=x,
                                    i_t_is_final_ref=ddt.is_final_ref[i_t],
                                    update_ddt=True)
    
    # calculate residual 
    # ddt_make_all_cube uses ddt.i_fit and only calculates those*/  
    x = x.reshape(ddt.model_gal.shape)
    r = make_all_cube(ddt, galaxy=x, sn=ddt.model_sn, sky=ddt.model_sky,
                          eta=ddt.model_eta)
    r = r[i_fit] - ddt.data[i_fit]
    wr = ddt.weight[i_fit] * r
    
    if ddt.verb:
        print "<ddt_penalty_g>:r %s, wr %s" % (np.sum(r), np.sum(wr))
  
    # Comment from Yorick DDT :
    # FIXME: The gradient MUST be reinitialized each time isn't it?
    grd = np.zeros(x.shape)
 
    # Likelihood 
    lkl_err = np.sum(wr*r)
    n_fit = i_fit.size
    for i_n in range(n_fit):
        i_t = i_fit[i_n]
        #if ddt.verb:
        #    print "<ddt_penalty_g>: calculating gradient for i_t=%d" % i_t
        tmp_x = ddt.r_inv(np.array([2.*wr[i_n,:,:,:]]))[0]
        grd += ddt.H(tmp_x, i_t)
  
    wr=[]
    tmp_x=[]
        
    galdiff = x - ddt.model_galprior

    # Regularization
    dw = galdiff[1:, :, :] - galdiff[:-1, :, :]
    dy = galdiff[:, 1:, :] - galdiff[:, :-1, :]
    dx = galdiff[:, :, 1:] - galdiff[:, :, :-1]
    rgl_err = (mu_xy * np.sum(dx**2) +
               mu_xy * np.sum(dy**2) +
               mu_wave * np.sum(dw**2))

    
    # TODO: These prob go into header if debug=1:
    #h_set, ddt.ddt_model, grd_rgl = grd2
    #h_set, ddt.ddt_model, grd_lkl = grd
    # TODO: These need to go into output file header:
    #h_set, ddt.ddt_model, lkl = lkl_err   
    #h_set, ddt.ddt_model, rgl = rgl_err
  
    grd += grd2;

    if ddt.verb:
        print "<ddt_penalty_g>: lkl %s, rgl %s \n" % (lkl_err, rgl_err)
  
    return rgl_err + lkl_err

def fit_model_all_epoch(ddt, maxiter=None, xmin=None):
    """fits galaxy (and thus extracts sn and sky)
    
    Parameters
    ----------
    ddt : DDT object
    
    Returns
    -------
    Nothing
    
    Notes
    -----
    Updates DDT object
    Assumes no_eta = True (seems to always be)
    """
    
    penalty = penalty_g_all_epoch
    x = copy.copy(ddt.model_gal)
    
    # TODO : This obviously needs to be fixed:
    #method = (OP_FLAG_UPDATE_WITH_GP |
    #          OP_FLAG_SHANNO_PHUA |
    #          OP_FLAG_MORE_THUENTE);
    mem = 3   

    if maxiter:
        print "<fit_model_all_epoch> starting the fit"
        # TODO: Placeholder in now for op_mnb
        #x_new = op_mnb(penalty, x, extra=ddt, xmin=xmin, maxiter=maxiter,
        #               method=method, mem=mem,
        #               verb=ddt.verb, ftol=ftol)
        x_new = scipy.optimize.fmin_cg(penalty, x, args=(ddt,)) 
    
    ddt.model_gal = x_new
    sn_sky = extract_eta_sn_sky_all(ddt, update_ddt=True, no_eta=True)
    
