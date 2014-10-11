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
    return ddt.r(np.array([model.psf_convolve(model, i_t)]))[0]


     
  
        
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
    

def make_sn_model(sn, ddt, i_t, offset=None):
    """offsets in spaxels
    """
    if not isinstance(offset, np.ndarray):
        offset = np.array([0.,0.])
    sn_model = np.zeros((ddt.nw,ddt.psf_ny, ddt.psf_nx))
    sn_model[:,ddt.model_sn_y, ddt.model_sn_x] = sn
    
    return model.psf_convolve(sn_model, i_t, offset=offset)
    
    
    
def penalty_g_all_epoch(x, model, data):
    """if i_t is not set, fits all the datacubes at once, else fits the 
        datacube considered
    Parameters
    ----------
    x : 3-d array 
        model of galaxy
    model : DDTModel 
    data : DDTData
    
    Returns
    -------
    penalty : float
    
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
    
    print "Fitting simultaneously %d exposures" % (data.nt)
    # TODO i_fit is an option in DDT (could be only some phases) (was ddt.i_fit)


    model.gal = x.reshape(model.gal.shape)
    # Extracts sn and sky 
    for i_t in range(data.nt):
        if i_t != data.master_final_ref:
            sn_sky = model.update_sn_and_sky(data, i_t)
    
    # calculate residual 
    # ddt_make_all_cube uses ddt.i_fit and only calculates those*/  
    r = np.empty_like(data.data)
    for i_t in range(data.nt):
        xcoords = np.arange(data.nx) - (data.nx - 1) / 2. + model.data_xctr[i_t]
        ycoords = np.arange(data.ny) - (data.ny - 1) / 2. + model.data_yctr[i_t]
        r[i_t] = model.evaluate(i_t, xcoords=xcoords, ycoords=ycoords, 
                                which='all')
    r -= data.data
    
    # Likelihood 
    lkl_err = np.sum(data.weight*r**2)
           
    galdiff = model.gal - model.galprior

    # Regularization
    dw = galdiff[1:, :, :] - galdiff[:-1, :, :]
    dy = galdiff[:, 1:, :] - galdiff[:, :-1, :]
    dx = galdiff[:, :, 1:] - galdiff[:, :, :-1]
    rgl_err = (mu_xy * np.sum(dx**2) +
               mu_xy * np.sum(dy**2) +
               mu_wave * np.sum(dw**2))
    

    # TODO: lkl_err and rgl_err need to go into output file header:
  
    return rgl_err + lkl_err

def fit_model_all_epoch(model, data, maxiter=1000, xmin=None):
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
    x = model.gal.reshape((model.gal.size))

    x_new = scipy.optimize.fmin_l_bfgs_b(penalty, x, args=(model, data), 
                                         approx_grad=True) 
    
    model.gal = x_new

