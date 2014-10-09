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
        cube[i_t,:,:,:] = ddt.r(np.array([model.psf_convolve(model, i_t)]))[0]
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
    
def make_model_cube(model, data, galaxy=None, sn=None, sky=None, eta=None):
    """Makes a cube out of model parts, returns part of model that overlaps
    with the data. Replaces make_all_cube/make_cube/make_g, provides only part 
    needed in calc_residual.
    
    Parameters
    ----------
    model : DDTModel
    galaxy : 3d array
    sn : 1d array
    sky : 1d array
    eta : 1d array or float?
    
    Returns
    -------
    4-d array 
        A cube for each exposure
    """
    if not isinstance(galaxy, np.ndarray):
        galaxy = model.gal
    if not isinstance(sn, np.ndarray):
        sn = model.sn
    if not isinstance(sky, np.ndarray):
        sky = model.sky
    if not isinstance(eta, np.ndarray):
        eta = model.eta

    full_cube = np.zeros_like(data.data)
    
    i_fit = np.arange(data.nt)
    for i_t in i_fit:
        model_i_t = galaxy*eta[i_t] + sky[i_t,:,None,None]
        model_i_t[:,model.model_sn_y, model.model_sn_x] += sn[i_t]
        full_cube[i_t,:,:,:] = model.psf_convolve(model_i_t, i_t)
        
    cube = main.r(full_cube)
    return cube

    
        
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
    i_fit = np.arange(data.nt)
    # Extracts sn and sky 
    for i_t in i_fit:
        sn_sky = extract_eta_sn_sky(data, i_t, no_eta=True,
                                    galaxy=x,
                                    i_t_is_final_ref=data.is_final_ref[i_t],
                                    update_ddt=True)
    
    # calculate residual 
    # ddt_make_all_cube uses ddt.i_fit and only calculates those*/  
    x = x.reshape(model.gal.shape)
    r = make_model_cube(model, data, galaxy=x, sn=model.sn, sky=model.sky,
                        eta=model.eta)
    r = r[i_fit] - data.data[i_fit]
    wr = data.weight[i_fit] * r
    
    if model.verb:
        print "<ddt_penalty_g>:r %s, wr %s" % (np.sum(r), np.sum(wr))
  
    # Comment from Yorick DDT :
    # FIXME: The gradient MUST be reinitialized each time isn't it?
    grd = np.zeros(x.shape)
 
    # Likelihood 
    lkl_err = np.sum(wr*r)
    
    for i_t in i_fit:
        #if ddt.verb:
        #    print "<ddt_penalty_g>: calculating gradient for i_t=%d" % i_t
        tmp_x = main.r_inv(np.array([2.*wr[i_n,:,:,:]]))[0]
        grd += model.psf_convolve(tmp_x, i_t)
  
    wr=[]
    tmp_x=[]
        
    galdiff = x - model.galprior

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
  
    grd += grd2

    return rgl_err + lkl_err

def fit_model_all_epoch(model, data, maxiter=None, xmin=None):
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
    x = copy.copy(model.gal)
    
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
        x_new = scipy.optimize.fmin_l_bfgs_b(penalty, x, args=(model, data), 
                                             approx_grad=True) 
    
    model.gal = x_new
    sn_sky = extract_eta_sn_sky_all(data, update_ddt=True, no_eta=True)
