import copy

import numpy as np
import scipy.optimize


        
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
    rgl_err = (model.mu_xy * np.sum(dx**2) +
               model.mu_xy * np.sum(dy**2) +
               model.mu_wave * np.sum(dw**2))
    

    # TODO: lkl_err and rgl_err need to go into output file header:
  
    return rgl_err + lkl_err

def fit_model_all_epoch(model, data, maxiter=1000):
    """fit galaxy, SN and sky and update the model accordingly, keeping
    registration fixed.
    
    Parameters
    ----------
    model : DDTModel
    data : DDTData

    """
    
    penalty = penalty_g_all_epoch
    x = model.gal.reshape((model.gal.size))

    x_new = scipy.optimize.fmin_l_bfgs_b(penalty, x, args=(model, data), 
                                         approx_grad=True) 
    
    model.gal = x_new

