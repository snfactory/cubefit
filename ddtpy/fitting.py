from __future__ import print_function, division

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
    """computes likelihood and regularization penalty for a given galaxy model
    
    Parameters
    ----------
    x : np.ndarray
        1-d array, flattened 3-d galaxy model
    model : DDTModel 
    data : DDTData
    
    Returns
    -------
    penalty : float
    gradient : np.ndarray
        1-d array of length x.size giving the gradient on penalty.

    Notes
    -----
    Used in op_mnb (DDT/OptimPack-1.3.2/yorick/OptimPack1.i)
    * Compute likelihood term and gradient on NORMALIZED x
    *       1. compute 4-D model: g(x)
    *       2. apply convolution: H.g(x)
    *       3. apply resampling: R.H.g(x)
    *       4. compute residuals and penalty
    *       5. compute gradient by transposing steps 3, 2, and 1
    """

    model.gal = x.reshape(model.gal.shape)

    lkl_penalty, lkl_grad = likelihood_penalty(model, data)
    rgl_penalty, rgl_grad = regularization_penalty(model, data)

    tot_penalty = lkl_penalty + rgl_penalty
    tot_grad = lkl_grad + rgl_grad
    
    return tot_penalty, tot_grad

def likelihood_penalty(model, data):
    """computes likelihood and likelihood gradient for galaxy model
    
    Parameters
    ----------
    x : np.ndarray
        1-d array, flattened 3-d galaxy model
    model : DDTModel 
    data : DDTData
    
    Returns
    -------
    penalty : float
    gradient : np.ndarray
        1-d array of length x.size giving the gradient on penalty.
    """

    # Extracts sn and sky 
    for i_t in range(data.nt):
        if i_t != data.master_final_ref:
            sn_sky = model.update_sn_and_sky(data, i_t)
    
    # calculate residual 
    # ddt_make_all_cube uses ddt.i_fit and only calculates those*/  

    lkl_err = 0.0
    grad = np.empty_like(model.gal)
    for i_t in range(data.nt):
        xcoords = np.arange(data.nx) - (data.nx-1)/2. + model.data_xctr[i_t]
        ycoords = np.arange(data.ny) - (data.ny-1)/2. + model.data_yctr[i_t]
        m = model.evaluate(i_t, xcoords=xcoords, ycoords=ycoords, 
                           which='all')
        r = data.data[i_t] - m
        wr = data.weight[i_t] * r
        lkl_err += np.sum(wr * r)

        # gradient
        grad += model.gradient_helper(i_t, wr, xcoords, ycoords)
        
    return lkl_err, grad.reshape(model.gal.size)


def regularization_penalty(model, data):
    """computes regularization penalty and gradient for a given galaxy model
    
    Parameters
    ----------
    x : np.ndarray
        1-d array, flattened 3-d galaxy model
    model : DDTModel 
    data : DDTData
    
    Returns
    -------
    penalty : float
    gradient : np.ndarray
        1-d array of length x.size giving the gradient on penalty.
    """

    galdiff = model.gal - model.galprior
    galdiff /= data.data_avg[:, None, None]
    dw = galdiff[1:, :, :] - galdiff[:-1, :, :]
    dy = galdiff[:, 1:, :] - galdiff[:, :-1, :]
    dx = galdiff[:, :, 1:] - galdiff[:, :, :-1]

    # Regularlization penalty term
    rgl_err = (model.mu_xy * np.sum(dx**2) +
               model.mu_xy * np.sum(dy**2) +
               model.mu_wave * np.sum(dw**2))
    
    # Gradient in regularization penalty term
    #
    # This is clearer when the loops are explicitly written out.
    # For a loop that goes over all adjacent elements in a given dimension,
    # one would do (pseudocode):
    # for i in ...:
    #     d = arr[i+1] - arr[i]
    #     penalty += hyper * d^2
    #     gradient[i+1] += 2 * hyper * d
    #     gradient[i]   -= 2 * hyper * d

    rgl_grad = np.zeros(galdiff.shape, dtype=np.float64)
    rgl_grad[:, :, 1:] += 2. * model.mu_xy * dx
    rgl_grad[:, :, :-1] -= 2. * model.mu_xy * dx
    rgl_grad[:, 1:, :] += 2. * model.mu_xy * dy
    rgl_grad[:, :-1,:] -= 2. * model.mu_xy * dy
    rgl_grad[1:, :, :] += 2. * model.mu_wave * dw
    rgl_grad[:-1, :, :] -= 2. * model.mu_wave * dw
    
    return rgl_err, rgl_grad.reshape(model.gal.size)

iteration = 0
def callback(x):
    global iteration
    print(iteration)
    iteration += 1


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

    #bounds = zip(np.ones(model.gal.size)*10e-6,
    #             np.ones(model.gal.size)*data.data.max())
    x, f, d = scipy.optimize.fmin_l_bfgs_b(penalty, x, args=(model, data), 
                                           approx_grad=False,# bounds = bounds,
                                           iprint=0, callback=callback) 
    
    model.gal = x.reshape(model.gal.shape)

    print("optimization finished\n"
          "function minimum: {:f}".format(f))
    print("info dict: ", d)

