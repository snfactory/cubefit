from __future__ import print_function, division

import copy

import numpy as np
import scipy.optimize


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
        m = model.evaluate(i_t, data.xctr[i], data.yctr[i],
                           (data.ny, data.nx), which='all')
        r = data.data[i_t] - m
        wr = data.weight[i_t] * r
        lkl_err += np.sum(wr * r)

        # gradient
        grad += model.gradient_helper(i_t, wr, data.xctr[i], data.yctr[i],
                                      (data.ny, data.nx))
        
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


def fit_model_all_epoch(model, data, maxiter=1000):
    """fit galaxy, SN and sky and update the model accordingly, keeping
    registration fixed.
    
    Parameters
    ----------
    model : DDTModel
    data : DDTData

    """
    
    x0 = model.gal.reshape((model.gal.size))

    #bounds = zip(np.ones(model.gal.size)*10e-6,
    #             np.ones(model.gal.size)*data.data.max())
    x, f, d = scipy.optimize.fmin_l_bfgs_b(penalty_g_all_epoch, x0,
                                           # bounds=bounds,
                                           args=(model, data)) 
    
    model.gal = x.reshape(model.gal.shape)

    print("optimization finished\n"
          "function minimum: {:f}".format(f))
    print("info dict: ", d)


# TODO: should we change this to use a general-purpose optimizer rather 
# than leastsq? Leastsq seems like a strange choice - its a non-linear
# problem.
def fit_position(data, model, i_t, maxiter=100):
    """Fit data position for epoch i_t, keeping galaxy model
    fixed. Doesn't modify model or data.

    Parameters
    ----------
    data : DDTData
    model : DDTModel
    i_t : int
        Epoch number.

    Returns
    -------
    x, y : float, float
        x and y position.
    """

    # Define a function that returns the sqrt(weight) * (data-model)
    # for the given epoch i_t, given the data position.
    # scipy.optimize.leastsq will minimize the sum of the squares of this
    # function's return value, so we're minimizing
    # sum(weight * residual^2), which seems reasonable.
    def objective_func(pos):
        m = model.evaluate(i_t, pos[0], pos[1], (data.ny, data.nx),
                           which='all')
        out = np.sqrt(data.weight[i_t]) * (data.data[i_t] - m)
        return np.ravel(out)

    pos0 = [data.xctr[i_t], data.yctr[i_t]]  # initial position

    pos, info = scipy.optimize.leastsq(objective_func, pos0)
    if info not in [1, 2, 3, 4]:
        raise RuntimeError("optimize.leastsq didn't converge properly")

    return pos[0], pos[1]
