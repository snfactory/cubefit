import numpy as np

from .fit_toolbox import extract_eta_sn_sky, make_all_cube
from .regul_toolbox import regul_g
import scipy.optimize 
import copy


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
        
    # regularization 
    grd2 = np.zeros(grd.shape)
    rgl_err = regul_g(ddt, x, grd2)
    
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
    
