import numpy as np

from .fit_toolbox import extract_eta_sn_sky, make_all_cube
from .regul_toolbox import regul_g

def penalty_g_all_epoch(x, grd, ddt):
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
    
    print "Fitting simultaneously %d exposures:" % (ddt.i_fit.size, ddt.i_fit)
    
    # Extracts sn and sky 
    n_fit = (ddt.i_fit).size
    for i_n in range(n_fit):
        i_t = ddt.i_fit[i_n]
        sn_sky = extract_eta_sn_sky(ddt, i_t, no_eta=ddt.no_eta,
                                        galaxy=x,
                                        i_t_is_final_ref=ddt.is_final_ref[i_t],
                                        update_ddt=TRUE)
  


    # calculate residual 
    # ddt_make_all_cube uses ddt.i_fit and only calculates those*/  
    r = make_all_cube(ddt, galaxy=x, sn=ddt.model_sn, sky=ddt.model_sky,
                          eta=ddt.model_eta)
    r = r[ddt.i_fit] - ddt.data[ddt.i_fit]
    wr = ddt.weight[ddt.i_fit] * r
  
    if ddt.verb:
        print "<ddt_penalty_g>:r %s, wr %s" (np.sum(r), np.sum(wr))
  

    # FIXME: The gradient MUST be reinitialized each time isn't it?
    grd = np.zeros(x.shape)
 
    # Likelihood 
    lkl_err = np.sum(wr*r)
    del r
    n_fit = ddt.i_fit.size
    tmp_H=ddt.H; 
    for i_n in range(n_fit):
        i_t = ddt.i_fit[i_n]
        if(ddt.verb){
            print "<ddt_penalty_g>: calculating gradient for i_t=%d" % i_t
    
        tmp_op = tmp_H[i_t]
        tmp_x = ddt.R( 2.*wr[i_n,:,:,:], 1 )
        grd +=  tmp_op(tmp_x, 1 )
  

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


    