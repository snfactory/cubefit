import numpy as np
import scipy.optimize

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


 