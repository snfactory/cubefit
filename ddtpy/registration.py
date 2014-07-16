import numpy as np


def shift_galaxy(ddt, offset, galaxy=None):
    """Galaxy offset in SPAXELS
    Parameters
    ----------
    ddt : DDT object
    offset : ? Presumably an array of 1-3 dimensions
    galaxy : 3d array 
    
    Returns
    -------
    galaxy_fix : 3d array
    """

    if not isinstance(galaxy, np.ndarray):
        galaxy = ddt.model_gal
  
    offset = offset
    dim_gal = galaxy.shape
    
    # This comes up in other places, will be done later
    phase_shift = fft_shift_phasor(dim_gal, offset, half=1)

    galaxy_fix = np.zeros(dim_gal)

    for i_l in range(dim_gal[0]):
        galaxy_fix[i_l,:,:] = ddt.FFT(ddt.FFT(galaxy(i_l,:,:))*phase_shift , 2)
    
    return galaxy_fix
    
    
def sn_galaxy_registration(ddt, i_t, verb=None, maxiter=None, job=None, 
                           mask_sn=0, recalculate=None):
    """          
    Warning: Offsets are in SPAXELS
          
    Parameters
    ----------       
    mask_sn : 
        level at which the SN starts being masked
    recalculate: 
        set it if you want to recalculate sky and SN each time
    
    Returns
    -------
    Notes
    -----
    job=0: fixed galaxy, registers SN
    job=1: fixed SN, registers galaxy
    job=2: registers both
    a(1) sn_offset_x
    a(2) sn_offet_y
    a(3) galaxy_offset_x
    a(4) galaxy_offset_y
    """
    if job == 0:
        print "<ddt_sn_galaxy_registration> SN registration"
        a = np.array([0.,0.])

    elif job == 1:
        print "<ddt_sn_galaxy_registration> galaxy registration"
        a = np.array([0.,0.])

    elif job == 2:
        print "<ddt_sn_galaxy_registration> galaxy and SN registration"
        a = np.array([0.,0.,0.,0.])
   
    sqrt_weight = (ddt.weight[i_t,:,:,:])**0.5
  
    # TODO: Fix this once op_nllsq is sorted out.
    extra = h_new(ddt=ddt,
                i_t=i_t,
                data=ddt.data[i_t,:,:,:],
                sqrt_weight = sqrt_weight,
                model=_sn_galaxy_registration_model,
                job=job, recalculate=recalculate,
                mask_sn=mask_sn)
  
    # FIND op_nllsq (optimpac probably)
    # Translation of this will probably depend on how optimpac wrapper works
    anew = op_nllsq(_registration_worker, a, extra=extra, verb=verb,
                    maxstep=maxiter)
    if job == 0:
        sn_offset = a
        galaxy_offset = np.array([0.,0.])
    elif job == 1:
        galaxy_offset = a 
        sn_offset = np.array([0.,0.])
    elif job == 2:
        sn_offset = np.array([a[0], a[1]])
        galaxy_offset = np.array([a[2], a[3]])

    model = make_offset_cube(ddt, i_t, galaxy_offset=galaxy_offset, 
                             sn_offset=sn_offset, recalculate=recalculate)
    return anew, a, model


# TODO: once op_llnsq above is sorted out, maybe this can be removed.
def _sn_galaxy_registration_model(a, ddt, i_t, job=None, recalculate=None):
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
    if ! job:
        sn_offset = a
    elif job == 1:
        galaxy_offset = a 
    elif job == 2:
        sn_offset = np.array([a[0], a[1]])
        galaxy_offset = np.array([a[2], a[3]])
    else:
        raise ValueError("<_ddt_sn_galaxy_registration_model> 
                            job %s not implemented" % job)
  

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
    ?
    """
    if extra.mask_sn:
        # SN PSF convolved by delta function 
        sn_mask_32 = make_sn_model(np.zeros(extra.ddt.ddt_model.n_l), 
                                   extra.ddt, extra.i_t)

        i_low = np.where(sn_mask_32 <= (extra.mask_sn * max(sn_mask_32)))
        sn_mask_32 *= 0.;
        sn_mask_32[i_low] = 1.;

        h_set, extra.ddt, sn_mask_32 = sn_mask_32

        sqrt_weight = extra.sqrt_weight*ddt.r(sn_mask_32)
    else:
        sqrt_weight = extra.sqrt_weight
  
  
    wr = sqrt_weight*(extra.data - extra.model(a, extra.ddt, extra.i_t, 
                                               job=extra.job, 
                                               recalculate=extra.recalculate))
  
    return wr


 