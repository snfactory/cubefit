import numpy as np
from copy import copy

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
        
    o = np.zero_like(ddt.data)
    m = np.zeros_like(ddt.data)
    
    
    if galaxy is None:
        galaxy = ddt.model_gal
        
    for i_t in range(ddt.nt):
        tmp_galaxy = shift_galaxy(ddt,
                                  [ddt.target_xp[i_t]-
                                   ddt.target_xp[ddt.final_ref[0]],
                                   ddt.target_yp[i_t]-
                                   ddt.target_yp[ddt.final_ref[0]]],
                                  galaxy=galaxy)
        o[i_t,:,:,:] = ddt.r(tmp_galaxy) 
        m[i_t,:,:,:] = make_cube(ddt, i_t, galaxy=galaxy, sn=sn, sky=sky, 
                                 eta=eta)

    resid_dict = {}
    resid_dict['d'] = copy(ddt.data)
    resid_dict['w'] = copy(ddt.weight)
    resid_dict['r'] = d - m
    resid_dict['wr'] = r * w
    resid_dict['lkl'] = wr * r
    resid_dict['m'] = m
    resid_dict['o'] = o
    resid_dict['l'] = copy(ddt.wave)
    
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
    if not isinstance(eta, np.ndarray):
        eta = ddt.model_eta
    
    # Get galaxy*eta + sn (in center) + sky
    model = make_g(galaxy, sn[i_t,:], sky[i_t,:], eta[i_t], ddt.sn_x, ddt.sn_y)
    
    return ddt.r(ddt.H(model, i_t))


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

    n_fit = ddt.i_fit.size
    tmp_H=ddt.H
    for i_n in range(n_fit):
        i_t = ddt.i_fit[i_n]
        model = make_g(galaxy, sn[i_t,:], sky[i_t,:], eta[i_t], 
                       ddt.sn_x, ddt.sn_y) 
        cube[i_t,:,:,:] = ddt.R( tmp_H[i_t](model) )
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
    
    return ddt.H(sn_model, i_t job=2, offset=offset)
    
    
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
    z_jlt = ddt.R(galaxy_model)
    
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
        if ( i_t == ddt.final_ref[0]):
            if ddt.fit_final_ref_sky:
                print "<ddt_extract_eta_sn_sky>: 
                       calculating sky on final ref %d" % i_t
                sky = ((wd-wz).sum(axis=-1).sum(axis=-1)/
                       w.sum(axis=-1).sum(axis=-1))
            else:
                print "<ddt_extract_eta_sn_sky>:
                       using final_ref_sky on exp %d" % i_t
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
        print "<ddt_extract_eta_sn_sky>: working on SN+Galaxy exposure %d" % i_t
        sn_psf = make_sn_model(np.ones(ddt.nw), ddt, i_t, offset=sn_offset)
        y_jlt = ddt.R(sn_psf)
    
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
            print "<ddt_extract_eta_sn_sky> WARNING: 
                    found null denominator in %d slices \n" % i_bad.size
            denom[i_bad] = 1.
      
            if sum(w.sum(axis=-1).sum(axis=-1)[i_bad]) != 0.0:
                raise ValueError("<ddt_extract_eta_sn_sky> ERROR: 
                                  found null denom for slices with non null
                                  weight")
      
    
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
        wz = wz(sum,sum,:)
        wzy = wzy(sum,sum,:)
        wzz = wzz(sum,sum,:)
    
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
                print "<ddt_extract_eta_sn_sky> WARNING: 
                       due to null denom, putting some sn and sky values to 0"
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
                print "WARNING: variance calculation with 
                       eta calculation not implemented"
                sky_var = np.ones(sky.shape)
                sn_var = np.ones(sn.shape)
            else:
                sky_var = np.ones(sky.shape)
                sn_var = np.ones(sn.shape)
      
            if isinstance(i_bad, np.ndarray):
                print "<ddt_extract_eta_sn_sky> WARNING: 
                       due to null denom, putting some eta, 
                       sn and sky values to 0"
                sky[i_bad] = 0.
                sn[i_bad] = 0.
                eta[i_bad] = 0.
                sky_var[i_bad] = 0.
                sn_var[i_bad] = 0.
    
    if update_ddt:
        raise ValueError("<extract_eta_sn_sky>:
                          need to write part that updates ddt")
                          
    extract_dict = {'sky': sky, 'sn': sn}
    return extract_dict # What do I want here? So far no other attributes used
    
  


    


    