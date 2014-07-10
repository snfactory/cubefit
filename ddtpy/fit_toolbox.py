import numpy as np
from copy import copy

from .registration import shift_galaxy
def calc_residual(ddt, galaxy=None, sn=None, sky=None, eta=None):
    """Returns residual?
    
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
    data_shape = ddt.data.shape
    
    o = np.zeros(data_shape)
    m = np.zeros(data_shape)
    
    
    if galaxy is None:
        galaxy = ddt.model_gal
        
    for i_t in range(ddt.nt):
        tmp_galaxy = shift_galaxy(ddt,
                                  [ddt.target_xp[i_t]-
                                   ddt.target_xp[ddt.final_ref[0]],
                                   ddt.target_yp[i_t]-
                                   ddt.target_yp[ddt.final_ref[0]]],
                                  galaxy=galaxy)
        o[i_t,:,:,:] = ddt.R(tmp_galaxy)
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
              galaxy_offset=None, H=None):
              
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
    
    if H is None:
        return ddt.R(ddt.H(ddt, i_t)(model)) #Sort this out
    else:
        return ddt.R((H(model)))
        
        
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
    

def get_H(ddt, i_t):
    """Function from ddt_setup.i, not sure where it should go.
    """
    ## i_t == 0 is okay in python version:
    #if i_t == 0:
    #    print "<ddt_get_H> WARNING: i_t == 0"
    #    i_t = ddt.nt
    return ddt.H[i_t]
    
def setup_H(ddt):
    """Also from ddt_setup.i
    """
    
    H = [ddt_H(ddt,1)] # Should 1 be zero here?
    for i_t in range(1, ddt.nt):
        H.append(ddt_H(ddt, i_t))
    
    return H

#### All functions below here need to be merged with Kyle's G, R work ####    
def ddt_H(ddt, i_t, psf=None):
    """Operator H corresponding to time i_t
    
    Parameters
    ----------
    ddt : DDT object
    i_t = int
    psf = ?
    
    Returns
    -------
    a function
    """
    
    print "<ddt_H> calculating H for i_t=%d" % i_t
    if not ddt.psf_rolled: # Whatever this is
        raise ValueError("<ddt_H> need the psf to be rolled!")
    if psf is None:
        psf = ddt.psf[i_t,:,:,:]
    else:
        print "<ddt_H> setting up new psf"
        
    return G_or_H(ddt, psf, ddt.psf_nx, ddt.psf_ny, ddt.psf.shape[-3], i_t)
    
def G_or_H(ddt, x, n_x, n_y, n_l, i_t):
    """this is where ADR is treated a s a phase shift in Fourier space
    Parameters
    ----------
    ddt: DDT object
    x : 3d array
    n_x, n_y, n_l, i_t : int
    
    Returns
    _______
    a function
    """
    
    ptr = np.zeros(n_l) # This is a pointer in DDT, not sure if ptr is used
    number = ptr.size
    FFT = ddt.FFT # Not set up yet in ddt object
    
    for k in range(number):
        # fft_shift_phasor is defined in Yorick_util/fft_shift.i
        phase_shift_apodize = fft_shift_phasor([2,ddt.psf_nx, ddt.psf_ny],
                                                [ddt.sn_offset_x[k,i_t],
                                                 ddt.sn_offset_y[k,i_t]],
                                                half=1,
                                                apodize=ddt.apodizer)
        ptr[k] = FFT(x[k,:,:] * phase_shift_apodize)
    
    return _convolve()
           
def _convolve(this, x, job, offset=None, n_x=None, n_y=None, apodize=None):
    """
    Notes
    -----
    job = 0: direct
    job = 1: gradient
    job = 2: add an offset, in SPAXELS
    """
    FFT = this.FFT
    local ptr
    eq_nocopy, ptr, this.ptr # Haven't yet figured out eq_nocopy
    number = ptr.size
    out = np.zeros(x.shape)
    
    if job == 0:
        for k in range(number):
            out[k,:,:] = FFT(ptr[k]*FFT(x[k,:,:]),2)
        return out
    elif job == 1:
        for k in range(number):
            out[k,:,:] = FFT( conj(ptr[k]) * FFT(x[k,:,:]),2)
        return out
    elif job == 2:
        if offset is None:
            raise ValueError("<_convolve> job=2 need offset")
        if n_x is None: n_x = 32
        if n_y is None: n_y = 32
        phase_shift_apodize = fft_shift_phasor([2, n_x, n_y], offset, half=1,
                                               apodize=apodize)
        for k in range(number):
            out[k,:,:] = FFT( ptr[k] * phase_shift_apodize*FFT(x[k,:,:]),2)
        return out 
    else: 
        raise ValueError("unsupported JOB")
    