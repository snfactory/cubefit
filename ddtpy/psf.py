from __future__ import division

import math
import numpy as np

def params_from_gs(es_psf, wave, wave_ref):
    """Return arrays of ellipticity, alpha PSF parameters.

    Parameters
    ----------
    es_psf : 2-d array

    wave : 1-d array
    wave_ref : float

    Returns
    -------
    ellipticity : 2-d array
    alpha : 2-d array
    """

    relwave = wave / wave_ref
  
    # one ellipticity per time (constant across wavelength)
    ellipticity = es_psf[:, 0]
    a0 = es_psf[:, 1]
    a1 = es_psf[:, 2]
    a2 = es_psf[:, 3]

    # duplicate ellipticity for each wavelength
    ellipticity = np.repeat(ellipticity[:, np.newaxis], len(wave), axis=1)
    
    relwave = wave / wave_ref - 1.
    alpha = (a0[:, np.newaxis] +
             a1[:, np.newaxis] * relwave +
             a2[:, np.newaxis] * relwave**2)

    return ellipticity, alpha

def gaussian_plus_moffat_psf_4d(x, y, ellipticity, alpha, angle=None):
    """Create a 2-d PSF for a 2-d array of parameters.

    Parameters
    ----------
    x, y : 1-d array of int
        Coordinates of each array element in x and y. The PSF is centered
        at coordinates (x,y) = (0.,0.).
    ellipticity : 2-d array
    alpha : 2-d array

    Returns
    -------
    x : 1-d array of int
    y : 1-d array of int
    psf : 4-d array

    Notes
    -----
    (0.,0.) is at the center of the field
    position of the center of the PSF by definition 
    sn_x = int((n_x + 1.0)/2.);
    sn_y = int((n_y + 1.0)/2.);
    """

    assert ellipticity.shape == alpha.shape

    nt, nw = ellipticity.shape
    ny = len(y)
    nx = len(x)

    if angle is None:
        angle = np.zeros(nt)

    # allocate output array
    psf = np.empty((nt, nw, ny, nx), dtype=np.float)

    for i_t in range(nt):
        for i_w in range(nw):
            e = ellipticity[i_t, i_w]
            a = alpha[i_t, i_w]
            slicepsf = gaussian_plus_moffat_psf(x, y, 0., 0., e, a,
                                                angle=angle[i_t])
            psf[i_t, i_w, :, :] = slicepsf / np.sum(slicepsf)

    return psf

def gaussian_plus_moffat_psf(x, y, x0, y0, ellipticity, alpha, angle=0.):
    """Evaluate a gaussian+moffat function on a 2-d grid. 

    Parameters
    ----------
    x : 1-d array
        x coordinates of output array
    y : 1-d array
        y coordinates of output array
    x0, y0 : float
    ellipticity: float
    alpha : float
    angle : float

    Returns
    -------
    psf : 2-d array
        The shape will be (len(y), len(x)) 
    """

    alpha = abs(alpha)
    ellipticity = abs(ellipticity)
  
    # Correlated params
    s1 = 0.215
    s0 = 0.545
    b1 = 0.345
    b0 = 1.685
    e1 = 0.0
    e0 = 1.04

    # Moffat
    sigma = s0 + s1*alpha
    beta  = b0 + b1*alpha
    eta   = e0 + e1*alpha

    # In the next line, (x-x0) and (y-y0) are both 1-d arrays.
    # Output arrays are 2-d, both with shape (len(y), len(x)).
    # dx, for example, gives the dx value at each point in the grid.
    dx, dy = np.meshgrid(x-x0, y-y0)

    # Offsets in rotated coordinate system (dx', dy')
    dx_prime = dx * math.cos(angle) - dy * math.sin(angle)
    dy_prime = dx * math.sin(angle) + dy * math.cos(angle)
    r2 = dx_prime**2 + ellipticity * dy_prime**2

    # Gaussian, Moffat
    gauss = np.exp(-r2 / (2. * sigma**2))
    moffat = (1. + r2 / alpha**2)**(-beta)

    # scalars normalization
    norm_moffat = 1./math.pi * math.sqrt(ellipticity) * (beta-1.) / alpha**2
    norm_gauss = 1./math.pi * math.sqrt(ellipticity) / (2. * eta * sigma**2)
    norm_psf = 1. / (1./norm_moffat + eta * 1./norm_gauss)

    return norm_psf * (moffat + eta*gauss)

def roll_psf(psf, dx, dy):
    """Rolls the psf by dx, dy in x and y coordinates.

    The equivalent of the Yorick version with job = 0 is

    dx, dy = (1 - sn_x, 1 - sn_y) [if sn_x = (nx+1)//2]
    or
    dx, dy = (-sn_x, -sn_y) [if sn_x = (nx-1)//2]  # python indexed

    job=1 is 

    Parameters
    ----------
    psf : 4-d array
    dx, dy : int
        Shift in x and y coordinates.

    Returns
    -------
    rolled_psf : 4-d array

    """

    assert psf.ndim == 4  # just checkin' (in Yorick this is also used for 3-d)
                          # but we haven't implemented that here.

    tmp = np.roll(psf, dy, 2)  # roll along y (axis = 2)
    return np.roll(tmp, dx, 3)  # roll along x (axis = 3)
