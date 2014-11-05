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

def gaussian_plus_moffat_psf_4d(shape, x0, y0, ellipticity, alpha,
                                angle=None):
    """Create a 2-d PSF for a 2-d array of parameters.

    Parameters
    ----------
    shape : 2-tuple
        (ny, nx) shape of spatial component of output array.
    x0, y0 : float
        Center of PSF in array coordinates. (0, 0) = centered on lower left
        pixel.
    ellipticity : 2-d array
    alpha : 2-d array

    Returns
    -------
    psf : 4-d array
        Shape is (nt, nw, ny, nx) where (nt, nw) is the shape of ellipticity
        and alpha.
    """

    assert ellipticity.shape == alpha.shape

    nt, nw = ellipticity.shape
    ny, nx = shape
    if angle is None:
        angle = np.zeros(nt)

    # allocate output array
    psf = np.empty((nt, nw, ny, nx), dtype=np.float)

    for i_t in range(nt):
        for i_w in range(nw):
            slicepsf = gaussian_plus_moffat_psf(shape, x0, y0,
                                                ellipticity[i_t, i_w],
                                                alpha[i_t, i_w], angle[i_t])
            slicepsf /= np.sum(slicepsf)  # normalize array sum to 1.0.
            psf[i_t, i_w, :, :] = slicepsf

    return psf

def gaussian_plus_moffat_psf(shape, x0, y0, ellipticity, alpha, angle):
    """Evaluate a gaussian+moffat function on a 2-d grid. 

    Parameters
    ----------
    shape : 2-tuple
        (ny, nx) of output array.
    x0, y0 : float
        Center of PSF in array coordinates. (0, 0) = centered on lower left
        pixel.
    ellipticity: float
    alpha : float
    angle : float

    Returns
    -------
    psf : 2-d array
        The shape will be (len(y), len(x)) 
    """

    ny, nx = shape
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

    # In the next line, output arrays are 2-d, both with shape (ny, nx).
    # dx, for example, gives the dx value at each point in the grid.
    dx, dy = np.meshgrid(np.arange(nx) - x0, np.arange(ny) - y0)

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
