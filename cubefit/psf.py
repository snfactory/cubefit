from __future__ import division

import math
import numpy as np

__all__ = ["gaussian_plus_moffat_psf", "psf_3d_from_params"]

def gaussian_plus_moffat_psf(shape, xctr, yctr, ellipticity, alpha, angle):
    """Evaluate a gaussian+moffat function on a 2-d grid. 

    Parameters
    ----------
    shape : 2-tuple
        (ny, nx) of output array.
    xctr, yctr : float
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
    dx, dy = np.meshgrid(np.arange(nx) - xctr, np.arange(ny) - yctr)

    # Offsets in rotated coordinate system (dx', dy')
    dx_prime = dx * math.cos(angle) - dy * math.sin(angle)
    dy_prime = dx * math.sin(angle) + dy * math.cos(angle)
    r2 = dx_prime**2 + ellipticity * dy_prime**2

    # Gaussian, Moffat
    gauss = np.exp(-r2 / (2. * sigma**2))
    moffat = (1. + r2 / alpha**2)**(-beta)

    # scalars normalization
    norm_moffat = 1./math.pi * math.sqrt(ellipticity) * (beta-1.) / alpha**2
    norm_gauss = 1./math.pi * math.sqrt(ellipticity) / (2. * sigma**2)
    norm_psf = 1. / (1./norm_moffat + eta * 1./norm_gauss)

    return norm_psf * (moffat + eta*gauss)


def psf_3d_from_params(params, wave, wave_ref, shape):
    """Create a wavelength-dependent Gaussian+Moffat PSF from given
    parameters.

    Parameters
    ----------
    params : 4-tuple
        Ellipticty and polynomial parameters in wavelength
    wave : np.ndarray (1-d)
        Wavelengths
    wave_ref : float
        Reference wavelength
    shape : 2-tuple
        (ny, nx) shape of spatial component of output array.

    Returns
    -------
    psf : 3-d array
        Shape is (nw, ny, nx) where (nw,) is the shape of wave array.
        PSF will be spatially centered in array.
    """

    relwave = wave / wave_ref - 1.
    ellipticity = params[0]
    alpha = params[1] + params[2]*relwave + params[3]*relwave**2

    nw = len(wave)
    ny, nx = shape
    xctr = (nx - 1) / 2.
    yctr = (ny - 1) / 2.
    psf = np.empty((nw, ny, nx), dtype=np.float)
    for i in range(nw):
        psf2d = gaussian_plus_moffat_psf(shape, xctr, yctr, ellipticity,
                                         alpha[i], 0.0)
        psf2d /= np.sum(psf2d)  # normalize array sum to 1.0.
        psf[i, :, :] = psf2d

    return psf


class DiscretePSF(object):
    """A Discrete (tabular) 3-d point spread function.
    
    Parameters
    ----------
    psf : ndarray (3-d)
        The PSF, which is assumed to be centered in the array. The shape
        should be (nw, ny, nx).
    """

    def __init__(self, psf):
        self.psf = psf


class GaussMoffatPSF(object):
    """A Gaussian plus Moffat function 3-d point spread function.

    This describes a separate analytic PSF at multiple (discrete) wavelengths.
    At each wavelength, the PSF is described by two parameters.

    Parameters
    ----------
    ellipticity : ndarray (1-d)
    alpha : ndarray (1-d)
    """

    def __init__(self, ellipticity, alpha, alpha=0.0):
        if not (len(ellipticity) == len(alpha)):
            raise ValueError("length of ellipticity and alpha must match")
        self.ellipticity == ellipticity
        self.alpha = alpha


    def __call__(self, shape, yctr, xctr, angle=0.0):
        """Sample the PSF onto a 3-d grid with 2-d shape `shape`.
        parameters.

        Parameters
        ----------
        params : 4-tuple
            Ellipticty and polynomial parameters in wavelength
        wave : np.ndarray (1-d)
            Wavelengths
        wave_ref : float
            Reference wavelength
        shape : 2-tuple
            (ny, nx) shape of spatial component of output array.

        Returns
        -------
        psf : 3-d array
            Shape is (nw, ny, nx) where (nw,) is the shape of wave array.
            PSF will be spatially centered in array.
        """

        nw = len(wave)
        ny, nx = shape
        xctr = (nx - 1) / 2.
        yctr = (ny - 1) / 2.
        psf = np.empty((nw, ny, nx), dtype=np.float)
        for i in range(nw):
            psf2d = gaussian_plus_moffat_psf(shape, xctr, yctr, ellipticity,
                                             alpha[i], 0.0)
            psf2d /= np.sum(psf2d)  # normalize array sum to 1.0.
            psf[i, :, :] = psf2d

        return psf
