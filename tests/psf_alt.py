"""This is a parallel implementation of GaussMoffatPSF in pure Python for
testing purposes."""

from __future__ import division

import math
import numpy as np


class GaussMoffatPSF:
    """A Gaussian plus Moffat function 3-d point spread function.

    This describes a separate analytic PSF at multiple (discrete) wavelengths.
    At each wavelength, the PSF is described by two parameters: ellipticity
    and alpha. These in turn determine the Gaussian and Moffat function
    parameters.

    Parameters
    ----------
    ellipticity : ndarray (1-d)
    alpha : ndarray (1-d)
    """

    def __init__(self, ellipticity, alpha):
        self.nw = len(ellipticity)
        if not len(alpha) == self.nw:
            raise ValueError("length of ellipticity and alpha must match")

        self.ellipticity = np.abs(ellipticity)
        self.alpha = np.abs(alpha)

        # Correlated params (determined externally)
        s0, s1 = 0.545, 0.215
        b0, b1 = 1.685, 0.345
        e0, e1 = 1.040, 0.0

        self.sigma = s0 + s1 * self.alpha  # Gaussian parameter
        self.beta  = b0 + b1 * self.alpha  # Moffat parameter
        self.eta   = e0 + e1 * self.alpha  # gaussian ampl. / moffat ampl.

    def __call__(self, shape, yctr, xctr, angle=0.0):
        """Evaluate a gaussian+moffat function on a 3-d grid. 

        Parameters
        ----------
        shape : 2-tuple
            (ny, nx) of output array.
        yctr, xctr : ndarray (1-d)
            Position of center of PSF relative to *center* of output array
            at each wavelength.

        Returns
        -------
        psf : 3-d array
            The shape will be (self.nw, shape[0], shape[1])
        """

        ny, nx = shape

        output = np.zeros((self.nw, ny, nx), dtype=np.float64)
        for i in range(self.nw):
            sigma_y = self.sigma[i]
            alpha_y = self.alpha[i]
            beta = self.beta[i]
            e = self.ellipticity[i]
            eta = self.eta[i]

            dy = np.arange(-ny/2.0 + 0.5 - yctr[i], ny/2.0 + 0.5 - yctr[i])
            dx = np.arange(-nx/2.0 + 0.5 - xctr[i], nx/2.0 + 0.5 - xctr[i])

            # Output arrays of numpy.meshgrid() are 2-d, both with
            # shape (ny, nx).  DX, for example, gives the dx value at
            # each point in the grid.
            DX, DY = np.meshgrid(dx, dy)

            # Offsets in rotated coordinate system (DX', DY')
            DXp = DX * math.cos(angle) - DY * math.sin(angle)
            DYp = DX * math.sin(angle) + DY * math.cos(angle)

            # We are defining, in the Gaussian,
            # sigma_y^2 / sigma_x^2 === ellipticity
            # and in the Moffat,
            # alpha_y^2 / alpha_x^2 === ellipticity
            sigma_x = math.sqrt(e) * sigma_y
            alpha_x = math.sqrt(e) * alpha_y

            # Gaussian normalized to 1.0
            g = 1. / (2. * math.pi * sigma_x * sigma_y) * \
                np.exp(-(DXp**2/(2.*sigma_x**2) + DYp**2/(2.*sigma_y**2)))

            # Moffat normalized to 1.0
            m = (beta - 1.) / (math.pi * alpha_x * alpha_y) * \
                (1. + DXp**2/alpha_x**2 + DYp**2/alpha_y**2)**-beta

            output[i, :, :] = (m + eta * g) / (1. + eta)

        return output


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
    norm_gauss = 1./math.pi * math.sqrt(ellipticity) / (2. * eta * sigma**2)
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
