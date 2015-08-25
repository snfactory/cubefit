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
            sigma_x = self.sigma[i]
            alpha_x = self.alpha[i]
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
            sigma_y = math.sqrt(e) * sigma_x
            alpha_y = math.sqrt(e) * alpha_x

            # Gaussian normalized to 1.0
            g = 1. / (2. * math.pi * sigma_x * sigma_y) * \
                np.exp(-(DXp**2/(2.*sigma_x**2) + DYp**2/(2.*sigma_y**2)))

            # Moffat normalized to 1.0
            m = (beta - 1.) / (math.pi * alpha_x * alpha_y) * \
                (1. + DXp**2/alpha_x**2 + DYp**2/alpha_y**2)**-beta

            output[i, :, :] = (m + eta * g) / (1. + eta)

        return output
