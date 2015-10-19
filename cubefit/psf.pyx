from __future__ import division
import numpy as np
cimport numpy as cnp
from libc.math cimport exp, sqrt, pow, M_PI
import cython

cnp.import_array()  # To access the numpy C-API.

__all__ = ["GaussMoffatPSF"]


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

    def __init__(self, ellipticity, alpha, subpix=1):
        self.nw = len(ellipticity)
        if not len(alpha) == self.nw:
            raise ValueError("length of ellipticity and alpha must match")

        self.ellipticity = np.abs(ellipticity)
        self.alpha = np.abs(alpha)
        self.subpix = subpix

        # Correlated params (determined externally)
        s0, s1 = 0.545, 0.215
        b0, b1 = 1.685, 0.345
        e0, e1 = 1.040, 0.0

        self.sigma = s0 + s1 * self.alpha  # Gaussian parameter
        self.beta  = b0 + b1 * self.alpha  # Moffat parameter
        self.eta   = e0 + e1 * self.alpha  # gaussian ampl. / moffat ampl.

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    def __call__(self, shape, double[:] yctr, double[:] xctr, int subpix=0):
        """Evaluate a gaussian+moffat function on a 3-d grid. 

        Parameters
        ----------
        shape : 2-tuple
            (ny, nx) of output array.
        yctr, xctr : ndarray (1-d)
            Position of center of PSF relative to *center* of output array
            at each wavelength.
        subpix : int, optional
            Subpixel sampling. If 0, default specified in constructor
            will be used.

        Returns
        -------
        psf : 3-d array
            The shape will be (self.nw, shape[0], shape[1])
        """

        # NOTE: for rotated coordinates, we would do
        #     dyp = dx * cos(angle) - dy * sin(angle)
        #     dxp = dx * sin(angle) + dy * cos(angle)
        # and then use dyp, dxp.

        cdef cnp.intp_t nw, ny, nx, i, j, k
        cdef double sigma_x, sigma_y, alpha_x, alpha_y
        cdef double yc, xc, dy, dx, sx, sy
        cdef double gnorm, mnorm, norm, g, m
        cdef double scale, area
        cdef double[:, :, :] outview
        cdef double[:] sigma, alpha, beta, ellipticity, eta

        if subpix == 0:
            subpix = self.subpix

        scale = 1. / subpix
        area = scale * scale

        # use memory views so we can index our arrays faster
        sigma = self.sigma
        alpha = self.alpha
        beta = self.beta
        ellipticity = self.ellipticity
        eta = self.eta

        nw = self.nw
        ny, nx = shape

        # allocate output buffer
        out = np.empty((nw, ny, nx), dtype=np.float64)
        outview = out

        for k in range(nw):

            # We are defining, in the Gaussian,
            # sigma_x^2 / sigma_y^2 === ellipticity
            # and in the Moffat,
            # alpha_x^2 / alpha_y^2 === ellipticity
            sigma_x = sigma[k]
            alpha_x = alpha[k]
            sigma_y = sigma_x / sqrt(ellipticity[k])
            alpha_y = alpha_x / sqrt(ellipticity[k])

            # normalizing pre-factors for gaussian and moffat
            gnorm = 1. / (2. * M_PI * sigma_x * sigma_y)
            mnorm = (beta[k] - 1.) / (M_PI * alpha_x * alpha_y)

            # normalization on (m + eta * g) [see below]
            norm = 1. / (1. / mnorm + eta[k] / gnorm)

            # center in pixel coordinates
            yc = yctr[k] + (ny-1) / 2.0
            xc = xctr[k] + (nx-1) / 2.0

            for j in range(ny):
                dy = j - yc
                for i in range(nx):
                    dx = i - xc

                    g = 0.0
                    m = 0.0
                    sy = dy - 0.5 + 0.5 * scale  # subpixel coordinates
                    while sy < dy + 0.5:
                        sx = dx - 0.5 + 0.5 * scale
                        while sx < dx + 0.5:
                            
                            # gaussian
                            g += exp(-(sx*sx/(2.*sigma_x*sigma_x) +
                                       sy*sy/(2.*sigma_y*sigma_y)))

                            # moffat
                            m += pow(1. + sx*sx/(alpha_x*alpha_x) +
                                          sy*sy/(alpha_y*alpha_y), -beta[k])
                            
                            sx += scale
                        sy += scale

                    # Note: eta is *apparently* defined as the scaling
                    # of the peak of the Gaussian relative to the peak
                    # of the Moffat. therefore eta is applied to the
                    # unnormalized function values (m and g) rather
                    # than the normalized values, which would look
                    # like `(mnorm * m + eta[k] * gnorm * g)`. The
                    # normalization constant `norm` accounts for the
                    # integral of `m + eta[k] * g`.
                    #
                    # `area` accounts for the subpixel area.
                    outview[k, j, i] = norm * area * (m + eta[k] * g)

        return out
