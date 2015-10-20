from __future__ import division
import numpy as np
cimport numpy as cnp
from libc.math cimport exp, sqrt, pow, M_PI
import cython

cnp.import_array()  # To access the numpy C-API.

__all__ = ["gaussian_moffat_psf"]

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def gaussian_moffat_psf(double[:] sigma, double[:] alpha, double[:] beta,
                        double[:] ellipticity, double[:] eta,
                        double[:] yctr, double[:] xctr, shape, int subpix=1):
        """Evaluate a gaussian+moffat function on each slice of a 3-d grid. 

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

        scale = 1. / subpix
        area = scale * scale

        nw = len(sigma)
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
