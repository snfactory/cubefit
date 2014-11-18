from __future__ import print_function

import numpy as np
from numpy.fft import fft2, ifft2

from .psf import gaussian_plus_moffat_psf_4d
from .utils import fft_shift_phasor_2d

__all__ = ["DDTModel"]

class DDTModel(object):
    """This class is the equivalent of everything else that isn't data
    in the Yorick version.

    Parameters
    ----------
    nt : 2-tuple of int
        Model dimension in time.
    wave : np.ndarray
        One-dimensional array of wavelengths in angstroms. Should match
        data.
    psf_ellipticity, psf_alpha : np.ndarray (2-d)
        Parameters characterizing the PSF at each time, wavelength. Shape
        of both must be (nt, len(wave)).
    adr_dx, adr_dy : np.ndarray (2-d)
        Atmospheric differential refraction in x and y directions, in spaxels,
        relative to reference wavelength.
    mu_xy : float
        Hyperparameter in spatial (x, y) coordinates. Used in penalty function
        when fitting model.
    mu_wave : float
        Hyperparameter in wavelength coordinate. Used in penalty function
        when fitting model.
    spaxel_size : float
        Spaxel size in arcseconds.
    skyguess : np.ndarray (2-d)
        Initial guess at sky. Sky is a spatially constant value, so the
        shape is (nt, len(wave)).

    Notes
    -----
    The spatial coordinate system used to align data and the model is
    an arbitrary grid on the sky where the origin is chosen to be at
    the true location of the SN. That is, in the model the SN is
    located at (0., 0.)  by definition and the location and shape of
    the galaxy is defined relative to the SN location. Similarly, the
    pointings/alignment of the data are defined relative to this
    coordinate system. For example, if a pointing is centered at
    coordinates (3.5, 2.5), this means that we believe the central
    spaxel of the data array to be 3.5 spaxels west and 2.5 spaxels
    north of the true SN position. (The units of the coordinate system
    are in spaxels for convenience.)
    """

    MODEL_SHAPE = (32, 32)

    def __init__(self, nt, wave, psf_ellipticity, psf_alpha, adr_dx, adr_dy,
                 mu_xy, mu_wave, spaxel_size, skyguess):

        ny, nx = self.MODEL_SHAPE
        nw, = wave.shape

        # Model shape
        self.nt = nt
        self.nw = nw
        self.ny = ny
        self.nx = nx

        # consistency checks on inputs
        s = (nt, nw)
        if not (psf_ellipticity.shape == s):
            raise ValueError("psf_ellipticity has wrong shape")
        if not (psf_alpha.shape == s):
            raise ValueError("psf_alpha has wrong shape")
        if not (adr_dx.shape == adr_dy.shape == s):
            raise ValueError("adr_dx and adr_dy shape must be (nt, nw)")

        # Coordinates of model grid
        array_xctr = (nx - 1) / 2.0
        array_yctr = (ny - 1) / 2.0
        self.xcoords = np.arange(nx, dtype=np.float64) - array_xctr
        self.ycoords = np.arange(ny, dtype=np.float64) - array_yctr

        # wavelength grid of the model (fixed; should be same as data)
        self.wave = wave

        # ADR part of the model (fixed)
        self.adr_dx = adr_dx
        self.adr_dy = adr_dy

        # PSF part of the model (fixed)
        self.psf = gaussian_plus_moffat_psf_4d(self.MODEL_SHAPE,
                                               array_xctr, array_yctr,
                                               psf_ellipticity, psf_alpha)

        # We now shift the PSF so that instead of being exactly
        # centered in the array, it is exactly centered on the lower
        # left pixel. We do this for using the PSf as a convolution kernel:
        # For convolution in Fourier space, the (0, 0) element of the kernel
        # is effectively the "center."
        # Note that this shifting is different than simply
        # creating the PSF centered at the lower left pixel to begin
        # with, due to wrap-around.
        self.conv = np.empty_like(self.psf)
        fshift = fft_shift_phasor_2d(self.MODEL_SHAPE,
                                     (-array_xctr, -array_yctr))
        for j in range(self.psf.shape[0]):
            for i in range(self.psf.shape[1]):
                tmp = self.psf[j, i, :, :]
                self.conv[j, i, :, :] = ifft2(fft2(tmp) * fshift).real

        # hyperparameters and spaxel size
        self.mu_xy = mu_xy
        self.mu_wave = mu_wave
        self.spaxel_size = spaxel_size

        # Galaxy and sky part of the model
        self.gal = np.zeros((nw, ny, nx))
        self.galprior = np.zeros((nw, ny, nx))
        self.sky = skyguess
        self.sn = np.zeros((nt, nw))
        self.eta = np.ones(nt)  # eta is transmission
        self.final_ref_sky = np.zeros(nw)

    def evaluate(self, i_t, xctr, yctr, shape, which='galaxy'):
        """Evalute the model on a grid for a single epoch.

        Parameters
        ----------
        i_t : int
            Epoch index.
        shape : tuple
            Two integers giving size of output array (ny, nx).
        xctr, yctr : float
            Center of output array in model coordinates.
        which : {'galaxy', 'snscaled', 'all'}
            Which part of the model to evaluate: galaxy-only or SN scaled to
            flux of 1.0?

        Returns
        -------
        x : np.ndarray (3-d)
            Shape is (nw, len(ycoords), len(xcoords)).
        """

        # min and max coordinates requested (inclusive)
        ny, nx = shape
        xmin = xctr - (nx - 1) / 2.
        xmax = xctr + (nx - 1) / 2.
        ymin = yctr - (ny - 1) / 2.
        ymax = yctr + (ny - 1) / 2.

        if (xmin < self.xcoords[0] or xmax > self.xcoords[-1] or
            ymin < self.ycoords[0] or ymax > self.ycoords[-1]):
            raise ValueError("requested coordinates out of model bounds")
        
        # Shift needed to put the model onto the requested coordinates.
        xshift = -(xmin - self.xcoords[0])
        yshift = -(ymin - self.ycoords[0])

        conv = self.conv[i_t]
        out = np.empty((self.nw, self.ny, self.nx), dtype=np.float64)
                                     
        for j in range(self.nw):
            fshift = fft_shift_phasor_2d(self.MODEL_SHAPE,
                                         (yshift + self.adr_dy[i_t, j],
                                          xshift + self.adr_dx[i_t, j]))

            if which == 'galaxy':
                tmp = ifft2(fft2(conv[j, :, :]) * fshift *
                            fft2(self.gal[j, :, :]))
                if not np.allclose(tmp.imag, 0., atol=1.e-14):
                    raise RuntimeError("IFFT returned non-real array.")
                out[j, :, :] = tmp.real

            elif which == 'snscaled':
                tmp = ifft2(fft2(psf[j, :, :]) * fshift)
                if not np.allclose(tmp.imag, 0., atol=1.e-14):
                    raise RuntimeError("IFFT returned non-real array.")
                out[j, :, :] = tmp.real

            elif which == 'all':
                tmp = ifft2(
                    fshift *
                    (fft2(self.gal[j, :, :]) * fft2(self.conv[j, :, :]) +
                     fft2(self.sn[i_t,j] * self.psf[j, :, :])))
                if not np.allclose(tmp.imag, 0., atol=1.e-14):
                    raise RuntimeError("IFFT returned non-real array.")
                out[j, :, :] = tmp.real + self.sky[i_t, j]

        # Return a slice that matches the data.
        return out[:, 0:ny, 0:nx]


    def gradient_helper(self, i_t, x, xctr, yctr, shape):
        """Not sure exactly what this does yet.

        Parameters
        ----------
        i_t : int
            Epoch index.
        x : np.ndarray (3-d)
            Same shape as *data* for single epoch (nw, ny, nx).
        xcoords : np.ndarray (1-d)
        ycoords : np.ndarray (1-d)

        Returns
        -------
        x : np.ndarray (3-d)
            Shape is (nw, len(ycoords), len(xcoords)).
        """

        # min and max coordinates requested (inclusive)
        ny, nx = shape
        xmin = xctr - (nx - 1) / 2.
        xmax = xctr + (nx - 1) / 2.
        ymin = yctr - (ny - 1) / 2.
        ymax = yctr + (ny - 1) / 2.

        if (xmin < self.xcoords[0] or xmax > self.xcoords[-1] or
            ymin < self.ycoords[0] or ymax > self.ycoords[-1]):
            raise ValueError("requested coordinates out of model bounds")
        
        xshift = -(xmin - self.xcoords[0])
        yshift = -(ymin - self.ycoords[0])

        # create output array
        out = np.zeros((self.nw, self.ny, self.nx), dtype=np.float64)
        out[:, :x.shape[1], :x.shape[2]] = x

        conv = self.conv[i_t]

        for j in range(self.nw):
            fshift = fft_shift_phasor_2d(self.MODEL_SHAPE,
                                         (yshift + self.adr_dy[i_t,j],
                                          xshift + self.adr_dx[i_t,j]))

            tmp = ifft2(np.conj(fft2(conv[j, :, :]) * fshift) *
                        fft2(out[j, :, :]))

            assert np.allclose(tmp.imag, 0., atol=1.e-14)

            out[j, :, :] = tmp.real

        return out
