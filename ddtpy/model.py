from __future__ import print_function

import numpy as np
from numpy.fft import fft2, ifft2

from .psf import gaussian_plus_moffat_psf_4d
from .utils import fft_shift_phasor_2d

__all__ = ["DDTModel"]

# TODO: take out these asserts eventually
def _assert_real(x):
    assert (np.all((x.imag == 0.) & (x.real == 0.)) or
            np.all(np.abs(x.imag / x.real) < 1.e-10))

class DDTModel(object):
    """This class is the equivalent of everything else that isn't data
    in the Yorick version.

    Parameters
    ----------
    nt : int
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
    sn_x_init, sn_y_init : float
        Initial SN position in model coordinates.
    skyguess : np.ndarray (2-d)
        Initial guess at sky. Sky is a spatially constant value, so the
        shape is (nt, len(wave)).
    mean_gal_spec : np.ndarray (1-d)
        Rough guess at average galaxy spectrum for use in regularization.
        Shape is (len(wave),).

    Notes
    -----
    The spatial coordinate system of the model is fixed to be aligned with
    the master reference of the data. The "spaxel size" of the model is
    fixed to be the same as the instrument.
    """

    MODEL_SHAPE = (32, 32)

    def __init__(self, nt, wave, psf_ellipticity, psf_alpha, adr_dx, adr_dy,
                 mu_xy, mu_wave, sn_x_init, sn_y_init, skyguess,
                 mean_gal_spec):

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
        # left pixel. We do this for using the PSF as a convolution kernel:
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

        # hyperparameters
        self.mu_xy = mu_xy
        self.mu_wave = mu_wave

        # Galaxy, sky, and SN part of the model
        self.gal = np.zeros((nw, ny, nx))
        self.galprior = np.zeros((nw, ny, nx))
        self.mean_gal_spec = mean_gal_spec
        self.sky = skyguess
        self.sn = np.zeros((nt, nw))  # SN spectrum at each epoch
        self.sn_x_init = sn_x_init  # position of SN in model coordinates
        self.sn_y_init = sn_x_init
        self.sn_x = sn_x_init
        self.sn_y = sn_x_init
        self.eta = np.ones(nt)  # eta is transmission; not currently used.

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

        # shift needed to put SN in right place in the model
        fshift_sn = fft_shift_phasor_2d(self.MODEL_SHAPE,
                                        (self.sn_y, self.sn_x))

        conv = self.conv[i_t]
        out = np.empty((self.nw, self.ny, self.nx), dtype=np.float64)
                                     
        for j in range(self.nw):
            fshift = fft_shift_phasor_2d(self.MODEL_SHAPE,
                                         (yshift + self.adr_dy[i_t, j],
                                          xshift + self.adr_dx[i_t, j]))

            if which == 'galaxy':
                tmp = ifft2(fft2(conv[j, :, :]) * fshift *
                            fft2(self.gal[j, :, :]))
                _assert_real(tmp)
                out[j, :, :] = tmp.real

            elif which == 'snscaled':
                tmp = ifft2(fft2(psf[j, :, :]) * fshift_sn * fshift)
                _assert_real(tmp)
                out[j, :, :] = tmp.real

            elif which == 'all':
                tmp = ifft2(
                    fshift *
                    (fft2(self.gal[j, :, :]) * fft2(self.conv[j, :, :]) +
                     self.sn[i_t,j] * fshift_sn * fft2(self.psf[j, :, :])))
                _assert_real(tmp)
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
            _assert_real(tmp)
            out[j, :, :] = tmp.real

        return out
