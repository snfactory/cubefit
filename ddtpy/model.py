from __future__ import print_function

import numpy as np
import pyfftw
from numpy.fft import fft2, ifft2

from .utils import fft_shift_phasor_2d

__all__ = ["AtmModel", "RegularizationPenalty"]

# -----------------------------------------------------------------------------
# Helper functions

# TODO: move these asserts to tests
def assert_real(x):
    if np.all((x.imag == 0.) & (x.real == 0.)):
        return
    absfrac = np.abs(x.imag / x.real)
    mask = absfrac < 1.e-3 #1.e-4
    if not np.all(mask):
        print(x.imag[~mask])
        print(x.real[~mask])
        raise RuntimeError("array not real: max imag/real = {:g}"
                           .format(np.max(absfrac)))

def yxoffset(shape1, shape2, ctr):
    """y, x offset between two 2-d arrays (their lower-left corners)
    with shape1 and shape2, where the array centers are offset by ctr.
    
    Examples
    --------
    >>> yxoffset((32, 32), (15, 15), (0., 0.))
    (8.5, 8.5)
    >>> yxoffset((32, 32), (15, 15), (1., 0.))
    (9.5, 8.5)
    >>> yxoffset((32, 32), (15, 15), (1., 0.))
    (9.5, 8.5)

    Raises
    ------
    ValueError : If the arrays don't completely overlap.
    """

    # min and max coordinates of first array
    ymin1 = -(shape1[0] - 1) / 2.
    ymax1 = (shape1[0] - 1) / 2.
    xmin1 = -(shape1[1] - 1) / 2.
    xmax1 = (shape1[1] - 1) / 2.

    # min and max coordinates requested (inclusive)
    ymin2 = ctr[0] - (shape2[0] - 1) / 2.
    ymax2 = ctr[0] + (shape2[0] - 1) / 2.
    xmin2 = ctr[1] - (shape2[1] - 1) / 2.
    xmax2 = ctr[1] + (shape2[1] - 1) / 2.

    if (xmin2 < xmin1 or xmax2 > xmax1 or
        ymin2 < ymin1 or ymax2 > ymax1):
        raise ValueError("second array not within first array")

    return ymin2 - ymin1, xmin2 - xmin1


# -----------------------------------------------------------------------------

class AtmModel(object):
    """Container for "Atmospheric conditions" - PSF and ADR for a single
    observation.

    Parameters
    ----------
    psf : ndarray (3-d)
        PSF as a function of wavelength - assumed to be spatially centered
        in array.
    adr_refract : ndarray (2-d)
        Array with shape (2, nw) where [0, :] corresponds to the refraction
        in y and [1, :] corresponds to the refraction in x at each wavelength.

    """

    def __init__(self, psf, adr_refract, fftw_threads=1):

        self.shape = psf.shape
        self.nw, self.ny, self.nx = psf.shape
        spatial_shape = self.ny, self.nx
        nbyte = pyfftw.simd_alignment

        # The attribute `fftconv` stores the Fourier-space array
        # necessary to convolve another array by the PSF. This is done
        # by mulitiplying the input array by `fftconv` in fourier
        # space.
        #
        # We shift the PSF so that instead of being exactly centered
        # in the array, it is exactly centered on the lower left
        # pixel.  (For convolution in Fourier space, the (0, 0)
        # element of the kernel is effectively the "center.")
        # Note that this shifting is different than simply
        # creating the PSF centered at the lower left pixel to begin
        # with, due to wrap-around.
        #
        #`ifft2(fftconv).real` would be the PSF in
        # real space, shifted to be centered on the lower-left pixel.
        shift = -(self.ny - 1) / 2., -(self.nx - 1) / 2.
        fshift = fft_shift_phasor_2d(spatial_shape, shift)
        self.fftconv = fft2(psf) * fshift
        self.fftconv = pyfftw.n_byte_align(self.fftconv, nbyte,
                                           dtype=np.complex64)

        # Check that ADR has the correct shape.
        assert adr_refract.shape == (2, self.nw)

        # Further shift the PSF by the ADR.
        for i in range(self.nw):
            shift = adr_refract[0, i], adr_refract[1, i]
            fshift = fft_shift_phasor_2d(spatial_shape, shift)
            self.fftconv[i, :, :] *= fshift

        # set up input and output arrays for performing forward and reverse
        # FFTs.
        self.fftin = pyfftw.n_byte_align_empty(self.shape, nbyte,
                                               dtype=np.complex64)
        self.fftout = pyfftw.n_byte_align_empty(self.shape, nbyte,
                                                dtype=np.complex64)
        self.fft = pyfftw.FFTW(self.fftin, self.fftout, axes=(1, 2),
                               threads=fftw_threads)
        self.ifft = pyfftw.FFTW(self.fftout, self.fftin, axes=(1, 2),
                                threads=fftw_threads,
                                direction='FFTW_BACKWARD')
        self.fftnorm = 1. / (self.ny * self.nx) 

    def evaluate_galaxy(self, galmodel, shape, ctr):
        """convolve, shift and sample the galaxy model"""

        # shift necessary to put model onto data coordinates
        offset = yxoffset((self.ny, self.nx), shape, ctr)
        fshift = fft_shift_phasor_2d((self.ny, self.nx),
                                     (-offset[0], -offset[1]))

        # ifft2(self.fftconv * fshift * fft2(galmodel))
        np.copyto(self.fftin, galmodel)  # copy input array to complex array
        self.fft.execute()  # populates self.fftout
        self.fftout *= fshift
        self.fftout *= self.fftconv
        self.ifft.execute() # populates self.fftin
        self.fftin *= self.fftnorm

        return self.fftin.real[:, 0:shape[0], 0:shape[1]]

    def evaluate_point_source(self, pos, shape, ctr):
        """Evaluate a point source at the given position."""

        # shift necessary to put model onto data coordinates
        offset = yxoffset((self.ny, self.nx), shape, ctr)
        fshift = fft_shift_phasor_2d((self.ny, self.nx),
                                     (-offset[0], -offset[1]))

        # Shift to move point source from 0, 0 in model coords to `pos`.
        fshift_point = fft_shift_phasor_2d((self.ny, self.nx), pos)

        # following block is like ifft2(fftconv * fshift_point * fshift)
        np.copyto(self.fftout, self.fftconv)
        self.fftout *= fshift_point * fshift * self.fftnorm
        self.ifft.execute()

        return self.fftin.real[:, 0:shape[0], 0:shape[1]]

    def gradient_helper(self, x, shape, ctr):
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

        # shift necessary to put model onto data coordinates
        offset = yxoffset((self.ny, self.nx), shape, ctr)
        fshift = fft_shift_phasor_2d((self.ny, self.nx),
                                     (-offset[0], -offset[1]))

        # create output array
        out = np.zeros((self.nw, self.ny, self.nx), dtype=np.float64)
        out[:, :x.shape[1], :x.shape[2]] = x

        for i in range(self.nw):
            tmp = ifft2(np.conj(self.fftconv[i, :, :] * fshift) *
                        fft2(out[i, :, :]))
            assert_real(tmp)
            out[i, :, :] = tmp.real

        return out


class RegularizationPenalty(object):
    """Callable that returns the penalty and gradient on it."""
    
    def __init__(self, galprior, mean_gal_spec, mu_xy, mu_wave):
        self.galprior = galprior
        self.mean_gal_spec = mean_gal_spec
        self.mu_xy = mu_xy
        self.mu_wave = mu_wave

    def __call__(self, galmodel):
        """Return regularization penalty and gradient for a given galaxy model.

        Parameters
        ----------
        TODO

        Returns
        -------
        penalty : float
        penalty_gradient : ndarray
            Gradient with respect to model galaxy
        """

        galdiff = galmodel - self.galprior
        galdiff /= self.mean_gal_spec[:, None, None]
        dw = galdiff[1:, :, :] - galdiff[:-1, :, :]
        dy = galdiff[:, 1:, :] - galdiff[:, :-1, :]
        dx = galdiff[:, :, 1:] - galdiff[:, :, :-1]

        # Regularlization penalty term
        val = (self.mu_xy * np.sum(dx**2) +
               self.mu_xy * np.sum(dy**2) +
               self.mu_wave * np.sum(dw**2))

        # Gradient in regularization penalty term
        #
        # This is clearer when the loops are explicitly written out.
        # For a loop that goes over all adjacent elements in a given dimension,
        # one would do (pseudocode):
        # for i in ...:
        #     d = arr[i+1] - arr[i]
        #     penalty += hyper * d^2
        #     gradient[i+1] += 2 * hyper * d
        #     gradient[i]   -= 2 * hyper * d

        grad = np.zeros_like(galdiff)
        grad[:, :, 1:] += 2. * self.mu_xy * dx
        grad[:, :, :-1] -= 2. * self.mu_xy * dx
        grad[:, 1:, :] += 2. * self.mu_xy * dy
        grad[:, :-1,:] -= 2. * self.mu_xy * dy
        grad[1:, :, :] += 2. * self.mu_wave * dw
        grad[:-1, :, :] -= 2. * self.mu_wave * dw

        return val, grad
