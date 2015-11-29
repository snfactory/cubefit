from __future__ import division

import numpy as np
from numpy.fft import fft2, ifft2
import pyfftw

from .utils import fft_shift_phasor_2d, yxoffset
from .psffuncs import gaussian_moffat_psf

__all__ = ["TabularPSF", "GaussianMoffatPSF"]


class PSFBase(object):
    """Base class for 3-d PSFs."""

    def __init__(self, A):
        """Set up arrays and FFTs for convolution.

        Parameters
        ----------
        A : ndarray (3-d)
            PSF, assumed to be centered in the array at the
            "reference wavelength."
        """

        self.nw, self.ny, self.nx = A.shape

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
        fshift = fft_shift_phasor_2d((self.ny, self.nx), shift)
        fftconv = fft2(A) * fshift

        # align on SIMD boundary.
        self.fftconv = pyfftw.n_byte_align(fftconv, pyfftw.simd_alignment,
                                           dtype=np.complex128)

        # set up input and output arrays for FFTs.
        self.fftin = pyfftw.n_byte_align_empty(A.shape,
                                               pyfftw.simd_alignment,
                                               dtype=np.complex128)
        self.fftout = pyfftw.n_byte_align_empty(A.shape,
                                                pyfftw.simd_alignment,
                                                dtype=np.complex128)

        # Set up forward and backward FFTs.
        self.fft = pyfftw.FFTW(self.fftin, self.fftout, axes=(1, 2),
                               threads=1)
        self.ifft = pyfftw.FFTW(self.fftout, self.fftin, axes=(1, 2),
                                threads=1, direction='FFTW_BACKWARD')

        self.fftnorm = 1. / (self.ny * self.nx) 

    def evaluate_galaxy(self, galmodel, shape, ctr, grad=False):
        """convolve, shift and sample the galaxy model"""

        # shift necessary to put model onto data coordinates
        offset = yxoffset((self.ny, self.nx), shape, ctr)
        fshift = fft_shift_phasor_2d((self.ny, self.nx),
                                     (-offset[0], -offset[1]), grad=grad)
        if grad:
            fshift, fshiftgrad = fshift
            fshiftgrad *= -1.  # make derivatives w.r.t. `ctr`.

        # calculate `fft(galmodel) * fftconv`
        np.copyto(self.fftin, galmodel)  # copy input array to complex array
        self.fft.execute()  # populates self.fftout
        self.fftout *= self.fftconv
        if grad:
            fftgal = np.copy(self.fftout)  # cache result for use in gradient.

        self.fftout *= fshift
        self.ifft.execute() # populates self.fftin
        self.fftin *= self.fftnorm
        gal = np.copy(self.fftin.real[:, 0:shape[0], 0:shape[1]])

        if grad:
            galgrad = np.empty((2,) + gal.shape, dtype=np.float64)
            for i in (0, 1):
                np.copyto(self.fftout, fftgal)
                self.fftout *= fshiftgrad[i]
                self.ifft.execute() # populates self.fftin
                self.fftin *= self.fftnorm
                galgrad[i] = self.fftin.real[:, 0:shape[0], 0:shape[1]]
            return gal, galgrad

        else:
            return gal

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
        fshift = np.asarray(fshift, dtype=np.complex128)

        # create output array
        out = np.zeros((self.nw, self.ny, self.nx), dtype=np.float64)
        out[:, :x.shape[1], :x.shape[2]] = x

        for i in range(self.nw):
            tmp = ifft2(np.conj(self.fftconv[i, :, :] * fshift) *
                        fft2(out[i, :, :]))
            out[i, :, :] = tmp.real

        return out


class TabularPSF(PSFBase):
    """PSF represented by an array."""

    def point_source(self, pos, shape, ctr, grad=False):
        """Evaluate a point source at the given position.

        If grad is True, return a 2-tuple, with the second item being
        a 4-d array of gradient with respect to
        ctr[0], ctr[1], pos[0], pos[1].
        """

        # shift necessary to put model onto data coordinates
        offset = yxoffset((self.ny, self.nx), shape, ctr)
        yshift, xshift = -offset[0], -offset[1]

        # Add shift to move point source from the lower left in the model array
        # to `pos` in model *coordinates*. Note that in model coordinates,
        # (0, 0) corresponds to array position (ny-1)/2., (nx-1)/2.
        yshift += (self.ny - 1) / 2. + pos[0]
        xshift += (self.nx - 1) / 2. + pos[1]

        fshift = fft_shift_phasor_2d((self.ny, self.nx), (yshift, xshift),
                                     grad=grad)
        if grad:
            fshift, fshiftgrad = fshift
            fshiftgrad *= -1.  # make derivatives w.r.t. `ctr`.

        # following block is like ifft2(fftconv * fshift)
        np.copyto(self.fftout, self.fftconv)
        self.fftout *= self.fftnorm * fshift
        self.ifft.execute()
        s = np.copy(self.fftin.real[:, 0:shape[0], 0:shape[1]])
        
        if grad:
            sgrad = np.empty((4,) + s.shape, dtype=np.float64)
            for i in (0, 1):
                np.copyto(self.fftout, self.fftconv)
                self.fftout *= self.fftnorm * fshiftgrad[i]
                self.ifft.execute()
                sgrad[i] = self.fftin.real[:, 0:shape[0], 0:shape[1]]
            sgrad[2:4] = -sgrad[0:2]
            return s, sgrad

        else:
            return s


class GaussianMoffatPSF(PSFBase):
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

    def __init__(self, sigma, alpha, beta, ellipticity, eta, yctr, xctr,
                 shape, subpix=1):

        if not (len(sigma) == len(alpha) == len(beta) == len(ellipticity) ==
                len(eta) == len(yctr) == len(xctr)):
            raise ValueError("length of input arrays must match")

        if not np.all(beta > 1.):
            raise ValueError("beta must be > 1")

        self.sigma = sigma
        self.alpha = alpha
        self.beta = beta
        self.ellipticity = ellipticity
        self.eta = eta
        self.yctr = yctr
        self.xctr = xctr

        self.subpix = subpix

        # Set up tabular PSF for galaxy convolution
        A = gaussian_moffat_psf(sigma, alpha, beta, ellipticity, eta,
                                yctr, xctr, shape, subpix=subpix)
        super(GaussianMoffatPSF, self).__init__(A)

    def point_source(self, pos, shape, ctr, grad=False):
        yctr = self.yctr + pos[0] - ctr[0]
        xctr = self.xctr + pos[1] - ctr[1]

        res = gaussian_moffat_psf(self.sigma, self.alpha, self.beta,
                                  self.ellipticity, self.eta, yctr, xctr,
                                  shape, subpix=self.subpix, grad=grad)

        if grad:
            s, sgrad_pos = res
            sgrad = np.empty((4,) + s.shape, dtype=np.float64)
            sgrad[0:2] = -sgrad_pos
            sgrad[2:4] = sgrad_pos
            return s, sgrad
            
        else:
            return res
