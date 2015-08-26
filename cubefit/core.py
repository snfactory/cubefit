"""Core data structures."""

import numpy as np
import pyfftw
from numpy.fft import fft2, ifft2

from .utils import fft_shift_phasor_2d, yxoffset

__all__ = ["DataCube", "AtmModel", "RegularizationPenalty"]


class DataCube(object):
    """A container for data and weight arrays.

    Attributes
    ----------
    data : ndarray (3-d)
    weight : ndarray (3-d)
    wave : ndarray (1-d)
    nw : int
        length of wave, data.shape[0], weight.shape[0]
    ny : int
        data.shape[1], weight.shape[1]
    nx : int
        data.shape[2], weight.shape[2]
    """
    
    def __init__(self, data, weight, wave, wavewcs=None):
        if data.shape != weight.shape:
            raise ValueError("shape of weight and data must match")
        if len(wave) != data.shape[0]:
            raise ValueError("length of wave must match data axis=1")
        if wavewcs is None:
            wavewcs = {}

        self.data = data
        self.weight = weight
        self.wave = wave
        self.nw, self.ny, self.nx = data.shape
        self.wavewcs = wavewcs


class AtmModel(object):
    """Container for "Atmospheric conditions" - PSF and ADR for a single
    observation.

    Parameters
    ----------
    psfarr : ndarray (3-d)
        PSF as a function of wavelength - assumed to be spatially centered
        in array. The shape (nw, ny, nx) determines the shape of the model.
    adr_refract : ndarray (2-d)
        Array with shape (2, nw) where [0, :] corresponds to the refraction
        in y and [1, :] corresponds to the refraction in x at each wavelength.
    dtype : {np.float64, np.float32}, optional
        Data type used internally. Default is np.float64.
    fftw_threads : int, optional
        Threads used by FFTW.
    psf : callable, optional
        If specified, a callable that will be used to evaluate a point source
        on the image. The callable should accept three parameters:
        (shape, yctr, xctr), where shape is a 2-tuple giving the spatial shape
        of the output array, and yctr and xctr are 1-d arrays of length nw
        giving the center of the psf relative to the center of the output
        grid. The callable should return a 3-d array of shape (nw, ny, nx) with
        the PSF centered at (yctr[i], xctr[i]) in each slice, relative to the
        array center.

    """

    def __init__(self, psfarr, adr_refract, psf=None):
        self.shape = psfarr.shape
        self.nw, self.ny, self.nx = psfarr.shape
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
        self.fftconv = fft2(psfarr) * fshift
        self.fftconv = pyfftw.n_byte_align(self.fftconv, nbyte,
                                           dtype=np.complex128)

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
                                               dtype=np.complex128)
        self.fftout = pyfftw.n_byte_align_empty(self.shape, nbyte,
                                                dtype=np.complex128)

        self.fft = pyfftw.FFTW(self.fftin, self.fftout, axes=(1, 2),
                               threads=1)
        self.ifft = pyfftw.FFTW(self.fftout, self.fftin, axes=(1, 2),
                                threads=1, direction='FFTW_BACKWARD')

        self.fftnorm = 1. / (self.ny * self.nx) 

        # Set up evaluate_point_source().
        if psf is None:
            self.evaluate_point_source = self._evaluate_point_source_fft
        else:
            self.psf = psf
            self.adr_refract = adr_refract
            self.evaluate_point_source = self._evaluate_point_source_func

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

    def _evaluate_point_source_fft(self, pos, shape, ctr, grad=False):
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

    def _evaluate_point_source_func(self, pos, shape, ctr):
        """Use the psf attribute to evalute a point source at the given
        position.

        Parameters
        ----------
        pos : tuple
            (y, x) position of point source in model frame (relative to center).
        shape : tuple
            (y, x) shape of output.
        ctr : tuple
            (y, x) position of center of output frame relative to center of
            model frame.
        """

        # where is the SN in the data frame (relative to center)?
        y = pos[0] - ctr[0] + self.adr_refract[0, :]
        x = pos[1] - ctr[1] + self.adr_refract[1, :]

        return self.psf(shape, y, x)

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
        val = (self.mu_wave * np.sum(dw**2) +
               self.mu_xy * np.sum(dy**2) +
               self.mu_xy * np.sum(dx**2))

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
        grad[1:, :, :] += 2. * self.mu_wave * dw
        grad[:-1, :, :] -= 2. * self.mu_wave * dw
        grad[:, 1:, :] += 2. * self.mu_xy * dy
        grad[:, :-1, :] -= 2. * self.mu_xy * dy
        grad[:, :, 1:] += 2. * self.mu_xy * dx
        grad[:, :, :-1] -= 2. * self.mu_xy * dx

        grad /= self.mean_gal_spec[:, None, None]  # put back normalization

        return val, grad
