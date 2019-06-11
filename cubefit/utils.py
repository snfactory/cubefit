"""General utilities."""

from __future__ import division

import numpy as np
from numpy import fft


def yxoffset(shape1, shape2, ctr):
    """y, x offset between two 2-d arrays (their lower-left corners)
    with shape1 and shape2, where the array centers are offset by ctr.
    
    Examples
    --------
    >>> yxoffset((32, 32), (15, 15), (0., 0.))
    (8.5, 8.5)
    >>> yxoffset((32, 32), (15, 15), (1., 0.))
    (9.5, 8.5)
    >>> yxoffset((32, 32), (15, 15), (0., 1.))
    (8.5, 9.5)

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

def yxbounds(shape1, shape2):
    """Bounds on the relative position of two arrays such that they overlap.

    Given the shapes of two arrays (second array smaller) return the range of
    allowed center offsets such that the second array is wholly contained in
    the first.

    Parameters
    ----------
    shape1 : tuple
        Shape of larger array.
    shape2 : tuple
        Shape of smaller array.

    Returns
    -------
    ybounds : tuple
         Length 2 tuple giving ymin, ymax.
    xbounds : tuple
         Length 2 tuple giving xmin, xmax.

    Examples
    --------
    >>> yxbounds((32, 32), (15, 15))
    (-8.5, 8.5), (-8.5, 8.5)

    """

    yd = (shape1[0] - shape2[0]) / 2.
    xd = (shape1[1] - shape2[1]) / 2.

    return (-yd, yd), (-xd, xd)


def fft_shift_phasor(n, d, grad=False):
    """Return a 1-d complex array of length `n` that, when mulitplied
    element-wise by another array in Fourier space, results in a shift
    by `d` in real space.

    Notes
    -----

    ABOUT THE NYQUIST FREQUENCY

    When fine shifting a sampled function by multiplying its FFT by a
    phase tilt, there is an issue about the Nyquist frequency for a
    real valued function when the number of samples N is even.  In
    this case, the Fourier transform of the function at Nyquist
    frequel N/2 must be real because the Fourier transform of the
    function is Hermitian and because +/- the Nyquist frequency is
    supported by the same array element.  When the shift is not a multiple
    of the sampling step size, the phasor violates this requirement.
    To solve for this issue with have tried different solutions:
 
      (a) Just take the real part of the phasor at the Nyquist
          frequency.  If FFTW is used, this is the same as doing
          nothing special with the phasor at Nyquist frequency.
 
      (b) Set the phasor to be 0 at the Nyquist frequency.
 
      (c) Set the phasor to be +/- 1 at the Nyquist frequency by
          rounding the phase at this frequency to the closest multiple
          of PI (same as rounding the shift to the closest integer
          value to compute the phasor at the Nyquist frequency).
 
      (d) Set the phasor to be 1 at the Nyquist frequency.
 
    The quality of the approximation can be assessed by computing
    the corresponding impulse response.
 
    The best method in terms of least amplitude of the side lobes is
    clearly (a), then (b), then (c) then (d).  The impulse response
    looks like a sinc for (a) and (b) whereas its shows strong
    distorsions around shifts equal to (k + 1/2)*s, where k is
    integer and s is the sample step size, for methods (c) and (d).
    The impulse response cancels at integer multiple of the sample
    step size for methods (a), (c) and (d) -- making them suitable as
    interpolation functions -- not for (b) which involves some
    smoothing (the phasor at Nyquist frequency is set to zero).
    """

    # fftfreq() gives frequency corresponding to each array element in
    # an FFT'd array (between -0.5 and 0.5). Multiplying by 2pi expands
    # this to (-pi, pi). Finally multiply by offset in array elements.
    f = 2. * np.pi * fft.fftfreq(n) * (d % n)

    result = np.cos(f) - 1j*np.sin(f)  # or equivalently: np.exp(-1j * f)

    # This is where we set the Nyquist frequency to be purely real (see above)
    if n % 2 == 0:
        result[int(n/2)] = np.real(result[int(n/2)])

    if grad:
        df = 2. * np.pi * fft.fftfreq(n)
        dresult = (-np.sin(f) -1j*np.cos(f)) * df
        if n % 2 == 0:
            dresult[n//2] = np.real(dresult[n//2])
        return result, dresult

    else:
        return result

def fft_shift_phasor_2d(shape, offset, grad=False):
    """Return phasor array used to shift an array (in real space) by
    multiplication in fourier space.
    
    Parameters
    ----------
    shape : (int, int)
        Length 2 iterable giving shape of array.
    offset : (float, float)
        Offset in array elements in each dimension.

    Returns
    -------
    z : np.ndarray (complex; 2-d)
        Complex array with shape ``shape``.

    """
    
    ny, nx = shape
    dy, dx = offset

    yphasor = fft_shift_phasor(ny, dy, grad=grad)
    xphasor = fft_shift_phasor(nx, dx, grad=grad)

    if grad:
        res = np.outer(yphasor[0], xphasor[0])
        dres_dy = np.outer(yphasor[1], xphasor[0])
        dres_dx = np.outer(yphasor[0], xphasor[1])
        resgrad = np.concatenate((dres_dy[None, :, :], dres_dx[None, :, :]))
        return res, resgrad

    else:
        return np.outer(yphasor, xphasor)
