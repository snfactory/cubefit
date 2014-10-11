"""General numeric utilities."""

import numpy as np
from numpy import fft


def stringify(obj):
    """Create a string representation.

    For now this just lists all the members of the object and for arrays,
    their shapes.
    """

    lines = ["Members:"]
    for name, val in obj.__dict__.iteritems():
        if isinstance(val, np.ndarray):
            info = "{0:s} array".format(val.shape)
        else:
            info = repr(val)
        lines.append("  {0:s} : {1}".format(name, info))
    return "\n".join(lines)


def fft_shift_phasor_2d(shape, offset):
    """Return a complex phasor suitable to fine shift an array be OFFSET
    
    Parameters
    ----------
    shape : iterable (int)
        Length 2 iterable giving shape of array.
    offset : iterable (float)
        Offset in array elements in each dimension.

    Returns
    -------
    z : np.ndarray (complex)
        Complex array with shape ``shape``.

    Notes
    -----

    ABOUT THE NYQUIST FREQUENCY

    When fine shifting a sampled function by multiplying its FFT by a
    phase tilt, there is an issue about the Nyquist frequency for a
    real valued function when the number of samples N is even.  In
    this case, the Fourier transform of the function at Nyquist
    frequel N/2 must be real because the Fourier transform of the
    function is Hermitian and because +/- the Nyquist frequency is
    supported by the same frequel.  When the shift is not a multiple
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
    
    if not len(offset) == len(shape) == 2:
        raise ValueError("length of offset and shape must be 2")

    m, n = shape
    offm, offn = offset

    # fftfreq() gives frequency corresponding to each array element in
    # an FFT'd array (between -0.5 and 0.5). Multiplying by 2pi expands
    # this to (-pi, pi). Finally multiply by offset in array elements.
    fm = 2. * np.pi * fft.fftfreq(m) * (offm % m)
    fn = 2. * np.pi * fft.fftfreq(n) * (offn % n)

    phasorm =  np.cos(fm) - 1j*np.sin(fm)
    phasorn =  np.cos(fn) - 1j*np.sin(fn)

    # This is the part where we reset the phasor at the nyquist frequency
    # to be real (see comments above).
    if m % 2 == 0:
        phasorm[m/2] = np.real(phasorm[m/2])
    if n % 2 == 0:
        phasorn[n/2] = np.real(phasorn[n/2])

    return np.outer(phasorm, phasorn)

