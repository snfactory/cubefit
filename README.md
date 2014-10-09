ddtpy
=====

Python implementation of the spectral data cube modeling code DDT

The name `ddtpy` is mostly a placeholder to distinguish this from the
original DDT implementation, during development.


Dependencies
------------

Currently written for Python 2.7.

* numpy
* fitsio - https://github.com/esheldon/fitsio


**FFT libraries** - Currently we're just using `numpy.fft`. Note that
PyFFTW is a wrapper for `fftw` while `numpy.fft` or `scipy.fftpack`
are wrappers for other FFT libraries. FFTW is supposedly the
fastest. We're The original Yorick code uses a FFTW wrapper and we may
want to switch at some point.
