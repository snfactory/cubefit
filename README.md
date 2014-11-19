DDTPy
=====

Python implementation of the spectral data cube modeling code DDT

Dependencies
------------

Currently written for Python 2.7.

* numpy
* scipy (for optimization)
* fitsio - https://github.com/esheldon/fitsio

_Note on FFT library:_ Currently we're just using `numpy.fft` which
Ithink wraps FFTPACK. FFTW is supposedly the fastest available library
(we're talking factors of ~2 here). The original Yorick code uses a
FFTW wrapper. We may want to switch to using PyFFTW at some point.

Documentation
-------------

http://ddtpy.readthedocs.org

License
-------

MIT