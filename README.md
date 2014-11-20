DDTPy
=====

Python implementation of the spectral data cube modeling code DDT

Dependencies
------------

* Python 2.7
* numpy
* scipy (for optimization)
* fitsio - https://github.com/esheldon/fitsio

_Note on FFT library:_ Currently we're just using `numpy.fft` which
Ithink wraps FFTPACK. FFTW is supposedly the fastest available library
(we're talking factors of ~2 here). The original Yorick code uses a
FFTW wrapper. We may want to switch to using PyFFTW at some point.

Install & Tests
---------------

```
setup.py install
```

Running tests requires the pytest package (available via `pip` or
`conda`).  Execute `py.test` in the root of the source code repository
or `test` directory.

Documentation
-------------

http://ddtpy.readthedocs.org

License
-------

MIT
