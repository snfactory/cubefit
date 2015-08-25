#!/usr/bin/env python
import os
from distutils.core import setup
from distutils.extension import Extension
import numpy

# Get __version__ from version.py without importing package itself.
with open('cubefit/version.py') as f:
    exec(f.read())

fname = os.path.join("cubefit", "psf.pyx")
USE_CYTHON = True
if not os.path.exists(fname):
    fname = fname.replace(".pyx", ".c")
    USE_CYTHON = False

exts = [Extension("cubefit.psf", [fname], include_dirs=[numpy.get_include()],
                  libraries=["m"])]
if USE_CYTHON:
    from Cython.Build import cythonize
    exts = cythonize(exts)

setup(name="cubefit", 
      version=__version__,
      description=("Fit combined supernova and galaxy model on a Nearby "
                   "Supernova Factory spectral data cube."),
      license="MIT",
      classifiers=["Topic :: Scientific/Engineering :: Astronomy",
                   "Intended Audience :: Science/Research"],
      url="https://github.com/snfactory/cubefit",
      author="Kyle Barbary, Seb Bongard, Clare Saunders",
      author_email="kylebarbary@gmail.com",
      packages=['cubefit'],
      ext_modules=exts,
      scripts=['scripts/cubefit',
               'scripts/cubefit-subtract',
               'scripts/cubefit-plot'])
