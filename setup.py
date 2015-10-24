#!/usr/bin/env python
import os
from setuptools import setup
from setuptools.extension import Extension
import numpy

# Get __version__ from version.py without importing package itself.
with open('cubefit/version.py') as f:
    exec(f.read())

fname = os.path.join("cubefit", "psffuncs.pyx")
USE_CYTHON = True
if not os.path.exists(fname):
    fname = fname.replace(".pyx", ".c")
    USE_CYTHON = False

exts = [Extension("cubefit.psffuncs", [fname],
                  include_dirs=[numpy.get_include()],
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
      entry_points={
          'console_scripts': [
              'cubefit = cubefit.main:cubefit',
              'cubefit-subtract = cubefit.main:cubefit_subtract',
              'cubefit-plot = cubefit.main:cubefit_plot'
          ]
      }
  )
