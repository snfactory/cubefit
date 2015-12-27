#!/usr/bin/env python
import os
import sys

from setuptools import setup
from setuptools.extension import Extension
from setuptools.command.test import test as TestCommand

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


class PyTest(TestCommand):
    """Enables setup.py test"""

    user_options = [('pytest-args=', 'a', "Arguments to pass to py.test")]

    def initialize_options(self):
        TestCommand.initialize_options(self)
        self.pytest_args = []

    def run_tests(self):
        #import here, because outside the eggs aren't loaded
        import pytest
        errno = pytest.main(self.pytest_args)
        sys.exit(errno)


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
      packages=['cubefit', 'cubefit.extern'],
      ext_modules=exts,
      scripts=['scripts/cubefit',
               'scripts/cubefit-subtract',
               'scripts/cubefit-plot'],
      cmdclass={'test': PyTest}
  )
