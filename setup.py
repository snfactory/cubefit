#!/usr/bin/env python
from distutils.core import setup

# Get __version__ from version.py without importing package itself.
with open('cubefit/version.py') as f:
    exec(f.read())

description = ("Fit supernova + galaxy model on a Nearby Supernova Factory "
               "spectral data cube.")

authors = ["Kyle Barbary", 
           "Seb Bongard",
           "Clare Saunders"]

classifiers = ["Topic :: Scientific/Engineering :: Astronomy",
               "Intended Audience :: Science/Research"]

setup(name="cubefit", 
      version=__version__,
      description=description,
      license="MIT",
      classifiers=classifiers,
      url="https://github.com/snfactory/cubefit",
      author=", ".join(authors),
      author_email="kylebarbary@gmail.com",
      packages=['cubefit'],
      scripts=['scripts/cubefit',
               'scripts/cubefit-subtract',
               'scripts/cubefit-plot'])
