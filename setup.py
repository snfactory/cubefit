#!/usr/bin/env python
from distutils.core import setup

data_files=[]

description = ("Fit supernova + galaxy model on a Nearby Supernova Factory "
               "spectral data cube.")

authors = ["Seb Bongard", "Kyle Barbary", "Clare Saunders"]

classifiers = ["Topic :: Scientific/Engineering :: Astronomy",
               "Intended Audience :: Science/Research"]

setup(name="cubefit", 
      version="0.1.0-dev",
      description=description,
      license="MIT",
      classifiers=classifiers,
      url="https://github.com/snfactory/cubefit",
      author=", ".join(authors),
      author_email="kylebarbary@gmail.com",
      packages=['cubefit'],
      scripts=['scripts/cubefit','scripts/cubefit-plot'],
      data_files=data_files)
