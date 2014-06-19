#!/usr/bin/env python
from distutils.core import setup

data_files=[]

description = ("Python implementation of the spectral data cube modeling "
               "code DDT")

authors = ["Seb Bongard", "Kyle Barbary", "Clare Saunders"]

classifiers = ["Topic :: Scientific/Engineering :: Astronomy",
               "Intended Audience :: Science/Research"]

setup(name="ddtpy", 
      version="0.1.0-dev",
      description=description,
      long_description=description,
      license="None",
      classifiers=classifiers,
      url="https://github.com/kbarbary/ddtpy",
      author=", ".join(authors),
      author_email="kylebarbary@gmail.com",
      requires=['numpy', 'scipy'],
      packages=['ddtpy'],
      scripts=['scripts/ddt'],
      data_files=data_files)
