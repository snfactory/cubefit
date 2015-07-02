Test Set of SNe
===============

This directory contains a makefile useful for running debugging tests
on a subset of the SNe. You need to get the test data and unpack it
into a `data` directory in this directory. After you do this, the
directory should look like this:

```
cubefit/testset/data/NAME1/NAME1.json
                          /FILE1.fits
                          /FILE2.fits
                          /...
                    /NAME2/NAME2.json
                          /FILE1.fits
                          /FILE2.fits
                          /...
                    /...
```

Run cubefit on some SNe
-----------------------

```
make PTF09fox_B
```

or run on multiple SNe using 2 jobs:

```
make -j 2 PTF09fox_B PTF09fox_R
```

Finally, `make all` will run all of them.

Make plots
----------

```
make PTF09fox_B-plots
```