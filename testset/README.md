Test Data Set
=============

This directory contains a Makefile useful for running debugging tests
on a set of test data. You need to get the test data and unpack it
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
make PTF09fox_B             # run one
make PTF09fox_B PTF09fox_R  # run two
make all                    # run all
```

Options (can be combined):

```
make -j 2 PTF09fox_B PTF09fox_R  # run in parallel on two cores
make LOGLEVEL=debug PTF09fox_B   # change the logging level
make LOGFILE=Y PTF09fox_B        # log to a file
```

Make plots
----------

```
make PTF09fox_B-plots
make PLOTEPOCHS=Y PTF09fox_B-plots  # make plots for individual epochs
```

To plot against the IDR, create a file `Make.user` in this directory,
with a single line giving the path to the root of the IDR, such as:

```
IDRPREFIX = ~/projects/cubefit/BEDELLv1
```
