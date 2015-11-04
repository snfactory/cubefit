"""Main entry points for scripts."""

from __future__ import print_function, division

from argparse import ArgumentParser
from collections import OrderedDict
from copy import copy
from datetime import datetime
import glob
import json
import logging
import math
import os

import numpy as np

from .version import __version__
from .psffuncs import gaussian_moffat_psf
from .psf import TabularPSF, GaussianMoffatPSF
from .io import read_datacube, write_results, read_results
from .fitting import (guess_sky, fit_galaxy_single, fit_galaxy_sky_multi,
                      fit_position_sky, fit_position_sky_sn_multi,
                      RegularizationPenalty)
from .extern import ADR, Hyper_PSF3D_PL


__all__ = ["cubefit", "setup_logging"]

MODEL_SHAPE = (32, 32)
SNIFS_LATITUDE = np.deg2rad(19.8228)
SPAXEL_SIZE = 0.43
MIN_NMAD = 2.5  # Minimum Number of Median Absolute Deviations above
                # the minimum spaxel value in fit_position
LBFGSB_FACTOR = 1e10
REFWAVE = 5000.  # reference wavelength in Angstroms for PSF params and ADR


def setup_logging(loglevel, logfname=None):

    # if loglevel isn't an integer, parse it as "debug", "info", etc:
    if not isinstance(loglevel, int):
        loglevel = getattr(logging, loglevel.upper(), None)
    if not isinstance(loglevel, int):
        print('Invalid log level: %s' % loglevel)
        exit(1)

    # remove logfile if it already exists
    if logfname is not None and os.path.exists(logfname):
        os.remove(logfname)

    logging.basicConfig(filename=logfname, format="%(levelname)s %(message)s",
                        level=loglevel)


def cubefit(argv=None):

    DESCRIPTION = "Fit SN + galaxy model to SNFactory data cubes."

    parser = ArgumentParser(prog="cubefit", description=DESCRIPTION)
    parser.add_argument("configfile",
                        help="configuration file name (JSON format)")
    parser.add_argument("outfile", help="Output file name (FITS format)")
    parser.add_argument("--dataprefix", default="",
                        help="path prepended to data file names; default is "
                        "empty string")
    parser.add_argument("--logfile", help="Write log to this file "
                        "(default: print to stdout)", default=None)
    parser.add_argument("--loglevel", default="info",
                        help="one of: debug, info, warning (default is info)")
    parser.add_argument("--diagdir", default=None,
                        help="If given, write intermediate diagnostic results "
                        "to this directory")
    parser.add_argument("--refitgal", default=False, action="store_true",
                        help="Add an iteration where galaxy model is fit "
                        "using all epochs and then data/SN positions are "
                        "refit")
    parser.add_argument("--mu_wave", default=0.07, type=float,
                        help="Wavelength regularization parameter. "
                        "Default is 0.07.")
    parser.add_argument("--mu_xy", default=0.001, type=float,
                        help="Spatial regularization parameter. "
                        "Default is 0.001.")
    parser.add_argument("--psftype", default="tabular",
                        help="Type of PSF: 'tabular' or 'gaussian-moffat'. "
                        "Currently, tabular means generate a tabular PSF from "
                        "gaussian-moffat parameters.")
    args = parser.parse_args(argv)

    setup_logging(args.loglevel, logfname=args.logfile)

    # record start time
    tstart = datetime.now()
    logging.info("cubefit v%s started at %s", __version__,
                 tstart.strftime("%Y-%m-%d %H:%M:%S"))
    tsteps = OrderedDict()  # finish time of each step.

    logging.info("parameters: mu_wave={:.3g} mu_xy={:.3g} refitgal={}"
                 .format(args.mu_wave, args.mu_xy, args.refitgal))
    logging.info("            psftype={}".format(args.psftype))

    logging.info("reading config file")
    with open(args.configfile) as f:
        cfg = json.load(f)

    # convert to radians (config file is in degrees)
    cfg["ha"] = np.deg2rad(cfg["ha"])
    cfg["dec"] = np.deg2rad(cfg["dec"])

    # basic checks on config contents.
    assert (len(cfg["filenames"]) == len(cfg["airmasses"]) ==
            len(cfg["pressures"]) == len(cfg["temperatures"]) ==
            len(cfg["xcenters"]) == len(cfg["ycenters"]) ==
            len(cfg["psf_params"]) == len(cfg["ha"]) == len(cfg["dec"]))

    # -------------------------------------------------------------------------
    # Load data cubes from the list of FITS files.

    nt = len(cfg["filenames"])

    logging.info("reading %d data cubes", nt)
    cubes = [read_datacube(os.path.join(args.dataprefix, fname))
             for fname in cfg["filenames"]]
    wave = cubes[0].wave
    wavewcs = cubes[0].wavewcs
    nw = len(wave)

    # assign some local variables for convenience
    refs = cfg["refs"]
    master_ref = cfg["master_ref"]
    if master_ref not in refs:
        raise ValueError("master ref choice must be one of the final refs (" +
                         " ".join(refs.astype(str)) + ")")
    nonmaster_refs = [i for i in refs if i != master_ref]
    nonrefs = [i for i in range(nt) if i not in refs]

    # Ensure that all cubes have the same wavelengths.
    if not all(np.all(cubes[i].wave == wave) for i in range(1, nt)):
        raise ValueError("all data must have same wavelengths")

    # -------------------------------------------------------------------------
    # PSF for each observation

    logging.info("setting up PSF for all %d epochs", nt)
    psfs = []
    for i in range(nt):

        # Get Gaussian+Moffat parameters at each wavelength.
        relwave = wave / REFWAVE - 1.0
        ellipticity = abs(cfg["psf_params"][i][0]) * np.ones_like(wave)
        alpha = np.abs(cfg["psf_params"][i][1] +
                       cfg["psf_params"][i][2] * relwave +
                       cfg["psf_params"][i][3] * relwave**2)

        # correlated parameters (coefficients determined externally)
        sigma = 0.545 + 0.215 * alpha  # Gaussian parameter
        beta  = 1.685 + 0.345 * alpha  # Moffat parameter
        eta   = 1.040 + 0.0   * alpha  # gaussian ampl. / moffat ampl.

        # Atmospheric differential refraction (ADR): Because of ADR,
        # the center of the PSF will be different at each wavelength,
        # by an amount that we can determine (pretty well) from the
        # atmospheric conditions and the pointing and angle of the
        # instrument. We calculate the offsets here as a function of
        # observation and wavelength and input these to the model.

        # Correction to parallactic angle and airmass for 2nd-order effects
        # such as MLA rotation, mechanical flexures or finite-exposure
        # corrections. These values have been trained on faint-std star
        # exposures.
        # 
        # TODO: get 'parang', 'airmass', 'channel' from database rather than
        # header for consistency with all other parameters.

        delta, theta = Hyper_PSF3D_PL.predict_adr_params(cubes[i].header)

        adr = ADR(cfg["pressures"][i], cfg["temperatures"][i], lref=REFWAVE,
                  delta=delta, theta=theta)
        adr_refract = adr.refract(0, 0, wave, unit=SPAXEL_SIZE)
        
        # adr_refract[0, :] corresponds to x, adr_refract[1, :] => y
        xctr, yctr = adr_refract

        # Tabular PSF
        if args.psftype == 'tabular':
            A = gaussian_moffat_psf(sigma, alpha, beta, ellipticity, eta,
                                    yctr, xctr, MODEL_SHAPE, subpix=1)
            psfs.append(TabularPSF(A))
        elif args.psftype == 'gaussian-moffat':
            psfs.append(GaussianMoffatPSF(sigma, alpha, beta, ellipticity, eta,
                                          yctr, xctr, MODEL_SHAPE, subpix=1))
        else:
            raise ValueError("unknown psf type: " + repr(args.psftype))

    # -------------------------------------------------------------------------
    # Initialize all model parameters to be fit

    galaxy = np.zeros((nw, MODEL_SHAPE[0], MODEL_SHAPE[1]), dtype=np.float64)
    sn = np.zeros((nt, nw), dtype=np.float64)  # SN spectrum at each epoch
    snctr = (0.0, 0.0)
    xctr = np.array(cfg["xcenters"])
    yctr = np.array(cfg["ycenters"])

    logging.info("guessing sky for all %d epochs", nt)
    skys = np.array([guess_sky(cube, npix=30) for cube in cubes])

    # -------------------------------------------------------------------------
    # Regularization penalty parameters

    # Calculate rough average galaxy spectrum from all final refs.
    spectra = np.zeros((len(refs), len(wave)), dtype=np.float64)
    for j, i in enumerate(refs):
        spectra[j] = np.average(cubes[i].data, axis=(1, 2)) - skys[i]
    mean_gal_spec = np.average(spectra, axis=0)

    galprior = np.zeros((nw, MODEL_SHAPE[0], MODEL_SHAPE[1]), dtype=np.float64)

    regpenalty = RegularizationPenalty(galprior, mean_gal_spec, args.mu_xy,
                                       args.mu_wave)

    tsteps["setup"] = datetime.now()

    # -------------------------------------------------------------------------
    # Fit just the galaxy model to just the master ref.

    data = cubes[master_ref].data - skys[master_ref][:, None, None]
    weight = cubes[master_ref].weight

    logging.info("fitting galaxy to master ref [%d]", master_ref)
    galaxy = fit_galaxy_single(galaxy, data, weight,
                               (yctr[master_ref], xctr[master_ref]),
                               psfs[master_ref], regpenalty, LBFGSB_FACTOR)

    if args.diagdir:
        fname = os.path.join(args.diagdir, 'step1.fits')
        write_results(galaxy, skys, sn, snctr, yctr, xctr,
                      cubes[0].data.shape, psfs, wavewcs, fname)

    tsteps["fit galaxy to master ref"] = datetime.now()

    # -------------------------------------------------------------------------
    # Fit the positions of the other final refs
    #
    # Here we only use spaxels where the *model* has significant flux.
    # We define "significant" as some number of median absolute deviations
    # (MAD) above the minimum flux in the model. We (temporarily) set the
    # weight of "insignificant" spaxels to zero during this process, then
    # restore the original weight after we're done.
    #
    # If there are less than 20 "significant" spaxels, we do not attempt to
    # fit the position, but simply leave it as is.

    logging.info("fitting position of non-master refs %s", nonmaster_refs)
    for i in nonmaster_refs:
        cube = cubes[i]

        # Evaluate galaxy on this epoch for purpose of masking spaxels.
        gal = psfs[i].evaluate_galaxy(galaxy, (cube.ny, cube.nx),
                                      (yctr[i], xctr[i]))

        # Set weight of low-valued spaxels to zero.
        gal2d = gal.sum(axis=0)  # Sum of gal over wavelengths
        mad = np.median(np.abs(gal2d - np.median(gal2d)))
        mask = gal2d > np.min(gal2d) + MIN_NMAD * mad
        if mask.sum() < 20:
            continue

        weight = cube.weight * mask[None, :, :]

        fctr, fsky = fit_position_sky(galaxy, cube.data, weight,
                                      (yctr[i], xctr[i]), psfs[i])
        yctr[i], xctr[i] = fctr
        skys[i] = fsky

    tsteps["fit positions of other refs"] = datetime.now()

    # -------------------------------------------------------------------------
    # Redo model fit, this time including all final refs.

    datas = [cubes[i].data for i in refs]
    weights = [cubes[i].weight for i in refs]
    ctrs = [(yctr[i], xctr[i]) for i in refs]
    psfs_refs = [psfs[i] for i in refs]
    logging.info("fitting galaxy to all refs %s", refs)
    galaxy, fskys = fit_galaxy_sky_multi(galaxy, datas, weights, ctrs,
                                         psfs_refs, regpenalty, LBFGSB_FACTOR)

    # put fitted skys back in `skys`
    for i,j in enumerate(refs):
        skys[j] = fskys[i]

    if args.diagdir:
        fname = os.path.join(args.diagdir, 'step2.fits')
        write_results(galaxy, skys, sn, snctr, yctr, xctr,
                      cubes[0].data.shape, psfs, wavewcs, fname)

    tsteps["fit galaxy to all refs"] = datetime.now()

    # -------------------------------------------------------------------------
    # Fit position of data and SN in non-references
    #
    # Now we think we have a good galaxy model. We fix this and fit
    # the relative position of the remaining epochs (which presumably
    # all have some SN light). We simultaneously fit the position of
    # the SN itself.

    logging.info("fitting position of all %d non-refs and SN position",
                 len(nonrefs))
    if len(nonrefs) > 0:
        datas = [cubes[i].data for i in nonrefs]
        weights = [cubes[i].weight for i in nonrefs]
        ctrs = [(yctr[i], xctr[i]) for i in nonrefs]
        psfs_nonrefs = [psfs[i] for i in nonrefs]
        fctrs, snctr, fskys, fsne = fit_position_sky_sn_multi(
            galaxy, datas, weights, ctrs, snctr, psfs_nonrefs, LBFGSB_FACTOR)

        # put fitted results back in parameter lists.
        for i,j in enumerate(nonrefs):
            skys[j, :] = fskys[i]
            sn[j, :] = fsne[i]
            yctr[j], xctr[j] = fctrs[i]

    tsteps["fit positions of nonrefs & SN"] = datetime.now()

    # -------------------------------------------------------------------------
    # optional step(s)

    if args.refitgal and len(nonrefs) > 0:

        if args.diagdir:
            fname = os.path.join(args.diagdir, 'step3.fits')
            write_results(galaxy, skys, sn, snctr, yctr, xctr,
                          cubes[0].data.shape, psfs, wavewcs, fname)

        # ---------------------------------------------------------------------
        # Redo fit of galaxy, using ALL epochs, including ones with SN
        # light.  We hold the SN "fixed" simply by subtracting it from the
        # data and fitting the remainder.
        #
        # This is slightly dangerous: any errors in the original SN
        # determination, whether due to an incorrect PSF or ADR model
        # or errors in the galaxy model will result in residuals. The
        # galaxy model will then try to compensate for these.
        #
        # We should look at the galaxy model at the position of the SN
        # before and after this step to see if there is a bias towards
        # the galaxy flux increasing.

        logging.info("fitting galaxy using all %d epochs", nt)
        datas = [cube.data for cube in cubes]
        weights = [cube.weight for cube in cubes]
        ctrs = [(yctr[i], xctr[i]) for i in range(nt)]

        # subtract SN from non-ref cubes.
        for i in nonrefs:
            s = psfs[i].point_source(snctr, datas[i].shape[1:3], ctrs[i])
            # do *not* use in-place operation (-=) here!
            datas[i] = cubes[i].data - sn[i, :, None, None] * s

        galaxy, fskys = fit_galaxy_sky_multi(galaxy, datas, weights, ctrs,
                                             psfs, regpenalty, LBFGSB_FACTOR)
        for i in range(nt):
            skys[i] = fskys[i]  # put fitted skys back in skys

        if args.diagdir:
            fname = os.path.join(args.diagdir, 'step4.fits')
            write_results(galaxy, skys, sn, snctr, yctr, xctr,
                          cubes[0].data.shape, psfs, wavewcs, fname)

        # ---------------------------------------------------------------------
        # Repeat step before last: fit position of data and SN in
        # non-references

        logging.info("re-fitting position of all %d non-refs and SN position",
                     len(nonrefs))
        datas = [cubes[i].data for i in nonrefs]
        weights = [cubes[i].weight for i in nonrefs]
        ctrs = [(yctr[i], xctr[i]) for i in nonrefs]
        psfs_nonrefs = [psfs[i] for i in nonrefs]
        fctrs, snctr, fskys, fsne = fit_position_sky_sn_multi(
            galaxy, datas, weights, ctrs, snctr, psfs_nonrefs, LBFGSB_FACTOR)

        # put fitted results back in parameter lists.
        for i,j in enumerate(nonrefs):
            skys[j] = fskys[i]
            sn[j, :] = fsne[i]
            yctr[j], xctr[j] = fctrs[i]

    # -------------------------------------------------------------------------
    # Write results

    logging.info("writing results to %s", args.outfile)
    write_results(galaxy, skys, sn, snctr, yctr, xctr,
                  cubes[0].data.shape, psfs, wavewcs, args.outfile)

    # time info
    logging.info("step times:")
    maxlen = max(len(key) for key in tsteps)
    fmtstr = "        %dm%02ds - %-" + str(maxlen) + "s"
    tprev = tstart
    for key, tstep in tsteps.items():
        t = (tstep - tprev).seconds
        logging.info(fmtstr, t//60, t%60, key)
        tprev = tstep

    tfinish = datetime.now()
    logging.info("finished at %s", tfinish.strftime("%Y-%m-%d %H:%M:%S"))
    t = (tfinish - tstart).seconds
    logging.info("took %dm%ds", t // 60, t % 60)


def cubefit_subtract(argv=None):
    DESCRIPTION = \
"""Subtract model determined by cubefit from the original data.

The "outnames" key in the supplied configuration file is used to
determine the output FITS file names. The input FITS header is passed
unaltered to the output file, with the following additions:
(1) A `HISTORY` entry. (2) `CBFT_SNX` and `CBFT_SNY` records giving
the cubefit-determined position of the SN relative to the center of
the data array (at the reference wavelength).

This script also writes fitted SN spectra to individual FITS files.
The "sn_outnames" configuration field determines the output filenames.
"""

    import shutil

    import fitsio

    prog_name = "cubefit-subtract"
    prog_name_ver = "{} v{}".format(prog_name, __version__)
    parser = ArgumentParser(prog=prog_name, description=DESCRIPTION)
    parser.add_argument("configfile", help="configuration file name "
                        "(JSON format), same as cubefit input.")
    parser.add_argument("resultfile", help="Result FITS file from cubefit")
    parser.add_argument("--dataprefix", default="",
                        help="path prepended to data file names; default is "
                        "empty string")
    parser.add_argument("--outprefix", default="",
                        help="path prepended to output file names; default is "
                        "empty string")
    args = parser.parse_args(argv)

    setup_logging("info")

    # get input & output filenames
    with open(args.configfile) as f:
        cfg = json.load(f)
    fnames = [os.path.join(args.dataprefix, fname)
              for fname in cfg["filenames"]]
    outfnames = [os.path.join(args.outprefix, fname)
                 for fname in cfg["outnames"]]

    # load results
    results = read_results(args.resultfile)
    epochs = results["epochs"]
    sny, snx = results["snctr"]
    if not len(epochs) == len(fnames) == len(outfnames):
        raise RuntimeError("number of epochs in result file not equal to "
                           "number of input and output files in config file")

    # subtract and write out.
    for fname, outfname, epoch in zip(fnames, outfnames, epochs):
        logging.info("writing %s", outfname)
        shutil.copy(fname, outfname)
        f = fitsio.FITS(outfname, "rw")
        data = f[0].read()
        data -= epoch["galeval"]
        f[0].write(data)
        f[0].write_history("galaxy subtracted by " + prog_name_ver)
        f[0].write_key("CBFT_SNX", snx - epoch['xctr'],
                       comment="SN x offset from center at ref wave [spaxels]")
        f[0].write_key("CBFT_SNY", sny - epoch['yctr'],
                       comment="SN y offset from center at ref wave [spaxels]")
        f.close()

    # output SN spectra to separate files.
    sn_outnames = [os.path.join(args.outprefix, fname)
                   for fname in cfg["sn_outnames"]]
    header = {"CRVAL1": results["wavewcs"]["CRVAL3"],
              "CRPIX1": results["wavewcs"]["CRPIX3"],
              "CDELT1": results["wavewcs"]["CDELT3"]}
    for outfname, epoch in zip(sn_outnames, epochs):
        logging.info("writing %s", outfname)
        if os.path.exists(outfname):  # avoid warning from clobber=True
            os.remove(outfname)
        with fitsio.FITS(outfname, "rw") as f:
            f.write(epoch["sn"], extname="sn", header=header)
            f[0].write_history("created by " + prog_name_ver)


def cubefit_plot(argv=None):
    DESCRIPTION = """Plot results and diagnostics from cubefit"""

    from .plotting import (plot_timeseries, plot_adr, plot_epoch, plot_sn,
                           plot_wave_slices)

    # arguments are the same as cubefit except an output 
    parser = ArgumentParser(prog="cubefit-plot", description=DESCRIPTION)
    parser.add_argument("configfile", help="configuration filename")
    parser.add_argument("resultfile", help="Result filename from cubefit")
    parser.add_argument("outprefix", help="output prefix")
    parser.add_argument("--dataprefix", default="",
                        help="path prepended to data file names; default is "
                        "empty string")
    parser.add_argument('-b', '--band', help='timeseries band (U, B, V). '
                        'Default is a 1000 A wide band in middle of cube.',
                        default=None, dest='band')
    parser.add_argument('--idrfiles', nargs='+', default=None,
                        help='Prefix of IDR. If given, the cubefit SN '
                        'spectra are plotted against the production values.')
    parser.add_argument("--diagdir", default=None,
                        help="If given, read intermediate diagnostic "
                        "results from this directory and include in plot(s)")
    parser.add_argument("--plotepochs", default=False, action="store_true",
                        help="Make diagnostic plots for each epoch")
    args = parser.parse_args(argv)

    # Read in data
    with open(args.configfile) as f:
        cfg = json.load(f)
    cubes = [read_datacube(os.path.join(args.dataprefix, fname), scale=False)
             for fname in cfg["filenames"]]

    results = OrderedDict()

    # Diagnostic results at each step
    if args.diagdir is not None:
        fnames = sorted(glob.glob(os.path.join(args.diagdir, "step*.fits")))
        for fname in fnames:
            name = os.path.basename(fname).split(".")[0]
            results[name] = read_results(fname)

    # Final result (don't fail if not available)
    if os.path.exists(args.resultfile):
        results["final"] = read_results(args.resultfile)

    # plot time series
    plot_timeseries(cubes, results, band=args.band,
                    fname=(args.outprefix + '_timeseries.png'))

    # Plot the x-y coordinates of the adr versus wavelength.
    plot_adr(cfg, cubes[0].wave, cubes, fname=(args.outprefix + '_adr.png'))

    # Plots that depend on final results being available.
    if 'final' in results:
        # plot wave slices, just for final refs.
        refcubes = [cubes[i] for i in cfg['refs']]
        refgaleval = results['final']['epochs']['galeval'][cfg['refs']]
        plot_wave_slices(refcubes, refgaleval,
                         fname=(args.outprefix + '_waveslice.png'))

        # Plot result spectra against IDR spectra.
        if args.idrfiles is not None:
            plot_sn(cfg['filenames'], results['final']['epochs']['sn'],
                    results['final']['wave'], args.idrfiles,
                    args.outprefix + '_sn.png')

        # Plot wave slices and sn, galaxy and sky spectra for all epochs.
        if args.plotepochs:
            for i_t in range(len(cubes)):
                plot_epoch(cubes[i_t], results['final']['epochs'][i_t],
                           fname=(args.outprefix + '_epoch%02d.png' % i_t))
