from __future__ import print_function, division

import os
import json
import math
from copy import copy
import logging
from datetime import datetime
from collections import OrderedDict

import numpy as np

from .version import __version__
from .psffuncs import gaussian_moffat_psf
from .psf import TabularPSF, GaussianMoffatPSF
from .io import read_datacube, write_results
from .adr import paralactic_angle
from .fitting import (guess_sky, fit_galaxy_single, fit_galaxy_sky_multi,
                      fit_position_sky, fit_position_sky_sn_multi,
                      RegularizationPenalty)
from .extern import ADR

__all__ = ["main", "setup_logging"]

MODEL_SHAPE = (32, 32)
SNIFS_LATITUDE = np.deg2rad(19.8228)
SPAXEL_SIZE = 0.43
MIN_NMAD = 2.5  # Minimum Number of Median Absolute Deviations above
                # the minimum spaxel value in fit_position
LBFGSB_FACTOR = 1e10
REFWAVE = 5000.  # reference wavelength in Angstroms for PSF params and ADR


def setup_logging(loglevel, logfname=None):
    if logfname is not None and os.path.exists(logfname):
        os.remove(logfname)
    logging.basicConfig(filename=logfname, format="%(levelname)s %(message)s",
                        level=loglevel)


def main(configfname, outfname, dataprefix="", logfname=None,
         loglevel=logging.INFO, diagdir=None, refitgal=False,
         mu_wave=0.07, mu_xy=0.001, psftype='tabular', **kwargs):
    """Run cubefit.

    Parameters
    ----------
    configfname : str
        Configuration file name (JSON format).
    outfname : str
        Output file name (FITS format).

    Optional Parameters
    -------------------
    dataprefix : str
        Path appended to data file names.
    logfname : str
        If supplied, write log to given file (otherwise print to stdout).
    loglevel : int
        One of logging.DEBUG, logging.INFO (default), logging.WARNING, etc.
    diagdir : str
        If given, write diagnostic output to this directory.
    refitgal : bool
        If true, run additional steps in algorithm.
    mu_wave, mu_xy : float
        Hyperparameters in wavelength and spatial directions.
    psftype : {'gaussian-moffat', 'tabular'}
        Type of psf

    Additional keyword arugments override parameters in config file
    (after config file is parsed).
    """

    setup_logging(loglevel, logfname=logfname)

    # record start time
    tstart = datetime.now()
    logging.info("cubefit v%s started at %s", __version__,
                 tstart.strftime("%Y-%m-%d %H:%M:%S"))
    tsteps = OrderedDict()  # finish time of each step.

    logging.info("parameters: mu_wave={:.3g} mu_xy={:.3g} refitgal={}"
                 .format(mu_wave, mu_xy, refitgal))

    # Read the config file and parse it into a nice dictionary.
    logging.info("reading config file")
    with open(configfname) as f:
        cfg = json.load(f)

    # convert to radians (config file is in degrees)
    cfg["ha"] = np.deg2rad(cfg["ha"])
    cfg["dec"] = np.deg2rad(cfg["dec"])

    # basic checks on config contents.
    assert (len(cfg["filenames"]) == len(cfg["airmasses"]) ==
            len(cfg["pressures"]) == len(cfg["temperatures"]) ==
            len(cfg["xcenters"]) == len(cfg["ycenters"]) ==
            len(cfg["psf_params"]) == len(cfg["ha"]) == len(cfg["dec"]))

    # Change any parameters that have been chosen at command line.
    # There is a check to ensure that we only try to set parameters that are
    # already in the config file.
    for key, val in kwargs.items():
        if key not in cfg:
            raise RuntimeError("key not in configuration: " + repr(key))
        if val is not None:
            cfg[key] = val

    # -------------------------------------------------------------------------
    # Load data cubes from the list of FITS files.

    nt = len(cfg["filenames"])

    logging.info("reading %d data cubes", nt)
    cubes = [read_datacube(os.path.join(dataprefix, fname))
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
        pa = paralactic_angle(cfg["airmasses"][i], cfg["ha"][i], cfg["dec"][i],
                              cfg["mla_tilt"], SNIFS_LATITUDE)
        adr = ADR(cfg["pressures"][i], cfg["temperatures"][i], lref=REFWAVE,
                  airmass=cfg["airmasses"][i], theta=pa)
        adr_refract = adr.refract(0, 0, wave, unit=SPAXEL_SIZE)
        
        # adr_refract[0, :] corresponds to x, adr_refract[1, :] => y
        xctr, yctr = adr_refract

        # Tabular PSF
        if psftype == 'tabular':
            A = gaussian_moffat_psf(sigma, alpha, beta, ellipticity, eta,
                                    yctr, xctr, MODEL_SHAPE, subpix=1)
            psfs.append(TabularPSF(A))
        elif psftype == 'gaussian-moffat':
            psfs.append(GaussianMoffatPSF(sigma, alpha, beta, ellipticity, eta,
                                          yctr, xctr, MODEL_SHAPE, subpix=1))
        else:
            raise ValueError("unknown psf type: " + repr(psftype))

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

    regpenalty = RegularizationPenalty(galprior, mean_gal_spec, mu_xy, mu_wave)

    tsteps["setup"] = datetime.now()

    # -------------------------------------------------------------------------
    # Fit just the galaxy model to just the master ref.

    data = cubes[master_ref].data - skys[master_ref][:, None, None]
    weight = cubes[master_ref].weight

    logging.info("fitting galaxy to master ref [%d]", master_ref)
    galaxy = fit_galaxy_single(galaxy, data, weight,
                               (yctr[master_ref], xctr[master_ref]),
                               psfs[master_ref], regpenalty, LBFGSB_FACTOR)

    if diagdir:
        fname = os.path.join(diagdir, 'step1.fits')
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

    if diagdir:
        fname = os.path.join(diagdir, 'step2.fits')
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

    if refitgal and len(nonrefs) > 0:

        if diagdir:
            fname = os.path.join(diagdir, 'step3.fits')
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

        if diagdir:
            fname = os.path.join(diagdir, 'step4.fits')
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

    logging.info("writing results to %s", outfname)
    write_results(galaxy, skys, sn, snctr, yctr, xctr,
                  cubes[0].data.shape, psfs, wavewcs, outfname)

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
