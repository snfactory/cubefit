from __future__ import print_function, division

import os
import json
import math
import cPickle as pickle
from copy import copy
import logging
from datetime import datetime

import numpy as np

from .psf import psf_3d_from_params
from .core import AtmModel, RegularizationPenalty
from .io import read_datacube, write_results
from .adr import paralactic_angle
from .fitting import (guess_sky, fit_galaxy_single, fit_galaxy_sky_multi,
                      fit_position_sky, fit_position_sky_sn_multi)
from .extern import ADR

__all__ = ["main", "parse_conf", "setup_logging"]

MODEL_SHAPE = (32, 32)
SNIFS_LATITUDE = np.deg2rad(19.8228)
WAVE_REF_DEFAULT = 5000.
MIN_NMAD = 2.5  # Minimum Number of Median Absolute Deviations above
                # the minimum spaxel value in fit_position
DTYPE = np.float64
LBFGSB_FACTOR = 1e10

def parse_conf(inconf):
    """Parse the raw input configuration dictionary. Return a new dictionary.
    """

    outconf = {}

    outconf["fnames"] = inconf["IN_CUBE"]
    nt = len(outconf["fnames"])

    outconf["outfnames"] = inconf["OUT_DATACUBE_SUBTRACTION_FILE"]

    # check apodizer flag because the code doesn't support it
    if inconf.get("FLAG_APODIZER", 0) >= 2:
        raise RuntimeError("FLAG_APODIZER >= 2 not implemented")

    outconf["spaxel_size"] = inconf["PARAM_SPAXEL_SIZE"]
    outconf["wave_ref"] = inconf.get("PARAM_LAMBDA_REF", WAVE_REF_DEFAULT)

    # index of master final ref. Subtract 1 for Python indexing.
    outconf["master_ref"] = inconf["PARAM_FINAL_REF"] - 1

    is_ref = np.array(inconf.get("PARAM_IS_FINAL_REF"))
    assert len(is_ref) == nt
    refs = np.flatnonzero(is_ref)

    # indicies of all final refs
    outconf["refs"] = refs

    # atmospheric conditions at each observation time.
    outconf["airmass"] = np.array(inconf["PARAM_AIRMASS"])
    outconf["p"] = np.asarray(inconf.get("PARAM_P", 615.*np.ones(nt)))
    outconf["t"] = np.asarray(inconf.get("PARAM_T", 2.*np.ones(nt)))

    # Position of the instrument (note that config file is in degrees)
    outconf["ha"] = np.deg2rad(np.array(inconf["PARAM_HA"]))
    outconf["dec"] = np.deg2rad(np.array(inconf["PARAM_DEC"]))

    # TODO: check that this is in radians!
    outconf["tilt"] = inconf["PARAM_MLA_TILT"]

    if inconf["PARAM_PSF_TYPE"] != "GS-PSF":
        raise RuntimeError("unrecognized PARAM_PSF_TYPE. "
                           "[G-PSF (from FITS files) not implemented.]")
    outconf["psfparams"] = inconf["PARAM_PSF_ES"]

    # If target positions are given in the config file, set them in
    # the model.  (Otherwise, the positions default to zero in all
    # exposures.)
    if "PARAM_TARGET_XP" in inconf:
        xctr_init = np.array(inconf["PARAM_TARGET_XP"])
    else:
        xctr_init = np.zeros(nt)
    if "PARAM_TARGET_YP" in inconf:
        yctr_init = np.array(inconf["PARAM_TARGET_YP"])
    else:
        yctr_init = np.zeros(nt)

    # In the input file, we *think* the coordinates are where the center of
    # the reference is relative to the exposure.

    # First, negate the coordinates, so that they give the position of
    # the center of each exposure relative to the center of the
    # reference.
    xctr_init = -xctr_init
    yctr_init = -yctr_init

    # Now, we guess that the SN is located at approximately the
    # average center of all exposures (supposing that on average, we
    # pointed perfectly at the SN). Determine the average center
    # position:
    avgxctr = np.average(xctr_init)
    avgyctr = np.average(yctr_init)

    # Finally, subtract this average position, effectively shifting
    # the model reference frame to be centered on this location. Our
    # guess at the SN position is then centered in this reference
    # frame.
    outconf["xctr_init"] = xctr_init - avgxctr
    outconf["yctr_init"] = yctr_init - avgyctr
    outconf["sn_x_init"] = 0.
    outconf["sn_y_init"] = 0.

    return outconf


def setup_logging(loglevel, logfname=None):
    if logfname is None:
        logfmt = "\033[1m\033[34m%(levelname)s:\033[0m %(message)s"
    else:
        if os.path.exists(logfname):
            os.remove(logfname)
        logfmt = "%(asctime)s %(levelname)s %(message)s"

    logging.basicConfig(filename=logfname, format=logfmt,
                        level=loglevel, datefmt="%Y-%m-%dT%H:%M:%S")

    # record start time
    tstart = datetime.now()
    logging.info("%s started at %s", progname,
                 tstart.strftime("%Y-%m-%d %H:%M:%S"))


def main(configfname, outfname, dataprefix="", logfname=None,
         loglevel=logging.INFO, diagdir=None, refitgal=False,
         mu_wave=5.e-4, mu_xy=1.e-5, **kwargs):
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

    Additional keyword arugments override parameters in config file
    (after config file is parsed).
    """

    # Set up logging
    setup_logging(loglevel, logfname=logfname)

    # record start time
    tstart = datetime.now()
    logging.info("cubefit started at %s",
                 tstart.strftime("%Y-%m-%d %H:%M:%S"))


    # Read the config file and parse it into a nice dictionary.
    logging.info("reading config file")
    with open(configfname) as f:
        cfg = json.load(f)
        cfg = parse_conf(cfg)

    # Change any parameters that have been chosen at command line.
    # There is a check to ensure that we only try to set parameters that are
    # already in the config file.
    for key, val in kwargs.iteritems():
        if key not in cfg:
            raise RuntimeError("key not in configuration: " + repr(key))
        if val is not None:
            cfg[key] = val

    logging.info("parameters: mu_wave={:.3g} mu_xy={:.3g} refitgal={}"
                 .format(mu_wave, mu_xy, refitgal))

    # -------------------------------------------------------------------------
    # Load data cubes from the list of FITS files.

    nt = len(cfg["fnames"])

    logging.info("reading %d data cubes", nt)
    cubes = [read_datacube(os.path.join(dataprefix, fname), dtype=DTYPE)
             for fname in cfg["fnames"]]
    wave = cubes[0].wave
    wavewcs = cubes[0].wavewcs
    nw = len(wave)

    # assign some local variables for convenience
    refs = cfg["refs"]
    master_ref = cfg["master_ref"]
    if master_ref not in refs:
        raise ValueError("master ref choice must be one of the final refs (" +
                         " ".join(refs.astype(str)) + ")")
    nonmaster_refs = refs[refs != master_ref]
    nonrefs = [i for i in range(nt) if i not in refs]

    # Ensure that all cubes have the same wavelengths.
    if not all(np.all(cubes[i].wave == wave) for i in range(1, nt)):
        raise ValueError("all data must have same wavelengths")

    # -------------------------------------------------------------------------
    # Atmospheric conditions for each observation

    logging.info("setting up PSF and ADR for all %d epochs", nt)
    atms = []
    for i in range(nt):

        # Create a 3-d cube representing the Point Spread Function (PSF)
        # as a function of wavelength.
        psf = psf_3d_from_params(cfg["psfparams"][i], wave, cfg["wave_ref"],
                                 MODEL_SHAPE)

        # Atmospheric differential refraction (ADR): Because of ADR,
        # the center of the PSF will be different at each wavelength,
        # by an amount that we can determine (pretty well) from the
        # atmospheric conditions and the pointing and angle of the
        # instrument. We calculate the offsets here as a function of
        # observation and wavelength and input these to the model.
        pa = paralactic_angle(cfg["airmass"][i], cfg["ha"][i], cfg["dec"][i],
                              cfg["tilt"], SNIFS_LATITUDE)
        adr = ADR(cfg["p"][i], cfg["t"][i], lref=cfg["wave_ref"],
                  airmass=cfg["airmass"][i], theta=pa)
        adr_refract = adr.refract(0, 0, wave, unit=cfg["spaxel_size"])

        # make adr_refract[0, :] correspond to y and adr_refract[1, :] => x
        adr_refract = np.flipud(adr_refract)

        atms.append(AtmModel(psf, adr_refract, dtype=DTYPE, fftw_threads=1))

    # -------------------------------------------------------------------------
    # Initialize all model parameters to be fit

    galaxy = np.zeros((nw, MODEL_SHAPE[0], MODEL_SHAPE[1]), dtype=DTYPE)
    sn = np.zeros((nt, nw), dtype=DTYPE)  # SN spectrum at each epoch
    snctr = (cfg["sn_y_init"], cfg["sn_x_init"])
    xctr = np.copy(cfg["xctr_init"])
    yctr = np.copy(cfg["yctr_init"])

    logging.info("guessing sky for all %d epochs", nt)
    skys = np.array([guess_sky(cube, npix=20) for cube in cubes])

    # -------------------------------------------------------------------------
    # Regularization penalty parameters

    # Calculate rough average galaxy spectrum from all final refs.
    spectra = np.zeros((len(refs), len(wave)), dtype=DTYPE)
    for j, i in enumerate(refs):
        spectra[j] = np.average(cubes[i].data, axis=(1, 2)) - skys[i]
    mean_gal_spec = np.average(spectra, axis=0)

    galprior = np.zeros((nw, MODEL_SHAPE[0], MODEL_SHAPE[1]), dtype=DTYPE)

    regpenalty = RegularizationPenalty(galprior, mean_gal_spec, mu_xy, mu_wave)

    # -------------------------------------------------------------------------
    # Fit just the galaxy model to just the master ref.

    data = cubes[master_ref].data - skys[master_ref][:, None, None]
    weight = cubes[master_ref].weight

    logging.info("fitting galaxy to master ref [%d]", master_ref)
    galaxy = fit_galaxy_single(galaxy, data, weight,
                               (yctr[master_ref], xctr[master_ref]),
                               atms[master_ref], regpenalty, LBFGSB_FACTOR)

    if diagdir:
        fname = os.path.join(diagdir, 'step1.fits')
        write_results(galaxy, skys, sn, snctr, yctr, xctr,
                      cubes[0].data.shape, atms, wavewcs, fname)


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
        gal = atms[i].evaluate_galaxy(galaxy, (cube.ny, cube.nx),
                                      (yctr[i], xctr[i]))

        # Set weight of low-valued spaxels to zero.
        gal2d = gal.sum(axis=0)  # Sum of gal over wavelengths
        mad = np.median(np.abs(gal2d - np.median(gal2d)))
        mask = gal2d > np.min(gal2d) + MIN_NMAD * mad
        if mask.sum() < 20:
            continue

        weight = cube.weight * mask[None, :, :]

        fctr, fsky = fit_position_sky(galaxy, cube.data, weight,
                                      (yctr[i], xctr[i]), atms[i])
        yctr[i], xctr[i] = fctr
        skys[i] = fsky

    # -------------------------------------------------------------------------
    # Redo model fit, this time including all final refs.

    datas = [cubes[i].data for i in refs]
    weights = [cubes[i].weight for i in refs]
    ctrs = [(yctr[i], xctr[i]) for i in refs]
    atms_refs = [atms[i] for i in refs]
    logging.info("fitting galaxy to all refs %s", refs)
    galaxy, fskys = fit_galaxy_sky_multi(galaxy, datas, weights, ctrs,
                                         atms_refs, regpenalty, LBFGSB_FACTOR)

    # put fitted skys back in `skys`
    for i,j in enumerate(refs):
        skys[j] = fskys[i]

    if diagdir:
        fname = os.path.join(diagdir, 'step2.fits')
        write_results(galaxy, skys, sn, snctr, yctr, xctr,
                      cubes[0].data.shape, atms, wavewcs, fname)

    # -------------------------------------------------------------------------
    # Fit position of data and SN in non-references
    #
    # Now we think we have a good galaxy model. We fix this and fit
    # the relative position of the remaining epochs (which presumably
    # all have some SN light). We simultaneously fit the position of
    # the SN itself.

    logging.info("fitting position of all %d non-refs and SN position",
                 len(nonrefs))
    datas = [cubes[i].data for i in nonrefs]
    weights = [cubes[i].weight for i in nonrefs]
    ctrs = [(yctr[i], xctr[i]) for i in nonrefs]
    atms_nonrefs = [atms[i] for i in nonrefs]
    fctrs, snctr, fskys, fsne = fit_position_sky_sn_multi(galaxy, datas,
                                                          weights, ctrs,
                                                          snctr, atms_nonrefs)

    # put fitted results back in parameter lists.
    for i,j in enumerate(nonrefs):
        skys[j, :] = fskys[i]
        sn[j, :] = fsne[i]
        yctr[j], xctr[j] = fctrs[i]

    # -------------------------------------------------------------------------
    # optional step(s)

    if refitgal:

        if diagdir:
            fname = os.path.join(diagdir, 'step3.fits')
            write_results(galaxy, skys, sn, snctr, yctr, xctr,
                          cubes[0].data.shape, atms, wavewcs, fname)

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
            psf = atms[i].evaluate_point_source(snctr, datas[i].shape[1:3],
                                                ctrs[i])
            snpsf = sn[i, :, None, None] * psf  # scaled PSF
            datas[i] = cubes[i].data - snpsf  # do *not* use in-place op (-=)!

        galaxy, fskys = fit_galaxy_sky_multi(galaxy, datas, weights, ctrs,
                                             atms, regpenalty, LBFGSB_FACTOR)
        for i in range(nt):
            skys[i] = fskys[i]  # put fitted skys back in skys

        if diagdir:
            fname = os.path.join(diagdir, 'step4.fits')
            write_results(galaxy, skys, sn, snctr, yctr, xctr,
                          cubes[0].data.shape, atms, wavewcs, fname)

        # ---------------------------------------------------------------------
        # Repeat step before last: fit position of data and SN in
        # non-references

        logging.info("re-fitting position of all %d non-refs and SN position",
                     len(nonrefs))
        datas = [cubes[i].data for i in nonrefs]
        weights = [cubes[i].weight for i in nonrefs]
        ctrs = [(yctr[i], xctr[i]) for i in nonrefs]
        atms_nonrefs = [atms[i] for i in nonrefs]
        fctrs, snctr, fskys, fsne = fit_position_sky_sn_multi(
            galaxy, datas, weights, ctrs, snctr, atms_nonrefs)

        # put fitted results back in parameter lists.
        for i,j in enumerate(nonrefs):
            skys[j] = fskys[i]
            sn[j, :] = fsne[i]
            yctr[j], xctr[j] = fctrs[i]

    # -------------------------------------------------------------------------
    # Write results

    logging.info("writing results to %s", outfname)
    write_results(galaxy, skys, sn, snctr, yctr, xctr,
                  cubes[0].data.shape, atms, wavewcs, outfname)

    tfinish = datetime.now()
    logging.info("finished at %s", tfinish.strftime("%Y-%m-%d %H:%M:%S"))
    t = (tfinish - tstart).seconds
    logging.info("took %dm%ds", t // 60, t % 60)
