from __future__ import print_function, division

import os.path
from copy import deepcopy
import json

import numpy as np
from numpy import fft
import math
import matplotlib as mpl
import pickle
mpl.use('Agg')

from .psf import params_from_gs, gaussian_plus_moffat_psf_4d
from .model import DDTModel
from .data import read_dataset, read_select_header_keys, DDTData
from .adr import paralactic_angle, differential_refraction
from .fitting import (guess_sky, fit_model, fit_position, fit_sky_and_sn,
                      fit_sky, fit_position_sn_sky)
from .extern import ADR

__all__ = ["main"]

MODEL_SHAPE = (32, 32)
SNIFS_LATITUDE = np.deg2rad(19.8228)
WAVE_REF_DEFAULT = 5000.
MAXITER_FIT_POSITION = 100  # Max iterations in fit_position
MAXMOVE_FIT_POSITION = 3.0  # maxmimum movement allowed in fit_position
MIN_NMAD = 2.5  # Minimum Number of Median Absolute Deviations above
                # the minimum spaxel value in fit_position

def parse_conf(inconf):
    """Parse the raw input configuration dictionary. Return a new dictionary.
    """

    outconf = {}

    outconf["fnames"] = inconf["IN_CUBE"]
    nt = len(outconf["fnames"])
    
    # check apodizer flag because the code doesn't support it
    if inconf.get("FLAG_APODIZER", 0) >= 2:
        raise RuntimeError("FLAG_APODIZER >= 2 not implemented")

    outconf["spaxel_size"] = inconf["PARAM_SPAXEL_SIZE"]
    outconf["wave_ref"] = inconf.get("PARAM_LAMBDA_REF", WAVE_REF_DEFAULT)  

    outconf["mu_xy"] = inconf["MU_GALAXY_XY_PRIOR"]
    outconf["mu_wave"] = inconf["MU_GALAXY_LAMBDA_PRIOR"]

    # index of master final ref. Subtract 1 for Python indexing.
    outconf["master_ref"] = inconf["PARAM_FINAL_REF"] - 1

    is_ref = inconf.get("PARAM_IS_FINAL_REF")
    assert len(is_ref) == nt
    refs = np.flatnonzero(is_ref)

    # indicies of all final refs
    outconf["refs"] = refs

    outconf["n_iter_galaxy_prior"] = inconf.get("N_ITER_GALAXY_PRIOR", None)

    # atmospheric conditions at each observation time.
    outconf["airmass"] = np.array(inconf["PARAM_AIRMASS"])
    outconf["p"] = np.asarray(inconf.get("PARAM_P", 615.*np.ones(nt)))
    outconf["t"] = np.asarray(inconf.get("PARAM_T", 2.*np.ones(nt)))
    outconf["h"] = np.asarray(inconf.get("PARAM_H", np.zeros(nt)))

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

    # In the input file the coordinates are w.r.t where we think the SN is
    # located. Shift them so that all the coordinates are w.r.t. the center
    # of the master final ref.
    xref = xctr_init[outconf["master_ref"]]
    yref = xctr_init[outconf["master_ref"]]
    outconf["xctr_init"] = xctr_init - xref
    outconf["yctr_init"] = yctr_init - yref
    outconf["sn_x_init"] = -xref
    outconf["sn_y_init"] = -yref

    return outconf


def main(filename, data_dir):
    """Do everything.

    Parameters
    ----------
    filename : str
        JSON-formatted config file.
    data_dir : str
        Directory containing FITS files given in the config file.
    """
    
    # Read the config file and parse it into a nice dictionary.
    with open(filename) as f:
        cfg = json.load(f)
        cfg = parse_conf(cfg)

    # TODO: This is a hack for the test sn. Generalize this?
    cfg["mu_xy"] = cfg["mu_xy"] / 10.

    # -------------------------------------------------------------------------
    # Load data cubes from the list of FITS files.

    cubes = [read_datacube(os.path.join(data_dir, fname))
             for fname in cfg["fnames"]]
    wave = cubes[0].wave
    nt = len(cubes)
    nw = len(wave)

    # assign some local variables for convenience
    refs = cfg["refs"]
    master_ref = cfg["master_ref"]
    nonmaster_refs = refs[refs != master_ref]
    nonrefs = [i for i in range(nt) if i not in refs]

    # Ensure that all cubes have the same wavelengths.
    if not all(cubes[i].wave == wave for i in range(1, len(cubes))):
        raise ValueError("all data must have same wavelengths")

    # -------------------------------------------------------------------------
    # Atmospheric conditions for each observation

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

        atms.append(AtmModel(psf, adr_refract))

    # -------------------------------------------------------------------------
    # Initialize all model parameters to be fit

    galaxy = np.zeros((nw, MODEL_SHAPE[0], MODEL_SHAPE[1]))
    skys = [guess_sky(cube, 2.0) for cube in cubes]
    sn = np.zeros((nt, nw))  # SN spectrum at each epoch
    sn_x = cfg["sn_x_init"]
    sn_y = cfg["sn_y_init"]
    xctr = np.copy(cfg["xctr_init"])
    yctr = np.copy(cfg["yctr_init"])

    # -------------------------------------------------------------------------
    # Regularization penalty parameters

    # Calculate rough average galaxy spectrum from all final refs.
    # TODO: use only spaxels that weren't masked in `guess_sky()`?
    spectra = np.zeros((len(refs), len(wave)))
    for i in refs:
        spectra[i] = np.average(cubes[i].data, axis=(1, 2)) - skys[i]
    mean_gal_spec = np.average(spectra, axis=0)

    galprior = np.zeros((nw, ny, nx))

    regpenalty = RegularizationPenalty(galprior, mean_gal_spec, mu_xy, mu_wave)

    # -------------------------------------------------------------------------
    # Fit just the galaxy model to just the master ref.

    data = cubes[master_ref].data - skys[master_ref][:, None, None]
    weight = cubes[master_ref].weight
    galaxy = fit_galaxy_single(galaxy, data, weight,
                               (yctr[master_ref], xctr[master_ref]),
                               atms[master_ref], regpenalty)

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

    exclude_from_fit = []
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

        fctr, sky[i] = fit_position_sky(galaxy, cube.data, weight,
                                        (yctr[i], xctr[i]), atms[i],
                                        MAXITER_FIT_POSITION)

        # Check if the position moved too much from initial position.
        # If it didn't move too much, update the model.
        # If it did, cut it from the fitting for the next step.
        dist = math.sqrt((fctr[0] - yctr[i])**2 + (fctr[1] - xctr[i])**2)
        if dist < MAXMOVE_FIT_POSITION:
            yctr[i] = fctr[0]
            xctr[i] = fctr[1]
        else:
            exclude_from_fit.append(i)


    # -------------------------------------------------------------------------
    # Redo model fit, this time including all final refs.

    datas = [cube.data[i] - skys[i][:, None, None] for i in refs]
    weights = [cube.weight[i] for i in refs]
    ctrs = [(yctr[i], xctr[i]) for i in refs]
    atms = [atms[i] for i in refs]
    galaxy = fit_galaxy_multi(galaxy, datas, weights, ctrs, atms, regpenalty)

    #pickle.dump(model, open('model2.pkl','w'))
    #fig = plot_timeseries(ddtdata, model)
    #fig.savefig("testfigure2.png")
    #fig.clear()
    #for i_t in np.flatnonzero(ddtdata.is_final_ref):
    #    fig2 = plot_wave_slices(ddtdata, model, i_t)
    #    fig2.savefig("testslices_%s.png" % i_t)
    #    fig2.clear()
    
    # Fit registration on just exposures with a supernova (not final refs)
    # ===================================================================

    # -------------------------------------------------------------------------
    # Fit position of data and SN in non-references
    #
    # Now we think we have a good galaxy model. We fix this and fit
    # the relative position of the remaining epochs (which presumably
    # all have some SN light). We simultaneously fit the position of
    # the SN itself.

    # `nonrefs` is indicies of non-final refs

    pos = fit_position_sn_sky(model, ddtdata, epochs)

    pickle.dump(model, open('model3.pkl','w'))
    pickle.dump(ddtdata, open('data3.pkl','w'))
    """
    model = pickle.load(open('model3.pkl','r'))
    ddtdata = pickle.load(open('data3.pkl','r'))
    """
    fig = plot_timeseries(ddtdata, model)
    fig.savefig("testfigure3.png")
    fig.clear()
    for i_t in epochs:
        fig2 = plot_wave_slices(ddtdata, model, i_t)
        fig2.savefig("testslices_%s.png" % i_t)
        fig2.clear()

    
    """
    for i_t in range(ddtdata.nt):
        if ddtdata.is_final_ref[i_t]:
            continue

        pos = fit_position(model, ddtdata, i_t)

        # Check if the position moved too much from initial position.
        # If it didn't move too much, update the model.
        # If it did, cut it from the fitting for the next step.
        dist = math.sqrt((pos[0] - ddtdata.xctr_init[i_t])**2 + 
                         (pos[1] - ddtdata.yctr_init[i_t])**2)
        
        if dist < maxmove_fit_position:
            ddtdata.xctr[i_t] = pos[0]
            ddtdata.yctr[i_t] = pos[1]
        else:
            include_in_fit[i_t] = False

        sky, sn = fit_sky_and_sn(model, ddtdata, i_t)
        model.sky[i_t,:] = sky
        model.sn[i_t,:] = sn
    """

    # Redo fit of galaxy
    # TODO: go back to DDT, check what should be fit here
    fit_model(model, ddtdata, np.arange(ddtdata.nt))
    fig = plot_timeseries(ddtdata, model)
    fig.savefig("testfigure4.png")
    
