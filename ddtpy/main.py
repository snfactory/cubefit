from __future__ import print_function, division

import os.path
import json
import math
import cPickle as pickle
from copy import copy

import numpy as np

from .psf import psf_3d_from_params
from .model import AtmModel, RegularizationPenalty
from .data import read_datacube
from .adr import paralactic_angle
from .fitting import (guess_sky, fit_galaxy_single, fit_galaxy_sky_multi,
                      fit_position_sky, fit_position_sky_sn_multi)
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

    is_ref = np.array(inconf.get("PARAM_IS_FINAL_REF"))
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
    outconf["xctr_init"] = -(xctr_init - xref)
    outconf["yctr_init"] = -(yctr_init - yref)
    outconf["sn_x_init"] = -(-xref)
    outconf["sn_y_init"] = -(-yref)

    return outconf


def result_dict(galaxy, skys, sn, snctr, yctr, xctr, dshape, atms):
    """Package parameters into a dictionary for saving to a pickle file."""

    # evaluate galaxy & PSF on data
    galeval = []
    psfeval = []
    for i in range(len(atms)):
        tg = atms[i].evaluate_galaxy(galaxy, dshape, (yctr[i], xctr[i]))
        galeval.append(tg.copy())
        tp = atms[i].evaluate_point_source(snctr, dshape, (yctr[i], xctr[i]))
        psfeval.append(tp.copy())
    
    return {'galaxy' : galaxy, 'snctr' : snctr,
            'ctrs' : zip(copy(yctr), copy(xyctr)),
            'skys' : copy(skys), 'sn' : sn.copy(),
            'galeval': galeval,
            'psfeval': psfeval}


def main(filename, data_dir, output_filename):
    """Do everything.

    Parameters
    ----------
    filename : str
        JSON-formatted config file.
    data_dir : str
        Directory containing FITS files given in the config file.
    output_filename : str
        File to write output to (currently in pickle form).
    """

    output_dict = {}
    
    # Read the config file and parse it into a nice dictionary.
    with open(filename) as f:
        cfg = json.load(f)
        cfg = parse_conf(cfg)

    # -------------------------------------------------------------------------
    # Load data cubes from the list of FITS files.

    cubes = [read_datacube(os.path.join(data_dir, fname))
             for fname in cfg["fnames"]]
    wave = cubes[0].wave
    nt = len(cubes)
    nw = len(wave)
    output_dict['Data'] = cubes

    # assign some local variables for convenience
    refs = cfg["refs"]
    master_ref = cfg["master_ref"]
    nonmaster_refs = refs[refs != master_ref]
    nonrefs = [i for i in range(nt) if i not in refs]
    output_dict['Refs'] = refs

    # Ensure that all cubes have the same wavelengths.
    if not all(np.all(cubes[i].wave == wave) for i in range(1, nt)):
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

        atms.append(AtmModel(psf, adr_refract, fftw_threads=1))

    # -------------------------------------------------------------------------
    # Initialize all model parameters to be fit

    galaxy = np.zeros((nw, MODEL_SHAPE[0], MODEL_SHAPE[1]))
    skys = [guess_sky(cube, 2.0) for cube in cubes]
    sn = np.zeros((nt, nw))  # SN spectrum at each epoch
    snctr = (cfg["sn_y_init"], cfg["sn_x_init"])
    xctr = np.copy(cfg["xctr_init"])
    yctr = np.copy(cfg["yctr_init"])

    # -------------------------------------------------------------------------
    # Regularization penalty parameters

    # Calculate rough average galaxy spectrum from all final refs.
    # TODO: use only spaxels that weren't masked in `guess_sky()`?
    spectra = np.zeros((len(refs), len(wave)))
    for j, i in enumerate(refs):
        spectra[j] = np.average(cubes[i].data, axis=(1, 2)) - skys[i]
    mean_gal_spec = np.average(spectra, axis=0)

    galprior = np.zeros((nw, MODEL_SHAPE[0], MODEL_SHAPE[1]))

    regpenalty = RegularizationPenalty(galprior, mean_gal_spec, cfg["mu_xy"],
                                       cfg["mu_wave"])

    # -------------------------------------------------------------------------
    # Fit just the galaxy model to just the master ref.
    
    data = cubes[master_ref].data - skys[master_ref][:, None, None]
    weight = cubes[master_ref].weight
    galaxy = fit_galaxy_single(galaxy, data, weight,
                               (yctr[master_ref], xctr[master_ref]),
                               atms[master_ref], regpenalty)
    
    output_dict['MasterRefFit'] = result_dict(galaxy, skys, sn, snctr,
                                              yctr, xctr,
                                              (cubes[0].ny, cubes[0].nx), atms)

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

        fctr, skys[i] = fit_position_sky(galaxy, cube.data, weight,
                                         (yctr[i], xctr[i]), atms[i])

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

    datas = [cubes[i].data for i in refs]
    weights = [cubes[i].weight for i in refs]
    ctrs = [(yctr[i], xctr[i]) for i in refs]
    atms_refs = [atms[i] for i in refs]
    galaxy, fskys = fit_galaxy_sky_multi(galaxy, datas, weights, ctrs,
                                         atms_refs, regpenalty)

    # put fitted skys back in `skys`
    for i,j in enumerate(refs):
        skys[j] = fskys[i]

    output_dict['AllRefFit'] = result_dict(galaxy, skys, sn, snctr,
                                           yctr, xctr,
                                           (cubes[0].ny, cubes[0].nx), atms)

        
    # -------------------------------------------------------------------------
    # Fit position of data and SN in non-references
    #
    # Now we think we have a good galaxy model. We fix this and fit
    # the relative position of the remaining epochs (which presumably
    # all have some SN light). We simultaneously fit the position of
    # the SN itself.

    datas = [cubes[i].data for i in nonrefs]
    weights = [cubes[i].weight for i in nonrefs]
    ctrs = [(yctr[i], xctr[i]) for i in nonrefs]
    atms_nonrefs = [atms[i] for i in nonrefs]
    fctrs, snctr, fskys, fsne = fit_position_sky_sn_multi(galaxy, datas,
                                                          weights, ctrs,
                                                          snctr, atms_nonrefs)

    # put fitted results back in parameter lists.
    for i,j in enumerate(nonrefs):
        skys[j] = fskys[i]
        sn[j, :] = fsne[i]
        yctr[j], xctr[j] = fctrs[i]

    output_dict['FinalFit'] = result_dict(galaxy, skys, sn, snctr,
                                          yctr, xctr,
                                          (cubes[0].ny, cubes[0].nx), atms)

    # -------------------------------------------------------------------------
    # Redo fit of galaxy, using ALL epochs.

    # TODO: go back to DDT, check what should be fit here

    # -------------------------------------------------------------------------
    # Dump results dictionary to pickle.
    
    if os.path.exists(output_filename) and os.path.isfile(output_filename):
        os.remove(output_filename)
    output_file = open(output_filename, 'wb')
    pickle.dump(output_dict, output_file, protocol=2)
    output_file.close()
