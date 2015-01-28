from __future__ import print_function, division

import os.path
from copy import deepcopy
import json

import numpy as np
from numpy import fft
import math

from .psf import params_from_gs, gaussian_plus_moffat_psf_4d
from .model import DDTModel
from .data import read_dataset, read_select_header_keys, DDTData
from .adr import paralactic_angle, differential_refraction
from .fitting import (guess_sky, fit_model, fit_position, fit_sky_and_sn,
                      fit_sky, fit_position_sn_sky)
from .extern import ADR

__all__ = ["main"]

SNIFS_LATITUDE = np.deg2rad(19.8228)

def main(filename, data_dir):
    """Do everything.

    Parameters
    ----------
    filename : str
        JSON-formatted config file.
    data_dir : str
        Directory containing FITS files given in the config file.
    """

    with open(filename) as f:
        conf = json.load(f)

    # check apodizer flag because the code doesn't support it
    if conf.get("FLAG_APODIZER", 0) >= 2:
        raise RuntimeError("FLAG_APODIZER >= 2 not implemented")

    spaxel_size = conf["PARAM_SPAXEL_SIZE"]
    
    # Reference wavelength. Used in PSF parameters and ADR.
    wave_ref = conf.get("PARAM_LAMBDA_REF", 5000.)

    # index of final ref. Subtract 1 due to Python zero-indexing.
    master_final_ref = conf["PARAM_FINAL_REF"] - 1

    # also want array with true/false for final refs.
    is_final_ref = np.array(conf.get("PARAM_IS_FINAL_REF"))

    n_iter_galaxy_prior = conf.get("N_ITER_GALAXY_PRIOR")

    # Load the header from the final ref or first cube
    fname = os.path.join(data_dir, conf["IN_CUBE"][master_final_ref])
    header = read_select_header_keys(fname)

    # Load data from list of FITS files.
    fnames = [os.path.join(data_dir, fname) for fname in conf["IN_CUBE"]]
    data, weight, wave = read_dataset(fnames)

    # Testing with only a couple wavelengths
    #data = data[:, 0:1, :, :]
    #weight = weight[:, 0:1, :, :]
    #wave = wave[0:1]

    # Zero-weight array elements that are NaN
    # TODO: why are there nans in here?
    mask = np.isnan(data)
    data[mask] = 0.0
    weight[mask] = 0.0

    # If target positions are given in the config file, set them in
    # the model.  (Otherwise, the positions default to zero in all
    # exposures.)
    if "PARAM_TARGET_XP" in conf:
        xctr_init = np.array(conf["PARAM_TARGET_XP"])
    else:
        xctr_init = np.zeros(ddtdata.nt)
    if "PARAM_TARGET_YP" in conf:
        yctr_init = np.array(conf["PARAM_TARGET_YP"])
    else:
        yctr_init = np.zeros(ddtdata.nt)

    # calculate all positions relative to master final ref
    xctr_init -= xctr_init[master_final_ref]
    yctr_init -= yctr_init[master_final_ref]
    sn_x_init = -xctr_init[master_final_ref]
    sn_y_init = -yctr_init[master_final_ref]

    ddtdata = DDTData(data, weight, wave, xctr_init, yctr_init,
                      is_final_ref, master_final_ref, header)

    # Load PSF model parameters. Currently, the PSF in the model is
    # represented by an arbitrary 4-d array that is constructed
    # here. The PSF depends on some aspects of the data, such as
    # wavelength.  If different types of PSFs need to do different
    # things, we may wish to represent the PSF with a class, called
    # something like GaussMoffatPSF.
    #
    # GS-PSF --> ES-PSF
    # G-PSF --> GR-PSF
    if conf["PARAM_PSF_TYPE"] == "GS-PSF":
        es_psf_params = np.array(conf["PARAM_PSF_ES"])
        psf_ellipticity, psf_alpha = params_from_gs(
            es_psf_params, ddtdata.wave, wave_ref)
    elif conf["PARAM_PSF_TYPE"] == "G-PSF":
        raise RuntimeError("G-PSF (from FITS files) not implemented")
    else:
        raise RuntimeError("unrecognized PARAM_PSF_TYPE")

    # The following section relates to atmospheric differential
    # refraction (ADR): Because of ADR, the center of the PSF will be
    # different at each wavelength, by an amount that we can determine
    # (pretty well) from the atmospheric conditions and the pointing
    # and angle of the instrument. We calculate the offsets here as a function
    # of observation and wavelength and input these to the model.

    # atmospheric conditions at each observation time.
    airmass = np.array(conf["PARAM_AIRMASS"])
    p = np.asarray(conf.get("PARAM_P", 615.*np.ones_like(airmass)))
    t = np.asarray(conf.get("PARAM_T", 2.*np.ones_like(airmass)))
    h = np.asarray(conf.get("PARAM_H", np.zeros_like(airmass)))

    # Position of the instrument
    ha = np.deg2rad(np.array(conf["PARAM_HA"]))   # config files in degrees
    dec = np.deg2rad(np.array(conf["PARAM_DEC"])) # config files in degrees
    tilt = conf["PARAM_MLA_TILT"]

    # differential refraction as a function of time and wavelength,
    # in arcseconds (2-d array).
    delta_r = differential_refraction(airmass, p, t, h, ddtdata.wave, wave_ref)
    delta_r /= spaxel_size  # convert from arcsec to spaxels
    pa = paralactic_angle(airmass, ha, dec, tilt, SNIFS_LATITUDE)
    adr_dx = np.zeros((ddtdata.nt, ddtdata.nw), dtype=np.float)
    adr_dy = np.zeros((ddtdata.nt, ddtdata.nw), dtype=np.float)


    for i_t in range(ddtdata.nt):
        adr = ADR(p[i_t], t[i_t], lref=wave_ref, airmass=airmass[i_t],
                  theta=pa[i_t])
        adr_refract = adr.refract(0, 0, wave, unit=spaxel_size)
        assert adr_refract.shape == (2, len(wave))
        adr_dx[i_t] = adr_refract[0]
        adr_dy[i_t] = adr_refract[1]

    # debug
    #adr_dx = -delta_r * np.sin(pa)[:, None]  # O'xp <-> - east
    #adr_dy = delta_r * np.cos(pa)[:, None]

    # Make a first guess at the sky level based on the data.
    skyguess = guess_sky(ddtdata, 2.0)

    # Calculate rough average galaxy spectrum from final refs
    # for use in regularization.
    refdata = ddtdata.data[ddtdata.is_final_ref]
    refdata -= skyguess[ddtdata.is_final_ref][:, :, None, None]
    mean_gal_spec = refdata.mean(axis=(0, 2, 3))

    # Initialize model
    model = DDTModel(ddtdata.nt, ddtdata.wave, psf_ellipticity, psf_alpha,
                     adr_dx, adr_dy, conf["MU_GALAXY_XY_PRIOR"]/10.,
                     conf["MU_GALAXY_LAMBDA_PRIOR"],
                     sn_x_init, sn_y_init, skyguess, mean_gal_spec)


    # Fit just the galaxy model to only the *master* final ref,
    # holding the sky fixed (logic to do that is inside
    # fit_model). The galaxy model is defined in the frame of the
    # master final ref.
    fit_model(model, ddtdata, [ddtdata.master_final_ref])

    # Test plotting
    from .plotting import plot_timeseries, plot_wave_slices
    fig = plot_timeseries(ddtdata, model)
    fig.savefig("testfigure.png")
    fig2 = plot_wave_slices(ddtdata, model, ddtdata.master_final_ref)
    fig2.savefig("testslices.png")
    #exit()

    # Fit registration on just the final refs
    # ==================================================================

    maxiter_fit_position = 100  # Max iterations in fit_position
    maxmove_fit_position = 3.0  # maxmimum movement allowed in fit_position
    mask_nmad = 2.5  # Minimum Number of Median Absolute Deviations above
                     # the minimum spaxel value in fit_position
    
    # Make a copy of the weight at this point, because we're going to
    # modify the weights in place for the next step.
    weight_orig = ddtdata.weight.copy()

    include_in_fit = np.ones(ddtdata.nt, dtype=np.bool)

    # /* Register the galaxy in the other final refs */
    # i_fit_galaxy_position=where(ddt.ddt_data.is_final_ref);
    # n_final_ref = numberof( i_fit_galaxy_position);
    # galaxy_offset = array(double, 2, ddt.ddt_data.n_t);
    
    # Loop over just the other final refs (not including master)
    is_other_final_ref = ddtdata.is_final_ref.copy()
    is_other_final_ref[ddtdata.master_final_ref] = False
    other_final_refs = np.flatnonzero(is_other_final_ref)

    for i_t in other_final_refs:
        
        m = model.evaluate(i_t, ddtdata.xctr[i_t], ddtdata.yctr[i_t],
                           (ddtdata.ny, ddtdata.nx), which='all')

        # The next few lines finds spaxels where the model is high and
        # sets the weight (in the data) for all other spaxels to zero

        tmp_m = m.sum(axis=0)  # Sum of model over wavelengths (result = 2-d)
        tmp_mad = np.median(np.abs(tmp_m - np.median(tmp_m)))

        # Spaxels where model is greater than minimum + 2.5 * MAD
        mask = tmp_m > np.min(tmp_m) + mask_nmad * tmp_mad
        
        # If there is less than 20 spaxels available, we don't fit
        # the position
        if mask.sum() < 20:
            continue

        # This sets weight to zero on spaxels where mask is False
        # (where spaxel sum is less than minimum + 2.5 MAD)
        ddtdata.weight[i_t] = ddtdata.weight[i_t] * mask[None, :, :]

        # Fit the position for this epoch, keeping the sky fixed.
        # TODO: should the sky be varied on each iteration?
        #       (currently, it is not varied)
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

    # Reset weight
    ddtdata.weight = weight_orig

    # Now that we have fit all their positions and reset the weights, 
    # recalculate sky for all other final refs.
    for i_t in other_final_refs:
        model.sky[i_t,:] = fit_sky(model, ddtdata, i_t)
        
    # Redo model fit, this time including all final refs.
    fit_model(model, ddtdata, np.flatnonzero(ddtdata.is_final_ref))


    fig = plot_timeseries(ddtdata, model)
    fig.savefig("testfigure2.png")
    for i_t in np.flatnonzero(ddtdata.is_final_ref):
        fig2 = plot_wave_slices(ddtdata, model, i_t)
        fig2.savefig("testslices_%s.png" % i_t)
    


    # list of non-final refs
    epochs = [i_t for i_t in range(ddtdata.nt)
              if not ddtdata.is_final_ref[i_t]]
    
    pos = fit_position_sn_sky(model, ddtdata, epochs)

    print(pos)

    # Fit registration on just exposures with a supernova (not final refs)
    # ===================================================================
    
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

    fig = plot_timeseries(ddtdata, model)
    fig.savefig("testfigure3.png")
    # Redo fit of galaxy
    # TODO: go back to DDT, check what should be fit here
    #fit_model(model, ddtdata)
    
