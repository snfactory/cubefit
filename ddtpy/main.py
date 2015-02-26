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

    # Load data from list of FITS files.
    fnames = [os.path.join(data_dir, fname) for fname in conf["IN_CUBE"]]
    cubes = [read_datacube(fname) for fname in fnames]
    wave = cubes[0].wave
    nt = len(cubes)
    nw = len(wave)

    # Ensure that all cubes have the same wavelengths.
    if not all(cubes[i].wave == wave for i in range(1, len(cubes))):
        raise ValueError("all data must have same wavelengths")

    # check apodizer flag because the code doesn't support it
    if conf.get("FLAG_APODIZER", 0) >= 2:
        raise RuntimeError("FLAG_APODIZER >= 2 not implemented")

    spaxel_size = conf["PARAM_SPAXEL_SIZE"]
    
    # Reference wavelength. Used in PSF parameters and ADR.
    wave_ref = conf.get("PARAM_LAMBDA_REF", 5000.)  

    mu_xy = conf["MU_GALAXY_XY_PRIOR"]/10.
    mu_wave = conf["MU_GALAXY_LAMBDA_PRIOR"]

    master_ref = conf["PARAM_FINAL_REF"] - 1  # index of master final
                                              # ref.  Subtract 1 for
                                              # Python indexing
    is_ref = conf.get("PARAM_IS_FINAL_REF")
    assert len(is_ref) == len(cubes)
    refs = np.flatnonzero(is_ref)  # indicies of all final refs
    nonmaster_refs = refs[refs != master_ref]  # indicies of all but master.

    n_iter_galaxy_prior = conf.get("N_ITER_GALAXY_PRIOR")

    # atmospheric conditions at each observation time.
    airmass = np.array(conf["PARAM_AIRMASS"])
    p = np.asarray(conf.get("PARAM_P", 615.*np.ones_like(airmass)))
    t = np.asarray(conf.get("PARAM_T", 2.*np.ones_like(airmass)))
    h = np.asarray(conf.get("PARAM_H", np.zeros_like(airmass)))

    # Position of the instrument
    ha = np.deg2rad(np.array(conf["PARAM_HA"]))   # config files in degrees
    dec = np.deg2rad(np.array(conf["PARAM_DEC"])) # config files in degrees
    tilt = conf["PARAM_MLA_TILT"]

    # PSF Parameters
    # GS-PSF --> ES-PSF
    # G-PSF --> GR-PSF
    if conf["PARAM_PSF_TYPE"] != "GS-PSF":
        raise RuntimeError("unrecognized PARAM_PSF_TYPE. "
                           "[G-PSF (from FITS files) not implemented.]")
    psfparams = conf["PARAM_PSF_ES"]


    # If target positions are given in the config file, set them in
    # the model.  (Otherwise, the positions default to zero in all
    # exposures.)
    if "PARAM_TARGET_XP" in conf:
        xctr_init = np.array(conf["PARAM_TARGET_XP"])
    else:
        xctr_init = np.zeros(nt)
    if "PARAM_TARGET_YP" in conf:
        yctr_init = np.array(conf["PARAM_TARGET_YP"])
    else:
        yctr_init = np.zeros(nt)

    # calculate all positions relative to master final ref
    xref = xctr_init[master_ref]
    yref = xctr_init[master_ref]
    xctr_init -= xref
    yctr_init -= yref
    sn_x_init = -xref
    sn_y_init = -yref

    # -------------------------------------------------------------------------
    # Point Spread Function (PSF)

    myctr = (MODEL_SHAPE[0] - 1) / 2.
    mxctr = (MODEL_SHAPE[1] - 1) / 2.
    psfs = []
    for i in range(len(psfparams)):
        psfs.append(psf_3d_from_params(psfparams[i], wave, wave_ref,
                                       MODEL_SHAPE, mxctr, myctr))

    # We now shift the PSF so that instead of being exactly
    # centered in the array, it is exactly centered on the lower
    # left pixel. We do this for using the PSF as a convolution kernel:
    # For convolution in Fourier space, the (0, 0) element of the kernel
    # is effectively the "center."
    # Note that this shifting is different than simply
    # creating the PSF centered at the lower left pixel to begin
    # with, due to wrap-around.
    #
    # mulitiplying an array by fftconv in fourier space will convolve the
    # array by the PSF. (`ifft2(fftconv).real` would be the shifted PSF in
    # real space.)
    fftconvs = []
    fshift = fft_shift_phasor_2d(MODEL_SHAPE, (-myctr, -mxctr))
    for psf in psfs:
        fftconv = np.empty(psf.shape, dtype=np.complex)
        for i in range(psf.shape[0]):
            fftconv[i, :, :] = fft2(psf[i, :, :]) * fshift
        fftconvs.append(fftconv)

    # -------------------------------------------------------------------------
    # Atmospheric differential refraction (ADR)

    # The following section relates to atmospheric differential
    # refraction (ADR): Because of ADR, the center of the PSF will be
    # different at each wavelength, by an amount that we can determine
    # (pretty well) from the atmospheric conditions and the pointing
    # and angle of the instrument. We calculate the offsets here as a function
    # of observation and wavelength and input these to the model.

    # OLD METHOD:
    # differential refraction as a function of time and wavelength,
    # in arcseconds (2-d array).
    #delta_r = differential_refraction(airmass, p, t, h, wave, wave_ref)
    #delta_r /= spaxel_size  # convert from arcsec to spaxels

    pa = paralactic_angle(airmass, ha, dec, tilt, SNIFS_LATITUDE)

    adr_dx = np.zeros((nt, nw))
    adr_dy = np.zeros((nt, nw))
    for i in range(nt):
        adr = ADR(p[i], t[i], lref=wave_ref, airmass=airmass[i], theta=pa[i])
        adr_refract = adr.refract(0, 0, wave, unit=spaxel_size)
        assert adr_refract.shape == (2, len(wave))
        adr_dx[i_t] = adr_refract[0]
        adr_dy[i_t] = adr_refract[1]

    # debug
    #adr_dx = -delta_r * np.sin(pa)[:, None]  # O'xp <-> - east
    #adr_dy = delta_r * np.cos(pa)[:, None]

    # -------------------------------------------------------------------------
    # Guesses

    # Make a first guess at the sky level based on the data.
    skyguesses = [guess_sky(cube, 2.0) for cube in cubes]

    # Calculate rough average galaxy spectrum from all final refs
    # for use in regularization.
    # TODO: use only spaxels that weren't masked in `guess_sky()`?
    spectra = np.zeros((len(refs), len(wave)))
    for i in refs:
        spectra[i] = np.average(cubes[i].data, axis=(1, 2)) - skyguesses[i]
    mean_gal_spec = np.average(spectra, axis=0)

    galprior = np.zeros((nw, ny, nx))

    # -------------------------------------------------------------------------
    # Model parameters

    galmodel = np.zeros((nw, MODEL_SHAPE[0], MODEL_SHAPE[1]))
    sn = np.zeros((nt, nw))  # SN spectrum at each epoch
    sn_x = sn_x_init
    sn_y = sn_y_init
    xctr = np.copy(xctr_init)
    yctr = np.copy(yctr_init)

    # Initialize model
    #model = DDTModel(ddtdata.nt, ddtdata.wave, psf_ellipticity, psf_alpha,
    #                 adr_dx, adr_dy, mu_xy, mu_wave,
    #                 sn_x_init, sn_y_init, skyguess, mean_gal_spec)

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
    fig.clear()
    fig2.clear()
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
    
    pickle.dump(model, open('model2.pkl','w'))
    fig = plot_timeseries(ddtdata, model)
    fig.savefig("testfigure2.png")
    fig.clear()
    for i_t in np.flatnonzero(ddtdata.is_final_ref):
        fig2 = plot_wave_slices(ddtdata, model, i_t)
        fig2.savefig("testslices_%s.png" % i_t)
        fig2.clear()
    
    

    # list of non-final refs
    epochs = [i_t for i_t in range(ddtdata.nt)
              if not ddtdata.is_final_ref[i_t]]
    
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

    # Redo fit of galaxy
    # TODO: go back to DDT, check what should be fit here
    fit_model(model, ddtdata, np.arange(ddtdata.nt))
    fig = plot_timeseries(ddtdata, model)
    fig.savefig("testfigure4.png")
    
