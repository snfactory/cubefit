from __future__ import print_function, division

from copy import deepcopy
import json

import numpy as np
from numpy import fft

from .psf import params_from_gs, gaussian_plus_moffat_psf_4d
from .model import DDTModel
from .data import read_dataset, read_select_header_keys, DDTData
from .adr import calc_paralactic_angle, differential_refraction
from .fitting import fit_model_all_epoch

__all__ = ["main"]


def main(filename):
    """Do everything.

    Parameters
    ----------
    filename : str
        JSON-formatted config file.
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
    i = master_final_ref if (master_final_ref >= 0) else 0
    header = read_select_header_keys(conf["IN_CUBE"][i])

    # Load data from list of FITS files.
    data, weight, wave = read_dataset(conf["IN_CUBE"])

    # Zero-weight array elements that are NaN
    # TODO: why are there nans in here?
    mask = np.isnan(data)
    data[mask] = 0.0
    weight[mask] = 0.0

    ddtdata = DDTData(data, weight, wave, is_final_ref, master_final_ref,
                      header, spaxel_size)

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
    paralactic_angle = calc_paralactic_angle(airmass, ha, dec, tilt)
    adr_dx = -delta_r * np.sin(paralactic_angle)[:, None]  # O'xp <-> - east
    adr_dy = delta_r * np.cos(paralactic_angle)[:, None]

    # Make a first guess at the sky level based on the data.
    sky = ddtdata.guess_sky(2.0)

    # Initialize model
    model = DDTModel((ddtdata.nt, ddtdata.nw), psf_ellipticity, psf_alpha,
                     adr_dx, adr_dy, spaxel_size,
                     conf["MU_GALAXY_XY_PRIOR"],
                     conf["MU_GALAXY_LAMBDA_PRIOR"], sky)

    # If target positions are given in the config file, set them in
    # the model.  (Otherwise, the positions default to zero in all
    # exposures.)
    if "PARAM_TARGET_XP" in conf:
        model.data_xctr[:] = np.array(conf["PARAM_TARGET_XP"])
    if "PARAM_TARGET_YP" in conf:
        model.data_yctr[:] = np.array(conf["PARAM_TARGET_YP"])



    # TODO : I Don't think this is needed anymore.                            
    # flag_apodizer = bool(conf.get("FLAG_APODIZER", 0))

    # Move the following operations to the model (when fitting?)
    # self.sn_offset_x_ref = deepcopy(self.target_xp)
    # self.sn_offset_y_ref = deepcopy(self.target_yp)
    # self.sn_offset_x = (self.sn_offset_x_ref[:, None] +  # 2-d arrays (nt, nw)
    #                     self.delta_r * self.adr_x[:, None])
    # self.sn_offset_y = (self.sn_offset_y_ref[:, None] +
    #                     self.delta_r * self.adr_y[:, None])
    
    fit = fit_model_all_epoch(model, ddtdata)
