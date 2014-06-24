from __future__ import print_function, division

import json

import numpy as np

from .psf import params_from_gs, gaussian_plus_moffat_psf_4d
from .io import read_dataset, read_select_header_keys
from .adr import calc_paralactic_angle, differential_refraction

__all__ = ["DDTData"]

class DDTData(object):
    """A class to hold DDT data."""

    def __init__(self, filename):
        """Initialize from a JSON-formatted file."""
        
        with open(filename) as f:
            conf = json.load(f)

        self.spaxel_size = conf["PARAM_SPAXEL_SIZE"]

        # index of final ref. Subtract 1 due to Python zero-indexing.
        self.final_ref = conf["PARAM_FINAL_REF"]-1

        # Load the header from the final ref or first cube
        idx = self.final_ref if (self.final_ref >= 0) else 0
        self.header = read_select_header_keys(conf["IN_CUBE"][idx])

        # Load data from FITS files.
        self.data, self.weight, self.wave = read_dataset(conf["IN_CUBE"])

        # save sizes for convenience
        self.nt = len(self.data)
        self.nw = len(self.wave)

        # Load PSF model
        # GS-PSF --> ES-PSF
        # G-PSF --> GR-PSF
        if conf["PARAM_PSF_TYPE"] == "GS-PSF":
            es_psf = np.array(conf["PARAM_PSF_ES"])
            ellipticity, alpha = params_from_gs(es_psf, self.wave)
            tmp = gaussian_plus_moffat_psf_4d(32, 32, ellipticity, alpha)
            self.psf_x, self.psf_y, self.psf = tmp
            self.psf_nx = len(self.psf_x)
            self.psf_ny = len(self.psf_y)
        elif conf["PARAM_PSF_TYPE"] == "G-PSF":
            raise RuntimeError("G-PSF (from FITS files) not implemented")
        else:
            raise RuntimeError("unrecognized PARAM_PSF_TYPE")

        # Initialize rest of model: galaxy, sky, SN
        self.model_gal = np.zeros((self.nw, self.psf_ny, self.psf_nx))
        self.model_galprior = np.zeros((self.nw, self.psf_ny, self.psf_nx))
        self.model_sky = np.zeros((self.nt, self.nw))
        self.model_sn = np.zeros((self.nt, self.nw))
        self.model_sn_x = (self.psf_nx + 1) // 2
        self.model_sn_y = (self.psf_ny + 1) // 2
        self.model_eta = np.ones(self.nt)
        self.model_final_ref_sky = np.zeros(self.nw)

        # atmospheric differential refraction
        airmass = np.array(conf["PARAM_AIRMASS"])
        ha = np.deg2rad(np.array(conf["PARAM_HA"]))   # config files in degrees
        dec = np.deg2rad(np.array(conf["PARAM_DEC"])) # config files in degrees
        tilt = conf["PARAM_MLA_TILT"]
        p = np.asarray(conf.get("PARAM_P", 615.*np.ones_like(airmass)))
        t = np.asarray(conf.get("PARAM_T", 2.*np.ones_like(airmass)))
        h = np.asarray(conf.get("PARAM_H", np.zeros_like(airmass)))
        wave_ref = conf.get("PARAM_LAMBDA_REF", 5000.)
        paralactic_angle = calc_paralactic_angle(airmass, ha, dec, tilt)
        delta_r = differential_refraction(airmass, p, t, h, self.wave,
                                          wave_ref)
        delta_r /= self.spaxel_size  # convert from arcsec to spaxels

        self.delta_r = delta_r
        self.adr_x = -1. * np.sin(paralactic_angle)
        self.adr_y = np.cos(paralactic_angle)
 
         # O'xp <-> - east
        self.delta_xp = -1. * delta_r * np.sin(paralactic_angle)[:, None]
        self.delta_yp = delta_r * np.cos(paralactic_angle)[:, None]
        self.paralactic_angle = paralactic_angle
        
        self.target_xp = np.asarray(conf.get("PARAM_TARGET_XP",
                                             np.zeros_like(airmass)))
        self.target_yp = np.asarray(conf.get("PARAM_TARGET_YP",
                                             np.zeros_like(airmass)))

        self.flag_apodizer = bool(conf.get("FLAG_APODIZER", 0))

        # TODO : setup FFT(s)

    def __str__(self):
        """Create a string representation.

        For now this just lists the sizes of all the member arrays.
        """

        lines = ["Members:"]
        for name, val in self.__dict__.iteritems():
            if isinstance(val, np.ndarray):
                info = "{0:s} array".format(val.shape)
            else:
                info = repr(val)
            lines.append("  {0:s} : {1}".format(name, info))
        return "\n".join(lines)
