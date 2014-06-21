from __future__ import print_function, division

import json

import numpy as np

from .psf import params_from_gs, gaussian_plus_moffat_psf_4d
from .io import read_dataset, read_select_header_keys

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

        # Initialize model
        self.model_gal = np.zeros((self.nw, self.psf_ny, self.psf_nx))
        self.model_galprior = np.zeros((self.nw, self.psf_ny, self.psf_nx))
        self.model_sky = np.zeros((self.nt, self.nw))
        self.model_sn = np.zeros((self.nt, self.nw))
        self.model_sn_x = (self.psf_nx + 1.) / 2.
        self.model_sn_y = (self.psf_ny + 1.) / 2.
        self.model_eta = np.ones(self.nt)
        self.model_final_ref_sky = np.zeros(self.nw)

    def __str__(self):
        """Create a string representation.

        For now this just lists the sizes of all the member arrays.
        """

        lines = ["Member arrays:"]
        for name in ["data", "weight", "wave", "psf", "psf_x", "psf_y",
                     "model_gal", "model_galprior", "model_sky", "model_sn",
                     "model_eta", "model_final_ref_sky"]:
            shape = getattr(self, name).shape
            lines.append("  {0:s} : {1:s}".format(name, shape))
        return "\n".join(lines)
