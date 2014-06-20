from __future__ import print_function

import json

import numpy as np
import fitsio

__all__ = ["DDTData"]

# TODO: move I/O stuff to separate module?

def read_datacube(filename):
    """Read a two-HDU FITS file into memory.

    Assumes 1st axis is data and 2nd axis is variance.

    Returns
    -------
    data : 3-d numpy array
    weight : 3-d numpy array
    wave : 1-d numpy array
        Wavelength coordinates along spectral (3rd) axis.
    """

    with fitsio.FITS(filename, "r") as f:
        header = f[0].read_header()
        data = f[0].read()
        variance = f[1].read()

    assert data.shape == variance.shape

    n = header["NAXIS3"]
    crpix = header["CRPIX3"]-1.0  # FITS is 1-indexed, numpy as 0-indexed 
    crval = header["CRVAL3"]
    cdelt = header["CDELT3"]
    wave = crval + cdelt * (np.arange(n) - crpix)

    weight = 1. / variance
    
    # TODO: check for variance <=0 - set weight to zero.
    #       (variance = Inf is OK - weight becomes 0)
    # TODO: check for NaN in weight, data, set weight to zero in these cases.
    #       check for Inf in data? set data, weight to zero?

    return data, weight, wave

def read_dataset(filenames):
    """Read multiple datacubes into 4-d arrays"""
    
    data, weight, wave = read_datacube(filenames[0])
    
    # Initialize output arrays.
    shape = (len(filenames),) + data.shape
    alldata = np.empty(shape, dtype=np.float)
    allweight = np.empty(shape, dtype=np.float)
    alldata[0] = data
    allweight[0] = weight

    for i, filename in enumerate(filenames[1:]):
        data, weight, _ = read_datacube(filename)
        alldata[i] = data
        allweight[i] = weight

    return alldata, allweight, wave

def psf_from_gs(psfdata, wave, spaxel_size):

    # only the 4 first elements of ES_PSF will be used, considering that
    # the caller knows which channel they are using.
    local_wave_ref = 5000. # From ES
  
    ellipticity = psfdata[:,0, np.newaxis] # still need to expand last axis to
                                           # match `wave`
    A0 = ES_PSF(2,..);
    A1 = ES_PSF(3,..);
    A2 = ES_PSF(4,..);

    wave_tmp = wave/local_wave_ref;
    #alpha = A0(-,..) + A1(-,..) * (lambda_tmp - 1) + A2(-,..) * (lambda_tmp - 1) * (lambda_tmp - 1); // From GS

}

class DDTData(object):
    """A class to hold DDT data"""

    def __init__(self, filename):
        """Initialize from a JSON-formatted file."""
        
        with open(filename) as f:
            conf = json.load(f)

        self.spaxel_size = conf["PARAM_SPAXEL_SIZE"]
        self.data, self.weight, self.wave = read_dataset(conf["IN_CUBE"])

        # Load PSF model
        # GS-PSF --> ES-PSF
        # G-PSF --> GR-PSF
        if conf["PARAM_PSF_TYPE"] == "GS-PSF":
            psfdata = np.array(conf["PARAM_PSF_ES"])
            psf = psf_from_gs(psfdata, self.wave, self.spaxel_size)
        elif conf["PARAM_PSF_TYPE"] == "G-PSF":
            psf = ddt_read_psf(conf["IN_PSF"], self.wave, self.spaxel_size)
        else:
            raise RuntimeError("unrecognized PARAM_PSF_TYPE")
