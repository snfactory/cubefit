import numpy as np
import fitsio

__all__ = ["DataCube", "read_datacube"]

RESCALE = 10**17

class DataCube(object):
    """A container for data and weight arrays.

    Attributes
    ----------
    data : ndarray (3-d)
    weight : ndarray (3-d)
    wave : ndarray (1-d)
    nw : int
        length of wave, data.shape[0], weight.shape[0]
    ny : int
        data.shape[1], weight.shape[1]
    nx : int
        data.shape[2], weight.shape[2]
    """
    
    def __init__(self, data, weight, wave):
        if data.shape != weight.shape:
            raise ValueError("shape of weight and data must match")
        if len(wave) != data.shape[0]:
            raise ValueError("length of wave must match data axis=1")
        self.data = data
        self.weight = weight
        self.wave = wave
        self.nw, self.ny, self.nx = data.shape



def read_datacube(filename):
    """Read a two-HDU FITS file into memory.

    Assumes 1st HDU is data and 2nd HDU is variance.
    Scale data by 10**17 so that optimizer works better.

    Returns
    -------

    data : 3-d numpy array
    weight : 3-d numpy array
    wave : 1-d numpy array
        Wavelength coordinates along spectral (3rd) axis.
    """

    with fitsio.FITS(filename, "r") as f:
        header = f[0].read_header()
        data = f[0].read() * RESCALE
        variance = f[1].read() 

    n = header["NAXIS3"]
    crpix = header["CRPIX3"]-1.0  # FITS is 1-indexed, numpy as 0-indexed 
    crval = header["CRVAL3"]
    cdelt = header["CDELT3"]
    wave = crval + cdelt * (np.arange(n) - crpix)

    weight = 1. / variance
    weight /= RESCALE**2
    # Zero-weight array elements that are NaN
    # TODO: why are there nans in here?
    mask = np.isnan(data)
    data[mask] = 0.0
    weight[mask] = 0.0

    # TODO: check for variance <=0 - set weight to zero.
    #       (variance = Inf is OK - weight becomes 0)
    #       check for Inf in data? set data, weight to zero?

    return DataCube(data, weight, wave)
