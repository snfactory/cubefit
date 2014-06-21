import numpy as np
import fitsio

def read_select_header_keys(filename):
    keys = ["OBJECT", "RA", "DEC", "EXPTIME", "EFFTIME", "AIRMASS",
            "LATITUDE", "HA", "TEMP", "PRESSURE", "CHANNEL", "PARANG",
            "DDTXP", "DDTYP"]

    f = fitsio.FITS(filename, "r")
    fullheader = f[0].read_header()
    f.close()
    
    return {key: fullheader.get(key, None) for key in keys}

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
