import numpy as np
import fitsio

__all__ = ["read_select_header_keys", "read_dataset", "DDTData"]

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
        alldata[i+1] = data
        allweight[i+1] = weight

    return alldata, allweight, wave


class DDTData(object):
    """This class is the equivalent of `ddt.ddt_data` in the Yorick version.

    Parameters
    ----------
    data : ndarray (4-d)
    weight : ndarray (4-d)
    wave : ndarray (1-d)
    is_final_ref : ndarray (bool)
    master_final_ref : int
    header : dict
    spaxel_size : float
    """
    
    def __init__(self, data, weight, wave, is_final_ref, master_final_ref,
                 header, spaxel_size):

        if len(wave) != data.shape[1]:
            raise ValueError("length of wave must match data axis=1")

        if len(is_final_ref) != data.shape[0]:
            raise ValueError("length of is_final_ref and data must match")

        if data.shape != weight.shape:
            raise ValueError("shape of weight and data must match")

        self.data = data
        self.weight = weight
        self.wave = wave
        self.nt, self.nw, self.ny, self.nx = self.data.shape

        self.is_final_ref = is_final_ref
        self.master_final_ref = master_final_ref
        self.header = header
        self.spaxel_size = spaxel_size
        self.data_avg = data[self.is_final_ref].mean(axis=(0, 2, 3))

    def guess_sky(self, sig, maxiter=10):
        """Guess sky based on lower signal spaxels compatible with variance

        Parameters
        ----------
        sig : float
            Number of standard deviations (not variances) to use as
            the clipping limit (on individual pixels).
        maxiter : int
            Maximum number of sigma-clipping interations. Default is 10.

        Returns
        -------
        sky : np.ndarray (2-d)        
            Sky level for each epoch and wavelength. Shape is (nt, nw).
        """

        nspaxels = self.data.shape[2] * self.data.shape[3]
        sky = np.zeros((self.nt, self.nw), dtype=np.float64)

        for i in range(self.nt):
            data = self.data[i]
            weight = np.copy(self.weight[i])
            var = 1.0 / weight

            # Loop until ind stops changing size or until a maximum
            # number of iterations.
            avg = None
            oldmask = None
            mask = None
            for j in range(maxiter):
                oldmask = mask
                
                # weighted average spectrum (masked array).
                # We use a masked array because some of the wavelengths 
                # may have all-zero weights for every pixel.
                # The masked array gets propagated so that `mask` is a
                # masked array of booleans!
                avg = np.ma.average(data, weights=weight, axis=(1, 2))
                deviation = data - avg[:, None, None]
                mask = deviation**2 > sig**2 * var

                # Break if the mask didn't change.
                if (oldmask is not None and
                    (mask.data == oldmask.data).all() and
                    (mask.mask == oldmask.mask).all()):
                    break

                # set weights of masked pixels to zero. masked elements
                # of the mask are *not* changed.
                weight[mask] = 0.0
                var[mask] = 0.0

            # convert to normal (non-masked) array. Masked wavelengths are 
            # set to zero in this process.
            sky[i] = np.asarray(avg)

        return sky
