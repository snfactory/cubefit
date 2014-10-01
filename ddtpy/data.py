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

    def guess_sky(self, nsigma):
        """Guess sky based on lower signal spaxels compatible with variance

        Parameters
        ----------
        nsigma : float
            Number of sigma to cut on.

        Returns
        -------
        sky : np.ndarray (2-d)        
            Sky level for each epoch and wavelength. Shape is (nt, nw).
        """

        maxiter = 10
        sky = np.zeros((nt, nw), dtype=np.float64)

        for i_t in range(nt):
            data = self.data[i_t]
            weight = self.weight[i_t]

            var = 1.0 / weight
            ind = np.zeros(data.size)
            formersize = None
            nspaxels = data.shape[1] * data.shape[2]

            # Loop until ind stops changing size or we do too many iterations.
            niter = 0
            while (ind.size != formersize) and (niter < maxiter):
                formersize = ind.size

                I = (data * weight).sum(axis=(1, 2))
                J = weight.sum(axis=(1, 2))
                i_Iok = np.where(J != 0.0)

                # if there are any wavelengths where weight is not zero
                # (presumably there are)
                if i_Iok[0].size != 0:
                    I[i_Iok] /= weight.sum(axis=-1).sum(axis=-1)[i_Iok]
                    sigma = (var.sum(axis=-1).sum(axis=-1)/nxny)**0.5
                    ind = np.where(abs(data - I[:,None,None]) > 
                                   nsigma*sigma[:,None,None])
                    if ind[0].size != 0:
                        data[ind] = 0.
                        var[ind] = 0.
                        weight[ind] = 0.
                    else:
                        break
                else:
                    break
                niter += 1

            sky[i_t] = I

        return sky
