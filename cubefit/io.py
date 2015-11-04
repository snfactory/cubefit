import logging
import os

import numpy as np
import fitsio

from .version import __version__

__all__ = ["DataCube", "read_datacube", "write_results", "read_results"]

SCALE_FACTOR = 1.e17


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
    
    def __init__(self, data, weight, wave, wavewcs=None, header=None):
        if data.shape != weight.shape:
            raise ValueError("shape of weight and data must match")
        if len(wave) != data.shape[0]:
            raise ValueError("length of wave must match data axis=1")
        if wavewcs is None:
            wavewcs = {}

        self.data = data
        self.weight = weight
        self.wave = wave
        self.nw, self.ny, self.nx = data.shape
        self.wavewcs = wavewcs
        self.header = header

def wcs_to_wave(hdr):
    pixcoords = np.arange(1., hdr["NAXIS3"] + 1.)  # FITS is 1-indexed
    wave = hdr["CRVAL3"] + hdr["CDELT3"] * (pixcoords - hdr["CRPIX3"])

    return wave


def read_datacube(filename, scale=True):
    """Read a two-HDU FITS file into memory.

    Assumes 1st HDU is data and 2nd HDU is variance.

    Parameters
    ----------
    filename : str
    scale : bool, opitonal
        Whether to scale the data by a universal scaling constant
        (global parameter).

    Returns
    -------
    cube : DataCube
        Object holding data, weight and wavelength arrays. dtype of internal
        arrays will match the storage datatype or the dtype keyword, if given.
    """

    with fitsio.FITS(filename, "r") as f:
        hdr = f[0].read_header()
        data = np.asarray(f[0].read(), dtype=np.float64)
        variance = np.asarray(f[1].read(), dtype=np.float64)

    wave = wcs_to_wave(hdr)
    weight = 1. / variance

    # Zero-weight array elements that are NaN
    mask = np.isnan(data) | np.isnan(weight)
    data[mask] = 0.0
    weight[mask] = 0.0

    # scale data so that optimizers work better. Results are descaled
    # when written out.
    if scale:
        data *= SCALE_FACTOR
        weight /= SCALE_FACTOR**2

    # TODO: check for variance <=0 - set weight to zero.
    #       (variance = Inf is OK - weight becomes 0)
    #       check for Inf in data? set data, weight to zero?

    # save the raw wavelength WCS solution for eventually writing out results.
    wavewcs = {"CRVAL3": hdr["CRVAL3"],
               "CRPIX3": hdr["CRPIX3"],
               "CDELT3": hdr["CDELT3"]}

    return DataCube(data, weight, wave, wavewcs=wavewcs, header=hdr)


def epoch_results(galaxy, skys, sn, snctr, yctr, xctr, dshape, psfs):
    """Package all by-epoch results into a single numpy structured array,
    amenable to writing out to a FITS file.

    Note that this format assumes that the data for all epochs has the same
    spatial shape. A different format would have to be used if this were not
    the case.
    """

    # This is a table with `nt` rows
    nt = len(psfs)
    dtype = [('yctr', 'f8'),
             ('xctr', 'f8'),
             ('sn', 'f4', sn.shape[1]),
             ('sky', 'f4', len(skys[0])),
             ('galeval', 'f4', dshape),
             ('sneval', 'f4', dshape)]

    epochs = np.zeros(nt, dtype=dtype)
    epochs['yctr'] = yctr
    epochs['xctr'] = xctr
    epochs['sn'] = sn
    epochs['sky'] = skys

    # evaluate galaxy & PSF on data
    for i in range(nt):
        epochs['galeval'][i] = psfs[i].evaluate_galaxy(galaxy, dshape[1:3],
                                                       (yctr[i], xctr[i]))
        epochs['sneval'][i] = psfs[i].evaluate_point_source(snctr, dshape[1:3],
                                                            (yctr[i], xctr[i]))

    # multiply by sn amplitude
    epochs['sneval'] *= sn[:, :, None, None]

    return epochs


def write_results(galaxy, skys, sn, snctr, yctr, xctr, dshape, psfs, wavewcs,
                  fname, descale=True):
    """Write results to a FITS file."""

    if descale:
        galaxy = galaxy / SCALE_FACTOR  # note: do NOT use in-place ops here!
        skys = skys / SCALE_FACTOR
        sn = sn / SCALE_FACTOR

    # Create epochs table.
    epochs = epoch_results(galaxy, skys, sn, snctr, yctr, xctr, dshape, psfs)

    if os.path.exists(fname):  # avoids warning doing FITS(..., clobber=True)
        os.remove(fname)

    with fitsio.FITS(fname, "rw") as f:
        f.write(galaxy, extname="galaxy", header=wavewcs)
        f[0].write_history("created by cubefit v" + __version__)
        f.write(epochs, extname="epochs",
                header={"SNY": snctr[0], "SNX": snctr[1]})


def read_results(fname):
    """Read results from a FITS file."""

    with fitsio.FITS(fname, "r") as f:
        galaxy_hdr = f[0].read_header()
        galaxy = f[0].read()
        epochs_hdr = f[1].read_header()
        epochs = f[1].read()

        wavewcs = {"CRVAL3": galaxy_hdr["CRVAL3"],
                   "CRPIX3": galaxy_hdr["CRPIX3"],
                   "CDELT3": galaxy_hdr["CDELT3"]}

    return {"galaxy": galaxy,
            "wavewcs": wavewcs,
            "wave": wcs_to_wave(galaxy_hdr),
            "epochs": epochs,
            "snctr": (epochs_hdr["SNY"], epochs_hdr["SNX"])}
