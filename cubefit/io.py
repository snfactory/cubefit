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
    
    def __init__(self, data, weight, wave, header=None):
        if data.shape != weight.shape:
            raise ValueError("shape of weight and data must match")
        if len(wave) != data.shape[0]:
            raise ValueError("length of wave must match data axis=1")

        self.data = data
        self.weight = weight
        self.wave = wave
        self.nw, self.ny, self.nx = data.shape
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
        header = f[0].read_header()
        data = np.asarray(f[0].read(), dtype=np.float64)
        variance = np.asarray(f[1].read(), dtype=np.float64)

    wave = wcs_to_wave(header)
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

    return DataCube(data, weight, wave, header=header)


def epoch_results(galaxy, skys, sn, snctr, yctr, xctr, yctr0, xctr0,
                  yctrbounds, xctrbounds, cubes, psfs):
    """Package all by-epoch results into a single numpy structured array,
    amenable to writing out to a FITS file.

    Note that this format assumes that the data for all epochs has the same
    spatial shape. A different format would have to be used if this were not
    the case.
    """
    dshape = cubes[0].data.shape

    # This is a table with `nt` rows
    nt = len(psfs)
    dtype = [('yctr', 'f8'),
             ('xctr', 'f8'),
             ('sn', 'f4', sn.shape[1]),
             ('sky', 'f4', skys.shape[1]),
             ('galeval', 'f4', dshape),
             ('sneval', 'f4', dshape),
             ('chisq', 'f8'),
             ('yctr0', 'f8'),
             ('xctr0', 'f8'),
             ('yctrbounds', 'f8', 2),
             ('xctrbounds', 'f8', 2)]

    epochs = np.zeros(nt, dtype=dtype)
    epochs['yctr'] = yctr
    epochs['xctr'] = xctr
    epochs['sn'] = sn
    epochs['sky'] = skys
    epochs['yctr0'] = yctr0
    epochs['xctr0'] = xctr0
    epochs['yctrbounds'] = yctrbounds
    epochs['xctrbounds'] = xctrbounds

    # evaluate galaxy & PSF on data
    for i in range(nt):
        epochs['galeval'][i] = psfs[i].evaluate_galaxy(galaxy, dshape[1:3],
                                                       (yctr[i], xctr[i]))
        epochs['sneval'][i] = psfs[i].point_source(snctr, dshape[1:3],
                                                   (yctr[i], xctr[i]))
        epochs['sneval'][i] *= sn[i, :, None, None]  # multiply by sn amplitude

        # Calculate chi squared.
        scene = (epochs['sky'][i, :, None, None] + epochs['galeval'][i] +
                 epochs['sneval'][i])
        epochs['chisq'][i] = np.sum(cubes[i].weight *
                                    (cubes[i].data - scene)**2)

    return epochs


def write_results(galaxy, skys, sn, snctr, yctr, xctr, yctr0, xctr0,
                  yctrbounds, xctrbounds, cubes, psfs, modelwcs, fname,
                  descale=True):
    """Write results to a FITS file."""

    if descale:
        galaxy = galaxy / SCALE_FACTOR  # note: do NOT use in-place ops here!
        skys = skys / SCALE_FACTOR
        sn = sn / SCALE_FACTOR

    # Create epochs table.
    epochs = epoch_results(galaxy, skys, sn, snctr, yctr, xctr, yctr0, xctr0,
                           yctrbounds, xctrbounds, cubes, psfs)

    if os.path.exists(fname):  # avoids warning doing FITS(..., clobber=True)
        os.remove(fname)

    with fitsio.FITS(fname, "rw") as f:
        f.write(galaxy, extname="galaxy", header=modelwcs)
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

    return {"galaxy": galaxy,
            "header": galaxy_hdr,
            "wave": wcs_to_wave(galaxy_hdr),
            "epochs": epochs,
            "snctr": (epochs_hdr["SNY"], epochs_hdr["SNX"])}
