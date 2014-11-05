from __future__ import print_function, division

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import NullLocator

__all__ = ["plot_timeseries"]

BAND_LIMITS = {'U': (3400., 3900.),
               'B': (4102., 5100.),
               'V': (6289., 7607.)}

STAMP_SIZE = 1.5

def plot_timeseries(data, model=None, band='B'):
    """Return a figure showing data and model.

    Parameters
    ----------
    data : DDTData
    model : DDTModel
    band : str
    """

    # one column for each data epoch, plus 2 extras for model
    ncol = data.nt + 2
    nrow = 4
    figsize = (STAMP_SIZE * ncol, STAMP_SIZE * nrow)
    fig = plt.figure(figsize=figsize)

    # upper and lower wavelength limits
    wmin, wmax = BAND_LIMITS[band]

    # plot model, if given
    if model is not None:
        ax = plt.subplot2grid((nrow, ncol), (0, 0), rowspan=2, colspan=2)
        mask = (model.wave > wmin) & (model.wave < wmax)
        image = np.average(model.gal[mask, :, :], axis=0)
        ax.imshow(image, vmin=image.min(), vmax=image.max(), cmap='Greys',
                  interpolation='nearest', origin='lower')
        ax.xaxis.set_major_locator(NullLocator())
        ax.yaxis.set_major_locator(NullLocator())

    # compute all images ahead of time so that we can set vmin, vmax
    # the same for all.
    wmin, wmax = BAND_LIMITS[band]
    mask = (data.wave > wmin) & (data.wave < wmax)
    images = np.average(data.data[:, mask, :, :], axis=1)
    vmin = images.min()
    vmax = images.max()

    for i_t in range(data.nt):
        ax = plt.subplot2grid((nrow, ncol), (0, i_t + 2))
        ax.imshow(images[i_t], vmin=vmin, vmax=vmax, cmap='Greys',
                  interpolation='nearest', origin='lower')
        ax.xaxis.set_major_locator(NullLocator())
        ax.yaxis.set_major_locator(NullLocator())

    fig.subplots_adjust(left=0.001, right=0.999, bottom=0.02, top=0.98,
                        hspace=0.01, wspace=0.01)

    return fig
