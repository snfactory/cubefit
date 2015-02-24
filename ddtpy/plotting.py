from __future__ import print_function, division

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
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

    # set limits all the same, or not
    #vmin = images.min()
    #vmax = images.max()
    vmin, vmax = None, None

    # compute model sampled to data frame
    predictions = np.empty_like(images)
    for i_t in range(data.nt):
        m = model.evaluate(i_t, data.xctr[i_t], data.yctr[i_t],
                           (data.ny, data.nx), which='all')
        predictions[i_t, :, :] = np.average(m[mask, :, :], axis=0)
    vmin, vmax = np.zeros(data.nt), np.zeros(data.nt)
    for i_t in range(data.nt):
        ax = plt.subplot2grid((nrow, ncol), (0, i_t + 2))
        vmin[i_t] = np.array([images[i_t], predictions[i_t],
                             images[i_t]-predictions[i_t]]).min()
        vmax[i_t] = np.array([images[i_t], predictions[i_t],
                             images[i_t]-predictions[i_t]]).max()
        ax.imshow(images[i_t], vmin=vmin[i_t], vmax=vmax[i_t], cmap='Greys',
                  interpolation='nearest', origin='lower')
        ax.xaxis.set_major_locator(NullLocator())
        ax.yaxis.set_major_locator(NullLocator())

    # model plot
    for i_t in range(data.nt):
        ax = plt.subplot2grid((nrow, ncol), (1, i_t + 2))
        ax.imshow(predictions[i_t], vmin=vmin[i_t], vmax=vmax[i_t], cmap='Greys',
                  interpolation='nearest', origin='lower')
        ax.xaxis.set_major_locator(NullLocator())
        ax.yaxis.set_major_locator(NullLocator())

    # residuals
    for i_t in range(data.nt):
        ax = plt.subplot2grid((nrow, ncol), (2, i_t + 2))
        ax.imshow(images[i_t] - predictions[i_t],
                  vmin=vmin[i_t], vmax=vmax[i_t], cmap='Greys',
                  interpolation='nearest', origin='lower')
        ax.xaxis.set_major_locator(NullLocator())
        ax.yaxis.set_major_locator(NullLocator())
        

    fig.subplots_adjust(left=0.001, right=0.999, bottom=0.02, top=0.98,
                        hspace=0.01, wspace=0.01)

    return fig


def plot_wave_slices(data, model, nt,
                     lambdas = [0, 100,200,300,400,500,600,700]):
    """Plot data, model and residual for a single epoch at a range of
    wavelength slices"""

    ncol = len(lambdas)
    nrow = 3
    figsize = (STAMP_SIZE * ncol, STAMP_SIZE * nrow)
    fig = plt.figure(figsize=figsize)
    fig, ax = plt.subplots(nrow, ncol)

    m = model.evaluate(nt, data.xctr[nt], data.yctr[nt],
                       (data.ny, data.nx), which='all')
    residual = data.data[nt] - m


    for s, l in enumerate(lambdas):
        data_slice = data.data[nt,l,:,:]
        model_slice = m[l]
        residual_slice = data_slice - model_slice

        vmin = np.array([data_slice,model_slice,residual_slice]).min()
        vmax = np.array([data_slice,model_slice,residual_slice]).max()

        ax[0,s].imshow(data_slice, vmin=vmin, vmax=vmax,
                       interpolation='nearest')
        ax[1,s].imshow(model_slice, vmin=vmin, vmax=vmax,
                       interpolation='nearest')
        im = ax[2,s].imshow(residual_slice, interpolation='nearest',
                       vmin = vmin, vmax=vmax)

        ax[0,s].xaxis.set_major_locator(NullLocator())
        ax[0,s].yaxis.set_major_locator(NullLocator())
        ax[1,s].xaxis.set_major_locator(NullLocator())
        ax[1,s].yaxis.set_major_locator(NullLocator())
        ax[2,s].xaxis.set_major_locator(NullLocator())
        ax[2,s].yaxis.set_major_locator(NullLocator())
        #cb = fig.colorbar(im, orientation='horizontal')
        #[l.set_rotation(45) for l in cb.ax.get_xticklabels()]
    
    fig.subplots_adjust(left=0.001, right=0.999, bottom=0.02, top=0.98,
                        hspace=0.01, wspace=0.01)

    return fig
