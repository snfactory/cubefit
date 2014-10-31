from __future__ import print_function, division

from matplotlib import pyplot as plt

bandlimits = {'B': (

def integrate_tophat(data, wmin, wmax):

def plot_timeseries(data, model=None, band='B'):
    """Return a figure showing data and model.

    Parameters
    ----------
    data : DDTData
    model : DDTModel
    band : str
    """

    STAMP_SIZE = 1.5

    # one column for each data epoch, plus 2 extras for model
    ncol = data.nt + 2
    nrow = 4
    figsize = (STAMP_SIZE * ncol, STAMP_SIZE * nrow)

    # Create figure and all the axes:
    fig, axes = plt.subplots(nrow, ncol, figsize=figsize, squeeze=False)
    fig.subplots_adjust(left=0.001, right=0.999, bottom=0.02, top=0.98,
                        hspace=0.01, wspace=0.01)

    for i_t in range(data.nt):
        ax = axes[0, i_t]

        
        ax.imshow(data.data[i_t]
