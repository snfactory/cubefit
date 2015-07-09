from __future__ import print_function, division

import numpy as np
import json
import fitsio
from glob import glob
#import matplotlib as mpl # Uncomment these if ddt run as batch job
#mpl.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.ticker import NullLocator

__all__ = ["plot_timeseries", "plot_wave_slices", "plot_sn"]

BAND_LIMITS = {'U': (3400., 3900.),
               'B': (4102., 5100.),
               'V': (6289., 7607.)}

STAMP_SIZE = 0.9
COLORMAP = 'bone'


def plot_timeseries(cubes, results, band='B', fname=None):
    """Return a figure showing data and model.

    Parameters
    ----------
    cubes : list of DataCube
    results : dict
        Dictionary of dictionaries. Dictionaries represent result
        after each step in the fit. Each dictionary contains keys
        'galaxy' (3-d array), 'snctr' (tuple), 'epochs' (structured array). 
    band : str
        Band over which to flatten 3-d cubes
    fname : str
        Output file name
    """

    nt = len(cubes)
    cube_shape = cubes[0].data.shape
    wave = cubes[0].wave

    ncol = nt + 2  # one column for each data epoch, plus 2 extras for model
    nrow = 1 + 2 * len(results)  # one row for the data, two for each
                                 # step in the fit (model and residual)
    figsize = (STAMP_SIZE * ncol, STAMP_SIZE * nrow)
    fig = plt.figure(figsize=figsize)

    # upper and lower wavelength limits
    wmin, wmax = BAND_LIMITS[band]
    wavemask = (wave > wmin) & (wave < wmax)

    # plot data for each epoch, keeping track of vmin/vmax for each.
    dataims = []
    datavmin = np.zeros(nt)
    datavmax = np.zeros(nt)
    for i_t, cube in enumerate(cubes):
        dataim = np.average(cube.data[wavemask, :, :], axis=0)
        datavmax[i_t] = 1.1*np.max(dataim)
        datavmin[i_t] = -0.2*np.max(dataim)
        ax = plt.subplot2grid((nrow, ncol), (0, i_t + 2))
        ax.imshow(dataim, cmap=COLORMAP, vmin=datavmin[i_t], vmax=datavmax[i_t],
                  interpolation='nearest', origin='lower')
        ax.xaxis.set_major_locator(NullLocator())
        ax.yaxis.set_major_locator(NullLocator())
        dataims.append(dataim)
        if i_t == 0:
            ax.set_ylabel('Data')

    # evaluate all scenes and residuals first, so we can set vmin/vmax uniformly
    scenes = []
    residuals = []
    masks = []
    for result in results.values():
        epochs = result['epochs']
        scenerow = []
        residualrow = []
        maskrow = []
        for i_t in range(nt):
            galeval = epochs['galeval'][i_t]
            sneval = epochs['sneval'][i_t]
            sky = epochs['sky'][i_t, :, None, None]
            scene = sky + galeval + sneval
            sceneim = np.average(scene[wavemask, :, :], axis=0)
            residim = dataims[i_t] - sceneim
            weightim = np.sum(cubes[i_t].weight[wavemask, :, :], axis=0)
            mask = weightim > 0.
            scenerow.append(sceneim)
            residualrow.append(residim)
            maskrow.append(mask)
        scenes.append(scenerow)
        residuals.append(residualrow)
        masks.append(maskrow)

    # Set residual vmin/vmax based on *last* row.
    residvmin = np.zeros(nt)
    residvmax = np.zeros(nt)
    for i_t in range(nt):
        vals = residuals[-1][i_t][masks[-1][i_t]]
        std = np.std(vals)
        residvmin[i_t] = -3. * std
        residvmax[i_t] = 3. * std


    for f, (key, result) in enumerate(results.iteritems()):
        galaxy = result['galaxy']
        epochs = result['epochs']

        # galaxy model
        image = np.average(galaxy[wavemask, :, :], axis=0)
        ax = plt.subplot2grid((nrow, ncol), (1+2*f,0), rowspan=2, colspan=2)
        ax.imshow(image, cmap=COLORMAP, interpolation='nearest', origin='lower')
        ax.xaxis.set_major_locator(NullLocator())
        ax.yaxis.set_major_locator(NullLocator())
        ax.set_ylabel(key)
        
        # evaluated model and residuals
        for i_t in range(nt):
            ax1 = plt.subplot2grid((nrow, ncol), (1+2*f,i_t+2))
            ax1.imshow(scenes[f][i_t], cmap=COLORMAP, interpolation='nearest',
                       origin='lower', vmin=datavmin[i_t], vmax=datavmax[i_t])

            ax2 = plt.subplot2grid((nrow, ncol), (1+2*f+1,i_t+2))
            ax2.imshow(residuals[f][i_t], cmap=COLORMAP, interpolation='nearest',
                       origin='lower', vmin=residvmin[i_t], vmax=residvmax[i_t])

            ax1.xaxis.set_major_locator(NullLocator())
            ax1.yaxis.set_major_locator(NullLocator())
            ax2.xaxis.set_major_locator(NullLocator())
            ax2.yaxis.set_major_locator(NullLocator())

            if i_t == 0:
                ax1.set_ylabel('Models', fontsize=8)
                ax2.set_ylabel('Residuals', fontsize=8)
                ax1.yaxis.set_label_coords(-0.02, 0.5)
                ax2.yaxis.set_label_coords(-0.02, 0.5)

    fig.subplots_adjust(left=0.02, right=0.999, bottom=0.02, top=0.98,
                        wspace=0.01)

    if fname is None:
        return fig
    
    plt.savefig(fname)
    plt.close()


def plot_wave_slices(cubes, galeval, indices=None, fname=None):
    """Return a figure showing wavelength slices of data and model.

    Parameters
    ----------
    ddt_output : pickle dictionary
        Dictionary with data and results of ddt fit at three steps
    fname : str
        Output file name
    """

    if len(cubes) != len(galeval):
        raise ValueError("length of cubes and galeval must match")

    cube_shape = cubes[0].data.shape
    wave = cubes[0].wave

    if indices is None:
        indices = np.arange(0, len(wave), 100)

    ncol = 2*len(cubes)  # two rows per final ref (model and residual)
    nrow = len(indices)  # one column for each wavelength slice
    figsize = (STAMP_SIZE * nrow, STAMP_SIZE * ncol)
    fig = plt.figure(figsize=figsize)

    # Plot data, model and residual for each final ref epoch at a range of
    # wavelength slices.
    for i in range(len(cubes)):
        for s, idx in enumerate(indices):
            data_slice = cubes[i].data[idx, :, :]
            model_slice = galeval[i][idx, : , :]
            residual_slice = data_slice - model_slice 

            vmin = None
            vmax = None
            ax1 = plt.subplot2grid((ncol, nrow), (i*2, s))
            ax2 = plt.subplot2grid((ncol, nrow), (i*2+1, s))

            ax1.imshow(model_slice, vmin=vmin, vmax=vmax,
                       interpolation='nearest', origin='lower')
            ax2.imshow(residual_slice, interpolation='nearest',
                       vmin=vmin, vmax=vmax, origin='lower')

            ax1.xaxis.set_major_locator(NullLocator())
            ax1.yaxis.set_major_locator(NullLocator())
            ax2.xaxis.set_major_locator(NullLocator())
            ax2.yaxis.set_major_locator(NullLocator())
            if s == 0:
                ax1.set_ylabel('%s Model' % i, fontsize=8)
                ax2.set_ylabel('%s Residual' % i, fontsize=8)
            if i == len(cubes) - 1:
                ax2.set_xlabel('$\lambda = %s$' % wave[idx], fontsize=10)

    fig.subplots_adjust(left=0.03, right=0.999, bottom=0.06, top=0.98,
                        hspace=0.01, wspace=0.01)

    if fname is None:
        return fig
    
    plt.savefig(fname)
    plt.close()


def plot_sn(sn_spectra, cfg, wave, idr_prefix, fname=None):
    """Return a figure with th SN

    Parameters
    ----------
    ddt_output : pickle dictionary
        Dictionary with data and results of ddt fit at three steps
    cfg : dict
        Raw configuration dictionary.
    idr_prefix : str
        Path to IDR directory.
    fname : str
        Output file name
    """

    sn_max = sn_spectra.max()

    in_cube_files = cfg["IN_CUBE_DDTNAME"]
    channel = cfg["PARAM_CHANNEL"]
    day_exp_nums = [file.split('_')[2:5] for file in in_cube_files]

    sn_name = cfg["PARAM_SN_NAME"]
    idr_files = glob(idr_prefix+'/%s*%s.fits' % (sn_name, channel))
    if len(idr_files) == 0:
        idr_files = glob(idr_prefix+'/*/%s/*%s.fits' % (sn_name, channel))
    try:
        assert len(idr_files) > 0
    except:
        raise ValueError("No files matching this sn in provided directory")
    phase_strings = [file.split('_')[-2] for file in idr_files]

    phases = [((-1 if phase_string[0] == 'M' else 1) *
               float(phase_string[1:])/1000.)
              for phase_string in phase_strings]
    phase_sort = np.array(phases).argsort()
    fig = plt.figure(figsize=(7,8))

    for p, phase_arg in enumerate(phase_sort):
        
        file = idr_files[phase_arg]
        phase = phases[phase_arg]
        
        with fitsio.FITS(file, 'r') as f:
            header = f[0].read_header()
            data = f[0].read()
            variance = f[1].read()

        n = header["NAXIS1"]
        #crpix = header["CRPIX1"]-1.0  # FITS is 1-indexed, numpy as 0-indexed 
        crval = header["CRVAL1"]
        cdelt = header["CDELT1"]
        sn_wave = crval + cdelt * (np.arange(n))# - crpix)

        file_day_exp = header["FILENAME"].split('_')[1:4]
        i_t_match = np.flatnonzero(np.array([day_exp == file_day_exp for
                                             day_exp in day_exp_nums]))

        plt.plot(sn_wave, data/sn_max + p/2., color='k')

        for i_t in i_t_match:
            plt.plot(wave, sn_spectra[i_t]/sn_max + p/2., color='r')
        plt.text(sn_wave[-20], p/2., 'Phase = '+str(phase))


    if fname is None:
        return fig
    
    plt.savefig(fname)
    plt.close()
