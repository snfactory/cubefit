from __future__ import print_function, division

import json
from glob import glob

import fitsio
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt
from matplotlib import gridspec
from matplotlib.ticker import NullLocator

from .extern import ADR, Hyper_PSF3D_PL
from .main import REFWAVE, SPAXEL_SIZE

__all__ = ["plot_timeseries", "plot_epoch", "plot_sn", "plot_adr"]

BAND_LIMITS = {'U': (3400., 3900.),
               'B': (4102., 5100.),
               'V': (6289., 7607.)}
COLORMAP = 'bone'

def plot_timeseries(cubes, results, band=None, fname=None, title='Timeseries'):
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

    # Set up figure and axes grid.
    ncol = nt + 3  # one column for each data epoch, 2 for model,
                   # 1 for blank space.
    nrow = 2 + 2 * len(results)  # one row for the data, two for each
                                 # step in the fit (model and residual),
                                 # one more for colorbar at the bottom
    left, right = 0.4, 0.2  # border in inches
    bottom, top = 0.5, 0.3  # border in inches
    wspace, hspace = 0.2, 0.2  # spacing in inches
    heights = nrow * [0.9]
    widths = ncol * [0.9]
    heights[-1] = 0.1  # shrink colorbar height
    widths[2] = 0.1  # width of blank space between model and epochs

    # set up figure and subplot grid.
    figwidth = left + right + sum(widths) + (ncol - 1) * wspace
    figheight = bottom + top + sum(heights) + (nrow - 1) * hspace
    fig = plt.figure(figsize=(figwidth, figheight))
    gs = gridspec.GridSpec(nrow, ncol, width_ratios=widths,
                           height_ratios=heights)
    gs.update(left=(left / figwidth),
              right=(1.0 - right / figwidth),
              bottom=(bottom / figheight),
              top=(1.0 - top / figheight),
              wspace=(wspace / figwidth * (ncol - 1)),
              hspace=(hspace / figheight * (nrow - 1)))

    # upper and lower wavelength limits (default is a 1000 Angstrom wide
    # band in the middle of the cube)
    if band is None:
        wmid = (wave[0] + wave[-1]) / 2.0
        wmin, wmax = wmid - 500.0, wmid + 500.0
    else:
        wmin, wmax = BAND_LIMITS[band]
    wavemask = (wave > wmin) & (wave < wmax)

    # Plot data for each epoch, keeping track of vmin/vmax for each.
    dataims = []
    datavmin = np.zeros(nt)
    datavmax = np.zeros(nt)
    for i_t, cube in enumerate(cubes):
        dataim = np.sum(cube.data[wavemask, :, :], axis=0)
        datavmax[i_t] = 1.1*np.max(dataim)
        datavmin[i_t] = -0.2*np.max(dataim)
        ax = plt.subplot(gs[0, i_t + 3])
        ax.imshow(dataim, vmin=datavmin[i_t], vmax=datavmax[i_t],
                  cmap=COLORMAP, interpolation='nearest', origin='lower')
        ax.xaxis.set_major_locator(NullLocator())
        ax.yaxis.set_major_locator(NullLocator())
        ax.set_title("epoch {:d}".format(i_t), fontsize=12)
        dataims.append(dataim)
        if i_t == 0:
            ax.set_ylabel('data', fontsize=12)

    # evaluate all scenes and residuals first, so we can set vmin/vmax
    # uniformly in the residuals (based on last row)
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
            sceneim = np.sum(scene[wavemask, :, :], axis=0)
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

    for j, (key, result) in enumerate(results.items()):
        galaxy = result['galaxy']
        epochs = result['epochs']

        # galaxy model
        image = np.sum(galaxy[wavemask, :, :], axis=0)
        ax = plt.subplot(gs[2*j+1:2*j+3, 0:2])
        ax.imshow(image, cmap=COLORMAP, interpolation='nearest',
                  origin='lower')
        ax.xaxis.set_major_locator(NullLocator())
        ax.yaxis.set_major_locator(NullLocator())
        ax.set_ylabel(key, fontsize=18)
        if j == len(results)-1:
            ax.set_xlabel("galaxy model", fontsize=12)

        # scene and residual for each epoch
        for i_t in range(nt):

            # scene
            ax1 = plt.subplot(gs[2*j+1, i_t+3])
            ax1.imshow(scenes[j][i_t],  vmin=datavmin[i_t], vmax=datavmax[i_t],
                       cmap=COLORMAP, interpolation='nearest', origin='lower')
            ax1.xaxis.set_major_locator(NullLocator())
            ax1.yaxis.set_major_locator(NullLocator())

            # residual
            ax2 = plt.subplot(gs[2*j+2, i_t+3])
            rp = ax2.imshow(residuals[j][i_t], vmin=residvmin[i_t],
                            vmax=residvmax[i_t], cmap=COLORMAP,
                            interpolation='nearest', origin='lower')
            ax2.xaxis.set_major_locator(NullLocator())
            ax2.yaxis.set_major_locator(NullLocator())

            # label first column
            if i_t == 0:
                ax1.set_ylabel('scene model', fontsize=11)
                ax2.set_ylabel('residual', fontsize=11)

            # colorbar (last row only)
            if j == len(results) - 1:
                ticklocs = [residvmin[i_t], 0., residvmax[i_t]]
                scenemax = scenes[j][i_t].max()
                ticklabels = ['%d%%' % (100 * residvmin[i_t] / scenemax),
                              '0%',
                              '%d%%' % (100 * residvmax[i_t] / scenemax)]
                tickalign =['left', 'center', 'right']

                cb = fig.colorbar(rp, cax=plt.subplot(gs[2*j+3, i_t+3]),
                                  ticks=ticklocs, orientation='horizontal')
                cb.ax.set_xticklabels(ticklabels, fontsize=10)
                for k, label in enumerate(cb.ax.get_xticklabels()):
                    label.set_horizontalalignment(tickalign[k])

    # figure text
    fig.text(left / figwidth, (figheight - top) / figheight, title,
             horizontalalignment='left', verticalalignment='top', fontsize=20)
    fig.text(left / figwidth, (figheight - top - 0.3) / figheight,
             "${:d} \\AA - {:d} \\AA$".format(int(wmin), int(wmax)),
             horizontalalignment='left', verticalalignment='top', fontsize=16)
    residtextx = 0.5 * (left + sum(widths[0:3]) + 3*wspace + figwidth - right)
    residtexty = 0.1
    fig.text(residtextx / figwidth, residtexty / figheight,
             "residual / max(scene)", horizontalalignment='center',
             verticalalignment='baseline')

    if fname is None:
        return fig
    
    plt.savefig(fname)
    plt.close()
    

def plot_epoch(cube, epoch, fname=None):
    """Return a figure with diagnostic plots for one epoch

    Parameters
    ----------
    cube : DataCube
    epoch : structured ndarray
        One row from the table of result['epochs'].
    """

    data = cube.data
    weight = cube.weight
    wave = cube.wave
    numslices = 5

    # plot parameters in physical units (in)
    left = 0.5
    right = 0.2
    bottom = 0.6
    top = 0.3
    wspace = 0.2
    hspace = 0.2
    height = 1.3  # stamp height
    widths = [1.3, 1.3, 1.3, 0.1, 0.8, 4.2]  # column widths: [data,
                                             # model, resid, colorbar,
                                             # (blank), spectrum]
    numcols = len(widths)
    numrows = numslices + 1

    # set up figure and subplot grid.
    figwidth = left + right + sum(widths) + (numcols - 1) * wspace
    figheight = bottom + top + numrows * height + (numrows - 1) * hspace
    fig = plt.figure(figsize=(figwidth, figheight))
    gs = gridspec.GridSpec(numrows, numcols, width_ratios=widths)
    gs.update(left=(left / figwidth),
              right=(1.0 - right / figwidth),
              bottom=(bottom / figheight),
              top=(1.0 - top / figheight),
              wspace=(wspace / figwidth * (numcols - 1)),
              hspace=(hspace / figheight * (numrows - 1)))

    # First row of stamps
    data_plot = plt.subplot(gs[0, 0])
    model_plot = plt.subplot(gs[0, 1])
    resid_plot = plt.subplot(gs[0, 2])

    wmin, wmax = wave[0], wave[-1]
    wavemask = (wave > wmin) & (wave < wmax)

    dataim = np.average(data[wavemask, :, :], axis=0)
    galeval = epoch['galeval']
    sneval = epoch['sneval']
    sky = epoch['sky'][:, None, None]
    scene = sky + galeval + sneval
    sceneim = np.average(scene[wavemask, :, :], axis=0)
    residim = dataim - sceneim

    datavmax = 1.1*np.max(dataim)
    datavmin = -0.2*np.max(dataim)
    weightim = np.sum(weight[wavemask, :, :], axis=0)
    mask = weightim > 0.
    vals = residim[mask]
    std = np.std(vals)
    residvmin = -3. * std
    residvmax = 3. * std

    data_plot.imshow(dataim, cmap=COLORMAP, vmin=datavmin, vmax=datavmax,
                     interpolation='nearest', origin='lower')
    model_plot.imshow(sceneim, cmap=COLORMAP, vmin=datavmin, vmax=datavmax,
                      interpolation='nearest', origin='lower')
    rp = resid_plot.imshow(residim, cmap=COLORMAP, 
                           vmin=residvmin, vmax=residvmax,
                           interpolation='nearest', origin='lower')
    cb = fig.colorbar(rp, cax=plt.subplot(gs[0, 3]),
                          ticks=[residvmin, 0, residvmax])

    cb.ax.set_yticklabels(['%.0f%%' % (100*residvmin/np.max(dataim)), '0%',
                           '%.0f%%' % (100*residvmax/np.max(dataim))],
                          fontsize='small')
    
    data_plot.xaxis.set_major_locator(NullLocator())
    data_plot.yaxis.set_major_locator(NullLocator())
    model_plot.xaxis.set_major_locator(NullLocator())
    model_plot.yaxis.set_major_locator(NullLocator())
    resid_plot.xaxis.set_major_locator(NullLocator())
    resid_plot.yaxis.set_major_locator(NullLocator())
    
    data_plot.set_ylabel('all\nwavelengths')
    data_plot.set_title('Data')
    model_plot.set_title('Model')
    resid_plot.set_title('Residual')

    metaslices = np.linspace(0, len(wave), numslices + 1)

    for i in range(numslices):
        sliceindices = np.arange(metaslices[i], metaslices[i+1], dtype=int)
        dataslice = np.average(data[sliceindices, :, :], axis=0)
        sceneslice = np.average(scene[sliceindices, :, :], axis=0)
        residslice = dataslice - sceneslice
        vmin, vmax = -.2*np.max(dataslice), 1.1*np.max(dataslice)

        data_plot = plt.subplot(gs[i+1, 0])
        model_plot = plt.subplot(gs[i+1, 1])
        resid_plot = plt.subplot(gs[i+1, 2])
        residmax = 3. * np.std(residslice[mask])
        residmin = -residmax
    
        data_plot.imshow(dataslice, cmap=COLORMAP, vmin=vmin, vmax=vmax,
                         interpolation='nearest', origin='lower')
        model_plot.imshow(sceneslice, cmap=COLORMAP, vmin=vmin, vmax=vmax,
                          interpolation='nearest', origin='lower')
        rp = resid_plot.imshow(residslice, cmap=COLORMAP, vmin=residmin,
                               vmax=residmax, interpolation='nearest', 
                               origin='lower')
        data_plot.xaxis.set_major_locator(NullLocator())
        data_plot.yaxis.set_major_locator(NullLocator())
        model_plot.xaxis.set_major_locator(NullLocator())
        model_plot.yaxis.set_major_locator(NullLocator())
        resid_plot.xaxis.set_major_locator(NullLocator())
        resid_plot.yaxis.set_major_locator(NullLocator())
        cb = fig.colorbar(rp, cax=plt.subplot(gs[i+1, 3]),
                          ticks=[residmin, 0, residmax])
        cb.ax.set_yticklabels(['%.0f%%' % (100*residmin/np.max(dataim)),
                               '0%',
                               '%.0f%%' % (100*residmax/np.max(dataim))],
                              fontsize='small')
        data_plot.set_ylabel('%d -\n %d $\AA$' % (wave[metaslices[i]],
                                                  wave[metaslices[i+1]-1]))

    spec = plt.subplot(gs[:, 5])
    spec.plot(wave, epoch['sn'], label='SN spectrum')
    spec.plot(wave, epoch['sky'], label='Sky spectrum')

    gal_ave = galeval.sum(axis=(1,2))
    spec.plot(wave, gal_ave, label = 'Galaxy spectrum (sum)')
    spec.set_xlim(wave[0], wave[-1])
    spec.legend(fontsize=9, frameon=False)
    spec.set_xlabel("wavelength ($\\AA$)")

    if fname is None:
        return fig
        
    plt.savefig(fname)
    plt.close()


def plot_sn(filenames, sn_spectra, wave, idrfilenames, outfname):
    """Return a figure with the SN

    Parameters
    ----------
    fname : str
        Output file name
    """

    sn_max = sn_spectra.max()

    day_exp_nums = [fname.split('_')[1:4] for fname in filenames]

    phase_strings = [fname.split('_')[-2] for fname in idrfilenames]

    print(phase_strings)
    phases = [((-1 if phase_string[0] == 'M' else 1) *
               float(phase_string[1:])/1000.)
              for phase_string in phase_strings]
    phase_sort = np.array(phases).argsort()
    fig = plt.figure(figsize=(7,8))

    for p, phase_arg in enumerate(phase_sort):
        
        file = idrfilenames[phase_arg]
        phase = phases[phase_arg]
        
        with fitsio.FITS(file, 'r') as f:
            header = f[0].read_header()
            data = f[0].read()
            variance = f[1].read()

        n = header["NAXIS1"]
        #crpix = header["CRPIX1"]-1.0  # FITS is 1-indexed, numpy as 0-indexed 
        crval = header["CRVAL1"]
        cdelt = header["CDELT1"]
        sn_wave = crval + cdelt * (np.arange(n)) # - crpix)

        file_day_exp = header["FILENAME"].split('_')[1:4]
        i_t_match = np.flatnonzero(np.array([day_exp == file_day_exp for
                                             day_exp in day_exp_nums]))

        plt.plot(sn_wave, data/sn_max + p/2., color='k')

        for i_t in i_t_match:
            plt.plot(wave, sn_spectra[i_t]/sn_max + p/2., color='r')
        plt.text(sn_wave[-20], p/2., 'Phase = '+str(phase))


    plt.savefig(outfname)
    plt.close()


def plot_adr(cubes, wave, fname=None):
    """Plot adr x and y vs. wavelength, and x vs y

    Parameters
    ----------
    cfg : dict
        Configuration contents.
    wave : 1-d array
    cubes : list of DataCube
        Used for header values only.
    """

    nt = len(cubes)

    fig = plt.figure()
    yplot = plt.subplot2grid((2, 2), (0, 0))
    xplot = plt.subplot2grid((2, 2), (0, 1))
    xyplot = plt.subplot2grid((2, 2), (1, 0), colspan=2)

    cm = plt.get_cmap("jet")

    for i in range(nt):
        # following lines same as in main.cubefit()
        delta, theta = Hyper_PSF3D_PL.predict_adr_params(cubes[i].header)
        adr = ADR(cubes[i].header['PRESSURE'], cubes[i].header['TEMP'],
                  lref=REFWAVE, delta=delta, theta=theta)
        adr_refract = adr.refract(0, 0, wave, unit=SPAXEL_SIZE)
        xctr, yctr = adr_refract

        yplot.plot(wave, yctr, color=cm(i/nt))
        xplot.plot(wave, xctr, color=cm(i/nt))
        xyplot.plot(xctr, yctr, color=cm(i/nt))

    yplot.set_ylabel('dY (spaxels)')
    xplot.set_ylabel('dX (spaxels)')
    yplot.set_xlabel('wavelength ($\\AA$)')
    xplot.set_xlabel('wavelength ($\\AA$)')
    xyplot.set_xlabel('dX')
    xyplot.set_ylabel('dY')
    plt.tight_layout()
    if fname is None:
        return fig
    
    plt.savefig(fname)
    plt.close()

        
        
