from __future__ import print_function, division

import numpy as np
import json
import fitsio
from glob import glob
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.ticker import NullLocator

__all__ = ["plot_timeseries", "plot_wave_slices", "plot_sn"]

BAND_LIMITS = {'U': (3400., 3900.),
               'B': (4102., 5100.),
               'V': (6289., 7607.)}

STAMP_SIZE = .9 #1.5
IDR_PREFIX = '/home/cmsaunders/ACEv2/training/'

def plot_timeseries(ddt_output, band='B', fname=None):
    """Return a figure showing data and model.

    Parameters
    ----------
    ddt_output : pickle dictionary
        Dictionary with data and results of ddt fit at three steps
    band : str
        Band over which to flatten 3-d cubes
    fname : str
        Output file name
    """

    data_cubes = ddt_output['Data']
    nt = len(data_cubes)
    cube_shape = data_cubes[0].data.shape
    wave = data_cubes[0].wave

    # one column for each data epoch, plus 2 extras for model
    ncol = nt + 2
    # one row for the data, two for each step in the fit (model and residual)
    nrow = 7
    figsize = (STAMP_SIZE * ncol, STAMP_SIZE * nrow)
    fig = plt.figure(figsize=figsize)

    # upper and lower wavelength limits
    wmin, wmax = BAND_LIMITS[band]
    mask = (wave > wmin) & (wave < wmax)

    for i_t, cube in enumerate(data_cubes):
        data_image = np.average(cube.data[mask, :, :], axis=0)
        ax = plt.subplot2grid((nrow, ncol), (0, i_t + 2))
        ax.imshow(data_image, #vmin=vmin[i_t], vmax=vmax[i_t],
                  cmap='Greys',
                  interpolation='nearest', origin='lower')
        ax.xaxis.set_major_locator(NullLocator())
        ax.yaxis.set_major_locator(NullLocator())
        if i_t == 0:
            ax.set_ylabel('Data')

    for f, fit_result in enumerate(['MasterRefFit', 'AllRefFit', 'FinalFit']):
        result = ddt_output[fit_result]
        print(fit_result, result.keys())   
        image = np.average(result['galaxy'][mask, :, :], axis=0)
        ax = plt.subplot2grid((nrow, ncol), (1+2*f,0), rowspan=2, colspan=2)
        ax.imshow(image, cmap='Greys', interpolation='nearest', origin='lower')
        ax.xaxis.set_major_locator(NullLocator())
        ax.yaxis.set_major_locator(NullLocator())
        ax.set_ylabel(fit_result)
        
        for i_t in range(nt):
            galaxy_model = result['galeval'][i_t]
            psf_model = result['psfeval'][i_t]
            prediction = (result['skys'][i_t][:, None, None] + galaxy_model +
                          result['sn'][i_t][:, None, None] * psf_model)
            residual = prediction - data_cubes[i_t].data
            
            prediction_image = np.average(prediction[mask, :, :], axis=0)
            residual_image = np.average(residual[mask, :, :], axis=0)
            ax1 = plt.subplot2grid((nrow, ncol), (1+2*f,i_t+2))
            ax1.imshow(prediction_image, cmap='Greys', interpolation='nearest',
                      origin='lower')
            ax2 = plt.subplot2grid((nrow, ncol), (1+2*f+1,i_t+2))
            ax2.imshow(residual_image, cmap='Greys', interpolation='nearest',
                      origin='lower')
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
                        #hspace=0.01,
                        wspace=0.01)

    if fname is None:
        return fig
    
    plt.savefig(fname)
    plt.close()


def plot_wave_slices(ddt_output,fname=None, slices = None):
    """Return a figure showing wavelength slices of final ref data and model.

    Parameters
    ----------
    ddt_output : pickle dictionary
        Dictionary with data and results of ddt fit at three steps
    fname : str
        Output file name
    slices : list
        Optional indices of wavelengths to plot.
    """

    data_cubes = ddt_output['Data']
    refs = ddt_output['Refs']
    nt = len(data_cubes)
    cube_shape = data_cubes[0].data.shape
    wave = data_cubes[0].wave
    if slices is not None:
        wave_slices = slices
    else:
        wave_slices = np.arange(0, len(wave), 100)

    # two rows per final ref (model and residual)
    ncol = 2*len(refs)
    #  one column for each wavelength slice
    nrow = len(wave_slices)
    figsize = (STAMP_SIZE * ncol, STAMP_SIZE * nrow)
    fig = plt.figure(figsize=figsize)

    """Plot data, model and residual for each final ref epoch at a range of
    wavelength slices"""

    for r, i_t in enumerate(refs):
        
        for s, slice in enumerate(wave_slices):
            data_slice = data_cubes[i_t].data[slice,:,:]
            model_slice = ddt_output['FinalFit']['galeval'][i_t][slice,:,:]
            residual_slice = data_slice - model_slice

            vmin = None
            vmax = None
            ax1 = plt.subplot2grid((nrow, ncol), (r*2, s))
            ax2 = plt.subplot2grid((nrow, ncol), (r*2+1, s))

            ax1.imshow(model_slice, vmin=vmin, vmax=vmax,
                       interpolation='nearest', origin='lower')
            ax2.imshow(residual_slice, interpolation='nearest',
                       vmin = vmin, vmax=vmax, origin='lower')

            ax1.xaxis.set_major_locator(NullLocator())
            ax1.yaxis.set_major_locator(NullLocator())
            ax2.xaxis.set_major_locator(NullLocator())
            ax2.yaxis.set_major_locator(NullLocator())
            if s == 0:
                ax1.set_ylabel('%s Model' % i_t, fontsize=8)
                ax2.set_ylabel('%s Residual' % i_t, fontsize=8)
            if r == len(refs) -1:
                ax2.set_xlabel('$\lambda =%s$' % wave[slice], fontsize=10)
    fig.subplots_adjust(left=0.03, right=0.999, bottom=0.04, top=0.98,
                        hspace=0.01, wspace=0.01)

    if fname is None:
        return fig
    
    plt.savefig(fname)
    plt.close()


def plot_sn(ddt_output, json_file, fname=None):
    """Return a figure showing wavelength slices of final ref data and model.

    Parameters
    ----------
    ddt_output : pickle dictionary
        Dictionary with data and results of ddt fit at three steps
    json_file : str
        Name of DDT input file, needed to get exposure numbers.
    fname : str
        Output file name
    slices : list
        Optional indices of wavelengths to plot.
    """

    data_cubes = ddt_output['Data']
    refs = ddt_output['Refs']
    nt = len(data_cubes)
    cube_shape = data_cubes[0].data.shape
    wave = data_cubes[0].wave
    sn_spectra = ddt_output['FinalFit']['sn']
    sn_max = max(sn_spec.max() for sn_spec in sn_spectra)

    cfg = json.load(open(json_file))
    in_cube_files = cfg["IN_CUBE"]
    channel = cfg["PARAM_CHANNEL"]
    day_exp_nums = [file.split('_')[2:5] for file in in_cube_files]

    sn_name = json_file.split('/')[-1].split('_')[0]
    idr_files = glob(IDR_PREFIX+'%s/*%s.fits' % (sn_name, channel))
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
