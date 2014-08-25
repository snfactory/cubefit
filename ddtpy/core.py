from __future__ import print_function, division

from copy import deepcopy
import json

import numpy as np

from .psf import params_from_gs, gaussian_plus_moffat_psf_4d, roll_psf
from .io import read_dataset, read_select_header_keys
from .adr import calc_paralactic_angle, differential_refraction
from .data_toolbox import sky_guess_all, fft_shift_phasor

__all__ = ["main"]


def stringify(obj):
    """Create a string representation.

    For now this just lists all the members of the object and for arrays,
    their shapes.
    """

    lines = ["Members:"]
    for name, val in obj.__dict__.iteritems():
        if isinstance(val, np.ndarray):
            info = "{0:s} array".format(val.shape)
        else:
            info = repr(val)
        lines.append("  {0:s} : {1}".format(name, info))
    return "\n".join(lines)


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




class DDTModel(object):
    """This class is the equivalent of everything else that isn't data
    in the Yorick version.

    Parameters
    ----------
    shape : 2-tuple of int
        Model dimensions in (time, wave). Time and wave must match
        that of the data.
    psf_ellipticity, psf_alpha : np.ndarray (2-d)
        Parameters characterizing the PSF at each time, wavelength. Shape
        of both must match `shape` parameter.
    adr_dx, adr_dy : np.ndarray (2-d)
        Atmospheric differential refraction in x and y directions, in spaxels,
        relative to reference wavelength.
    spaxel_size : float
        Spaxel size in arcseconds.
    """
    MODEL_SHAPE = 32, 32

    def __init__(self, shape, psf_ellipticity, psf_alpha, adr_dx, adr_dy,
                 spaxel_size)):

        ny, nx = MODEL_SHAPE
        nt, nw = shape

        if psf_ellipticity.shape != shape:
            raise ValueError("psf_ellipticity has wrong shape")
        if psf_alpha.shape != shape:
            raise ValueError("psf_alpha has wrong shape")

        self.nt = nt
        self.nw = nw
        self.ny = ny
        self.nx = nx
        self.gal = np.zeros((nw, ny, nx))
        self.galprior = np.zeros((nw, ny, nx))
        self.sky = np.zeros((nt, nw))
        self.sn = np.zeros((nt, nw))
        self.eta = np.ones(nt)
        self.final_ref_sky = np.zeros(nw)

        # initialize PSF part of the model

        # Make up a coordinate system for the model array
        offx = int((nx-1) / 2.)
        offy = int((ny-1) / 2.)
        xcoords = np.arange(-offx, nx - offx)  # x coordinates on array
        ycoords = np.arange(-offy, ny - offy)  # y coordinates on array
        self.psf = gaussian_plus_moffat_psf_4d(xcoords, ycoords,
                                               psf_ellipticity, psf_alpha)

        # sn is "by definition" at array position where coordinates = (0,0)
        # model_sn_x = offx
        # model_sn_y = offy

        # This moves the center of the PSF from array coordinates
        # (model_sn_x, model_sn_y) -> (0, 0) [lower left pixel]
        # I don't know why this is done.
        self.psf = roll_psf(self.psf, -self.model_sn_x, -self.model_sn_y)
        self.psf_rolled = True

        self.adr_dx = adr_dx
        self.adr_dy = adr_dy
        self.spaxel_size = spaxel_size


def main(filename):
    """Do everything.

    Parameters
    ----------
    filename : str
        JSON-formatted config file.
    """

    with open(filename) as f:
        conf = json.load(f)

    # check apodizer flag because the code doesn't support it
    if conf.get("FLAG_APODIZER", 0) >= 2:
        raise RuntimeError("FLAG_APODIZER >= 2 not implemented")

    spaxel_size = conf["PARAM_SPAXEL_SIZE"]
    
    # Reference wavelength. Used in PSF parameters and ADR.
    wave_ref = conf.get("PARAM_LAMBDA_REF", 5000.)

    # index of final ref. Subtract 1 due to Python zero-indexing.
    master_final_ref = conf["PARAM_FINAL_REF"]-1

    # also want array with true/false for final refs.
    is_final_ref = np.array(conf.get("PARAM_IS_FINAL_REF"))

    n_iter_galaxy_prior = conf.get("N_ITER_GALAXY_PRIOR")

    # Load the header from the final ref or first cube
    i = master_final_ref if (master_final_ref >= 0) else 0
    header = read_select_header_keys(conf["IN_CUBE"][i])

    # Load data from list of FITS files.
    data, weight, wave = read_dataset(conf["IN_CUBE"])

    # Zero-weight array elements that are NaN
    # TODO: why are there nans in here?
    mask = np.isnan(data)
    data[mask] = 0.0
    weight[mask] = 0.0

    ddtdata = DDTData(data, weight, wave, is_final_ref, master_final_ref,
                      header, spaxel_size)

    # Load PSF model parameters. Currently, the PSF in the model is represented
    # by an arbitrary 4-d array that is constructed here. The PSF depends
    # on some aspects of the data, such as wavelength.
    # If different types of PSFs need to do different things, we may wish to
    # represent the PSF with a class, called something like GaussMoffatPSF.
    #
    # GS-PSF --> ES-PSF
    # G-PSF --> GR-PSF
    if conf["PARAM_PSF_TYPE"] == "GS-PSF":
        es_psf_params = np.array(conf["PARAM_PSF_ES"])
        psf_ellipticity, psf_alpha = psf_params_from_gs(
            es_psf_params, ddtdata.wave, wave_ref)
    elif conf["PARAM_PSF_TYPE"] == "G-PSF":
        raise RuntimeError("G-PSF (from FITS files) not implemented")
    else:
        raise RuntimeError("unrecognized PARAM_PSF_TYPE")

    # The following section relates to atmospheric differential
    # refraction (ADR): Because of ADR, the center of the PSF will be
    # different at each wavelength, by an amount that we can determine
    # (pretty well) from the atmospheric conditions and the pointing
    # and angle of the instrument. We calculate the offsets here as a function
    # of observation and wavelength and input these to the model.

    # atmospheric conditions at each observation time.
    airmass = np.array(conf["PARAM_AIRMASS"])
    p = np.asarray(conf.get("PARAM_P", 615.*np.ones_like(airmass)))
    t = np.asarray(conf.get("PARAM_T", 2.*np.ones_like(airmass)))
    h = np.asarray(conf.get("PARAM_H", np.zeros_like(airmass)))

    # Position of the instrument
    ha = np.deg2rad(np.array(conf["PARAM_HA"]))   # config files in degrees
    dec = np.deg2rad(np.array(conf["PARAM_DEC"])) # config files in degrees
    tilt = conf["PARAM_MLA_TILT"]

    # differential refraction as a function of time, wavelength (2-d array)
    # in arcseconds.
    delta_r = differential_refraction(airmass, p, t, h, ddtdata.wave, wave_ref)
    delta_r /= spaxel_size  # convert from arcsec to spaxels
    paralactic_angle = calc_paralactic_angle(airmass, ha, dec, tilt)
    adr_dx = -delta_r * np.sin(paralactic_angle)[:, None]  # O'xp <-> - east
    adr_dy = delta_r * np.cos(paralactic_angle)[:, None]

    # Get initial position of SN.
    target_xp = np.asarray(conf.get("PARAM_TARGET_XP",
                                    np.zeros_like(airmass)))
    target_yp = np.asarray(conf.get("PARAM_TARGET_YP",
                                    np.zeros_like(airmass)))

    # Initialize model
    model = DDTModel((data.nt, data.nw), psf_ellipticity, psf_alpha,
                     adr_dx, adr_dy, spaxel_size)

    # TODO : I Don't think this is needed anymore.                            
    # flag_apodizer = bool(conf.get("FLAG_APODIZER", 0))

    # Move the following operations to the model (when fitting?)
    # self.sn_offset_x_ref = deepcopy(self.target_xp)
    # self.sn_offset_y_ref = deepcopy(self.target_yp)
    # self.sn_offset_x = (self.sn_offset_x_ref[:, None] +  # 2-d arrays (nt, nw)
    #                     self.delta_r * self.adr_x[:, None])
    # self.sn_offset_y = (self.sn_offset_y_ref[:, None] +
    #                     self.delta_r * self.adr_y[:, None])

# -----------------------------------------------------------------------------
# old 
# -----------------------------------------------------------------------------

        # TODO : setup FFT(s)
        # Placeholder FFT:
        self.FFT = np.fft.fft


        # if no pointing error, the supernova is at  
        # int((N_x + 1) / 2 ), int((N_y + 1) / 2 )
        # in Yorick indexing for both the MLA and the model

        offset_x = int((self.psf_nx-1.) / 2.) - int((self.data_nx-1.) / 2.)
        offset_y = int((self.psf_ny-1.) / 2.) - int((self.data_ny-1.) / 2.)

        self.range_x = slice(offset_x, offset_x + self.data_nx)
        self.range_y = slice(offset_y, offset_y + self.data_ny)

        # equilvalent of ddt_setup_apodizer() in Yorick,
        # with without implementing all of it:
        # if self.flag_apodizer < 2:
        #     self.apodizer = None
        #     self.psf_enlarge = None
        # else:
        #     raise RuntimeError("FLAG_APODIZER >= 2 not implemented")

        # This makes a first guess at the sky by recursively removing outliers
        self.guess_sky = sky_guess_all(self.data, self.weight, self.nw, self.nt)

        # equivalent of ddt_setup_regularization...
        # In original, these use "sky=guess_sky", but I can't find this defined.
        # Similarly, can't find DDT_CHEAT_NO_NORM
        self.regul_galaxy_xy = RegulGalaxyXY(self.data[self.final_ref], 
                                            self.weight[self.final_ref],
                                            conf["MU_GALAXY_XY_PRIOR"],
                                            sky=self.guess_sky[self.final_ref])
        self.regul_galaxy_lambda = RegulGalaxyLambda(
                                        self.data[self.final_ref], 
                                        self.weight[self.final_ref],
                                        conf["MU_GALAXY_LAMBDA_PRIOR"],
                                        sky=self.guess_sky[self.final_ref]) 
        self.verb = True


    # TODO: make a better name for this.
    def r(self, x):
        """Returns just the section of the model (x) that overlaps the data.

        This is the same as the Yorick ddt.R operator with job = 0.
        In Yorick, `x` could be 2 or more dimensions; here it must be 4-d.
        """
        return x[:, :, self.range_y, self.range_x]

    def r_inv(self, x):
        """creates a model with MLA data at the right location.

        This is the same as the Yorick ddt.R operator with job = 1.
        """
        shape = (x.shape[0], x.shape[1], self.psf_ny, self.psf_nx)
        y = np.zeros(shape, dtype=np.float32)
        y[:, :, self.range_y, self.range_x] = x
        return y
        
    def H(self, x, i_t, offset=None):
        """Convolve x with psf
        this is where ADR is treated as a phase shift in Fourier space.
        
        Parameters
        ----------
        x : 3-d array
        i_t : int
        offset : 1-d array
        
        Returns
        -------
        3-d array
        """
        if not self.psf_rolled:
            raise ValueError("<ddt_H> need the psf to be rolled!")
        
        psf = self.psf[i_t]
        if len(x.shape) == 1:
            x = x.reshape(self.model_gal.shape)
        ptr = np.zeros(psf.shape)
        number = ptr.shape[0]

        for k in range(number):
            
            phase_shift_apodize = fft_shift_phasor(
                                        [self.psf_ny, self.psf_nx],
                                        [self.sn_offset_y[i_t,k],
                                         self.sn_offset_x[i_t,k]],
                                        half=1, apodize=self.apodizer)
            ptr[k] = self.FFT(psf[k,:,:] * phase_shift_apodize)

        return self._convolve(ptr, x, offset=offset)
                            
    def _convolve(self, ptr, x, offset=None):
        """This convolves two functions using DDT.FFT
        Will need to be adapted if other FFT needs to be an option
        
        Parameters
        ----------
        ptr : 1-d array
        x : 3-d array
        offset : 1-d array
        
        Returns
        -------
        out : 3-d array
        
        Notes
        -----
        job = 0: direct
        job = 1: gradient
        job = 2: add an offset, in SPAXELS
        """
        number = ptr.shape[0]
        out = np.zeros(x.shape)
        
        if offset == None:
            for k in range(number):
                # TODO: Fix when FFT is sorted out:
                #out[k,:,:] = self.FFT(ptr[k] * self.FFT(x[k,:,:]),2)
                out[k,:,:] = self.FFT(ptr[k]*self.FFT(x[k,:,:]))
            return out

        else:
            phase_shift_apodize = fft_shift_phasor(
                                               [self.psf_ny, self.psf_nx], 
                                               offset, half=1,
                                               apodize=self.apodizer)
            for k in range(number):
                out[k,:,:] = self.FFT(ptr[k] * phase_shift_apodize *
                                      self.FFT(x[k,:,:]))    
                #out[k,:,:] = self.FFT(ptr[k] * phase_shift_apodize *
                #                      self.FFT(x[k,:,:]),2)
            return out 


def sky_guess_all(ddt_data, ddt_weight, nw, nt):
    """guesses sky with lower signal spaxels compatible with variance
    Parameters
    ----------
    ddt_data, ddt_weight : 4d arrays
    nw, nt : int
    Returns
    -------
    sky : 2d array
    Notes
    -----
    sky_cut: number of sigma used for the sky guess, was unused option in DDT
    """

    sky = np.zeros((nt, nw))
    sky_cut = 2.0
    
    for i_t in range(nt):
        data = ddt_data[i_t]
        weight = ddt_weight[i_t]
        
        var = 1./weight
        ind = np.zeros(data.size)
        prev_npts = 0
        niter_max = 10
        niter = 0
        nxny = data.shape[1]*data.shape[2]
        while (ind.size != prev_npts) and (niter < niter_max):
            prev_npts = ind.size
            #print "<ddt_sky_guess> Sky: prev_npts %d" % prev_npts
            I = (data*weight).sum(axis=-1).sum(axis=-1)
            i_Iok = np.where(weight.sum(axis=-1).sum(axis=-1) != 0.0)
            if i_Iok[0].size != 0:
                I[i_Iok] /= weight.sum(axis=-1).sum(axis=-1)[i_Iok]
                sigma = (var.sum(axis=-1).sum(axis=-1)/nxny)**0.5
                ind = np.where(abs(data - I[:,None,None]) > 
                               sky_cut*sigma[:,None,None])
                if ind[0].size != 0:
                    data[ind] = 0.
                    var[ind] = 0.
                    weight[ind] = 0.
                else:
                    #print "<ddt_sky_guess> no more ind"
                    break
            else:
                break
            niter += 1
                
        sky[i_t] = I
        
    return sky
    
