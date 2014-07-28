from __future__ import print_function, division

from copy import deepcopy
import json

import numpy as np

from .psf import params_from_gs, gaussian_plus_moffat_psf_4d, roll_psf
from .io import read_dataset, read_select_header_keys
from .adr import calc_paralactic_angle, differential_refraction
from .regul_toolbox import RegulGalaxyXY, RegulGalaxyLambda
from .data_toolbox import sky_guess_all
__all__ = ["DDT"]

class DDT(object):
    """This class is analagous to the ddt object that gets created by
    ddt_setup_ddt() in Yorick.

    The namespace has been flattened relative to the Yorick version.
    For example, suppose we have an instance of this class named `ddt`.
    Rather than `ddt.ddt_data.data` (as in the Yorick version), we have
    `ddt.data`.
    """

    def __init__(self, filename):
        """Initialize from a JSON-formatted file."""
        
        with open(filename) as f:
            conf = json.load(f)

        self.spaxel_size = conf["PARAM_SPAXEL_SIZE"]

        # index of final ref. Subtract 1 due to Python zero-indexing.
        self.final_ref = conf["PARAM_FINAL_REF"]-1
        # also want array with true/false for final refs.
        self.is_final_ref = np.array(conf.get("PARAM_IS_FINAL_REF"))

        # Load the heidader from the final ref or first cube
        idx = self.final_ref if (self.final_ref >= 0) else 0
        self.header = read_select_header_keys(conf["IN_CUBE"][idx])

        # Load data from FITS files.
        self.data, self.weight, self.wave = read_dataset(conf["IN_CUBE"])
        # TODO: why are there nans in here?
        zz = np.where(np.isnan(self.data))
        self.data[zz] = 0.0
        self.weight[zz] = 0.0
        
        # save sizes for convenience
        self.nt = len(self.data)
        self.nw = len(self.wave)
        self.data_ny = self.data.shape[2]  # 15 for SNIFS
        self.data_nx = self.data.shape[3]  # 15 for SNIFS

        # Load PSF model
        # GS-PSF --> ES-PSF
        # G-PSF --> GR-PSF
        if conf["PARAM_PSF_TYPE"] == "GS-PSF":
            self.psf_nx, self.psf_ny = 32, 32
            offx = int((self.psf_nx-1) / 2.)
            offy = int((self.psf_ny-1) / 2.)

            # x, y coordinates on psf array (1-d)
            self.psf_x = np.arange(-offx, self.psf_nx - offx)
            self.psf_y = np.arange(-offy, self.psf_ny - offy)

            # sn is "by definition" at array position where coordinates = (0,0)
            self.model_sn_x = offx
            self.model_sn_y = offy

            es_psf = np.array(conf["PARAM_PSF_ES"])
            ellipticity, alpha = params_from_gs(es_psf, self.wave)
            self.psf = gaussian_plus_moffat_psf_4d(self.psf_x, self.psf_y,
                                                   ellipticity, alpha)
        elif conf["PARAM_PSF_TYPE"] == "G-PSF":
            raise RuntimeError("G-PSF (from FITS files) not implemented")
        else:
            raise RuntimeError("unrecognized PARAM_PSF_TYPE")

        # Initialize rest of model: galaxy, sky, SN
        self.model_gal = np.zeros((self.nw, self.psf_ny, self.psf_nx))
        self.model_galprior = np.zeros((self.nw, self.psf_ny, self.psf_nx))
        self.model_sky = np.zeros((self.nt, self.nw))
        self.model_sn = np.zeros((self.nt, self.nw))
        self.model_eta = np.ones(self.nt)
        self.model_final_ref_sky = np.zeros(self.nw)
        self.n_iter_galaxy_prior = conf.get("N_ITER_GALAXY_PRIOR")

        # atmospheric differential refraction
        airmass = np.array(conf["PARAM_AIRMASS"])
        ha = np.deg2rad(np.array(conf["PARAM_HA"]))   # config files in degrees
        dec = np.deg2rad(np.array(conf["PARAM_DEC"])) # config files in degrees
        tilt = conf["PARAM_MLA_TILT"]
        p = np.asarray(conf.get("PARAM_P", 615.*np.ones_like(airmass)))
        t = np.asarray(conf.get("PARAM_T", 2.*np.ones_like(airmass)))
        h = np.asarray(conf.get("PARAM_H", np.zeros_like(airmass)))
        wave_ref = conf.get("PARAM_LAMBDA_REF", 5000.)
        paralactic_angle = calc_paralactic_angle(airmass, ha, dec, tilt)
        delta_r = differential_refraction(airmass, p, t, h, self.wave,
                                          wave_ref) # 2-d, shape = (nt, nw)
        delta_r /= self.spaxel_size  # convert from arcsec to spaxels

        self.delta_r = delta_r
        self.adr_x = -1. * np.sin(paralactic_angle)
        self.adr_y = np.cos(paralactic_angle)
 
         # O'xp <-> - east
        self.delta_xp = -1. * delta_r * np.sin(paralactic_angle)[:, None]
        self.delta_yp = delta_r * np.cos(paralactic_angle)[:, None]
        self.paralactic_angle = paralactic_angle
        
        self.target_xp = np.asarray(conf.get("PARAM_TARGET_XP",
                                             np.zeros_like(airmass)))
        self.target_yp = np.asarray(conf.get("PARAM_TARGET_YP",
                                             np.zeros_like(airmass)))

        self.flag_apodizer = bool(conf.get("FLAG_APODIZER", 0))

        # TODO : setup FFT(s)
        # Placeholder FFT:
        self.FFT = np.fft.fft

        # this was done in ddt_setup_R in Yorick
        self.sn_offset_x_ref = deepcopy(self.target_xp)
        self.sn_offset_y_ref = deepcopy(self.target_yp)

        # 2-d arrays in (time, wave)
        self.sn_offset_x = (self.sn_offset_x_ref[:, None] +
                            self.delta_r * self.adr_x[:, None])
        self.sn_offset_y = (self.sn_offset_y_ref[:, None] +
                            self.delta_r * self.adr_y[:, None])

        # if no pointing error, the supernova is at  
        # int((N_x + 1) / 2 ), int((N_y + 1) / 2 )
        # in Yorick indexing for both the MLA and the model

        offset_x = int((self.psf_nx-1.) / 2.) - int((self.data_nx-1.) / 2.)
        offset_y = int((self.psf_ny-1.) / 2.) - int((self.data_ny-1.) / 2.)

        self.range_x = slice(offset_x, offset_x + self.data_nx)
        self.range_y = slice(offset_y, offset_y + self.data_ny)

        # equilvalent of ddt_setup_apodizer() in Yorick,
        # with without implementing all of it:
        if self.flag_apodizer < 2:
            self.apodizer = None
            self.psf_enlarge = None
        else:
            raise RuntimeError("FLAG_APODIZER >= 2 not implemented")

        # This moves the center of the PSF from array coordinates
        # (model_sn_x, model_sn_y) -> (0, 0) [lower left pixel]
        # I don't know why this is done.
        self.psf = roll_psf(self.psf, -self.model_sn_x, -self.model_sn_y)
        self.psf_rolled = True
        
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

    def __str__(self):
        """Create a string representation.

        For now this just lists all the members of the object and for arrays,
        their shapes.
        """

        lines = ["Members:"]
        for name, val in self.__dict__.iteritems():
            if isinstance(val, np.ndarray):
                info = "{0:s} array".format(val.shape)
            else:
                info = repr(val)
            lines.append("  {0:s} : {1}".format(name, info))
        return "\n".join(lines)

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
        
    def H(self, x, i_t, job=0, offset=None):
        """Convolve x with psf
        this is where ADR is treated as a phase shift in Fourier space.
        
        Parameters
        ----------
        x : 3-d array
        i_t : int
        job : int
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
            # TODO: fix when fft_shift_phasor written
            phase_shift_apodize = 1  #fft_shift_phasor(
            #                            [self.psf_ny, self.pst_nx, 2],
            #                            [self.sn_offset_y[i_t,k],
            #                             self.sn_offset_x[i_t,k]],
            #                            half=1, apodize=self.apodizer)
            ptr[k] = self.FFT(psf[k,:,:] * phase_shift_apodize)
            
        return self._convolve(ptr, x, job=job, offset=offset)
                            
    def _convolve(self, ptr, x, job=0, offset=None):
        """This convolves two functions using DDT.FFT
        Will need to be adapted if other FFT needs to be an option
        
        Parameters
        ----------
        ptr : 1-d array
        x : 3-d array
        job : int, see below
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
        
        if job == 0:
            for k in range(number):
                # TODO: Fix when FFT is sorted out:
                #out[k,:,:] = self.FFT(ptr[k] * self.FFT(x[k,:,:]),2)
                out[k,:,:] = self.FFT(ptr[k]*self.FFT(x[k,:,:]))
            return out
        elif job == 1:
            for k in range(number):
                # out[k,:,:] = self.FFT(np.conj(ptr[k]) * self.FFT(x[k,:,:]),2)
                out[k,:,:] = self.FFT(np.conj(ptr[k]) * self.FFT(x[k,:,:]))
            return out
        elif job == 2:
            if offset is None:
                raise ValueError("<_convolve> job=2 need offset")
        
            phase_shift_apodize = 1#fft_shift_phasor(
            #                                   [self.psf_ny, self.psf_nx, 2], 
            #                                   offset, half=1,
            #                                   apodize=apodize)
            for k in range(number):
                out[k,:,:] = self.FFT(ptr[k] * phase_shift_apodize *
                                      self.FFT(x[k,:,:]))    
                #out[k,:,:] = self.FFT(ptr[k] * phase_shift_apodize *
                #                      self.FFT(x[k,:,:]),2)
            return out 
        else: 
            raise ValueError("unsupported JOB")
            
