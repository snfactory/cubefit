


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
    mu_xy : float
    mu_wave : float
    sky : np.ndarray (2-d)

    Notes
    -----
    The spatial coordinate system used to align data and the model is
    an arbitrary grid on the sky where the origin is chosen to be at
    the true location of the SN. That is, in the model the SN is
    located at (0., 0.)  by definition and the location and shape of
    the galaxy is defined relative to the SN location. Similarly, the
    pointings/alignment of the data are defined relative to this
    coordinate system. For example, if a pointing is centered at
    coordinates (3.5, 2.5), this means that we believe the central
    spaxel of the data array to be 3.5 spaxels west and 2.5 spaxels
    north of the true SN position. (The units of the coordinate system
    are in spaxels for convenience.)
    """

    MODEL_SHAPE = (32, 32)

    def __init__(self, shape, psf_ellipticity, psf_alpha, adr_dx, adr_dy,
                 spaxel_size, mu_xy, mu_wave, sky):

        ny, nx = MODEL_SHAPE
        nt, nw = shape

        if psf_ellipticity.shape != shape:
            raise ValueError("psf_ellipticity has wrong shape")
        if psf_alpha.shape != shape:
            raise ValueError("psf_alpha has wrong shape")

        # Model shape
        self.nt = nt
        self.nw = nw
        self.ny = ny
        self.nx = nx

        # Galaxy and sky part of the model
        self.gal = np.zeros((nw, ny, nx))
        self.galprior = np.zeros((nw, ny, nx))
        self.sky = sky
        self.sn = np.zeros((nt, nw))
        self.eta = np.ones(nt)  # TODO: what is eta again?
        self.final_ref_sky = np.zeros(nw)

        # PSF part of the model
        self.psf = gaussian_plus_moffat_psf_4d(MODEL_SHAPE, 15.0, 15.0,
                                               psf_ellipticity, psf_alpha)

        # Pointing of data. This is the location of the central spaxel of the
        # data with respect to the position of the SN in the model. (The
        # position of the SN in the model defines the coordinate system; see
        # comments in docstring.)
        self.data_xctr = np.zeros(nt, dtype=np.float64)
        self.data_yctr = np.zeros(nt, dtype=np.float64)

        # Make up a coordinate system for the model array
        #offx = int((nx-1) / 2.)
        #offy = int((ny-1) / 2.)
        #xcoords = np.arange(-offx, nx - offx)  # x coordinates on array
        #ycoords = np.arange(-offy, ny - offy)  # y coordinates on array

        # sn is "by definition" at array position where coordinates = (0,0)
        # model_sn_x = offx
        # model_sn_y = offy

        # This moves the center of the PSF from array coordinates
        # (model_sn_x, model_sn_y) -> (0, 0) [lower left pixel]
        # I don't know why this is done.
        #self.psf = roll_psf(self.psf, -self.model_sn_x, -self.model_sn_y)
        #self.psf_rolled = True

        self.adr_dx = adr_dx
        self.adr_dy = adr_dy
        self.spaxel_size = spaxel_size
        self.mu_xy = mu_xy
        self.mu_wave = mu_wave
