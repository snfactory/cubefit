"""Integration tests, testing main scripts."""

from __future__ import division

import json
import math
import os

import numpy as np
import fitsio

import cubefit


def add_gaussian(A, y, x, sigma_y, sigma_x, angle, scale):
    """Add a 2-d gaussian to 2-d array A."""
    
    gnorm = 1. / (2. * np.pi * sigma_x * sigma_y)

    ny, nx = A.shape
    dy = np.arange(-ny/2.0 + 0.5 - y, ny/2.0 + 0.5 - y)
    dx = np.arange(-nx/2.0 + 0.5 - x, nx/2.0 + 0.5 - x)
    DX, DY = np.meshgrid(dx, dy)

    # Offsets in rotated coordinate system (DX', DY')
    DXp = DX * np.cos(angle) - DY * np.sin(angle)
    DYp = DX * np.sin(angle) + DY * np.cos(angle)

    norm = scale / (2. * np.pi * sigma_x * sigma_y)
    A += norm * np.exp(-DXp**2 / (2.*sigma_x**2) - DYp**2 / (2.*sigma_y**2))


def rand_galaxy(nw, ny, nx):
    """Create a random 3-d galaxy with `nw` wavelengths.
    For now, this is just a few gaussians."""

    NCOMP = 2  # number of gaussians

    # Spatial parameters
    y = np.random.uniform(-8., 8., size=NCOMP)
    x = np.random.uniform(-8., 8., size=NCOMP)
    sigma_y = np.random.uniform(1.5, 8., size=NCOMP)
    sigma_x = np.random.uniform(1.5, 8., size=NCOMP)
    angle = np.random.uniform(0., 2. * np.pi, size=NCOMP)
    scale = np.random.uniform(1., 3., size=NCOMP)

    # Evaluate spatial shape
    A = np.zeros((ny, nx), dtype=np.float64)
    for i in range(NCOMP):
        add_gaussian(A, y[i], x[i], sigma_y[i], sigma_x[i], angle[i],
                     scale[i])

    # 1-d spectrum
    idx = np.linspace(0., 2. * np.pi, nw)
    spectrum = np.ones(nw) + 0.5 * np.cos(idx) + 0.3 * np.cos(2. * idx)

    return A * spectrum[:, None, None]


def generate_data():

    np.random.seed(0)

    MODEL_SHAPE = 32, 32
    NW = 100
    DATA_SHAPE = 15, 15
    NOBS = 5  # total observations
    REFS = [3, 4]  # indicies of refs
    MASTER_REF = 4  # index of master ref
    NOISE_RATIO = 0.01  # gaussian noise (sigma) vs peak value
    MINWAVE = 3200.
    MAXWAVE = 5500.

    wavewcs = {"CRVAL3": MINWAVE,
               "CRPIX3": 1,
               "CDELT3": (MAXWAVE - MINWAVE) / (NW - 1)}
    wave = np.linspace(MINWAVE, MAXWAVE, NW)

    # generate a random galaxy
    galaxy = rand_galaxy(NW, MODEL_SHAPE[0], MODEL_SHAPE[1])
    galaxy *= 1.0 / galaxy.max()  # scale so that max is 1.0

    # other "true" parameters
    yctr = np.random.uniform(-1.0, 1.0, size=NOBS)
    xctr = np.random.uniform(-1.0, 1.0, size=NOBS)
    yctr[MASTER_REF] = xctr[MASTER_REF] = 0.
    sky = [np.random.uniform(0.3, 0.7) * np.ones(NW) for _ in range(NOBS)]
    snctr = tuple(np.random.uniform(-2.0, 2.0, size=2))
    #snctr = (-2., 2.)

    # From SALT2 model, just for fun.
    sn_at_max = np.array([
        2.08427421e-13,   1.86628398e-13,   1.75669185e-13,
        1.76010069e-13,   1.89093943e-13,   2.14098436e-13,
        2.33239478e-13,   2.39518273e-13,   2.42990348e-13,
        2.56805649e-13,   2.78747706e-13,   2.99311151e-13,
        3.17882688e-13,   3.41603875e-13,   3.73262869e-13,
        4.02475485e-13,   4.17639939e-13,   4.18153305e-13,
        4.02418210e-13,   3.70403413e-13,   3.23825151e-13,
        2.63144182e-13,   2.15327657e-13,   2.01715968e-13,
        2.11842186e-13,   2.22485094e-13,   2.37142345e-13,
        2.85856207e-13,   3.72601798e-13,   4.50776414e-13,
        4.97265778e-13,   5.12373740e-13,   4.96411978e-13,
        4.61409587e-13,   4.50327390e-13,   4.65526813e-13,
        4.88532974e-13,   5.13970887e-13,   5.25563978e-13,
        5.11016121e-13,   4.81698395e-13,   4.60163230e-13,
        4.44310652e-13,   4.17221510e-13,   3.76715874e-13,
        3.33998367e-13,   2.94036252e-13,   2.69135693e-13,
        2.74209885e-13,   3.00515100e-13,   3.18046848e-13,
        3.25159663e-13,   3.32508457e-13,   3.43238508e-13,
        3.58420175e-13,   3.79003222e-13,   3.98917471e-13,
        4.03071804e-13,   3.92478403e-13,   3.81625416e-13,
        3.73621088e-13,   3.60710140e-13,   3.37113890e-13,
        3.08011878e-13,   2.85166301e-13,   2.69053011e-13,
        2.59916709e-13,   2.57747120e-13,   2.52876291e-13,
        2.37748353e-13,   2.17883563e-13,   2.08799993e-13,
        2.10746591e-13,   2.13573823e-13,   2.13954749e-13,
        2.12855495e-13,   2.11547843e-13,   2.11368561e-13,
        2.20102902e-13,   2.38804121e-13,   2.58285929e-13,
        2.71623629e-13,   2.79734279e-13,   2.85494055e-13,
        2.88993949e-13,   2.83819330e-13,   2.66621916e-13,
        2.40891059e-13,   2.16583951e-13,   1.94578199e-13,
        1.83149626e-13,   1.86441692e-13,   1.98908233e-13,
        2.03541344e-13,   1.99190111e-13,   1.89698964e-13,
        1.77758833e-13,   1.66318493e-13,   1.70576795e-13,
        1.92384176e-13])
    sn_at_max *= 1.0 / sn_at_max.max()  # scale to a max of 1.0
    
    # SN spectrum in each epoch
    sn = [0.2 * sn_at_max,
          1.0 * sn_at_max,
          5.0 * sn_at_max,
          np.zeros_like(sn_at_max),
          np.zeros_like(sn_at_max)]

    psf_params = [[1.753438, 1.751164, -0.119736, 1.717277],
                  [1.549305, 1.714546, -0.087633, 1.704836],
                  [1.531459, 1.932268, -0.181707, 1.867311],
                  [1.694011, 2.389584, -0.36936,  2.211401],
                  [1.612515, 2.276546, -0.323514, 2.127089]]

    cubes = []
    for i in range(NOBS):
        # atmospheric conditions used to create PSF; they also go in header
        header = {'AIRMASS': np.random.uniform(1., 1.5),
                  'TEMP': np.random.uniform(0., 5.),
                  'PRESSURE': np.random.uniform(615., 620.),
                  'PARANG': np.random.uniform(-180., 180.),
                  'CHANNEL': 'B'}
        header.update(wavewcs)

        # create a PSF for convolving & shifting the galaxy model to create
        # the data.
        sigma, alpha, beta, ellipticity, eta, yc, xc = \
            cubefit.main.snfpsfparams(wave, psf_params[i], header)
        A = cubefit.psffuncs.gaussian_moffat_psf(
            sigma, alpha, beta, ellipticity, eta,
            yc, xc, MODEL_SHAPE, subpix=1)
        psf = cubefit.TabularPSF(A)

        # create data from sky + galaxy + SN
        g = psf.evaluate_galaxy(galaxy, DATA_SHAPE, (yctr[i], xctr[i]))
        s = psf.point_source(snctr, DATA_SHAPE, (yctr[i], xctr[i]))
        data = sky[i][:, None, None] + g + sn[i][:, None, None] * s

        # add error
        error = NOISE_RATIO * (g + sky[i][:, None, None]).max()
        data += np.random.normal(scale=error, size=data.shape)
        weight = (1. / error**2) * np.ones_like(data)

        cube = cubefit.DataCube(data, weight, wave, header=header)
        cubes.append(cube)

    # Build a configuration dictionary that we'll need in order to run
    # cubefit.
    conf = {"xcenters": [0.0 for _ in range(NOBS)],
            "ycenters": [0.0 for _ in range(NOBS)],
            "psf_params": psf_params,
            "refs": REFS,
            "master_ref": MASTER_REF}

    return conf, cubes


def write_datacube(cube, fname):

    SCALE_FACTOR = 1.e17

    if os.path.exists(fname):
        os.remove(fname)

    scaled_data = (1. / SCALE_FACTOR) * cube.data
    scaled_var = (1. / SCALE_FACTOR**2) / cube.weight

    f = fitsio.FITS(fname, "rw")
    f.write(np.asarray(scaled_data, dtype=np.float32), header=cube.header)
    f.write(np.asarray(scaled_var, dtype=np.float32))
    f.close()


def test_cubefit():
    
    # make a temp directory to hold the data
    #dirname = tempfile.mkdtemp()
    testdir = os.path.dirname(os.path.abspath(__file__))
    dirname = os.path.join(testdir, "temp")
    if not os.path.exists(dirname):
        os.path.mkdir(dirname)

    conf, cubes = generate_data()

    # input/output paths
    filenames = [os.path.join(dirname, "epoch{:02d}.fits".format(i))
                 for i in range(len(cubes))]
    outnames = [os.path.join(dirname, "epoch{:02d}_sub.fits".format(i))
                for i in range(len(cubes))]
    sn_outnames = [os.path.join(dirname, "epoch{:02d}_sn.fits".format(i))
                   for i in range(len(cubes))]
    conf["filenames"] = filenames
    conf["outnames"] = outnames
    conf["sn_outnames"] = sn_outnames

    # write data
    for i in range(len(cubes)):
        write_datacube(cubes[i], filenames[i])

    # write config file
    configfname = os.path.join(dirname, "conf.json")
    with open(configfname, 'w') as f:
        json.dump(conf, f)

    # run cubefit
    resultfname = os.path.join(dirname, "result.fits")
    cubefit.cubefit(argv=[configfname, resultfname])
#                          "--mu_xy", "0.0", "--mu_wave", "0.0"])

    # run cubefit-subtract
    cubefit.cubefit_subtract([configfname, resultfname])

    # run cubefit-plot
    plotprefix = os.path.join(dirname, "plot")
    cubefit.cubefit_plot(argv=[configfname, resultfname, plotprefix,
                               "--plotepochs"])

if __name__ == "__main__":
    test_cubefit()
