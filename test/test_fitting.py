from __future__ import print_function

import numpy as np
from numpy.testing import assert_allclose

import ddtpy


def test_determine_sky_and_sn():
    truesky = 3. * np.ones((10,))
    truesn = 2. * np.ones((10,))

    gal = np.ones((10, 5, 5))
    psf = np.zeros((10, 5, 5))
    psf[:, 3, 3] = 1.  # psf is a single pixel

    data = gal + truesky[:, None, None] + truesn[:, None, None] * psf
    weight = np.ones_like(data)
    
    sky, sn = ddtpy.fitting.determine_sky_and_sn(gal, psf, data, weight)

    print(sky)
    print(sn)
    
