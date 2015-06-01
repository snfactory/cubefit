import math
import numpy as np

MMHG_PER_MBAR = 760./1013.25
ARCSEC_PER_RADIAN = 206265.

__all__ = ["differential_refraction", "paralactic_angle"]

def paralactic_angle(airmass, ha, dec, tilt, lat):
    """Return paralactic angle in radians, including MLA tilt

    Parameters
    ----------
    airmass : float
    ha : float
    dec : float
    tilt : float
    lat : float
        Earth latitude of instrument in radians.
    """

    cos_z = 1./airmass

    # TODO: original Yorick code says this breaks when ha = 0
    #       but not clear to me why that would be.
    sin_z = math.sqrt(1. - cos_z**2)

    # treat individually cases where airmass == 1.
    if sin_z == 0.:
        sin_paralactic = 0.
        cos_paralactic = 1.
    else:
        sin_paralactic = math.sin(ha) * math.cos(lat) / sin_z
        cos_paralactic = (math.sin(lat)*math.cos(dec) -
                          math.cos(lat)*math.sin(dec)*math.cos(ha)) / sin_z

    # add the tilt
    # alpha = paralactic - tilt
    sin_xy = sin_paralactic*math.cos(tilt) - math.sin(tilt)*cos_paralactic
    cos_xy = cos_paralactic*math.cos(tilt) + math.sin(tilt)*sin_paralactic
    
    # Consistency test (TODO: Move this to tests?)
    # Is this test correct? shouldn't sin^2 + cos^2 = 1?
    one_xy = sin_xy**2 + cos_xy**2 - 1.
    if one_xy >= 1. and airmass > 1.01:
        raise RuntimeError("something went wrong with paralactic angle calc")

    # arctan gives an angle between -pi and pi
    # and paralactic_angle is between 0 and 2pi, positive from North toward
    # East.
    # NB: due to numerical resolution, tan(pi/2.) is well defined.
    return math.atan2(sin_xy, cos_xy)


def water_vapor_pressure(t):
    """Vapor pressure of water in mBar, given temperature in Celcius

    Parameters
    ----------
    t : array or float
    """
    t = t + 273.15
    return np.exp(-7205.76 / t) / 100. * (50292. / t)**6.2896


def refraction_index_of_air_minus_one(p, t, h, wave):
    """Return n(lambda) - 1, where n(lambda) is the refraction index of air.
    
    Based on Edlen formula:
 
    n - 1 = 6.4328e-5 + 2.94981e-2/(1.46e2 - w) + 2.554e-4/(4.1e-1 - w)
 
    where w = (1 micron / lambda)^2.

    Parameters
    ----------
    wave : 1-d array
        Wavelength in Angstroms. 
    p : 1-d array
        Pressure in mBar
    t : 1-d array
        Temperature in Celcius
    h : 1-d array
        Humidity in percentage (values in range 0 - 100)

    Returns
    -------
    value : 2-d array
        shape is (len(p), len(wave))
    """

    # expand inputs if < 1d
    wave = np.atleast_1d(wave)
    p = np.atleast_1d(p)
    t = np.atleast_1d(t)
    h = np.atleast_1d(h)
    assert p.ndim == t.ndim == h.ndim == wave.ndim == 1
    assert p.shape == t.shape == h.shape

    wave = wave * 1.e-4      # convert wavelength to microns
    invwave2 = (1./wave)**2  # inverse microns squared

    n = (6.4328e-5 +
         2.94981e-2 / (1.46e2 - invwave2) +
         2.554e-4 / (4.1e1 - invwave2))   # 1-d array

    # pressure and temperature correction (1-d array):
    p = p*MMHG_PER_MBAR
    pt_corr = (((1.049 - 0.0157*t) * 1.e-6 * p + 1.) /
               (720.883*(1. + 0.003661*t)) * p)
  
    # humidity correction, depends on wavelength (2-d array)
    pvapor = water_vapor_pressure(t) * h / 100. * MMHG_PER_MBAR
    h_corr = (1.e-6 * pvapor[:, None] /
              (1. + 0.003661*t[:, None]) *
              (0.0624 - 0.00068*invwave2))

    return pt_corr[:, None] * n - h_corr

def differential_refraction(airmass, p, t, h, wave, wave_ref):
    """Differential refraction in arcseconds.

    Parameters
    ----------
    airmass, p, t, h : 1-d array
        Observing parameters: airmass, pressure, temperature, humidty
    wave : 1-d array
        Wavelengths
    wave_ref : float
        Reference wavelength

    Returns
    -------
    delta_r : 2-d array
        2-d array of shape (n_airmass, n_wave)
    """
  
    n2 = refraction_index_of_air_minus_one(p, t, h, wave)
    n1 = refraction_index_of_air_minus_one(p, t, h, wave_ref)
    ndiff = (n2 - n1) * ARCSEC_PER_RADIAN  # 2-d array

    tan_z = np.sqrt(airmass**2 - 1.)  # 1-d array
  
    return tan_z[:, None] * ndiff
