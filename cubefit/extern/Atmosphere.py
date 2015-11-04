# -*- coding: utf-8 -*-
# Copied from SNfactory/Offline/Toolbox/Atmosphere.py
# cvs v1.23 on 2015 Nov 4.

# Modifications:
# - import statement for Coords
# - dict.iteritems -> dict.items to support Python 3.
# - `@add_attrs` decorators commented out.

import numpy as N
from . import Coords

RAD2DEG = 57.295779513082323            # 180/pi


#@add_attrs(source='NIST/IAPWS')
def _saturationVaporPressureOverWater(T):
    """See :func:`saturationVaporPressure`"""

    K1 = 1.16705214528e+03
    K2 = -7.24213167032e+05
    K3 = -1.70738469401e+01
    K4 = 1.20208247025e+04
    K5 = -3.23255503223e+06
    K6 = 1.49151086135e+01
    K7 = -4.82326573616e+03
    K8 = 4.05113405421e+05
    K9 = -2.38555575678e-01
    K10 = 6.50175348448e+02

    t = T + 273.15                      # °C → K
    x = t + K9 / (t - K10)
    A = x**2 + K1 * x + K2
    B = K3 * x**2 + K4 * x + K5
    C = K6 * x**2 + K7 * x + K8
    X = -B + N.sqrt(B**2 - 4 * A * C)
    psv = 1e6 * (2 * C / X)**4

    return psv


#@add_attrs(source='NIST/IAPWS')
def _saturationVaporPressureOverIce(T):
    """See :func:`saturationVaporPressure`"""

    A1 = -13.928169
    A2 = 34.7078238
    th = (T + 273.15) / 273.16
    Y = A1 * (1 - th**-1.5) + A2 * (1 - th**-1.25)
    psv = 611.657 * N.exp(Y)

    return psv


#@add_attrs(source='NIST/IAPWS')
def saturationVaporPressure(T=2.):
    """Compute saturation vapor pressure [Pa] for temperature *T* [°C]
    according to Edlén Calculation of the Index of Refraction from
    NIST 'Refractive Index of Air Calculator'.

    Source: http://emtoolbox.nist.gov/Wavelength/Documentation.asp
    """

    t = N.atleast_1d(T)
    psv = N.where(t >= 0,
                  _saturationVaporPressureOverWater(t),
                  _saturationVaporPressureOverIce(t))

    return psv                          # [Pa]


#@add_attrs(source='NIST/IAPWS')
def refractiveIndexMEdlen(lbda, P=617., T=2., RH=0):
    """Compute refractive index at vacuum wavelength *lbda* [Å] for
    pressure *P* [mbar], temperature *T* [°C] and relative humidity
    *RH* [%] according to (modified) Edlén Calculation of the Index of
    Refraction from NIST 'Refractive Index of Air Calculator'.  CO2
    concentration is fixed to 450 µmol/mol.

    Note that *Mauna-Loa* CO2 concentration rises steadily from 370 to
    390 ppm between 2000 and 2010.

    Source: http://emtoolbox.nist.gov/Wavelength/Documentation.asp
    """

    A = 8342.54
    B = 2406147.
    C = 15998.
    D = 96095.43
    E = 0.601
    F = 0.00972
    G = 0.003661

    iml2 = (lbda * 1e-4)**-2              # 1/(lambda in microns)**2
    nsm1e2 = 1e-6 * (A + B / (130. - iml2) + C / (38.9 - iml2))  # (ns - 1)*1e2
    # P in mbar = 1e2 Pa
    X = (1. + 1e-6 * (E - F * T) * P) / (1. + G * T)
    n = 1. + P * nsm1e2 * X / D             # ref. index corrected for P,T

    if RH:                              # Humidity correction
        pv = RH / 100. * saturationVaporPressure(T)  # [Pa]
        n -= 1e-10 * (292.75 / (T + 273.15)) * (3.7345 - 0.0401 * iml2) * pv

    return n


# Aliases, for backward-compatibility
refractiveIndex = refractiveIndexMEdlen  # Default
atmosphericIndex = refractiveIndex


# Atmospheric Differential Refraction ==============================

class ADR(object):

    """
    ADR model for given pressure [mbar], temperature [°C] (and
    relative humidity [%]).

    Airmass is treated in the plane-parallel approximation.
    """

    def __init__(self, P=617., T=2., RH=0, **kwargs):
        """ADR_model(P, T, RH, [lref=, delta=, theta=, airmass=,
        parangle=, zd=])."""

        assert 550 < P < 650 and -20 < T < 20, \
            "Non-std pressure (%.0f mbar) or temperature (%.0f°C)" % (P, T)
        self.P = P                      # Pressure [mbar]
        self.T = T                      # Temperature [°C]
        self.RH = RH                    # Relative humidity [%]
        if 'lref' in kwargs:
            self.set_ref(lref=kwargs.pop('lref'))  # Ref. wavelength [Å]
        else:
            self.set_ref()
        self.set_param(**kwargs)        # Initialization from anonymous params

    def __str__(self):

        s = "ADR [ref: %.0f A]: P=%.0f mbar, T=%.0fC" % \
            (self.lref, self.P, self.T)
        if self.RH:
            s += ", RH=%.0f%%" % self.RH
        if self.isSet:
            s += ", airmass=%.2f, parangle=%.1f deg" % \
                 (self.get_airmass(), self.get_parangle())

        return s

    def set_ref(self, lref=5000.):
        """Set reference wavelength [Å] for ADR."""

        self.lref = lref                # [Å]
        self.nref = refractiveIndex(self.lref, P=self.P, T=self.T, RH=self.RH)

    def set_param(self, **kwargs):
        """Set ADR parameters, which could be:

        - delta: ADR power = tan(zenithal distance),
        - theta: parallactic angle [rad],
        - airmass: plane-parallel approximation,
        - parangle: parallactic angle in deg,
        - zd: (supposedly true) zenithal distance [deg].
        """

        for k, val in kwargs.items():
            if k == 'delta':                           # tan(zenithal distance)
                self.delta = val
            # Parallactic angle [rad]
            elif k == 'theta':
                self.theta = val
            elif k == 'airmass':                       # Plane-parallel airmass
                self.delta = N.tan(N.arccos(1. / val))
            # Parallactic angle [deg]
            elif k == 'parangle':
                self.theta = val / RAD2DEG
            # Zenithal distance [deg]
            elif k == 'zd':
                self.delta = N.tan(val / RAD2DEG)
            else:
                raise ValueError("Unknown parameter '%s'" % k)

        self.isSet = hasattr(self, 'delta') and hasattr(self, 'theta')

    def get_scale(self, lbda, **kwargs):
        """
        Return ADR scale [arcsec] for wavelength(s) `lbda` [A].

        Anonymous `kwargs` will be propagated to :meth:`set_param`.
        """

        if kwargs:                      # Update parameters if needed
            self.set_param(**kwargs)

        lbda = N.atleast_1d(lbda)       # (nlbda,)

        # Approximate ADR to 1st order in (n - 1). The difference
        # between exact expression and 1st-order approximation reaches
        # 4e-9 at 3200 A.
        # dz = self.nref - \
        #      refractiveIndex(lbda, P=self.P, T=self.T, RH=self.RH)
        # Exact ADR expression
        dz = (refractiveIndex(lbda, P=self.P, T=self.T, RH=self.RH)**-2 -
              self.nref**-2) * 0.5

        return dz * RAD2ARC  # (nlbda,) [arcsec]

    def refract(self, x, y, lbda, backward=False, unit=1., **kwargs):
        """If forward (default), return refracted position(s) at
        wavelength(s) *lbda* [Å] from reference position(s) *x*,*y*
        (in units of *unit* in arcsec).  Return shape is
        (2,[nlbda],[npos]), where nlbda and npos are the number of
        input wavelengths and reference positions.

        If backward, one should have `len(x) == len(y) ==
        len(lbda)`. Return shape is then (2,npos).

        Coordinate *x* is counted westward, and coordinate *y* counted
        northward (standard North-up, Est-left orientation).

        Anonymous `kwargs` will be propagated to :meth:`set_param`.
        """

        x0 = N.atleast_1d(x)                 # (npos,)
        y0 = N.atleast_1d(y)
        assert len(x0) == len(y0), "Incompatible x and y vectors."
        npos = len(x0)

        assert self.isSet, "ADR parameters are not yet initialized."

        dz = self.delta * self.get_scale(lbda, **kwargs) / unit  # [unit]

        if backward:
            nlbda = len(N.atleast_1d(lbda))
            assert npos == nlbda, "Incompatible x,y and lbda vectors."
            x = x0 - dz * N.sin(self.theta)
            y = y0 + dz * N.cos(self.theta)  # (nlbda=npos,)
            out = N.vstack((x, y))           # (2,npos)
        else:
            dz = dz[:, N.newaxis]            # (nlbda,1)
            x = x0 + dz * N.sin(self.theta)  # (nlbda,npos)
            y = y0 - dz * N.cos(self.theta)  # (nlbda,npos)
            out = N.dstack((x.T, y.T)).T     # (2,nlbda,npos)

        return out.squeeze()                 # (2,[nlbda],[npos])

    def get_zd(self, delta=None):
        """Get zenithal distance [deg]."""

        return N.arctan(self.delta if delta is None else delta) * RAD2DEG

    def get_airmass(self, delta=None):
        """Get airmass from plane-parallel approximation."""

        return 1 / N.cos(N.arctan(self.delta if delta is None else delta))

    def get_parangle(self, theta=None):
        """Get parallactic angle [deg]."""

        return RAD2DEG * (self.theta if theta is None else theta)

    def blurring(self, lbda, hamid, dec, exptime):
        """Compute finite-time x,y-blurring at wavelength *lbda* [Å],
        for an object followed at declination *dec* [deg] and mid-hour
        angle *hamid* [deg] during *exptime* [s].
        """

        # ZDs, parangles and ADR offsets at start (1) and end (2) of exposure
        z1, p1 = Coords.hadec2zdpar(
            hamid - Coords.ten(0, 0, exptime / 2.) * 15, dec)
        off1 = self.refract(0, 0, lbda, zd=z1, parangle=p1)  # (2,[nlbda])
        z2, p2 = Coords.hadec2zdpar(
            hamid + Coords.ten(0, 0, exptime / 2.) * 15, dec)
        off2 = self.refract(0, 0, lbda, zd=z2, parangle=p2)

        return (off2 - off1).squeeze()      # (2,[nlbda])

    def plot(self, hamid, dec, exptime, lbda=None, ax=None):
        """Plot mean ADR and ADR blurring."""

        import matplotlib.pyplot as PP

        if lbda is None:
            # Wavelength ramp
            lbda = N.logspace(N.log10(3200), N.log10(10000), 20)

        # Mean ADR
        z0, p0 = Coords.hadec2zdpar(hamid, dec)
        x0, y0 = self.refract(0, 0, lbda, zd=z0, parangle=p0)

        # ADR at start of exposure [spx]
        z1, p1 = Coords.hadec2zdpar(
            hamid - Coords.ten(0, 0, exptime / 2.) * 15, dec)
        x1, y1 = self.refract(0, 0, lbda, zd=z1, parangle=p1)

        # ADR at end of exposure [spx]
        z2, p2 = Coords.hadec2zdpar(
            hamid + Coords.ten(0, 0, exptime / 2.) * 15, dec)
        x2, y2 = self.refract(0, 0, lbda, zd=z2, parangle=p2)

        if ax is None:
            ax = PP.subplot(1, 1, 1)          # Default axes

        ax.set(aspect='equal', adjustable='datalim',
               title="HAmid = %.2f deg, Dec=%.2f deg, ExpTime = %ds" %
               (hamid, dec, exptime),
               xlabel="dx [arcsec]", ylabel="dy [arcsec]")

        ax.scatter(x1, y1, c=lbda, s=30, marker='>', edgecolors='none',
                   cmap=PP.matplotlib.cm.Spectral_r,
                   label="Start: airmass=%.3f, parangle=%.1fdeg" %
                   (airmass(z1), p1))
        ax.plot(x0, y0, c='0.8', zorder=-1,
                label="Mid: airmass=%.3f, parangle=%.1fdeg" %
                (airmass(z0), p0))
        ax.scatter(x2, y2, c=lbda, s=30, marker='<', edgecolors='none',
                   cmap=PP.matplotlib.cm.Spectral_r,
                   label="End: airmass=%.3f, parangle=%.1fdeg" %
                   (airmass(z2), p2))

        r, t = Coords.rec2pol(x2 - x1, y2 - y1)
        sc = ax.scatter(x2 - x1, y2 - y1, c=lbda, s=30,
                        marker='d', facecolors='none',
                        cmap=PP.matplotlib.cm.Spectral_r,
                        label="Max blurring: %.2f arcsec" % r.max())

        ax.legend(loc='best', scatterpoints=1, frameon=False)

        cax = ax.figure.colorbar(sc, ax=ax)
        cax.set_label(u"Wavelength [Å]")

        return ax
