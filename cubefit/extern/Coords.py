# -*- coding: utf-8 -*-
# Select functions copied from SNfactory/Offline/Toolbox/Astro/Coords.py
# cvs v1.28 on 2015 Nov 4.

import numpy as N

RAD2DEG = 180./N.pi       # 180/pi
RAD2ARC = RAD2DEG * 3600  # RAD2DEG * 3600 = 206265


# Angle handling ==============================

def ten(*args):
    """
    Convert sexagesimal angle [-]DD[,[-]MM[,[-]SS]] or string
    "[-]DD[:[-]MM[:[-]SS]]" to decimal degrees.  If the input is a tuple, the
    sign (if any) should be set on the first non-null value.

    >>> ten(0, -23, 34)
    -0.39277777777777778
    >>> ten("-0:23:34")
    -0.39277777777777778
    """

    # Single string arg case
    if len(args) == 1 and isinstance(args[0], basestring):
        # Split the string 'DD:MM:SS' or 'DD MM SS' in tokens
        if ':' in args[0]:
            toks = args[0].split(':')
        elif ' ' in args[0]:
            toks = args[0].split(' ')
        else:
            toks = [args[0], ]

        # Check whether any of the tokens starts with a '-'
        sign = -1 if [True for tok in toks if tok.startswith('-')] else 1
        try:
            args = [sign * float(tok) for tok in toks]
        except ValueError:
            raise ValueError("ten() takes up to 3 numbers DD[,MM[,SS.S]] or "
                             "one 'DD[:MM[:SS.S]]' string as input")

    if not 1 <= len(args) <= 3:
        raise ValueError("ten() takes up to 3 numbers DD[,MM[,SS.S]] or "
                         "one 'DD[:MM[:SS.S]]' string as input")
    if len(args) != 3:          # Complete args with trailing 0s
        args = list(args) + [0] * (3 - len(args))

    # Main case: 3 numeric args
    sign = -1 if (min(args) < 0) else 1
    try:
        dec = abs(args[0]) + abs(args[1]) / 60. + abs(args[2]) / 3600.
        dec *= sign
    except TypeError:
        raise ValueError("ten() takes 3 numbers DD,MM,SS.S or "
                         "one 'DD:MM:SS.S' string as input")

    return dec


# Angular coordinate conversions ==============================

def rec2pol(x, y, deg=False):
    """
    Conversion of rectangular *(x, y)* to polar *(r, theta)* coordinates.
    """

    r = N.hypot(x, y)
    t = N.arctan2(y, x)
    if deg:                             # Convert to radians
        t *= RAD2DEG

    return r, t


def hadec2zdpar(ha, dec, phi=19.823056, deg=True):
    """
    Conversion of equatorial coordinates *(ha, dec)* (in degrees if *deg*) to
    zenithal distance and parallactic angle *(zd, par)* (in degrees if *deg*),
    for a given geodetic latitude *phi* (in degrees if *deg*).
    """

    if deg:                             # Convert to radians
        ha = ha / RAD2DEG
        dec = dec / RAD2DEG
        phi = phi / RAD2DEG

    cha, sha = N.cos(ha), N.sin(ha)
    cdec, sdec = N.cos(dec), N.sin(dec)
    cphi, sphi = N.cos(phi), N.sin(phi)

    sz_sp =               cphi * sha
    sz_cp = sphi * cdec - cphi * cha * sdec
    cz =    sphi * sdec + cphi * cha * cdec

    sz, p = rec2pol(sz_cp, sz_sp)
    r, z = rec2pol(cz,    sz)

    assert N.allclose(r, 1), "Precision error"

    if deg:
        z *= RAD2DEG
        p *= RAD2DEG

    return z, p
