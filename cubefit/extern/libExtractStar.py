# Select class/methods copied from
# SNFactory/Offline/pySNIFS/lib/libExtractStar.py
# cvs v1.63 on 2015 Nov 4.
# import statements added, all other code copied directly.

import numpy as N
from . import Coords as TA  # just to get TA.RAD2DEG!


class Hyper_PSF3D_PL(object):

    @classmethod
    def predict_adr_params(cls, inhdr):
        """
        Predict ADR parameters delta and theta [rad] from header `inhdr`
        including standard keywords `AIRMASS`, `PARANG` (parallactic
        angle [deg]), and `CHANNEL`.
        """

        # 0th-order estimates
        delta0 = N.tan(N.arccos(1. / inhdr['AIRMASS']))
        theta0 = inhdr['PARANG'] / TA.RAD2DEG  # Parallactic angle [rad]

        # 1st-order corrections from ad-hoc linear regressions
        sinpar = N.sin(theta0)
        cospar = N.cos(theta0)
        X = inhdr['CHANNEL'][0].upper()  # 'B' or 'R'
        if X == 'B':                      # Blue
            ddelta1 = -0.00734 * sinpar + 0.00766
            dtheta1 = -0.554 * cospar + 3.027  # [deg]
        elif X == 'R':                    # Red
            ddelta1 = +0.04674 * sinpar + 0.00075
            dtheta1 = +3.078 * cospar + 4.447  # [deg]
        else:
            raise KeyError("Unknown channel '%s'" % inhdr['CHANNEL'])

        # Final predictions
        delta = delta0 + ddelta1
        theta = theta0 + dtheta1 / TA.RAD2DEG  # [rad]

        return delta, theta
