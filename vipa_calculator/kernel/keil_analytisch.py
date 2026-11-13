"""Wedge function, analytic: line of the exponential window with quadratic phase (chirp) in the continuum.

    I(delta) = |int_0^M exp(-gamma x) exp(i(delta x + beta x^2)) dx|^2   (closed form via erf with complex argument,
                                                                         evaluated with the Faddeeva function),
folded periodically over 2 pi; the overlap width Omega50 is found numerically (one root). This gives the width
D_c(gamma, M, beta) of the chirped window without any fitted constant.

Library module (no command-line use, no files): profil(delta, gam, M, beta) returns the line, omega50(gam, M, beta)
the overlap half width s in rad (finesse = 2 pi / s).
Inputs: gam = damping per round trip, M = number of round trips (window length), beta = chirp in rad per round trip^2.
"""
import numpy as np
from scipy.special import erf, wofz
from scipy.optimize import brentq
def profil(delta, gam, M, beta, nwrap=3):
    """Intensity of the chirped exponential window at the phases delta, periodised by summing the field integrals
    of the neighbouring orders k = -nwrap..nwrap (intensities added). Returns I(delta)."""
    d = np.asarray(delta, float); I = np.zeros_like(d)
    for k in range(-nwrap, nwrap + 1):
        a = gam - 1j*(d + 2*np.pi*k)
        if beta <= 0:
            A = (1 - np.exp(-a*M))/a
        # beta <= 0: plain truncated exponential; beta > 0: stable via the Faddeeva function w(z) = exp(-z^2) erfc(-iz):
        # int = sqrt(pi)/(2k) [ w(-i z0) - exp(i beta M^2 - a M) w(i(kM - z0)) ]
        else:
            kk = np.sqrt(beta)*np.exp(-1j*np.pi/4); z0 = kk*(-1j*a/(2*beta))
            A = np.sqrt(np.pi)/(2*kk)*(wofz(-1j*z0) - np.exp(1j*beta*M**2 - a*M)*wofz(1j*(kk*M - z0)))
        # overflow at extreme arguments is set to zero (those terms are negligible)
        with np.errstate(all='ignore'): I += np.nan_to_num(np.abs(A)**2, nan=0.0, posinf=0.0, neginf=0.0)
    return I
def omega50(gam, M, beta, n=20001):
    """Overlap width: shift s at which sum min(I, I shifted by s) has dropped to half the total energy.
    Returns s in rad (pi if the line is wider than half an order)."""
    x = np.linspace(-np.pi, np.pi, n)[:-1]; I = profil(x, gam, M, beta); tot = I.sum(); dx = x[1] - x[0]
    def ov(s):
        """Overlap sum of the line with itself shifted by s (rounded to the grid)."""
        k = int(round(s/dx)); return np.minimum(I, np.roll(I, -k)).sum()
    if ov(np.pi - dx) >= 0.5*tot: return np.pi          # line wider than half an order: resolution limit F = 2
    return brentq(lambda s: ov(s) - 0.5*tot, dx, np.pi - dx, xtol=dx/50)
