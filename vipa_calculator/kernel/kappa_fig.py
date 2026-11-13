"""Exact characteristic quantities of the figure band for one geometry (helper module).

Purpose
    Provides kappa_fig(), which computes the characteristic quantities (kappa_q = variance of the random chirp,
    kappa_par = residual resonance, kappa_perp = strip offset) of the figure band of the three-band surface model
    exactly, i.e. from the actual spot tracks of the central order. The figure is modelled by the Zernike terms
    of degree 2-4 listed in flaeche_dreiband.FIG_TERMS, each with RMS 1 over the diameter D_m; the result is the
    mean over these terms (equally strong terms).

Inputs
    Function arguments: R2, L [mm], tilt th [deg], wedge al [arcsec], wavelength wl [nm], beam width bl [mm],
    optional diameter D_m [m] of the normalisation circle (default: plate diameter 2*surface_radius(L)) and the
    number of strips ny across the chord.

Outputs
    numpy array (kappa_q, kappa_par, kappa_perp) for 2 k0 sigma_fig = 1 (scale with (2 k0 sigma_fig)^2).

Run
    Import only: from kappa_fig import kappa_fig.
"""
import sys; pass; import numpy as np
import pupil_theory_v7 as PT, run_full_grid_v7 as G
from theorie_grid_dreiband import order_tracks, kennzahlen_realisierung
from flaeche_dreiband import FIG_TERMS
from zernike_hr_surface import zernike
def kappa_fig(R2,L,th,al,wl,bl,D_m=None,ny=61):
    """Exact characteristic quantities of the figure (Zernike degree 2-4, equally strong terms, RMS 1 over D_m) for
    one geometry, averaged over the terms. Returns the array (kappa_q, kappa_par, kappa_perp) for 2 k0 sigma_fig = 1."""
    # central order (angle closest to the axis)
    thx=PT.order_angles(R2,L,th,wl,bl); thx=float(thx[np.argmin(np.abs(thx))])
    # spot positions X (round trip x strip), strip coordinates y, field U and strip groups g of this order
    U,X,y,g,dx,ch=order_tracks(R2,L,th,al,wl,bl,thx,ny); Y=np.repeat(y[None,:],X.shape[0],0)
    if D_m is None: D_m=2*G.surface_radius(L)
    # normalised polar coordinates on the Zernike circle; spots outside the circle get height 0
    rho=np.sqrt(X**2+Y**2)/(D_m/2); phi=np.arctan2(Y,X); inside=rho<=1
    kz=np.zeros(3)
    for n,m in FIG_TERMS:
        h=zernike(n,m,np.minimum(rho,1),phi)*inside
        kz+=np.array(kennzahlen_realisierung(h,U,g))
    # (kappa_q, kappa_par, kappa_perp) for 2k0*sigma_fig = 1
    return kz/len(FIG_TERMS)
