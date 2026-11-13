"""Detection models across the dispersion direction (y, chord of the aperture).

Purpose
-------
General form (Hopkins form over the strips rho of the chord):
    I(delta) = sum_{rho,rho'} Gamma_{rho rho'} E_rho(delta) E_rho'(delta)^*
with Gamma = mu_ill(dy) * mu_det(dy):
  mu_ill: coherence of the illumination across the chord (van Cittert-Zernike, round fibre core):
          2 J1(u)/u, u = pi d_core dy / (lambda f_c);  d_core -> 0: coherent (= 1)
  mu_det: detector filter, pixel of width p in the focal plane: sinc(p dy/(lambda f));
          p -> 0: point on the axis (= 1); integrated over the whole y spot: delta_{rho rho'}.
Limiting cases: "axis" (Gamma = 1, point detection on the axis) and "integrated" (Gamma = identity).

Contents
--------
  gamma_matrix  builds the coherence matrix Gamma for the strip positions y.
  intensity     computes the line I(delta) from the per-strip pupils in one of the three modes.

Inputs/outputs: library module, no files.

"""
import numpy as np
from scipy.special import j1

def gamma_matrix(y, lam, d_core=0.0, f_c=75e-3, pixel=0.0, f=75e-3, integrated=False):
    """Returns the (n, n) matrix Gamma_{rho rho'} for strip positions y (m), wavelength lam (m),
    fibre core diameter d_core and collimator focal length f_c, pixel width and camera focal
    length f (all in m). integrated=True returns the identity (detector integrates over y)."""
    dy = np.abs(y[:, None] - y[None, :])
    if integrated:
        return np.eye(len(y))
    G = np.ones_like(dy)
    if d_core > 0:
        # Airy-type coherence 2 J1(u)/u; the inner where avoids division by zero at u = 0 (limit 1)
        u = np.pi * d_core * dy / (lam * f_c); G = G * np.where(u > 1e-12, 2 * j1(u) / np.where(u > 1e-12, u, 1), 1.0)
    if pixel > 0:
        # np.sinc(x) = sin(pi x)/(pi x)
        G = G * np.sinc(pixel * dy / (lam * f))
    return G

def intensity(V, g, NF, mode="axis", Gamma=None, ny=None):
    """Returns the line intensity I(delta) on NF FFT points.
    V: (M, n_rays) pupil per partial ray (round trip x strip), g: core group per partial ray
    (different groups add incoherently).
    mode 'axis' | 'integrated' | 'gamma' (then Gamma is used for the y points of one core group)."""
    if mode == "axis":
        # Coherent sum over the strips of a group, then intensity; groups add incoherently
        return sum(np.abs(np.fft.fft(V[:, g == k].sum(1), NF)) ** 2 for k in np.unique(g))
    if mode == "integrated":
        # Every strip adds incoherently
        return (np.abs(np.fft.fft(V, NF, axis=0)) ** 2).sum(1)
    I = 0
    for k in np.unique(g):
        E = np.fft.fft(V[:, g == k], NF, axis=0)          # (NF, ny)
        # Hermitian form E Gamma E^* for every frequency f
        I = I + np.real(np.einsum('fr,rs,fs->f', E, Gamma, E.conj()))
    return I
