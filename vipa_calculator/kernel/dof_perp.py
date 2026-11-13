"""Strip offset as a quadratic form: eigenvalues, degrees of freedom (Satterthwaite) and median (Wilson-Hilferty),
computed from the geometry and the surface model alone, without drawing random surfaces.

Purpose
    The strip offset (the spread of the strip means of the surface phase across the chord) is a quadratic form
    in the Gaussian surface height. Its distribution therefore follows from the eigenvalues of the weighted,
    mean-removed covariance matrix of the strip means. This module computes these covariances and eigenvalues
    for the three bands of the surface model (mid band: power-law spectrum; figure: 12 Zernike terms;
    fine band: white), and turns them into mean, effective degrees of freedom and the median/mean factor.

Functions
    _geom(...)        round-trip weights, strip weights and hit positions of the order closest to the axis
    cov_parts(...)    covariances of the strip means per band (unit phase variance) and strip weights
    eig_total(C, wy)  eigenvalues of the quadratic form for a (weighted sum of) covariance(s)
    eig_perp(...)     eigenvalues per band directly
    median_factor(p)  mean, degrees of freedom and median/mean from (scale, eigenvalues) pairs

Inputs/outputs
    library module (no files, no printout); imported by other scripts (e.g. exit_surface.py, finesse_vipa.py).

Used in the main document: appendix A.38 (R46: Der Streifenversatz als quadratische Form – Mittel, Freiheitsgrade
und Median). Pages: 199.
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import numpy as np
import pupil_theory_v7 as PT, run_full_grid_v7 as G
from theorie_grid_dreiband import order_tracks
from flaeche_dreiband import FIG_TERMS, FMAX_DEFAULT
from zernike_hr_surface import zernike

def _geom(R2, L, th, al, wl, bl, ny=61):
    """Geometry of the order closest to the axis (order_tracks with ny chord strips).

    Returns (w, wy, X, Y, dx): round-trip weights w (power, normalised), strip weights wy across the chord,
    hit positions X, Y per round trip and strip (only round trips with nonzero power), round-trip step dx.
    """
    thx = PT.order_angles(R2, L, th, wl, bl); thx = float(thx[np.argmin(np.abs(thx))])
    U, X, y, g, dx, ch = order_tracks(R2, L, th, al, wl, bl, thx, ny)
    W = np.abs(U)**2; wm = W.sum(1); keep = wm > 0; U = U[keep]; X = X[keep]; W = W[keep]
    w = W.sum(1)/W.sum(); wy = W.sum(0)/W.sum(); Y = np.repeat(y[None, :], X.shape[0], 0)
    return w, wy, X, Y, dx

def cov_parts(R2, L, th, al, wl, bl, D_m, alpha_s, nf=160, nphi=64, ny=61):
    """Covariances of the strip means per band, each for unit phase variance (2 k0 sigma)^2 = 1.

    Returns (Cm, Cf, Ce, wy): mid band (power-law spectrum), figure (12 Zernike terms), fine band (white),
    and the strip weights wy. Weight and sum the parts, then pass the sum to eig_total.
    """
    w, wy, X, Y, dx = _geom(R2, L, th, al, wl, bl, ny)
    # start point of each strip and the (common) step of the hits along the round trips
    x0 = X[0]; y0 = Y[0]; mrow = (X - x0[None, :])[:, 0]; n = len(wy)
    # polar grid of spatial frequencies from 1/D_m to f_max; power-law weights f^(2-alpha_s) (spectrum f^-alpha_s
    # times the area element f df on a logarithmic grid), normalised to variance 1
    f = np.geomspace(1/D_m, FMAX_DEFAULT, nf); ph = (np.arange(nphi) + 0.5)*2*np.pi/nphi
    F, PH = np.meshgrid(f, ph, indexing="ij"); fx = (F*np.cos(PH)).ravel(); fy = (F*np.sin(PH)).ravel()
    dens = (F**(2 - alpha_s)).ravel(); dens /= dens.sum()
    # a(f): round-trip weighted phasor sum along the track; the covariance of two strip means is the spectral
    # integral of |a|^2 times the phase factor between their start points
    a = (w[None, :]*np.exp(2j*np.pi*np.outer(fx, mrow))).sum(1); s = dens*np.abs(a)**2
    V = np.exp(2j*np.pi*(np.outer(x0, fx) + np.outer(y0, fy))); Cm = np.real((V*s[None, :]) @ V.conj().T)
    # figure: strip mean of each Zernike term over the mirror disc (diameter D_m); terms of equal weight
    rho = np.sqrt(X**2 + Y**2)/(D_m/2); phi = np.arctan2(Y, X); inside = rho <= 1
    Z = np.array([w @ (zernike(nn, mm, np.minimum(rho, 1), phi)*inside) for nn, mm in FIG_TERMS])
    # fine band: independent per hit, so the strip mean has variance sum w^2 on the diagonal
    Cf = Z.T @ Z/len(FIG_TERMS); Ce = np.eye(n)*np.sum(w**2)
    return Cm, Cf, Ce, wy

def eig_total(C, wy):
    """Eigenvalues (clipped at 0) of the quadratic form of the strip offset for covariance C of the strip means.

    The projector P removes the weighted mean over the strips; the form is symmetrised with sqrt(wy) weights.
    """
    n = len(wy); P = np.eye(n) - np.outer(np.ones(n), wy); Wh = np.sqrt(wy)
    Mx = (Wh[:, None]*P) @ C @ (P.T*Wh[None, :]); return np.clip(np.linalg.eigvalsh(0.5*(Mx + Mx.T)), 0, None)

def eig_perp(R2, L, th, al, wl, bl, D_m, alpha_s, nf=160, nphi=64, ny=61):
    """Eigenvalues of the quadratic form of the strip offset per band, for (2 k0 sigma)^2 = 1.

    Returns (lam_mid, lam_fig, lam_fein): mid band (power law), figure (12 Zernike terms), fine band (white).
    """
    w, wy, X, Y, dx = _geom(R2, L, th, al, wl, bl, ny)
    # hits relative to the start point of each strip
    x0 = X[0]; y0 = Y[0]; mrel = X - x0[None, :]
    # P: removes the weighted mean over the strips; Wh = sqrt of the strip weights
    n = len(wy); P = np.eye(n) - np.outer(np.ones(n), wy); Wh = np.sqrt(wy)
    def lam_of(C):
        Mx = (Wh[:, None]*P) @ C @ (P.T*Wh[None, :]); ev = np.linalg.eigvalsh(0.5*(Mx + Mx.T)); return np.clip(ev, 0, None)
    # mid band: C_rr' = int S |a|^2 cos(2pi(f_x dx0 + f_y dy)) d^2f, variance 1
    f = np.geomspace(1/D_m, FMAX_DEFAULT, nf); ph = (np.arange(nphi) + 0.5)*2*np.pi/nphi
    F, PH = np.meshgrid(f, ph, indexing="ij"); fx = (F*np.cos(PH)).ravel(); fy = (F*np.sin(PH)).ravel()
    # S f df dphi, with f df = f^2 dln f on the logarithmic grid
    dlnf = np.log(f[1]/f[0]); dens = (F**(2 - alpha_s)).ravel()*dlnf*(2*np.pi/nphi)
    dens /= dens.sum()
    # the steps are the same for all strips
    mrow = mrel[:, 0]
    a = (w[None, :]*np.exp(2j*np.pi*np.outer(fx, mrow))).sum(1); s = dens*np.abs(a)**2
    V = np.exp(2j*np.pi*(np.outer(x0, fx) + np.outer(y0, fy)))
    C = np.real((V*s[None, :]) @ V.conj().T)
    lam_mid = lam_of(C)
    # figure: strip mean of each Zernike term
    rho = np.sqrt(X**2 + Y**2)/(D_m/2); phi = np.arctan2(Y, X); inside = rho <= 1
    # shape (12, n)
    Z = np.array([w @ (zernike(nn, mm, np.minimum(rho, 1), phi)*inside) for nn, mm in FIG_TERMS])
    # shape (12, n): mean-removed, weighted strip means of each term
    B = (Z @ (P.T*Wh[None, :]))
    lam_fig = np.clip(np.linalg.eigvalsh(B @ B.T/len(FIG_TERMS)), 0, None)
    # fine band: independent per hit, variance 1 -> variance of the strip mean is sum w^2
    lam_fein = lam_of(np.eye(n)*np.sum(w**2))
    return lam_mid, lam_fig, lam_fein

def median_factor(parts):
    """Mean, degrees of freedom and median of a weighted sum of chi-square terms.

    parts: list of (scale, eigenvalues). Returns (mu, k, median/mean): mean mu = sum s*lambda, effective degrees of
    freedom k = 2 mu^2 / Var (Satterthwaite), and the median/mean ratio (1 - 2/(9k))^3 (Wilson-Hilferty).
    """
    mu = sum(s*l.sum() for s, l in parts); var = sum(2*(s**2)*(l**2).sum() for s, l in parts)
    k = 2*mu**2/var if var > 0 else np.inf
    return mu, k, (1 - 2/(9*k))**3
