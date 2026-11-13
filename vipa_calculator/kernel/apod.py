"""Closed-form apodisation of the coherent chord sum, and the characteristic quantities of the common phase.

Purpose
    The coherent sum over the strips rho of the chord at round trip m has the amplitude
        A_m = |sum_rho w_rho exp(i psi_{rho m})|
    with the accumulated phase psi. For Gaussian-distributed phases, ln A_m ~ -V_m/2, where V_m is the weighted
    variance of the accumulated phase over the strips of the chord at round trip m. V_m is a quadratic form of the
    surface height:
      - mid band: spectral integral with the sum filter |S_m(f_x)|^2 and the cross filter
        1 - |sum_rho w_rho e^{2 pi i (f_x x0 + f_y y)}|^2;
      - figure: sum over the Zernike terms;
      - fine band: diagonal (uncorrelated from hit point to hit point).
    a_A = -(linear coefficient) of the weighted quadratic least-squares fit of -V_m/2 in m, i.e. the apodisation
    rate per round trip.

Functions
    V_m            expected phase variance across the chord per round trip, and the round-trip weights
    a_A            apodisation rate from V_m
    A_rms_profile  exact expected power of the coherent chord sum per round trip (includes the speckle floor)
    a_A_rms        apodisation rate from A_rms_profile
    kz_gemeinsam   characteristic quantities of the common (chord-averaged) phase: residual resonance and
                   variance of the random chirp

Units: lengths in m (D_m, sigma values), L in mm, tilt in deg, wedge alpha in arcsec, wavelength in nm, beam width in mm.
Library module (no command-line use); imported by other scripts.

Used in the main document: appendix A.41 (R49: Die Apodisation der kohärenten Sehnensumme in geschlossener Form).
Pages: 205.
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import numpy as np
from dof_perp import _geom
from flaeche_dreiband import FIG_TERMS, FMAX_DEFAULT
from zernike_hr_surface import zernike

def V_m(R2, L_mm, th_deg, alpha_as, wl_nm, bl_mm, D_m, sig_fig, sig_mid, alpha_s, sig_fein, nf=160, nphi=64):
    """Return (V, w): V[m] = weighted variance across the strips of the accumulated round-trip phase at round trip m,
    summed over the three bands (sigma values in m, alpha_s = PSD exponent of the mid band); w = round-trip weights."""
    # w: weight per round trip, wy: weight per strip, X/Y: hit points (round trip x strip), dx: step per round trip.
    w, wy, X, Y, dx = _geom(R2, L_mm, th_deg, alpha_as, wl_nm, bl_mm)
    # c2 = (2 k0)^2 converts height variance into phase variance (reflection: phase 2 k0 h).
    k0 = 2*np.pi/(wl_nm*1e-9); c2 = (2*k0)**2; M = len(w); m = np.arange(M)
    wr = wy/wy.sum(); V = np.zeros(M); x0 = X[0]; y0 = Y[0]
    if sig_mid > 0:
        # Polar grid of spatial frequencies (log-spaced from 1/D to f_max); weights of a power-law PSD f^-alpha_s
        # including the Jacobian f df on a log grid (f^2), normalised to unit total variance.
        f = np.geomspace(1/D_m, FMAX_DEFAULT, nf); ph = (np.arange(nphi) + 0.5)*2*np.pi/nphi
        F, PH = np.meshgrid(f, ph, indexing="ij"); fx = (F*np.cos(PH)).ravel(); fy = (F*np.sin(PH)).ravel()
        dens = (F**(2 - alpha_s)).ravel(); dens /= dens.sum()
        # Cross filter: removes the part of each Fourier component that is common to all strips of the chord.
        G = 1 - np.abs(np.exp(2j*np.pi*(np.outer(fx, x0) + np.outer(fy, y0))) @ wr)**2          # cross filter
        # Sum filter of the accumulation over m+1 hit points with step dx: |S_m|^2 = sin^2((m+1) u)/sin^2(u), u = pi f_x dx
        # (sin(u) is guarded against zero).
        u = np.pi*fx*dx; s = np.sin(u); s = np.where(np.abs(s) < 1e-12, 1e-12, s)
        Sm = np.sin(np.outer(m + 1, u))**2/s[None, :]**2                                          # |S_m|^2
        V += c2*sig_mid**2*(Sm @ (dens*G))
    if sig_fig > 0:
        # Figure: random Zernike terms with equal variance sig_fig^2/len(FIG_TERMS) each, evaluated on the plate disc.
        rho = np.sqrt(X**2 + Y**2)/(D_m/2); phi = np.arctan2(Y, X); inside = rho <= 1
        for nn, mm_ in FIG_TERMS:
            # accumulated over round trips, shape (M, ns)
            Z = np.cumsum(zernike(nn, mm_, np.minimum(rho, 1), phi)*inside, axis=0)
            mu = Z @ wr; V += c2*sig_fig**2/len(FIG_TERMS)*(((Z - mu[:, None])**2) @ wr)
    if sig_fein > 0:
        # Fine band: independent per hit point, so the accumulated variance grows like m+1; the weighted variance
        # across strips carries the factor 1 - sum w^2.
        V += c2*sig_fein**2*(m + 1)*(1 - np.sum(wr**2))
    return V, w

def a_A(R2, L_mm, th_deg, alpha_as, wl_nm, bl_mm, D_m, sig_fig, sig_mid, alpha_s, sig_fein):
    """Return (a_A, V): apodisation rate a_A = -(linear coefficient) of the weighted quadratic fit of -V_m/2 in m
    (clipped at 0), and the variance profile V_m."""
    V, w = V_m(R2, L_mm, th_deg, alpha_as, wl_nm, bl_mm, D_m, sig_fig, sig_mid, alpha_s, sig_fein)
    # Weighted least squares with weights w via rows scaled by sqrt(w).
    M = len(w); m = np.arange(M, dtype=float); y = -V/2
    X = np.stack([np.ones(M), m, m**2], 1); s = np.sqrt(w)[:, None]
    c = np.linalg.lstsq(s*X, s[:, 0]*y, rcond=None)[0]
    return max(-c[1], 0.0), V

def A_rms_profile(R2, L_mm, th_deg, alpha_as, wl_nm, bl_mm, D_m, sig_fig, sig_mid, alpha_s, sig_fein, nf=60, nphi=32):
    """Expected power of the coherent chord sum per round trip, A_rms,m^2 = sum w w' exp(-D_{rho rho'}(m)/2),
    with the structure function D of the accumulated phase between two strips (Gaussian, exact). Includes the speckle floor.
    Returns (A_rms, w): the RMS amplitude per round trip and the round-trip weights."""
    w, wy, X, Y, dx = _geom(R2, L_mm, th_deg, alpha_as, wl_nm, bl_mm)
    k0 = 2*np.pi/(wl_nm*1e-9); c2 = (2*k0)**2; M = len(w); m = np.arange(M)
    wr = wy/wy.sum(); x0 = X[0]; y0 = Y[0]; ns = len(wr)
    C = np.zeros((M, ns, ns))
    if sig_mid > 0:
        f = np.geomspace(1/D_m, FMAX_DEFAULT, nf); ph = (np.arange(nphi) + 0.5)*2*np.pi/nphi
        F, PH = np.meshgrid(f, ph, indexing="ij"); fx = (F*np.cos(PH)).ravel(); fy = (F*np.sin(PH)).ravel()
        dens = (F**(2 - alpha_s)).ravel(); dens /= dens.sum()
        E = np.exp(2j*np.pi*(np.outer(x0, fx) + np.outer(y0, fy)))
        u = np.pi*fx*dx; s = np.sin(u); s = np.where(np.abs(s) < 1e-12, 1e-12, s)
        # Covariance between strips per round trip: spectral integral with the sum filter |S_m|^2.
        for i in range(M):
            Sm = np.sin((i + 1)*u)**2/s**2
            C[i] += c2*sig_mid**2*np.real((E*(dens*Sm)[None, :]) @ E.conj().T)
    if sig_fig > 0:
        rho = np.sqrt(X**2 + Y**2)/(D_m/2); phi = np.arctan2(Y, X); inside = rho <= 1
        for nn, mm_ in FIG_TERMS:
            Z = np.cumsum(zernike(nn, mm_, np.minimum(rho, 1), phi)*inside, axis=0)
            C += c2*sig_fig**2/len(FIG_TERMS)*np.einsum('mi,mj->mij', Z, Z)
    if sig_fein > 0:
        C += c2*sig_fein**2*(m + 1)[:, None, None]*np.eye(ns)[None]
    # Structure function D_ij = C_ii + C_jj - 2 C_ij; for Gaussian phases <exp(i(psi_i - psi_j))> = exp(-D_ij/2).
    d = np.einsum('mii->mi', C); Dst = d[:, :, None] + d[:, None, :] - 2*C
    A2 = np.einsum('i,mij,j->m', wr, np.exp(-Dst/2), wr)
    return np.sqrt(A2), w

def a_A_rms(*args, **kw):
    """Return (a_A, A): apodisation rate from the weighted quadratic fit of ln A_rms,m in m (clipped at 0), and A_rms."""
    A, w = A_rms_profile(*args, **kw); M = len(w); m = np.arange(M, dtype=float)
    X = np.stack([np.ones(M), m, m**2], 1); s = np.sqrt(w)[:, None]
    c = np.linalg.lstsq(s*X, s[:, 0]*np.log(np.maximum(A, 1e-12)), rcond=None)[0]
    return max(-c[1], 0.0), A

def kz_gemeinsam(R2, L_mm, th_deg, alpha_as, wl_nm, bl_mm, D_m, sig_fig, sig_mid, alpha_s, sig_fein, nf=60, nphi=32):
    """Characteristic quantities of the common (chord-averaged) phase for the coherent chord sum: residual resonance
    sigma_res^2 and random-chirp variance Var q of the steps nu_m = sum_rho w_rho nu_{rho m}, after removing mean and
    slope with the window weights. Mid band: spectral integral with the cross filter
    |sum_rho w_rho e^{2 pi i (f_x x0 + f_y y)}|^2; figure: Zernike sum; fine band diagonal.
    Returns (sigma_res^2, Var q) in rad^2."""
    w, wy, X, Y, dx = _geom(R2, L_mm, th_deg, alpha_as, wl_nm, bl_mm)
    k0 = 2*np.pi/(wl_nm*1e-9); c2 = (2*k0)**2; M = len(w); m = np.arange(M, dtype=float); m = m - np.sum(w*m)
    # Weighted linear fit [1, m] (m centred): L maps a sequence to (mean, slope), H = B L is the hat matrix.
    B = np.vstack([np.ones(M), m]).T; G = B.T @ (w[:, None]*B); L = np.linalg.solve(G, (B*w[:, None]).T); H = B @ L
    wr = wy/wy.sum(); x0 = X[0]; y0 = Y[0]; mrow = (X - x0[None, :])[:, 0]
    s2 = 0.0; vq = 0.0
    if sig_mid > 0:
        f = np.geomspace(1/D_m, FMAX_DEFAULT, nf); ph = (np.arange(nphi) + 0.5)*2*np.pi/nphi
        F, PH = np.meshgrid(f, ph, indexing="ij"); fx = (F*np.cos(PH)).ravel(); fy = (F*np.sin(PH)).ravel()
        dens = (F**(2 - alpha_s)).ravel(); dens /= dens.sum()
        # Q: power of a Fourier component that survives the chord average (complement of the cross filter).
        Q = np.abs(np.exp(2j*np.pi*(np.outer(fx, x0) + np.outer(fy, y0))) @ wr)**2
        # Per round trip the component advances by the step along x; R = residual after mean+slope, co = fit coefficients.
        A = np.exp(2j*np.pi*np.outer(mrow, fx)); R = A - H @ A; co = L @ A
        s2 += c2*sig_mid**2*np.sum(dens*Q*np.sum(w[:, None]*np.abs(R)**2, 0)); vq += c2*sig_mid**2*np.sum(dens*Q*np.abs(co[1])**2)
    if sig_fig > 0:
        rho = np.sqrt(X**2 + Y**2)/(D_m/2); phi = np.arctan2(Y, X); inside = rho <= 1
        for nn, mm_ in FIG_TERMS:
            z = (zernike(nn, mm_, np.minimum(rho, 1), phi)*inside) @ wr; co = L @ z; r = z - H @ z
            s2 += c2*sig_fig**2/len(FIG_TERMS)*np.sum(w*r**2); vq += c2*sig_fig**2/len(FIG_TERMS)*co[1]**2
    if sig_fein > 0:
        # Fine band: chord-averaged steps are independent with variance c2 sig_fein^2 sum w^2; P_ projects out mean+slope.
        ss = c2*sig_fein**2*np.sum(wr**2); P_ = np.eye(M) - H
        s2 += ss*np.trace(np.diag(w) @ P_ @ P_.T); vq += ss*np.sum(L[1]**2)
    return s2, vq
