"""Exit surface of the partially reflecting (PR) plate.

Purpose
    The phase (n-1) k0 h_E(x_m) of the exit surface acts once on each contribution and does not accumulate over
    the round trips. Over the window it is decomposed by a weighted least-squares fit in m with 1, m, m^2 per strip:
      linear part     -> shift of the strip          -> strip covariance C_exit (adds to C of the gap)
      quadratic part  -> chirp beta_E                -> variance Var beta_E (adds to the chirp scatter)
      remainder       -> non-accumulating phase      -> pedestal p = 1 - exp(-sigma_rest^2), or the Kubo line
    All three are quadratic forms of the height: spectral integral (mid band), Zernike sum (figure),
    diagonal (fine band).

Functions
    exit_kz(...)              characteristic quantities of the exit surface (dict, see there)
    sockel_faktor(p)          finesse factor of a flat pedestal on a Lorentz line
    sockel_faktor_voigt(...)  the same for a Voigt line
    rechteck_faktor(...)      finesse factor of rectangular convolutions (transverse wedge, pixel) of a Voigt line
    kubo_rest_width(...)      overlap width of the ensemble line with the non-accumulating residual phase (Kubo form)

Inputs/outputs
    library module (no files, no printout); imported by finesse_vipa.py and others.
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import numpy as np
from dof_perp import _geom
from flaeche_dreiband import FIG_TERMS, FMAX_DEFAULT
from zernike_hr_surface import zernike

def exit_kz(R2, L_mm, th_deg, alpha_as, wl_nm, bl_mm, D_m, n, sig_fig, sig_mid, alpha_s, sig_fein, nf=160, nphi=64, D_line=None):
    """Characteristic quantities of the exit surface for one geometry and surface model.

    Arguments: R2 mirror reflectivity, L_mm, th_deg, alpha_as, wl_nm, bl_mm geometry (as in _geom), D_m mirror
    diameter [m], n refractive index of the plate, sig_fig, sig_mid, sig_fein rms heights of the three bands [m],
    alpha_s spectral exponent, D_line line width for the high-pass of the halo (None = no filter).
    Returns a dict: rest (weighted residual phase variance outside the line, rad^2), var_chirp (variance of the
    chirp beta_E), C (strip covariance from the linear part), wy (strip weights), Cm (covariance of the residual
    along the round trips, averaged over the strips, for the Kubo line), w (round-trip weights).
    """
    w, wy, X, Y, dx = _geom(R2, L_mm, th_deg, alpha_as, wl_nm, bl_mm)
    # phase per height: (n-1) k0; m centred on the weighted mean round trip
    k0 = 2*np.pi/(wl_nm*1e-9); c2 = ((n - 1)*k0)**2
    M = len(w); m = np.arange(M, dtype=float); m = m - np.sum(w*m)
    # L (3, M): weighted least-squares coefficients for 1, m, m^2
    B = np.vstack([np.ones(M), m, m**2]).T; Wd = w; G = B.T @ (Wd[:, None]*B); L = np.linalg.solve(G, (B*Wd[:, None]).T)
    # projector (M, M) onto the fitted polynomial
    H = B @ L
    # hits on the exit surface lie half a step further than on the mirror
    XP = X + dx/2; x0 = XP[0]; y0 = Y[0]; mrow = (XP - x0[None, :])[:, 0]; ns = len(wy)
    # Cm: covariance of the residual along the round trips (averaged over the strips), for the Kubo line
    out = dict(rest=0.0, var_chirp=0.0, C=np.zeros((ns, ns)), wy=wy, Cm=np.zeros((M, M)), w=Wd)
    # halo part: only residual components with a frequency (in the round-trip number) above the line width end up
    # outside the line; hp is the high-pass weight om^2/(om^2 + D_line^2)
    NFFT = 1 << int(np.ceil(np.log2(8*M))); om = 2*np.pi*np.fft.fftfreq(NFFT)
    hp = np.ones(NFFT) if D_line is None else om**2/(om**2 + D_line**2)
    sw = np.sqrt(Wd)
    # res: (M,) or (M, K), complex or real -> weighted, high-pass filtered residual energy per column
    def halo(res):
        r = res*sw[:, None] if res.ndim == 2 else (res*sw)[:, None]
        Rf = np.fft.fft(r, NFFT, axis=0); e = np.sum(np.abs(Rf)**2*hp[:, None], 0)/NFFT
        return e if res.ndim == 2 else e[0]
    if sig_mid > 0:
        # mid band: power-law spectrum on a polar frequency grid, scaled to the phase variance ((n-1) k0 sigma_mid)^2
        f = np.geomspace(1/D_m, FMAX_DEFAULT, nf); ph = (np.arange(nphi) + 0.5)*2*np.pi/nphi
        F, PH = np.meshgrid(f, ph, indexing="ij"); fx = (F*np.cos(PH)).ravel(); fy = (F*np.sin(PH)).ravel()
        dens = (F**(2 - alpha_s)).ravel(); dens /= dens.sum(); dens *= c2*sig_mid**2
        # (M, K): one Fourier component per column along the track
        A = np.exp(2j*np.pi*np.outer(mrow, fx))
        # weighted halo residual per component
        R = A - H @ A; rest = halo(R)
        out["Cm"] += np.real((R*dens[None, :]) @ R.conj().T)
        # (3, K): constant, linear, quadratic coefficient per component
        Cf = L @ A
        out["rest"] += np.sum(dens*rest); out["var_chirp"] += np.sum(dens*np.abs(Cf[2])**2)
        # strip covariance from the linear part (shift), with the phase factor between strip start points
        V = np.exp(2j*np.pi*(np.outer(x0, fx) + np.outer(y0, fy))); s = dens*np.abs(Cf[1])**2
        out["C"] += np.real((V*s[None, :]) @ V.conj().T)
    if sig_fig > 0:
        # figure: each of the 12 Zernike terms carries 1/12 of the figure variance
        rho = np.sqrt(XP**2 + Y**2)/(D_m/2); phi = np.arctan2(Y, XP); inside = rho <= 1
        # shape (12, M, ns)
        Z = np.array([zernike(nn, mm_, np.minimum(rho, 1), phi)*inside for nn, mm_ in FIG_TERMS])
        s2 = c2*sig_fig**2/len(FIG_TERMS); wr = wy/wy.sum()
        for z in Z:
            co = L @ z; res = z - H @ z
            out["rest"] += s2*np.sum(wr*halo(res)); out["var_chirp"] += s2*np.sum(wr*co[2]**2)
            out["Cm"] += s2*((res*wr[None, :]) @ res.T)
            out["C"] += s2*np.outer(co[1], co[1])
    if sig_fein > 0:
        # fine band: white, independent per hit
        s2 = c2*sig_fein**2
        # white: sum over the unit vectors
        Pm = np.eye(M) - H; out["rest"] += s2*np.sum(halo(Pm))
        out["Cm"] += s2*(Pm @ Pm.T)
        out["var_chirp"] += s2*np.sum(L[2]**2); out["C"] += s2*np.eye(ns)*np.sum(L[1]**2)
    return out

def sockel_faktor(p):
    """Finesse factor caused by a flat pedestal holding the energy fraction p (Lorentz line).

    Solves Omega = p + (1-p) Omega_line = 1/2 for the line overlap and returns the ratio of widths
    (p is clipped to [0, 0.49]).
    """
    p = min(max(p, 0.0), 0.49); Ol = (0.5 - p)/(1 - p); return 1/np.tan(np.pi/2*(1 - Ol))

def sockel_faktor_voigt(p, gam_L, sig_G):
    """Finesse factor caused by a flat pedestal p for a Voigt line (Lorentz half width gam_L, Gauss sigma sig_G).

    Overlap Omega(s) = p + (1-p) * 2 int_{s/2}^inf V; returns the ratio of the widths at Omega = 1/2 without and
    with the pedestal (root finding with brentq).
    """
    from scipy.special import voigt_profile
    from scipy.integrate import quad
    from scipy.optimize import brentq
    p = min(max(p, 0.0), 0.49)
    if p <= 0: return 1.0
    # tail(s): twice the Voigt area beyond s/2 (overlap of the line with itself shifted by s, for a symmetric line)
    tail = lambda s: quad(lambda x: voigt_profile(x, max(sig_G, 1e-12*gam_L), gam_L), s/2, np.inf, limit=200)[0]*2
    scale = 2*gam_L + 2.355*sig_G
    s0 = brentq(lambda s: tail(s) - 0.5, 1e-9*scale, 50*scale)
    s1 = brentq(lambda s: p + (1 - p)*tail(s) - 0.5, 1e-9*scale, 500*scale)
    return s0/s1


def rechteck_faktor(W, gam_L, sig_G):
    """Finesse factor caused by rectangular convolutions of a Voigt line, computed numerically.

    W: one width or a list of widths (e.g. the transverse wedge across the chord and the pixel width, in phase).
    Returns the ratio of the Omega50 widths: Voigt alone / Voigt convolved with the rectangles.
    """
    from scipy.special import voigt_profile
    Ws = [w for w in (W if np.ndim(W) else [W]) if w > 0]
    if not Ws: return 1.0
    # fine grid (2^17 points) wide enough for the Lorentz wings; convolutions done by FFT
    sg = max(sig_G, 1e-9*gam_L); span = 60*(2*gam_L + 2.355*sg + sum(Ws)); n = 1 << 17
    x = np.linspace(-span, span, n); dx = x[1] - x[0]
    V = voigt_profile(x, sg, gam_L); Pf = np.fft.rfft(np.fft.ifftshift(V))
    for w in Ws: Pf = Pf*np.fft.rfft(np.fft.ifftshift(np.where(np.abs(x) <= w/2, 1.0/w, 0.0)))*dx
    P = np.fft.fftshift(np.fft.irfft(Pf, n=n))
    # s50: shift s at which the overlap of the normalised line with its shifted copy is 1/2 (bisection)
    def s50(p):
        p = p/(p.sum()*dx); lo, hi = 0.0, span/2
        for _ in range(45):
            s = 0.5*(lo + hi); k = int(round(s/dx)); ov = np.minimum(p[k:], p[:-k]).sum()*dx
            if ov > 0.5: lo = s
            else: hi = s
        return 0.5*(lo + hi)
    return s50(V)/s50(P)


def kubo_rest_width(Cm, w, NFFT=None):
    """Overlap widths of the ensemble line of the window w with a non-accumulating residual phase (Kubo form).

    Cm is the covariance of the residual phase along the round trips; the Kubo form is exact for Gaussian phase.
    Returns (D_rest, D_0): the Omega50 width with the residual and that of the clean window line (both in rad).
    This replaces the pedestal of the exit residual.
    """
    from scipy.optimize import brentq
    # coherence matrix G_mn = sqrt(w_m w_n) exp(-Var(phi_m - phi_n)/2); G0 without residual phase
    M = len(w); u = np.sqrt(w); d = np.diag(Cm); V = d[:, None] + d[None, :] - 2*Cm
    G = np.outer(u, u)*np.exp(-0.5*np.maximum(V, 0)); G0 = np.outer(u, u)
    NFFT = NFFT or (1 << int(np.ceil(np.log2(16*M))))
    # omega: the line is the Fourier transform of the diagonal sums of G (autocorrelation); bisection for the shift
    # at which the overlap of the line with its shifted copy falls to 1/2; returns the width in rad
    def omega(Gm):
        diag = np.array([np.trace(Gm, k) for k in range(M)]); c = np.zeros(NFFT); c[0] = diag[0]; c[1:M] = diag[1:]; c[NFFT-M+1:] = diag[1:][::-1]
        I = np.real(np.fft.fft(c)); I = np.maximum(I, 0); tot = I.sum()
        def ov(k): return np.minimum(I, np.roll(I, -k)).sum()
        if ov(NFFT//2) >= 0.5*tot: return np.pi
        lo, hi = 1, NFFT//2
        while hi - lo > 1:
            mid = (lo + hi)//2
            if ov(mid) >= 0.5*tot: lo = mid
            else: hi = mid
        return 2*np.pi*lo/NFFT
    return omega(G), omega(G0)
