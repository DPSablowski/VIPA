"""Motional narrowing of the residual resonance.

The resonance position nu_m (phase step per round trip, after removing mean and slope) broadens the line only as far
as it acts as a fixed shift over the coherence length of the line, k* = W = 2/D round trips. For a frequency
component omega of the step sequence this is the fraction
    K(omega) = sin^2(k* omega/2) / (k*^2 sin^2(omega/2))       (Fejer kernel; 1 for slow, ~1/(k* omega)^2 for fast components),
because the phase accumulated over k* round trips has the variance sum_omega S(omega) sin^2(k* omega/2)/sin^2(omega/2) = k*^2 sigma_eff^2.
sigma_res,eff^2 = sum_omega S_res(omega) K(omega): spectral integral (mid band), Zernike sum (figure), mean over omega (fine band).

Library module (no command-line use, no files). RestSpektrum(...) precomputes the residual spectra for one geometry;
RestSpektrum.sig2(...) returns the (effective) residual resonance variance sig2 in rad^2 with the Fejer kernel,
sig2_kern(...) the same with the alternative kernel kern_omega. Uses dof_perp._geom for the weights and hit points
and the three-band surface model flaeche_dreiband.
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import numpy as np
from dof_perp import _geom
from flaeche_dreiband import FIG_TERMS, FMAX_DEFAULT
from zernike_hr_surface import zernike

def fejer(om, k):
    """Fejer kernel K(omega) = sin^2(k omega/2) / (k^2 sin^2(omega/2)) (guarded against division by zero)."""
    s = np.sin(om/2); s = np.where(np.abs(s) < 1e-12, 1e-12, s)
    return np.sin(k*om/2)**2/(k**2*s**2)

class RestSpektrum:
    """Residual of the phase steps per spectral component (after weighted removal of mean and slope over the round trips),
    computed once per geometry. Attributes: mid_rest / mid_om (mid band weights and step frequencies), fig / om_fft
    (mean residual power spectrum of the figure terms), fein_rest (residual fraction of the white fine band)."""
    def __init__(self, R2, L_mm, th_deg, alpha_as, wl_nm, bl_mm, D_m, alpha_s, nf=160, nphi=64):
        w, wy, X, Y, dx = _geom(R2, L_mm, th_deg, alpha_as, wl_nm, bl_mm)
        self.k0 = 2*np.pi/(wl_nm*1e-9); M = len(w); m = np.arange(M, dtype=float); m -= np.sum(w*m)
        # H = weighted least-squares projector onto mean and slope in m; the residual operator is 1 - H
        B = np.vstack([np.ones(M), m]).T; L = np.linalg.solve(B.T @ (w[:, None]*B), (B*w[:, None]).T); H = B @ L
        self.w = w; self.M = M
        # mid band: polar grid of spatial frequencies (1/D_m .. f_max) with power-law density f^(2 - alpha_s) per log bin
        f = np.geomspace(1/D_m, FMAX_DEFAULT, nf); ph = (np.arange(nphi) + 0.5)*2*np.pi/nphi
        F, PH = np.meshgrid(f, ph, indexing="ij"); fx = (F*np.cos(PH)).ravel()
        dens = (F**(2 - alpha_s)).ravel(); dens /= dens.sum()
        om = np.angle(np.exp(2j*np.pi*fx*dx))                              # frequency of the step sequence (folded to -pi..pi)
        A = np.exp(1j*np.outer(m, om)); R = A - H @ A
        self.mid_rest = dens*np.sum(w[:, None]*np.abs(R)**2, 0); self.mid_om = om
        rho = np.sqrt(X**2 + Y**2)/(D_m/2); phi = np.arctan2(Y, X); inside = rho <= 1; wr = wy/wy.sum()
        self.fig = []                                                       # figure: residual per term and strip, spectrum via FFT
        NFFT = 1 << int(np.ceil(np.log2(8*M))); self.om_fft = 2*np.pi*np.fft.fftfreq(NFFT); sw = np.sqrt(w)
        for nn, mm_ in FIG_TERMS:
            Z = zernike(nn, mm_, np.minimum(rho, 1), phi)*inside            # (M, ns)
            Rz = Z - H @ Z; P = np.abs(np.fft.fft(Rz*sw[:, None], NFFT, axis=0))**2/NFFT
            self.fig.append(P @ wr)
        self.fig = np.mean(self.fig, 0)                                     # mean residual power spectrum of the figure (variance 1 per term mean)
        # white fine band: residual fraction = weighted trace of the residual projector
        P_ = np.eye(M) - H; self.fein_rest = np.trace(np.diag(w) @ P_ @ P_.T)
    def sig2(self, sig_fig, sig_mid, sig_fein, D=None):
        """Residual resonance variance [rad^2] for the band rms values (in m). D = None: full residual (K = 1);
        otherwise motional narrowing with the Fejer kernel at k* = 2/D."""
        c2 = (2*self.k0)**2
        if D is None: Kmid = 1.0; Kfig = 1.0; Kfein = 1.0
        else:
            k = 2.0/D; Kmid = fejer(self.mid_om, k); Kfig = fejer(self.om_fft, k)
            om = np.linspace(-np.pi, np.pi, 4001); Kfein = np.mean(fejer(om, k))
        s_mid = c2*sig_mid**2*np.sum(self.mid_rest*Kmid)
        s_fig = c2*sig_fig**2*np.sum(self.fig*Kfig)
        s_fein = c2*sig_fein**2*self.fein_rest*Kfein
        return s_mid + s_fig + s_fein


def kern_omega(om, D):
    """Effective fraction of a step component of frequency omega in the static broadening (alternative kernel):
    slow (omega << D) it acts as a scatter of the resonance position (loss sigma^2/(2 D^2) in F), fast as a phase
    modulation with side bands outside the line, whose energy p = sigma^2/(4 sin^2(omega/2)) forms a pedestal
    (loss (pi/2) p). Equating gives K = pi D^2/(4 sin^2(omega/2)); smooth form: K = 1/(1 + 4 sin^2(omega/2)/(pi D^2)).
    No fitted constant. Returns K(omega)."""
    return 1.0/(1.0 + 4*np.sin(om/2)**2/(np.pi*D**2))

def sig2_kern(RS, sig_fig, sig_mid, sig_fein, D):
    """Residual resonance variance [rad^2] as RestSpektrum.sig2, but weighted with kern_omega instead of the Fejer kernel."""
    c2 = (2*RS.k0)**2
    s_mid = c2*sig_mid**2*np.sum(RS.mid_rest*kern_omega(RS.mid_om, D))
    s_fig = c2*sig_fig**2*np.sum(RS.fig*kern_omega(RS.om_fft, D))
    om = np.linspace(-np.pi, np.pi, 4001); s_fein = c2*sig_fein**2*RS.fein_rest*np.mean(kern_omega(om, D))
    return s_mid + s_fig + s_fein
