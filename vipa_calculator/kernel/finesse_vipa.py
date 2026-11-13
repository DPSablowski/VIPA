"""Finesse of an incoherently illuminated VIPA in the overlap measure (Omega50), closed-form chain of the main document.

Purpose
    finesse(...) evaluates the closed-form equation for the finesse of a real VIPA: clean line (edge and decay),
    wedge and random chirp, residual resonance of the surface error with the strip coherence factor g^2,
    strip offset across the chord, source width (Gauss, Lorentz, rectangle), temperature, detector (pixel, MTF),
    transverse wedge, exit surface of the PR plate and coating. The widths are combined with the Voigt rule
    F = 2 pi / (0.5346 D + sqrt(0.2166 D^2 + G^2)), D = Lorentz part, G = Gauss part. No covariance calculation
    is needed in the default settings.

Units
    Inputs in SI units (lengths in m, angle of incidence in degrees, wedge angles in arc seconds).

Functions
    phi_h(...)                   phase per reflection 2 k0 sigma_h
    korrektur(...), korrektur_dx(...)
                                 optional correction of the systematic deviation from tables of the grid runs
    gueltigkeit(...)             check of the range of validity
    finesse(...)                 the closed-form finesse (main function)
    finesse_streifensumme(...)   semi-numerical variant: sum of independent simulated strips (R55 step 6, R58 step 5)
    coating_sigma_eq(...), r2_effektiv(...)
                                 coating: equivalent mirror height and effective reflectivity (R51)

Inputs/outputs
    library module (no files, no printout); imported by the evaluation scripts.
"""
import numpy as np
from kappa_geschlossen import kappas
# diameter of the fibre core image at the entrance [m] (shortens the usable plate length by D_CORE/2)
D_CORE = 75e-6
from scipy.special import betaincinv as _bii
# chirp statistics over the plates (R56 step 10): the median of the finesse over the plates corresponds not to the
# median of the chirp but approximately to its MEAN ABSOLUTE VALUE, because the median of a sum of right-skewed
# contributions (D^2 = Dc^2 + g^2 sigma^2 + ...) moves towards the sum of the means (measured: median D^2 = 0.91 x sum
# of the means). Gauss: E|x| = sqrt(2/pi) = 0.798; figure (sphere, d = 12): 0.815.
from scipy.special import gammaln as _gln, erf
# analytic line (Fresnel/Faddeeva), overlap width found with one root search (R59)
from keil_analytisch import omega50 as _omega50_profil
CHIRP_MID = float(np.sqrt(2/np.pi))
# = 0.8147: mean |projection| of a unit vector in d = 12 Zernike dimensions, times sqrt(12)
MEDIAN_FIG_CHIRP = float(np.exp(_gln(6.0) - _gln(6.5))/np.sqrt(np.pi)*np.sqrt(12))
# median variants of the two chirp factors (0.6745 / 0.7129), kept for comparison
CHIRP_MID_MEDIAN = 0.6745; FIG_CHIRP_MEDIAN = float(np.sqrt(_bii(0.5, (12 - 1)/2, 0.5))*np.sqrt(12))
# adiabatic coefficient of the figure residual on a single strip (local calculations with pure figure); option figur_rest="adiabatisch"
G_A2_FIGUR = 1.2
# share of a single strip in the ensemble broadening (g^2(n)/g^2_inf = c1 + (1-c1)(1-1/n)^1.15); option figur_rest="c1"
C1_STREIFEN = 0.23
# narrow rectangle in the Voigt rule: Gaussian line of the same variance W^2/12 (R62)
C_G2_RECHTECK = 1.349**2/12
# rectangular vs. Gaussian distribution of the strip offsets in the Voigt rule (derived: Omega width W/2 vs. 1.349 sigma)
RECHTECK_GEWICHT = (np.sqrt(3)/1.349)**2
# default kappas of the figure term (calculation R41), provisional; normally computed by kappa_fig
KFIG = dict(par=0.13, q=0.005, perp=0.04)
# median/mean of the strip offset (three-band grid); used for the exit surface
M_PERP = 0.68
# w(v) for keil="funktion": W_EXP = threshold of the wedge function, derived from the exact chirped Airy line (w(t) = 0.35-0.37 for t = 0.3-1, R57); 2.71/1.54/4.67 from the analytic line (R42)
W_EXP, W_A, V_C, K_W = 0.360, 2.71, 1.54, 4.67
# D0 = [(A/M)^p + (2 pi/F_R)^p]^(1/p): A = overlap width of the Dirichlet kernel (exact, R57), p fitted (the only constant of the clean line)
A_RAND, P_RAND = 3.004, 2.6
# systematic deviation e = F_eq/F_theory - 1 per range (main document, appendix C.1, paragraph "Gültigkeit und systematische
# Abweichungen je Bereich"; run with one surface per point,
# 1200 points x 100 surfaces), used by korrektur(): key = source width 0 or 6 pm; rows R2 < 0.95 / >= 0.95;
# columns phi_h <= 0.7 / 0.7-1 / 1-1.5 / > 1.5 (with the factor g_phi, computed up to 2 rad). Only the last column needs a correction.
E_TAB = {0: [[0.00, +0.02, 0.00, +0.11], [-0.02, 0.00, -0.04, +0.09]],
         6: [[0.00, +0.02, +0.02, +0.17], [-0.02, -0.01, -0.01, +0.05]]}

def phi_h(wl, sigma_fig=0.0, sigma_mid=0.0, sigma_fein=0.0):
    """Return the phase per reflection 2 k0 sigma_h (Eq. phi_h of the main document) in rad.

    The equation holds without correction up to 1.5 rad (with the factor g_phi).
    """
    return 4*np.pi/wl*np.sqrt(sigma_fig**2 + sigma_mid**2 + sigma_fein**2)

# systematic deviation versus round-trip step dx = 2 h tan(theta) (rows R2 < 0.95 / >= 0.95; columns dx < 0.3 / 0.3-0.6 /
# 0.6-1.2 / > 1.2 mm), from the h grid (600 points x 200 surfaces, h = 0.5-12 mm), range of validity (calculation R55);
# key = source width 0, 2 or 6 pm
E_DX = {0: [[0.00, -0.045, -0.036, -0.063], [0.00, -0.080, -0.130, -0.187]],
        2: [[0.00, -0.028, -0.006, -0.023], [0.00, -0.045, -0.065, -0.102]],
        6: [[0.00, -0.010, 0.000, -0.013], [0.00, -0.036, -0.047, -0.068]]}

def korrektur_dx(F, R2, dx, dlam_Q_pm=0.0):
    """Correction of the systematic deviation at a large round-trip step dx [m]: F_korr = F/(1+e).

    e is interpolated between source widths 0, 2 and 6 pm. Returns (F_korr, e).
    """
    i = 0 if R2 < 0.95 - 1e-9 else 1; dxm = dx*1e3
    j = 0 if dxm < 0.3 else (1 if dxm < 0.6 else (2 if dxm < 1.2 else 3))
    pts = [0.0, 2.0, 6.0]; vals = [E_DX[0][i][j], E_DX[2][i][j], E_DX[6][i][j]]
    e = float(np.interp(min(max(dlam_Q_pm, 0.0), 6.0), pts, vals))
    return F/(1 + e), e

def korrektur(F, R2, phi, dlam_Q_pm=0.0):
    """Correction F_korr = F/(1+e) with e from the range table E_TAB; linear in the source width between 0 and 6 pm.

    Returns (F_korr, e).
    """
    i = 0 if R2 < 0.95 - 1e-9 else 1
    j = 0 if phi <= 0.7 else (1 if phi <= 1.0 else (2 if phi <= 1.5 else 3))
    a = min(max(dlam_Q_pm/6.0, 0.0), 1.0); e = (1-a)*E_TAB[0][i][j] + a*E_TAB[6][i][j]
    return F/(1+e), e

def gueltigkeit(R2, L_eff, theta_i_deg, alpha_as, wl, h=2e-3, phi=0.0, var_q=0.0, sigma_mid=None, sigma_fig=0.0, alpha_s=3.0, D_m=None):
    """Check of the range of validity (R58): phi_mf <= 1.5 (mid + fine band), phi_fig <= 1.5, t~ <= 30,
    and with a wedge v >= 2 or N_F <= 4.

    Returns a dict with v, t, N_F and the flags phi_ok, chirp_ok, plateau, grosschirp (large chirp), and, if sigma_mid is
    given, phi_mf, phi_fig, fig_ok and the hit-shift quantities eps_track, dx_track, track_ok.
    """
    # geometry: round-trip step dx, round trips M, decay v = gamma M, reflection finesse F_R, clean width D0
    th = np.deg2rad(theta_i_deg); k0 = 2*np.pi/wl; dx = 2*h*np.tan(th); M = (L_eff - D_CORE/2)/dx
    r2 = np.sqrt(R2); gam = -np.log(r2); v = gam*M; FR = np.pi*np.sqrt(r2)/(1-r2)
    D0 = ((A_RAND/M)**P_RAND + (2*np.pi/FR)**P_RAND)**(1/P_RAND)
    # wedge chirp beta, random chirp q/2 (median), normalised chirp t and Fresnel number N_F of the plate
    beta = 2*k0*np.deg2rad(alpha_as/3600)*h*np.sin(th); sq = 0.5*np.sqrt(var_q); bt = np.sqrt(beta**2 + (0.6745*sq)**2)
    t = bt*(2/D0)**2; NF = bt*M**2/np.pi
    plateau = (alpha_as > 0) and (v < 2) and (NF > 4)
    out = dict(v=v, t=t, N_F=NF, phi_ok=phi <= 1.5, chirp_ok=(t <= 30) and not plateau, plateau=plateau, grosschirp=t > 30)
    # phi: phase per reflection of mid and fine band (phi_mf) -- the figure counts separately: phi_fig <= 1.5 (sigma_fig <~ lambda/8), see fig_ok
    if sigma_mid is not None:
        k0_ = 2*np.pi/wl; out["phi_mf"] = 2*k0_*np.sqrt(sigma_mid**2 + (0.0)**2); out["phi_fig"] = 2*k0_*sigma_fig; out["phi_ok"] = out["phi_mf"] <= 1.5; out["fig_ok"] = out["phi_fig"] <= 1.5
    if sigma_mid is not None:
        # note (R52): shift of the hits by the deflection at the surface slope (Eq. trackshift) and the resulting phase error eps [rad];
        # eps <= 0.8: deviation phase screen / ray tracing < 2 % in the median (P90 4 %); above that only an estimate
        from track_eps import eps_track
        # default mirror diameter: just covers the plate length and a 5.5 mm wide beam
        if D_m is None: D_m = 2*1.02*np.sqrt((L_eff/2)**2 + (2.75e-3)**2)
        Nw, dxt, eps, fr, ss, sf = eps_track(R2, M, theta_i_deg, wl*1e9, h, D_m, alpha_s, sigma_mid, sigma_fig)
        out.update(eps_track=eps, dx_track=dxt, track_ok=eps <= 0.8)
    return out

from contextlib import contextmanager

@contextmanager
def _plattenabstand(h):
    """Context manager: set the mirror spacing h in the kernel module for the duration of a calculation
    (tracks for the figure sum, strip covariance, exit surface) and restore the old value afterwards.
    """
    import pupil_theory_v7 as PT
    from theorie_grid_dreiband import set_h
    alt = PT.h; set_h(h)
    try: yield
    finally: set_h(alt)

def finesse(R2, L_eff, theta_i_deg, alpha_as, wl, h=2e-3, blende=3.5e-3, f_cyl=75e-3,
            sigma_fig=0.0, sigma_mid=0.0, alpha_s=3.0, D_m=None, sigma_fein=0.0, dlam_Q=0.0, f_G_T=0.0, details=False, kfig=None, perp_exact=False,
            n_PR=1.457, sigma_E_fig=0.0, sigma_E_mid=0.0, alpha_s_E=3.0, sigma_E_fein=0.0, best_focus=False, kohaerent=False, k_sat=6.07, sehne=None, alpha_y_as=0.0, pixel_rad=0.0, koeffizient="neff_geschlossen", dlam_L=0.0,
            kernbild=None, dlam_R=0.0, sigma_H=None, sigma_P=None, rho_fig=0.0, det_mtf_rad=0.0, dT_dx=0.0, dT_dy=0.0, cte=0.55e-6, dn_dT=0.0, n_gap=1.0, sigma_T=0.0, keil="analytisch", exit_modell="kubo", perp_form="doppelfilter", f_max=1/0.1e-3, feinband="lorentz", rechteck="regel", g2_fig=None, fig_perp="rechteck", figur_rest="kohaerenz", ell_fig_rel=0.15, kstar_fig="D0"):
    """Closed-form finesse of the VIPA in the overlap measure Omega50. Returns F, or (F, dict of intermediate quantities) if details=True.

    Main arguments: R2 mirror reflectivity, L_eff effective plate length [m], theta_i_deg angle of incidence [deg],
    alpha_as wedge [arcsec], wl wavelength [m], h mirror spacing [m], sigma_fig / sigma_mid / sigma_fein rms heights of
    the three surface bands [m], alpha_s spectral exponent of the mid band, D_m mirror diameter [m], dlam_Q Gaussian
    source width (FWHM) [m], f_G_T additional Gaussian term [rad].

    f_max [1/m]: upper limit of the mid band (default 1/0.1 mm; above it the roughness counts as fine band sigma_fein). It
    enters the normalisation of the spectrum (noticeable only for alpha_s near 2) and the alias term of the strip offset
    (R60 step 8); for thick plates (h >= 8 mm) with alpha_s <= 2.5 a factor 2 in f_max changes the alias term by about a third.
    kernbild: diameter of the core image [m] (default 75 um). dlam_R: full width of a rectangular source spectrum [m] (filter;
    the overlap width of a rectangle is half its width). sigma_H, sigma_P: tuples (sigma_fig, sigma_mid, sigma_fein) [m] of the
    two gap surfaces (HR and PR); they override sigma_fig/mid/fein: mid and fine band independent (variances add), figure with
    correlation rho_fig.
    keil: "analytisch" (default): D0 and D_c are the overlap widths of the analytic line of the exponential window with
    quadratic phase (Fresnel integral via the Faddeeva function, periodically folded; one root search, no constant; also valid
    on the Fresnel plateau and for a large chirp to about 6 %); "funktion": the closed wedge function with
    D0 = [(3.004/M)^2.6 + (2 pi/F_R)^2.6]^(1/2.6), w(v) and D_inf (R42, R57), only for t~ <= 30 and outside the plateau.
    det_mtf_rad: Gaussian width (FWHM, far-field angle) of the detector transfer function (MTF); enters the Gaussian term like the pixel.
    dT_dx, dT_dy [K/m]: temperature gradient over the plate along / across the dispersion; with cte (of the spacer or the plate,
    1/K), dn_dT [1/K] and n_gap it gives a wedge alpha_T = h (n_gap cte + dn_dT) dT/dx (added to alpha) or a transverse wedge.
    sigma_T [K]: temperature fluctuation between exposures -> Gaussian term f_G,T = 2.3548 |d delta/dT| sigma_T with
    d delta/dT = 2 k0 h cos(theta) (n_gap cte + dn_dT), general in h (added in quadrature to f_G_T).
    blende: collimated beam width across the dispersion = chord of the central order; sehne overrides it (e.g. a side order
    of a round aperture: c*sqrt(1-(theta_x/theta_div)^2)). The divergence in the dispersion direction (NA) does not enter.
    Further options (see the comments in the body): best_focus, kohaerent, koeffizient, perp_exact, perp_form, feinband,
    rechteck, fig_perp, figur_rest, kstar_fig, exit_modell, the exit-surface heights sigma_E_* and the Lorentz source dlam_L.
    """
    if sehne is not None: blende = sehne
    th_ = np.deg2rad(theta_i_deg)
    # temperature gradient as wedge / transverse wedge: gap h(x) = h (1 + cte dT) plus refractive index
    if dT_dx != 0.0 or dT_dy != 0.0:
        k_T = h*(n_gap*cte + dn_dT)
        alpha_as = alpha_as + np.rad2deg(k_T*dT_dx)*3600; alpha_y_as = alpha_y_as + np.rad2deg(k_T*dT_dy)*3600
    # temperature fluctuation between exposures (temperature chapter), general in h
    if sigma_T > 0:
        ddel_dT = 2*(2*np.pi/wl)*h*np.cos(th_)*(n_gap*cte + dn_dT); f_G_T = np.sqrt(f_G_T**2 + (2.3548*abs(ddel_dT)*sigma_T)**2)
    # HR and PR surfaces given separately
    if sigma_H is not None or sigma_P is not None:
        sH = sigma_H if sigma_H is not None else (0.0, 0.0, 0.0); sP = sigma_P if sigma_P is not None else (0.0, 0.0, 0.0)
        sigma_fig = float(np.sqrt(max(sH[0]**2 + sP[0]**2 + 2*rho_fig*sH[0]*sP[0], 0.0)))
        sigma_mid = float(np.sqrt(sH[1]**2 + sP[1]**2)); sigma_fein = float(np.sqrt(sH[2]**2 + sP[2]**2))
    # feinband: "quadratur" (fine band counted in residual resonance, strip offset and n_eff) or "lorentz" (default, R61): the fine
    # band is uncorrelated from reflection to reflection, the accumulated phase is a random walk; its coherence e^{-s^2 |m-m'|/2}
    # convolves the whole line with a Lorentz line of width s^2 = (2 k0 sigma_fein)^2 -- like a Lorentz source, D -> D + s^2; no constant
    sigma_fein_L = 0.0
    if feinband == "lorentz": sigma_fein_L = sigma_fein; sigma_fein = 0.0
    d_core = D_CORE if kernbild is None else kernbild
    # alpha_y_as: transverse wedge (across the dispersion) in arc seconds; shifts the strip lines uniformly over the chord
    # (rectangular distribution of width W = 2 k0 alpha_y c)
    th = np.deg2rad(theta_i_deg); k0 = 2*np.pi/wl
    dx = 2*h*np.tan(th); M = (L_eff - d_core/2)/dx
    r2 = np.sqrt(R2); gam = -np.log(r2); v = gam*M; FR = np.pi*np.sqrt(r2)/(1-r2)
    # free spectral range in wavelength [m]; beta = quadratic phase coefficient of the wedge per round trip
    FSR = wl**2/(2*h*np.cos(th))
    beta = 2*k0*np.deg2rad(alpha_as/3600)*h*np.sin(th)
    # chord of the central order
    chord = blende
    if D_m is None: D_m = 2*1.02*np.sqrt((L_eff/2)**2+(2.75e-3)**2)
    # characteristic quantities of the mid band (closed form); without mid band or figure all zero
    K = kappas(R2, L_eff, theta_i_deg, chord, D_m, alpha_s, h=h, perp_form=perp_form, f_max=f_max) if sigma_mid > 0 or sigma_fig > 0 else dict(kappa_par=0, kappa_q=0, kappa_perp=0, S=1, L_w=0, xi=0)
    # kappas of the figure term from the Zernike calculation (unless given)
    if kfig is None and sigma_fig > 0:
        from kappa_fig import kappa_fig
        with _plattenabstand(h): kq_, kp_, kperp_ = kappa_fig(R2, L_eff*1e3, theta_i_deg, alpha_as, wl*1e9, blende*1e3, D_m)
        kfig = dict(par=kp_, q=kq_, perp=kperp_)
    KF = kfig if kfig is not None else KFIG
    # fine band: kappa_par = kf, kappa_perp = kpf (white); c2 = (2 k0)^2 converts height variance to phase variance.
    # sig2 = residual resonance (phase variance), varq = variance of the random chirp
    kf = 1 - 1.5*(1-R2)/(1+R2); kpf = (1-R2)/(1+R2); c2 = (2*k0)**2
    sig2 = c2*(K["kappa_par"]*sigma_mid**2 + KF["par"]*sigma_fig**2 + kf*sigma_fein**2)
    # parts of the residual resonance (figure separate: option g2_fig)
    sig2_fig = c2*KF["par"]*sigma_fig**2; sig2_mid = sig2 - sig2_fig
    varq = c2*(K["kappa_q"]*sigma_mid**2 + KF["q"]*sigma_fig**2)
    # perp_form (passed to kappas): "doppelfilter" (default: window length of equal area L_perp = pi N_Sigma dx, alias term with the
    # hole of the chord, chord filter 1 - 1/(1 + z^2/3 + z^4/15) with ln Z interpolation; R60 steps 8-10), "doppelfilter_phi" (with
    # the factor Phi(alpha_s)), "doppelfilter_ohne_alias", "doppelfilter_0710" (length L_w, alias without chord),
    # "doppelfilter_0710_ohne_alias" or "fit2" (form with l_perp, two constants)
    exit_on = (sigma_E_fig > 0 or sigma_E_mid > 0 or sigma_E_fein > 0)
    EX = None
    if exit_on:
        # exit surface (R48): residual -> pedestal or Kubo line, curvature -> chirp, linear part per strip -> strip offset
        from exit_surface import exit_kz, sockel_faktor_voigt
        with _plattenabstand(h): EX = exit_kz(R2, L_eff*1e3, theta_i_deg, alpha_as, wl*1e9, blende*1e3, D_m, n_PR, sigma_E_fig, sigma_E_mid, alpha_s_E, sigma_E_fein)
    if perp_exact and (sigma_mid > 0 or sigma_fig > 0 or sigma_fein > 0 or exit_on):
        # strip offset: covariance of the strip means (spectral integral or Zernike sum), median by Satterthwaite/Wilson-Hilferty
        from dof_perp import cov_parts, eig_total, median_factor
        with _plattenabstand(h): Cm, Cf, Ce, wy_ = cov_parts(R2, L_eff*1e3, theta_i_deg, alpha_as, wl*1e9, blende*1e3, D_m, alpha_s)
        Ctot = c2*(sigma_mid**2*Cm + sigma_fig**2*Cf + sigma_fein**2*Ce) + (EX["C"] if EX is not None else 0)
        lam = eig_total(Ctot, wy_)
        mu_p, k_p, fac_p = median_factor([(1.0, lam)]); sperp2 = fac_p*mu_p
    else:
        # closed main form: fitted/closed kappa_perp; perp_exact=True instead: semi-numerical (covariance, eigenvalues).
        # Median per band: ln 2 for the mid band (two modes across the chord: tilt and curvature, median of chi^2_2/2), 1/2 for the
        # figure (one dominant mode), 1 for the fine band (many independent strips) (R53).
        # Figure: tilt across the chord -> strip lines uniformly distributed (rectangle, Omega width W/2 = 1.732 sigma instead of
        # 1.349 sigma for a Gaussian); in the Voigt rule as a Gaussian equivalent with the weight (1.732/1.349)^2 (R53 step 8)
        # figure: variance with median factor 1/2
        sperp2_mid = c2*(np.log(2)*K["kappa_perp"]*sigma_mid**2 + 1.0*kpf*sigma_fein**2); sperp2_figG = c2*0.5*KF["perp"]*sigma_fig**2
        # fig_perp = "regel": the rectangle of the figure goes through the rectangle rule (R62) like filter source, transverse wedge
        # and pixel: narrow compared with the line as a Gaussian of equal variance (weight 1.349^2), wide as a rectangle (weight 3).
        # W_fig = sqrt(12 Var) = full width of the rectangle.
        # fig_perp = "fenster": the weight follows the window shape. Per plate the strip offset of the figure counts as a rectangle
        # (W/2, weight 3) for the edge-limited window (v < 1) and with its variance (Gauss, 1.349^2) for the long decay window (v > 2);
        # measure of the window shape without a constant: N_Sigma/M = 1/(M Sum w_m^2) (1 for the uniform window, -> 0 for the long one).
        # fig_perp = "rechteck" (default): always the rectangle weight.
        from kperp_closed import summe_w2
        f_fenster = 1.0/(max(M, 1.0)*summe_w2(R2, max(M, 2.0)))
        w_fig = {"rechteck": RECHTECK_GEWICHT, "fenster": 1.0 + (RECHTECK_GEWICHT - 1.0)*min(f_fenster, 1.0)}.get(fig_perp, 1.0)
        sperp2 = sperp2_mid + w_fig*sperp2_figG
        W_fig = np.sqrt(12*sperp2_figG) if (fig_perp == "regel" and sperp2_figG > 0) else 0.0
        if fig_perp == "regel": sperp2 = sperp2_mid
        # exit surface: mean scatter of the strip shifts, with the same median factor
        if EX is not None:
            wr_ = EX["wy"]/EX["wy"].sum(); Cx = EX["C"]; sperp2 += M_PERP*float(np.sum(wr_*np.diag(Cx)) - wr_ @ Cx @ wr_)
    aA = 0.0
    if kohaerent and (sigma_mid > 0 or sigma_fig > 0 or sigma_fein > 0):
        # coherent chord sum (laser, point detector): apodisation from the structure function of the strip phases (R49),
        # saturated gamma_A = gamma + k gamma tanh(a_A/(k gamma)); no strip offset
        from apod import a_A_rms
        with _plattenabstand(h): aA, _ = a_A_rms(R2, L_eff*1e3, theta_i_deg, alpha_as, wl*1e9, blende*1e3, D_m, sigma_fig, sigma_mid, alpha_s, sigma_fein, nf=40, nphi=24)
        gam = gam + k_sat*gam*np.tanh(aA/(k_sat*gam)); r2 = np.exp(-gam); FR = np.pi*np.sqrt(r2)/(1-r2); v = gam*M; sperp2 = 0.0
    # clean line width D0 (no wedge): analytic overlap width or the closed edge/decay form
    D0 = _omega50_profil(gam, M, 0.0) if keil == "analytisch" else ((A_RAND/M)**P_RAND + (2*np.pi/FR)**P_RAND)**(1/P_RAND)
    sq = 0.5*np.sqrt(varq); sE2 = EX["var_chirp"] if EX is not None else 0.0
    # median of the chirp: mid and fine band Gaussian (0.6745); figure with fixed rms per surface -> projection of a unit vector in
    # d = 12 Zernike dimensions, median |x| sqrt(d) from the beta distribution = 0.713 (R56, derived)
    vq_fig = c2*KF["q"]*sigma_fig**2; vq_rest = max(varq - vq_fig, 0.0)
    # effective chirp = mean absolute value of beta + q/2 over the plates (folded normal distribution, R24; R56 step 13): s = standard
    # deviation of q/2, figure weighted with (0.815/0.798)^2 (sphere instead of Gauss); E|beta + X| = s sqrt(2/pi) exp(-beta^2/2s^2)
    # + beta erf(beta/(sqrt2 s)); for beta = 0: 0.798 s
    s_q2 = 0.25*(vq_rest + (MEDIAN_FIG_CHIRP/CHIRP_MID)**2*vq_fig) + sE2; s_eff = np.sqrt(s_q2)
    # best focus: wedge and mean chirp are cancelled
    bt = 0.0 if best_focus else (s_eff*np.sqrt(2/np.pi)*np.exp(-beta**2/(2*s_q2)) + beta*erf(beta/(np.sqrt(2)*s_eff)) if s_eff > 0 else abs(beta))
    # wedge function (keil="funktion"): D_inf, normalised chirp t and threshold w(v); Dc = width with chirp
    Dinf = bt/gam*np.log(2/(1+np.exp(-2*v))); t = bt*(2/D0)**2; w = W_EXP + W_A/(1+(v/V_C)**K_W)
    Dc = (_omega50_profil(gam, M, bt) if bt > 0 else D0) if keil == "analytisch" else D0 + Dinf*t**2/(t**2+w**2)
    # phase per reflection of all bands; n_eff = effective number of independent strips (NaN if not computed)
    phi = 4*np.pi/wl*np.sqrt(sigma_fig**2 + sigma_mid**2 + sigma_fein**2)
    n_eff = np.nan
    if koeffizient == "neff_geschlossen" and (sigma_mid > 0 or sigma_fein > 0 or sigma_fig > 0) and not kohaerent:
        # closed strip coherence (R54): structure function of the power law (alpha_s = 2 logarithmic), line mean over k* = 2/D0
        # round trips (2F1 or elementary), chord mean via the incomplete gamma function with the exponent at the coherence length;
        # the figure via its strip offset (D_fig_c)
        from neff_closed import n_eff_closed
        kstar = max(2, int(round(2/D0)))
        # figur_rest = "kohaerenz" (default, R63): the figure decouples the strips with its own strip offset (tilt across the
        # dispersion); its structure function 12 k*^2 sigma_perp,fig^2 (u/c)^2 adds to that of the mid band. No new number.
        # kstar_fig: number of round trips over which the strip offset of the figure counts; "D0" (default), "Dc" (chirped line),
        # "geo" (geometric mean)
        ks_f = {"D0": kstar, "Dc": 2/Dc, "geo": 2/np.sqrt(D0*Dc)}.get(kstar_fig, kstar)
        D_fig_c = 12*ks_f**2*c2*KF["perp"]*sigma_fig**2 if (figur_rest == "kohaerenz" and sigma_fig > 0) else 0.0
        # option "kohaerenz_rest" (second counting, R63 step 6): not the shift but the difference of the residuals decouples,
        # D_psi,fig(u) = k*^2 sigma_res,fig^2 (u/ell)^2 with ell = ell_fig_rel D_m (structure function of the residual of the Zernike
        # figure: 0.13 to 0.23 D_m).
        if figur_rest == "kohaerenz_rest" and sigma_fig > 0: D_fig_c = kstar**2*sig2_fig*(blende/(ell_fig_rel*D_m))**2
        n_eff, _, _ = n_eff_closed(kstar, dx, D_m, alpha_s, sigma_mid, sigma_fein, blende, wl, f_max=f_max, D_fig_c=D_fig_c)
        # sum line (R54 step 5, main form): one strip stays concentrated (g^2 = 0), many independent strips give the ensemble line
        # with c_inf = 1 + phi_mf^2 (second order of the coherence mu = e^{-phi^2}); in between the fraction 1 - 1/n_eff. No constant.
        phi_mf_ = 4*np.pi/wl*np.sqrt(sigma_mid**2 + sigma_fein**2)
        g2 = (1 + phi_mf_**2)*(1 - 1/max(n_eff, 1.0)); g_phi = np.sqrt(max(g2, 0.0))
    elif koeffizient == "neff" and (sigma_mid > 0 or sigma_fig > 0 or sigma_fein > 0) and not kohaerent:
        # semi-numerical form of the coefficient (R58 step 6): g^2 = 0.36 (1 + ln n_eff), n_eff = 1/A_rms^2(2/D0) from the strip
        # coherence (apod.py, numerical); one constant (0.36 = coefficient of the single strip)
        from strip_sum_mc import n_eff_of
        with _plattenabstand(h): n_eff, _ = n_eff_of(R2, L_eff*1e3, theta_i_deg, alpha_as, wl*1e9, blende*1e3, D_m, sigma_fig, sigma_mid, alpha_s, sigma_fein, D0)
        g2 = 0.36*(1 + np.log(max(n_eff, 1.0))); g_phi = np.sqrt(g2)
    else:
        # higher-order factor (R58 steps 3 and 7): the variable is the phase per reflection of mid and fine band phi_mf (short
        # structures decorrelate the strips); the figure contributes little (with phi_h instead of phi_mf up to -68 % for a large figure)
        phi_mf = 4*np.pi/wl*np.sqrt(sigma_mid**2 + sigma_fein**2); phi_fig_ = 4*np.pi/wl*sigma_fig
        # (the figure would contribute like the mid band up to about 1.5 rad and then saturate: phi_fig,eff = 1.5 tanh(phi_fig/1.5); not used here)
        # figure not in the phase factor: its effect is the rectangular distribution of the strip offset (R53 step 8)
        g_phi = np.exp(phi_mf**2/4)
    # figur_rest (R63): counting of the strips for the residual resonance.
    #   "kohaerenz"      default: the figure decouples the strips with its own strip offset; its structure function
    #                    12 k*^2 sigma_perp,fig^2 (u/c)^2 adds to that of the mid band in n_eff (see n_eff_closed above); for the
    #                    figure alone 1/n = sqrt(pi/a) erf(sqrt a) - (1 - e^-a)/a, a = 6 k*^2 sigma_perp,fig^2. No new number.
    #   "kohaerenz_rest" (option, one number ell_fig_rel): counting via the difference of the residuals; equally good on the data.
    #   "hauptform"      n_eff only from mid and fine band. Where the mid band does not decouple the strips, the figure residual is
    #                    lost with g^2 -> 0 (pure figure +35 %, edge-limited grid points with large figure +20 %); kept for comparison.
    #   "adiabatisch"    (not recommended): own broadening of the single strip, g_fig^2 = g^2 + (1/n_eff) 1.2 x^2/(1 + x^2),
    #                    x = sigma_res,fig/D; fits the pure figure on thin plates, overcorrects thick plates with small chord by 14 to 28 %.
    #   "c1"             (not recommended): g_fig^2 = (1 + phi_mid^2)(1 - (1 - c1)/n_eff), c1 = 0.23; too small for the pure figure (+24 %).
    # g2_fig: fixed coefficient for the figure residual (only for tests); None = one coefficient for the whole residual resonance.
    n_ = max(n_eff, 1.0) if np.isfinite(n_eff) else 1.0
    if g2_fig is None and figur_rest == "c1":
        g2_fig = (1 + (4*np.pi/wl*sigma_mid)**2)*(1 - (1 - C1_STREIFEN)/n_)
    if g2_fig is None and figur_rest == "adiabatisch" and sig2_fig > 0:
        x2_ = sig2_fig/(Dc**2 + g_phi**2*sig2_mid); g2_fig = g_phi**2 + G_A2_FIGUR*x2_/(1 + x2_)/n_
    # Lorentz-like width D: chirped clean width and residual resonance (weighted with g^2) added in quadrature
    D = np.sqrt(Dc**2 + (g_phi**2)*sig2) if g2_fig is None else np.sqrt(Dc**2 + (g_phi**2)*sig2_mid + g2_fig*sig2_fig)
    # Gaussian source width (in phase, rad) and temperature term
    fG = np.sqrt((2*np.pi*dlam_Q/FSR)**2 + f_G_T**2)
    # rectangular spectrum: overlap width fR/2 in the Gaussian term (instead of 0.573 f_G)
    fR = 2*np.pi*dlam_R/FSR
    # Lorentz source (laser line, dlam_L = full width at half maximum in wavelength) and fine band as Lorentz: Lorentz widths add, D -> D + f_L (R50)
    fL = 2*np.pi*dlam_L/FSR + (2*k0*sigma_fein_L)**2; D_vor = D; D = D + fL
    # detector MTF as a Gaussian in the far field: Gaussian width (FWHM, angle) -> Gaussian term, in phase 2 pi det_mtf_rad Delta x / lambda
    if det_mtf_rad > 0: fG = np.sqrt(fG**2 + (2*np.pi*det_mtf_rad*dx/wl)**2)
    # rectangular convolutions of the line (R62): filter source f_R, figure rectangle W_fig, transverse wedge W_q = 2 k0 alpha_y c
    # (incoherent only) and pixel W_p = 2 pi pixel_rad Delta x / lambda (pixel_rad = angle of one pixel in the far field =
    # pixel / camera focal length; one order = lambda/Delta x).
    Ws = []
    if locals().get("W_fig", 0.0) > 0 and not kohaerent: Ws.append(W_fig)
    if alpha_y_as != 0.0 and not kohaerent: Ws.append(abs(2*k0*np.deg2rad(alpha_y_as/3600)*blende))
    if pixel_rad > 0: Ws.append(2*np.pi*pixel_rad*dx/wl)
    p_sockel = 0.0; kubo = EX is not None and exit_modell == "kubo"
    # R48 step 9: residual of the exit phase as a Kubo line instead of a pedestal; additive broadening of the line
    if kubo:
        from exit_surface import kubo_rest_width
        D_rest, D_0w = kubo_rest_width(EX["Cm"], EX["w"]); D_add = max(D_rest - D_0w, 0.0)
        D = D + D_add; Dc = Dc + D_add
    # Gaussian term of the Voigt rule: source/temperature/MTF (0.573 = Omega50 width per FWHM of a Gaussian) and strip offset
    # (1.349 sigma = Omega50 width of a Gaussian distribution of rms sigma)
    G2 = (0.573*fG)**2 + 1.349**2*sperp2
    if rechteck == "regel":
        # rectangle rule (default): a rectangle of full width W enters the Gaussian term of the Voigt rule as (c W)^2, with
        #   c^2 = (c_G^2 + c_R^2 u)/(1 + u),  u = (W/(3 Omega_0))^2,  Omega_0 = width without the rectangles,
        # c_G^2 = 1.349^2/12 (narrow rectangle = Gaussian line of equal variance), c_R = 1/2 (wide rectangle: overlap width W/2);
        # at most 2.4 % from the exact convolution Voigt (x) rectangle (W/Omega_0 = 0.1 ... 20), the number 3 fitted to it. No quadrature.
        # Voigt rule for the Omega50 width: 0.5346 D + sqrt(0.2166 D^2 + G^2); F = 2 pi / width.
        Om0 = 0.5346*D + np.sqrt(0.2166*D**2 + G2); G2r = G2
        for W_ in [fR] + Ws:
            if W_ > 0: u_ = (W_/(3*Om0))**2; G2r += (C_G2_RECHTECK + 0.25*u_)/(1 + u_)*W_**2
        F = 2*np.pi/(0.5346*D + np.sqrt(0.2166*D**2 + G2r))
    else:
        # rechteck="numerisch": filter source as (f_R/2)^2 in the Gaussian term, transverse wedge and pixel as a numerical convolution Voigt (x) rectangles
        F = 2*np.pi/(0.5346*D + np.sqrt(0.2166*D**2 + G2 + (0.5*fR)**2))
        if Ws:
            from exit_surface import rechteck_faktor
            F = F*rechteck_faktor(Ws, D/2, np.sqrt(G2)/1.349)
    # line wider than half the free spectral range: finesse set to its minimum 2
    if kubo and D >= np.pi: F = 2.0
    # exit-surface residual as a flat pedestal (exit_modell other than "kubo"): high-pass at the final line width D
    if EX is not None and not kubo:
        with _plattenabstand(h): EXh = exit_kz(R2, L_eff*1e3, theta_i_deg, alpha_as, wl*1e9, blende*1e3, D_m, n_PR, sigma_E_fig, sigma_E_mid, alpha_s_E, sigma_E_fein, D_line=D)
        p_sockel = 1 - np.exp(-EXh["rest"])
        sigG = np.sqrt((0.573*fG)**2 + 1.349**2*sperp2)/1.349
        F = F*sockel_faktor_voigt(p_sockel, D/2, sigG)
    # intermediate quantities for evaluations (widths in rad of round-trip phase)
    if details:
        return F, dict(M=M, v=v, FR=FR, FSR_pm=FSR*1e12, beta=beta, S=K["S"], L_w_mm=K["L_w"]*1e3, xi=K["xi"], kappa_par=K["kappa_par"], kappa_q=K["kappa_q"],
                       kappa_perp=K["kappa_perp"], sig_res=np.sqrt(sig2), s_q=sq, fL=fL, sperp=np.sqrt(sperp2), D0=D0, beta_t=bt, t=t, w=w, Dinf=Dinf, Dc=Dc, D=D, fG=fG,
                       F=F, dlam_pm=FSR*1e12/F, a_A=aA, n_eff=n_eff, g2=g_phi**2, p_sockel=p_sockel, sig2_mid=sig2_mid, sig2_fig=sig2_fig, sperp2_mid=locals().get("sperp2_mid", np.nan), sperp2_figG=locals().get("sperp2_figG", np.nan), G2_rest=(0.573*fG)**2, fR=fR, sd_chirp_E=np.sqrt(EX["var_chirp"]) if EX is not None else 0.0)
    return F


def finesse_streifensumme(R2, L_eff, theta_i_deg, alpha_as, wl, h=2e-3, blende=3.5e-3, sigma_fig=0.0, sigma_mid=0.0, alpha_s=3.0, D_m=None,
                          sigma_fein=0.0, dlam_Q=0.0, n_real=24, seed=2, details=False):
    """Semi-numerical finesse (calculation R55 step 6, R58 step 5): the line of the chord as the sum of n_eff independent strips.

    Each strip is simulated in 1D from the residual spectrum of the geometry (decay, edge, wedge and random chirp of the plate);
    n_eff = 1/A_rms^2(2/D0) from the coherence of the strips (apod.py), strip offset as a Gaussian convolution (median by
    Satterthwaite), source as a Gaussian convolution. No fitted constant apart from the edge (only for D0 = coherence length) and
    no phase factor; explains the trend with the round-trip step, the phase per reflection and the aperture with one mechanism.
    Computing time a few seconds. Returns F, or (F, dict) if details=True.
    """
    from res_eff import RestSpektrum
    from kappa_fig import kappa_fig
    from strip_sum_mc import F_sum_mc, n_eff_of
    from dof_perp import cov_parts, eig_total, median_factor
    from kappa_geschlossen import kappas
    if D_m is None: D_m = 2*1.02*np.sqrt((L_eff/2)**2 + (2.75e-3)**2)
    th = np.deg2rad(theta_i_deg); k0 = 2*np.pi/wl; dx = 2*h*np.tan(th); M = (L_eff - D_CORE/2)/dx
    r2 = np.sqrt(R2); gam = -np.log(r2); FR = np.pi*np.sqrt(r2)/(1 - r2); D0 = ((A_RAND/M)**P_RAND + (2*np.pi/FR)**P_RAND)**(1/P_RAND)
    beta = 2*k0*np.deg2rad(alpha_as/3600)*h*np.sin(th); FSR = wl**2/(2*h*np.cos(th)); c2 = (2*k0)**2
    K = kappas(R2, L_eff, theta_i_deg, blende, D_m, alpha_s, h=h)
    with _plattenabstand(h):
        RS = RestSpektrum(R2, L_eff*1e3, theta_i_deg, alpha_as, wl*1e9, blende*1e3, D_m, alpha_s, nf=100, nphi=48)
        ne, _ = n_eff_of(R2, L_eff*1e3, theta_i_deg, alpha_as, wl*1e9, blende*1e3, D_m, sigma_fig, sigma_mid, alpha_s, sigma_fein, D0)
        Cm, Cf, Ce, wy_ = cov_parts(R2, L_eff*1e3, theta_i_deg, alpha_as, wl*1e9, blende*1e3, D_m, alpha_s)
        kq_, kp_, kperp_ = kappa_fig(R2, L_eff*1e3, theta_i_deg, alpha_as, wl*1e9, blende*1e3, D_m) if sigma_fig > 0 else (0.0, 0.0, 0.0)
    # median strip offset from the eigenvalues; variance of the random chirp; Monte Carlo sum of the strips
    lam = eig_total(c2*(sigma_mid**2*Cm + sigma_fig**2*Cf + sigma_fein**2*Ce), wy_); mu_p, k_p, fac_p = median_factor([(1.0, lam)]); sperp2 = fac_p*mu_p
    varq = c2*(K["kappa_q"]*sigma_mid**2 + kq_*sigma_fig**2)
    F = F_sum_mc(RS, sigma_fig, sigma_mid, sigma_fein, gam, beta, ne, sperp2=sperp2, src_pm=dlam_Q*1e12, FSR_pm=FSR*1e12, n_real=n_real, seed=seed, var_q=varq)
    return (F, dict(n_eff=ne, D0=D0, sperp2=sperp2, var_q=varq, M=M)) if details else F


# ---- coating (calculation R51) ----
# h_eq = S lambda eps: Ta2O5/SiO2 (n 2.10/1.46); HR 0.41, PR (top layer tuned to R2) 0.42-0.48; transfer matrix, coating_tmm.py
S_PHASE_BESCHICHTUNG = 0.45

def coating_sigma_eq(eps_rms, wl, s_phase=S_PHASE_BESCHICHTUNG):
    """Return the equivalent mirror height [m] of a relative layer thickness deviation eps_rms (rms, all layers scaled uniformly).

    sigma_eq = s_phase * wl * eps_rms. Acts like a figure error of the surface (smooth, in the figure band); per coated surface,
    the variances add.
    """
    return s_phase*wl*eps_rms

def r2_effektiv(R2_eintritt, R2_ende, L_eff, h, theta_i_deg, n=2.0):
    """Return the effective reflectivity for an R2 that changes linearly over the plate (thickness gradient of the partial mirror).

    Mean over the first n/gamma round trips from the entrance (rule from calculation R51; +-3 % in F for |dR2| <= 0.01 over
    the plate); solved by a short fixed-point iteration.
    """
    dx = 2*h*np.tan(np.deg2rad(theta_i_deg)); R2e = R2_eintritt
    for _ in range(4):
        gam = -0.5*np.log(R2e); xl = min(n*dx/gam, L_eff); R2e = R2_eintritt + (R2_ende - R2_eintritt)*0.5*xl/L_eff
    return R2e
