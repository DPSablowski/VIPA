"""Closed-form characteristic quantities kappa_par, kappa_q, kappa_perp for a stationary surface with a power-law
spectrum S2(f) ~ f^(-alpha_s), 1/D_m <= f <= f_max (variance 1), round-trip step dx, round-trip number M,
reflectivity R2, chord c.

Purpose
    The kappas are the fractions of the surface phase variance that end up as residual resonance (kappa_par),
    as random chirp (kappa_q) and as strip offset (kappa_perp). They enter the finesse equation
    (finesse_vipa.py). Library module; run directly it prints the values of the standard VIPA.

Derivation (calculation, checked against the spectral integrals):
  Window weights w_m ~ R2^m, m < M.  S = Var_w(m) = R2/(1-R2)^2 - M^2 R2^M/(1-R2^M)^2  (truncated geometric series).
  With x = S (2 pi f_x dx)^2 the transfer functions along the round trips are (for v = gamma M >~ 2.5 to a few percent):
      mean removed:  T_mean = x/(1+x)        (exact for the infinite exponential window, 1 - |W(omega)|^2)
      slope:         T_q    = x/(S (1+x)^2)
      residual:      T_res  = T_mean - S T_q = x^2/(1+x)^2
  Angular mean over the direction of the surface wave (x = c cos^2 phi, c = (f/f_w)^2, f_w = 1/(2 pi sqrt(S) dx)):
      <T_mean> = 1 - (1+c)^(-1/2),   <S T_q> = c/(2 (1+c)^(3/2)).
  Radial integral over the power law, lower limit c0 = xi^2 with xi = L_w/D_m, L_w = 2 pi sqrt(S) dx, u0 = sqrt(1 + xi^2):
      alpha_s = 3:  kappa_mean = 1 + xi - u0,            kappa_Sq = (xi/2) (1 - xi/u0)
      alpha_s = 4:  kappa_mean = xi^2 [Lg(u0) - 1/(u0+1)], kappa_Sq = xi^2 [Lg(u0) - 1/u0],  Lg(u) = (1/2) ln((u+1)/(u-1))
      alpha_s = 2:  (upper limit r = f_max D_m, u1 = sqrt(1 + xi^2 r^2))
                    kappa_mean = ln((u1+1)/(u0+1)) / ln r,   kappa_Sq = (1/u0 - 1/u1) / (2 ln r)
      other alpha_s: ln kappa interpolated quadratically in alpha_s through the three exact values (error <= 9 %, mostly 2-6 %);
                    mean_sq_hyp gives the exact value for any alpha_s with the hypergeometric function 2F1.
  kappa_par = kappa_mean - kappa_Sq,   kappa_q = kappa_Sq / S.
  kappa_perp (mean), fitted form "fit4" (four constants per alpha_s; P90 7 / 5 / 12 % for 2.5 / 3 / 3.5):
      kappa_perp = g0 (c/D_m) (1 + L_w/c)^(-p) (1 + xi)^(-q) (1 + c/D_m)^(s)
      alpha_s = 2.5: g0 3.808, p 0.517, q 0.635, s -3.000 | 3.0: 2.567, 0.546, 0.411, -1.506 | 3.5: 1.619, 0.544, 0.268, -0.192
      (linear in alpha_s in between). The standard is the closed double-filter form (see kappas, perp_form).
  Fine band (white, rms sigma_fein): kappa_par = 1 (more precisely 1 - 1.5 (1-R2)/(1+R2)), kappa_q ~ 0, kappa_perp = (1-R2)/(1+R2).

Run
    python3 kappa_geschlossen.py   (prints the kappas of the standard VIPA, alpha_s = 3)

"""
import numpy as np
from scipy.special import hyp2f1

# default mirror spacing h = 2 mm
H_PLATE = 2e-3


def S_fenster(R2, M):
    """Return S = Var_w(m), the weighted variance of the round-trip number for the truncated geometric window R2^m, m < M."""
    r = R2; return r / (1 - r) ** 2 - M ** 2 * r ** M / (1 - r ** M) ** 2


def _mean_sq(alpha_s, xi, r_max):
    """Return (kappa_mean, kappa_Sq) with the elementary formulas for alpha_s = 2, 3, 4 (no upper limit for 3 and 4);
    other alpha_s by quadratic interpolation of ln kappa through the three exact values.
    """
    u0 = np.sqrt(1 + xi ** 2); Lg = lambda u: 0.5 * np.log((u + 1) / (u - 1))
    if alpha_s == 2:
        u1 = np.sqrt(1 + (xi * r_max) ** 2); return np.log((u1 + 1) / (u0 + 1)) / np.log(r_max), (1 / u0 - 1 / u1) / (2 * np.log(r_max))
    if alpha_s == 3:
        return 1 + xi - u0, xi / 2 * (1 - xi / u0)
    if alpha_s == 4:
        return xi ** 2 * (Lg(u0) - 1 / (u0 + 1)), xi ** 2 * (Lg(u0) - 1 / u0)
    # shape (3, 2)
    vals = np.array([_mean_sq(a, xi, r_max) for a in (2, 3, 4)])
    out = []
    for k in (0, 1):
        p = np.polyfit([2, 3, 4], np.log(vals[:, k]), 2); out.append(float(np.exp(np.polyval(p, alpha_s))))
    return tuple(out)


def _I_tail(c, a, b):
    """Return int_c^inf t^(a-1) (1+t)^(-b) dt = x^p/p 2F1(p, 1-a; p+1; x), p = b - a > 0, x = 1/(1+c)
    (substitution x = 1/(1+t)).
    """
    p = b - a; x = 1 / (1 + c); return x ** p / p * hyp2f1(p, 1 - a, p + 1, x)


def mean_sq_hyp(alpha_s, xi, r_max):
    """Return (kappa_mean, kappa_Sq) for any alpha_s > 1, with both limits c0 = xi^2, c1 = (xi r_max)^2
    (Gauss hypergeometric function 2F1).
    """
    # N: normalisation (spectral integral without filter); J1, J2: integrals with the angle-averaged filters
    c0 = xi ** 2; c1 = (xi * r_max) ** 2; h = alpha_s / 2
    N = np.log(c1 / c0) if abs(h - 1) < 1e-12 else (c0 ** (1 - h) - c1 ** (1 - h)) / (h - 1)
    a1 = 1 - h; J1 = _I_tail(c0, a1, 0.5) - _I_tail(c1, a1, 0.5)
    a2 = 2 - h; J2 = _I_tail(c0, a2, 1.5) - _I_tail(c1, a2, 1.5)
    return (N - J1) / N, 0.5 * J2 / N


def kappas(R2, L_eff, theta_i_deg, chord, D_m, alpha_s, f_max=1 / 0.1e-3, h=H_PLATE, perp_form="doppelfilter"):
    """All characteristic quantities for one geometry; all lengths in metres.

    Arguments: R2 reflectivity, L_eff effective plate length, theta_i_deg angle of incidence, chord, D_m mirror
    diameter, alpha_s spectral exponent, f_max upper spatial frequency, h mirror spacing, perp_form form of kappa_perp:
      "doppelfilter" (standard), "doppelfilter_ohne_alias", "doppelfilter_taylor", "doppelfilter_phi",
      "doppelfilter_0710", "doppelfilter_0710_ohne_alias" (older double-filter variants, kept for comparison),
      anything else -> the two-constant form "fit2".
    Returns a dict with kappa_par, kappa_q, kappa_mean, kappa_perp (mean), kappa_perp_alias, kappa_perp_fit2,
    kappa_perp_fit4, l_perp, S, L_w, L_perp, xi, M, dx.
    """
    # round-trip step dx = 2 h tan(theta), number of round trips M = L_eff/dx, window length L_w = 2 pi sqrt(S) dx
    th = np.deg2rad(theta_i_deg); dx = 2 * h * np.tan(th); M = L_eff / dx
    S = S_fenster(R2, M); L_w = 2 * np.pi * np.sqrt(S) * dx; xi = L_w / D_m
    km, ksq = mean_sq_hyp(alpha_s, xi, f_max * D_m)
    # four-constant fit of kappa_perp per alpha_s (form "fit4"), linear interpolation in alpha_s
    tab = {2.0: (4.378, 0.459, 0.939, -4.598), 2.5: (3.808, 0.517, 0.635, -3.000), 3.0: (2.567, 0.546, 0.411, -1.506), 3.5: (1.619, 0.544, 0.268, -0.192), 4.0: (1.060, 0.519, 0.191, 0.893)}
    ks = sorted(tab); a = float(np.clip(alpha_s, ks[0], ks[-1])); lo = max(k for k in ks if k <= a); hi = min(k for k in ks if k >= a)
    g0, p, q, s = (np.array(tab[lo]) if hi == lo else np.array(tab[lo]) + (a - lo) / (hi - lo) * (np.array(tab[hi]) - np.array(tab[lo])))
    kperp = g0 * (chord / D_m) * (1 + L_w / chord) ** -p * (1 + xi) ** -q * (1 + chord / D_m) ** s
    # strip offset, form with two constants (calculation R53): F_mean = 1 - kappa_mean is the fraction of the variance
    # in the window mean (closed form, 2F1); of it, the fraction [c/(c + l_perp)]^2 varies across the chord, with the
    # decorrelation length l_perp = (0.146 + 0.095 (alpha_s - 3)) D_m sqrt(1 + xi); exponent 2 = variance of a tilt across
    # the chord. Against the exact spectral integral: 8 % in the median (P90 22 %) at 697 grid points, without a trend in
    # alpha_s (the form with four constants per alpha_s: 7 %, P90 27 %).
    l_perp = (0.146 + 0.095 * (alpha_s - 3.0)) * D_m * np.sqrt(1 + xi)
    kperp_fit2 = (1 - km) * (chord / (chord + l_perp)) ** 2
    # strip offset derived in closed form (kperp_closed.py): double filter window x chord, angular mean and radial integral
    # elementary (alpha_s = 2, 3, 4). Against the integral 7 % rms, against the exact covariance median 0.99; more accurate
    # than the form with l_perp (which gave only 0.5 of the exact value at xi < 0.1).
    from kperp_closed import kappa_perp_closed, kappa_perp_alias, L_perp, kappa_perp_taylor
    # standard form perp_form "doppelfilter":
    #  (1) window length of the strip offset with equal area, L_perp = pi N_w dx with N_w = 1/Sum w_m^2 (instead of
    #      2 pi sqrt(S) dx): equal for the untruncated window, up to 1/0.577 longer at the plate edge (M < 2 N_eff);
    #  (2) alias lines of the periodic window at f_x = n/dx, with the hole of the chord filter (at dx >~ c the chord
    #      removes 10-25 % of them).
    # Both without a fitted constant. "doppelfilter_ohne_alias": (1) without the alias term; "doppelfilter_0710": older
    # variant (variance-equal length L_w, alias without the chord); "doppelfilter_0710_ohne_alias" accordingly.
    # Alias term only for alpha_s >= 2 (below, the cut-off at f_max determines the band).
    Mw = max(M, 2.0); alt = perp_form.startswith("doppelfilter_0710"); mit_alias = perp_form in ("doppelfilter", "doppelfilter_0710", "doppelfilter_taylor", "doppelfilter_phi") and alpha_s >= 2.0
    L_p = L_w if alt else L_perp(R2, Mw, dx)
    k_alias = (kappa_perp_alias(alpha_s, R2, max(int(round(M)), 2), dx, D_m, f_max=f_max) if alt else kappa_perp_alias(alpha_s, R2, Mw, dx, D_m, f_max=f_max, c=chord)) if (perp_form.startswith("doppelfilter") and alpha_s >= 2.0) else 0.0
    # base term: chord filter as a Taylor form up to z^4 (complex poles, real part), intermediate alpha_s via ln Z
    # (standard); the variant with the factor Phi(alpha_s) is perp_form="doppelfilter_phi" (and the _0710 variants).
    k_grund = kappa_perp_closed(alpha_s, L_p, chord, D_m, f_max=f_max) if (perp_form == "doppelfilter_phi" or alt) else kappa_perp_taylor(alpha_s, L_p, chord, D_m, f_max=f_max)
    kperp_neu = (k_grund + (k_alias if mit_alias else 0.0)) if perp_form.startswith("doppelfilter") else kperp_fit2
    if not perp_form.startswith("doppelfilter"): L_p = L_w
    return dict(kappa_par=km - ksq, kappa_q=ksq / S, kappa_mean=km, kappa_perp=kperp_neu, kappa_perp_alias=k_alias, kappa_perp_fit2=kperp_fit2, kappa_perp_fit4=kperp, l_perp=l_perp, S=S, L_w=L_w, L_perp=L_p, xi=xi, M=M, dx=dx)


# demo: standard VIPA (R2 = 0.9, L_eff = 25 mm, 4 deg, chord 3.5 mm, D_m = 26.1 mm, alpha_s = 3)
if __name__ == "__main__":
    k = kappas(0.9, 25e-3, 4.0, 3.5e-3, 26.1e-3, 3.0)
    print("Standard-VIPA, alpha_s = 3:", {a: round(float(b), 5) for a, b in k.items()})
