"""Strip offset kappa_perp of the mid band in closed form.

Purpose
    The strip offset is the variance fraction of the mid-band surface that survives the double filter
        window (along the plate, exponential) x chord (across, uniform).
    The chord filter 1 - sinc^2 is approximated rationally by (z^2/3)/(1+z^2/3); the angular mean is then elementary
    and the radial integral is elementary for the spectral exponents alpha_s = 2, 3, 4. In between, ln kappa is
    interpolated quadratically (as for kappa_mean). With
        u = f L_w, rho = (c'/L_w)^2, c' = pi c/sqrt(3),
        g(u;rho) = rho [sqrt(1+u^2) - (1+rho u^2)^(-1/2)] / (1+rho+rho u^2),
    kappa = int u^(1-alpha) g du / int u^(1-alpha) du over the band from 1/D_m to f_max.
    The module provides three pieces:
      - kappa_perp_closed: rational chord filter with a fitted factor per alpha_s (FAKTOR_ALPHA);
      - kappa_perp_alias: the aliasing contribution of the round-trip sampling (the window samples the surface only
        every dx, so its filter is periodic in f_x);
      - kappa_perp_taylor: version without fitted factor (Taylor chord filter, ln Z interpolation).

Inputs / outputs
    Library module, no files. All lengths in metres, frequencies in 1/m.

Usage
    import kperp_closed as KP; KP.kappa_perp_closed(alpha_s, L_w, c, D_m)

Used in the main document: appendix A.45 (R53: Der Streifenversatz in geschlossener Form mit zwei Konstanten);
appendix A.52 (R60: Der Streifenversatz des Mittelbands als Doppelfilter – geschlossen ohne Dekorrelationslänge).
Pages: 215, 238, 239, 240.
"""
import numpy as np
# Mean correction for the error of the rational approximation of sinc^2
FAKTOR_SINC = 1.17
# Factor per spectral exponent (fitted on a grid of 2900 points against the double-filter integral): absorbs the
# rational approximation of sinc^2 and the interpolation between the exact exponents; linear in between
FAKTOR_ALPHA = {2.0: 1.214, 2.5: 1.337, 3.0: 1.156, 3.5: 1.029, 4.0: 1.067}
def faktor(alpha_s):
    """Return the correction factor for alpha_s (linear interpolation in FAKTOR_ALPHA, clipped to 2..4)."""
    ks = sorted(FAKTOR_ALPHA); a = float(np.clip(alpha_s, ks[0], ks[-1])); return float(np.interp(a, ks, [FAKTOR_ALPHA[k] for k in ks]))
def _J1(x):
    """Return the antiderivative int dx/(x^3 sqrt(1+x^2)) (helper for alpha = 4)."""
    return -np.sqrt(1+x*x)/(2*x*x) + 0.5*np.log((1+np.sqrt(1+x*x))/x)
def _J2(x):
    """Return the antiderivative int dx/(x sqrt(1+x^2)) = ln(x/(1+sqrt(1+x^2)))."""
    return np.log(x/(1+np.sqrt(1+x*x)))
def I_alpha(alpha, xi, rho, u1=np.inf):
    """Return int_xi^u1 u^(1-alpha) g(u;rho) du / int_xi^u1 u^(1-alpha) du for alpha in (2, 3, 4).

    Fa, Fb are antiderivatives of the two terms of g; inf_a, inf_b their limits for u -> infinity
    (for alpha = 2 the upper limit u1 is finite because the band integral diverges logarithmically)."""
    A = 1+rho; sA = np.sqrt(A); sr = np.sqrt(rho)
    if alpha == 3:
        Fa = lambda u: (rho/A)*(-np.sqrt(1+u*u)/u + np.arctanh(u/(sA*np.sqrt(1+u*u)))/sA)
        Fb = lambda u: (rho/A)*(-np.sqrt(1+rho*u*u)/u - np.arctanh(rho*u/(sA*np.sqrt(1+rho*u*u)))/sA)
        inf_a = (rho/A)*(-1+np.arctanh(1/sA)/sA); inf_b = (rho/A)*(-sr-np.arctanh(sr/sA)/sA)
        I = (inf_a-inf_b)-(Fa(xi)-Fb(xi)); N = 1/xi
    elif alpha == 4:
        J3 = lambda u: np.arctan(sr*np.sqrt(1+u*u))/sr
        Fa = lambda u: rho*((1/A)*_J1(u) + (1/A**2)*_J2(u) - (rho/A**2)*J3(u))
        K3 = lambda u: np.arctan(np.sqrt(1+rho*u*u)/sr)/rho**1.5
        Fb = lambda u: rho*((1/A)*rho*_J1(sr*u) - (rho/A**2)*_J2(sr*u) + (rho**2/A**2)*K3(u))
        inf_a = rho*(-(rho/A**2)*(np.pi/2)/sr); inf_b = rho*((rho**2/A**2)*(np.pi/2)/rho**1.5)
        I = (inf_a-inf_b)-(Fa(xi)-Fb(xi)); N = 1/(2*xi*xi)
    elif alpha == 2:
        J3 = lambda u: np.arctan(sr*np.sqrt(1+u*u))/sr
        Fa = lambda u: rho*(1/A)*(_J2(u)+J3(u))
        Fb = lambda u: rho*(1/A)*(_J2(sr*u) - np.arctan(np.sqrt(1+rho*u*u)/sr)/sr)
        I = (Fa(u1)-Fb(u1))-(Fa(xi)-Fb(xi)); N = np.log(u1/xi)
    else: raise ValueError
    return I/N
def kappa_perp_closed(alpha_s, L_w, c, D_m, f_max=1/0.1e-3):
    """Return the strip offset kappa_perp (variance fraction) of the mid band, closed form with fitted factor.

    alpha_s: PSD exponent; L_w: window length [m]; c: chord [m]; D_m: aperture diameter [m] (lowest frequency 1/D_m);
    f_max: upper band frequency [1/m] (default 1/(0.1 mm))."""
    cp = np.pi*c/np.sqrt(3); rho = (cp/L_w)**2; xi = L_w/D_m; u1 = L_w*f_max
    vals = np.array([I_alpha(a, xi, rho, u1) for a in (2, 3, 4)])
    if alpha_s in (2, 3, 4): k = vals[(2, 3, 4).index(alpha_s)]
    else:
        # Quadratic interpolation of ln kappa through the exact exponents 2, 3, 4
        p = np.polyfit([2, 3, 4], np.log(np.maximum(vals, 1e-300)), 2); k = float(np.exp(np.polyval(p, alpha_s)))
    return faktor(alpha_s)*k

# ---- Aliasing of the round-trip sampling: the window samples the surface only every dx, so its filter |a(f_x)|^2 is
# periodic (Poisson kernel = sum of Lorentz lines at f_x = n/dx with mass Sum w_m^2 / dx). Each alias line lets the
# surface power at f_x = n/dx pass into the strip mean. Contribution per line: mass x int S(f_n, f_y) [chord filter] df_y
# over the disc |f| <= f_max (elementary for alpha_s = 2, 3, 4).
# Two forms of the f_y integral: _Iy neglects the chord (valid only for f_n c >> 1; too large for thick plates, where
# f_n c goes down to about 0.6); _Iy_c includes the hole of the chord filter as a Lorentz hole of the same depth and
# area as sinc^2(f_y c), half width f_c = 1/(pi c); against quadrature with exact sinc^2 the ratio is 1.02
# (P10/P90 1.00/1.03). The window length is the equal-area length L_perp = pi N_Sigma dx (N_Sigma = 1/Sum w_m^2)
# rather than 2 pi sqrt(S) dx, which would overestimate kappa_perp at the plate edge by 24-43 %.
# Check on the thick-plate grid (219 points): kappa_perp closed/theory 1.00 (P10/P90 0.95/1.06), all groups 0.98-1.03.
def _Iy(alpha, fn, b):
    """Return int_-b^b (fn^2 + y^2)^(-alpha/2) dy (chord neglected); elementary for alpha = 2, 3, 4, otherwise
    log-linear interpolation between the neighbouring integer exponents."""
    if alpha == 2: return 2*np.arctan(b/fn)/fn
    if alpha == 3: return 2*b/(fn**2*np.sqrt(fn**2+b**2))
    if alpha == 4: return b/(fn**2*(fn**2+b**2)) + np.arctan(b/fn)/fn**3
    lo = int(np.floor(alpha)); hi = lo+1
    if lo < 2: lo, hi = 2, 3
    if hi > 4: lo, hi = 3, 4
    a = np.log(_Iy(lo, fn, b)); c = np.log(_Iy(hi, fn, b)); return float(np.exp(a + (alpha-lo)*(c-a)))
def _Iy_c(alpha, a, b, q):
    """Return int_-b^b y^2/(y^2+q^2) (a^2+y^2)^(-alpha/2) dy: like _Iy, but with the hole of the chord filter
    (Lorentz hole of half width q); elementary for alpha = 2, 3, 4, otherwise log-linear interpolation."""
    d2 = a*a - q*q
    # Avoid the removable singularity at a = q by a tiny shift of q
    if abs(d2) < 1e-6*a*a: q = q*(1+1e-3); d2 = a*a - q*q
    if alpha == 2: return 2*(a*np.arctan(b/a) - q*np.arctan(b/q))/d2
    if alpha == 4: return (a*a/d2)*(b/(a*a*(a*a+b*b)) + np.arctan(b/a)/a**3) + (q*q/d2**2)*(2*np.arctan(b/a)/a - 2*np.arctan(b/q)/q)
    if alpha == 3:
        # The sign of a^2 - q^2 decides between arctan and log form
        T = b/np.sqrt(a*a+b*b); d = np.sqrt(abs(d2))
        L = 2*np.arctan(d*T/q)/(q*d) if d2 > 0 else np.log((q+d*T)/(q-d*T))/(q*d)
        return 2*T/a**2 + (q*q/a**2)*(2*T/d2 - (a*a/d2)*L)
    lo = int(np.floor(alpha)); hi = lo+1
    if lo < 2: lo, hi = 2, 3
    if hi > 4: lo, hi = 3, 4
    A = np.log(_Iy_c(lo, a, b, q)); C = np.log(_Iy_c(hi, a, b, q)); return float(np.exp(A + (alpha-lo)*(C-A)))
def summe_w2(R2, M):
    """Return Sum w_m^2 of the window w_m ~ R2^m, m < M, normalised to unit sum; 1/Sum w_m^2 = N_w is the number of
    effective reflections."""
    r = R2; return (1-r)*(1+r**M)/((1+r)*(1-r**M))
def L_perp(R2, M, dx):
    """Return the window length of the strip offset: Lorentz line of the same height (1) and the same area
    (Sum w_m^2/dx) as the line of the window filter, L_perp = pi N_w dx. For the untruncated window this equals
    2 pi sqrt(S) dx (up to (1-R2)^2/8); at the plate edge (M < 2 N_eff) it is longer by 1/eta,
    eta = 2 sqrt(S) Sum w_m^2 (1 ... 0.577 for the uniform window)."""
    return np.pi*dx/summe_w2(R2, M)
def kappa_perp_alias(alpha_s, R2, M, dx, D_m, f_max=1/0.1e-3, c=None):
    """Return the alias contribution to kappa_perp (variance fraction, for unit variance) for the window
    w_m ~ R2^m, m < M, round-trip offset dx [m], aperture D_m [m], power law alpha_s.

    c: chord [m]. With c the chord filter removes from each alias line the hole at f_y = 0 (Lorentz hole of the same
    depth and the same area 1/c as sinc^2(f_y c): half width f_c = 1/(pi c)); without c the chord is neglected
    (up to 25 % too large for dx >~ c)."""
    # Normalisation N = int_{1/D_m}^{f_max} f^(1-alpha_s) df of the isotropic band
    N = (D_m**(alpha_s-2) - f_max**(2-alpha_s))/(alpha_s-2) if abs(alpha_s-2) > 1e-9 else np.log(f_max*D_m)
    w2 = summe_w2(R2, M)
    s = 0.0; n = 1
    # Sum over the alias lines f_n = n/dx inside the band; b = half length of the f_y chord of the disc |f| <= f_max
    while n/dx < f_max:
        fn = n/dx; b = np.sqrt(f_max**2 - fn**2); s += _Iy(alpha_s, fn, b) if c is None else _Iy_c(alpha_s, fn, b, 1/(np.pi*c)); n += 1
    # Factor 2 for the lines at +-n
    return 2*(w2/dx)*s/(2*np.pi*N)


# ---- Version without fitted factor: chord filter 1 - sinc^2 z ~ 1 - 1/(1 + z^2/3 + z^4/15) (Taylor series of sinc^2
# up to z^4, asymptote 1). The partial-fraction decomposition in z^2 has the complex conjugate poles b_{1,2}
# (b1 + b2 = 1/3, b1 b2 = 1/15), so kappa = 2 Re[A1 I(rho1)] with rho_i = 3 b_i rho, A1 = b1/(b1 - b2). Between
# alpha_s = 2, 3, 4 not ln kappa but ln of the numerator Z = int u^(1-alpha) g du is interpolated quadratically
# (a Laplace transform, hence log-convex); the denominator is exact.
# Against the double-filter integral with exact sinc^2 (adaptive angular mean): 0.96-0.99 per alpha_s, without a constant.
B1 = 1/6 + 1j*np.sqrt(4/15 - 1/9)/2; B2 = np.conj(B1); A1 = B1/(B1 - B2)
def _Z_alpha_c(alpha, xi, rho, u1=np.inf):
    """Return the numerator integral int_xi^u1 u^(1-alpha) g(u;rho) du for complex rho (alpha = 2, 3, 4);
    principal branches, checked against quadrature."""
    A = 1+rho; sA = np.sqrt(A+0j); sr = np.sqrt(rho+0j); ath = lambda x: np.arctanh(x+0j); sq = lambda x: np.sqrt(x+0j)
    if alpha == 3:
        Fa = lambda u: (rho/A)*(-sq(1+u*u)/u + ath(u/(sA*sq(1+u*u)))/sA)
        Fb = lambda u: (rho/A)*(-sq(1+rho*u*u)/u - ath(rho*u/(sA*sq(1+rho*u*u)))/sA)
        inf_a = (rho/A)*(-1+ath(1/sA)/sA); inf_b = (rho/A)*(-sr-ath(sr/sA)/sA)
        return (inf_a-inf_b)-(Fa(xi)-Fb(xi))
    J1 = lambda x: -sq(1+x*x)/(2*x*x) + 0.5*np.log((1+sq(1+x*x))/x); J2 = lambda x: np.log(x/(1+sq(1+x*x))); J3 = lambda u: np.arctan(sr*sq(1+u*u))/sr
    if alpha == 4:
        Fa = lambda u: rho*((1/A)*J1(u) + (1/A**2)*J2(u) - (rho/A**2)*J3(u))
        K3 = lambda u: np.arctan(sq(1+rho*u*u)/sr)/rho**1.5
        Fb = lambda u: rho*((1/A)*rho*J1(sr*u) - (rho/A**2)*J2(sr*u) + (rho**2/A**2)*K3(u))
        inf_a = rho*(-(rho/A**2)*(np.pi/2)/sr); inf_b = rho*((rho**2/A**2)*(np.pi/2)/rho**1.5)
        return (inf_a-inf_b)-(Fa(xi)-Fb(xi))
    if alpha == 2:
        Fa = lambda u: rho*(1/A)*(J2(u)+J3(u)); Fb = lambda u: rho*(1/A)*(J2(sr*u) - np.arctan(sq(1+rho*u*u)/sr)/sr)
        return (Fa(u1)-Fb(u1))-(Fa(xi)-Fb(xi))
    raise ValueError
def _N_alpha(alpha, xi, u1):
    """Return the denominator int_xi^u1 u^(1-alpha) du (logarithmic for alpha = 2)."""
    return np.log(u1/xi) if abs(alpha-2) < 1e-9 else (xi**(2-alpha) - u1**(2-alpha))/(alpha-2)
def kappa_perp_taylor(alpha_s, L_w, c, D_m, f_max=1/0.1e-3):
    """Return the strip offset of the mid band from the double filter without fitted factor (Taylor chord filter,
    ln Z interpolation). Arguments as in kappa_perp_closed."""
    rho = (np.pi*c/np.sqrt(3)/L_w)**2; xi = L_w/D_m; u1 = L_w*f_max
    Z = np.array([2*np.real(A1*_Z_alpha_c(a, xi, 3*B1*rho, u1 if a == 2 else np.inf)) for a in (2, 3, 4)])
    Z = np.maximum(Z, 1e-300)
    if alpha_s in (2, 3, 4): z = Z[(2, 3, 4).index(alpha_s)]
    else: z = float(np.exp(np.polyval(np.polyfit([2, 3, 4], np.log(Z), 2), alpha_s)))
    return float(z/_N_alpha(alpha_s, xi, u1))
