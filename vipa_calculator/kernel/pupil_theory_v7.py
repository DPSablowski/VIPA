"""pupil_theory_v7.py -- round-trip pupil theory of the VIPA (the "theory" against which the closed equation and the
ray tracing are compared).

Geometry as in run_full_grid_v7.py: Zernike disk of radius G.surface_radius(L_eff), which encloses the whole
illuminated strip.

Physical model:
  - An order at field angle theta_x is formed by the rays that leave the plate at theta_x
    (virtual sources = mirror images of the focus, the same for all rays). No cone rays with other angles.
  - Partial beams rho of an order: points of the core image (N_CORE, added incoherently) x points of the y chord
    of the round aperture at theta_x (N_Y, added coherently).
  - Fixed ray paths (phase-screen approximation), wedge angle drift as in the ray-tracing pipeline, hard plate edge.

States a = (rho, m) (m = round trip):  P_a = u_a exp(i c.B_a),  u_a = w r2^m 1[alive] e^{i psi_wedge},
c = Zernike coefficients of the surface, B_a = accumulated reflection phase per unit coefficient.
Ensemble (exact for Gaussian Zernike coefficients c ~ N(0, s^2 1)):
  <I(delta)> = sum_{core groups g} sum_{a,b in g} u_a u_b* exp(-s^2/2 |B_a-B_b|^2) e^{i(m_a-m_b)delta}
(by default the "sphere" ensemble is used instead: |c| fixed, as each pipeline surface is scaled exactly to sigma_w).
Single realization: hierarchical decomposition (polynomial orders 2..K in m, and cross terms in eta up to J, are
drawn as samples; the rest is treated in the ensemble), K/J rule.

Main functions:
  configure(...)                 set up raytrace_2d as in the grid pipeline (needed for the order positions)
  order_angles(...)              field angles of the orders measured by the grid pipeline
  order_states(...)              U (M,nr), B (M,nr,NACT), core groups, MEAN_VAR
  T_matrix(U,B,s2,groups)        ensemble kernel over the rest (only pairs within the same core group)
  spectrum_from_T(T, ph)         I(delta)
  measure(d, I)                  (F_flankfit, F_eqwidth) as in the pipeline (circular centroid, one period)
  F_clean / F_ensemble / F_single_K / K_rule / F_single_KJ / KJ_rule
Further: semi-closed approximations of the single realization (instantaneous resonance: random chirp, apodization,
common residual resonance; F_semi_*, F_two_part*, chirp_apod_stats, F_semi_final), scatter loss (scatter_loss) and
the closed Fresnel/Faddeeva form of the clean line (fresnel_line).
Library module, no main program. Units: wavelengths in nm, lengths L and beam width bl in mm, angles in deg,
wedge alpha in arcsec, internal lengths in m.

"""
import sys, os
import numpy as np
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
pass
import run_full_grid_v7 as G
import closed_model3 as CM

h = 2e-3                      # mirror spacing h [m]
theta_div_override = None     # divergence half angle in the dispersion direction [rad]; None: from beam width and f_cyl (round beam)
apertur = "rund"              # 'rund' (round): chord = bl*sqrt(1-(thx/theta_div)^2); 'rechteck' (rectangle): chord = bl for all orders
WEDGE_PHASE = "exact"         # wedge phase of the pupil: "exact" = mirror-image phase k0 d_A [sin(Theta) - sin(Theta - 2 m alpha)]
                              # (series in alpha: a m + beta m^2 - 4/3 k0 alpha^2 h cos(Theta) m^3 + ..., main document eq.
                              # wedge_series); "hitpoint" = the first-order sum 2 k0 alpha cos(theta) x_j over the hit points used
                              # up to 21-10-2026 (gives beta m(m+1) and a cubic term of the wrong sign, +4/3); for old runs only.


def wedge_phase(m, a, Theta, x0, k0, hh=None):
    """Exact wedge phase of round trip m (mirror images of the focus on a circle about the wedge edge).

    m: round-trip index (int); a: wedge angle [rad] (sign: spacing h + a x grows with x); Theta: direction of the
    order against the plate normal, per ray (array, = theta_i + thx) [rad]; x0: entrance position of each ray on the
    plate [m] (x = 0 is the plate centre, where the spacing is hh); k0 vacuum wave number [1/m]; hh: spacing at x = 0
    [m] (default: module value h).
    Returns the phase [rad] per ray relative to the plane etalon of spacing hh, i.e. without the term 2 k0 hh cos(Theta) m
    that is the resonance variable delta:
        2 k0 a cos(Theta) x0 (m + 1)                    entrance spacing h0 = hh + a x0: constant + linear shift a m
      + 2 k0 h0 [cos(Theta) m (sin(2ma)/(2ma) - 1) + sin(Theta) sin^2(ma)/a]
                                                        = beta m^2 - 4/3 k0 a^2 h0 cos(Theta) m^3 + ...  (beta = 2 k0 a h0 sin(Theta)).
    The second line is k0 (h0/a)[sin(Theta) - sin(Theta - 2 m a)] - 2 k0 h0 cos(Theta) m, written without cancellation.
    The constant 2 k0 a cos(Theta) x0 is the same convention as in the hit-point form."""
    hh = h if hh is None else hh
    lin = 2 * k0 * a * np.cos(Theta) * x0 * (m + 1)
    if a == 0 or m == 0:
        return lin
    h0 = hh + a * x0; xx = m * a
    t1 = m * (np.sinc(2 * xx / np.pi) - 1.0)          # sin(2ma)/(2ma) - 1, np.sinc(z) = sin(pi z)/(pi z)
    t2 = np.sin(xx) ** 2 / a
    return lin + 2 * k0 * h0 * (np.cos(Theta) * t1 + np.sin(Theta) * t2)

def theta_div_of(bl_mm):
    """Return the divergence half angle [rad] for beam width bl_mm (or the override value)."""
    return float(theta_div_override) if theta_div_override is not None else float(np.arctan(bl_mm * 1e-3 / (2 * f_cyl)))

def chord_factor(thx, bl_mm):
    """Return chord length / beam width at field angle thx (1 for a rectangular aperture)."""
    if apertur == "rechteck": return 1.0
    td = theta_div_of(bl_mm); return float(np.sqrt(max(0.0, 1.0 - (thx / td) ** 2)))
f_cyl = 75e-3                 # focal length of the cylinder lens [m]


# ------------------------------------------------------------------ geometry / orders
def configure(R2, L_mm, th_deg, wl_nm):
    """Set the module raytrace_2d exactly as process_one_point of the grid pipeline does; returns the module."""
    import raytrace_2d as rt2d
    rt2d.wavelength = wl_nm * 1e-9; rt2d.k0 = 2 * np.pi / rt2d.wavelength
    rt2d.R2 = R2; rt2d.r2 = np.sqrt(R2)
    rt2d.tilt = th_deg; rt2d.theta = np.deg2rad(th_deg)
    rt2d.Delta_x = 2 * rt2d.h * np.tan(rt2d.theta); rt2d.L_step = 2 * rt2d.h / np.cos(rt2d.theta)
    rt2d.L_eff = L_mm * 1e-3; rt2d.R_mirror = L_mm * 1e-3 / 2
    return rt2d


def order_angles(R2, L_mm, th_deg, wl_nm, bl_mm):
    """Return the field angles theta_x [rad] of the orders measured by the grid pipeline (same selection)."""
    rt2d = configure(R2, L_mm, th_deg, wl_nm)
    centers, _ = G.order_centers(th_deg, bl_mm)
    if len(centers) == 0:
        # No order centre inside the divergence cone (thin plate, small beam, long wavelength: angular order spacing > cone width).
        # The wavelength is then thought to be shifted by less than half an order (relative < 1e-3): order on the axis.
        return np.array([0.0])
    return np.arctan(centers / rt2d.F)


_GRID_CACHE = {}


def _gridded_basis(R):
    """Return (xg, grids): the Zernike basis on the pipeline grid (GRID_N x GRID_N, 0 outside the disk of radius R), as in
    build_surface_no_astig; it is interpolated bilinearly as in raytrace_order. Cached per radius."""
    key = (round(R, 9), G.GRID_N)
    if key not in _GRID_CACHE:
        from zernike_hr_surface import zernike
        xg = np.linspace(-R, R, G.GRID_N); X, Y = np.meshgrid(xg, xg)
        rho = np.sqrt(X ** 2 + Y ** 2) / R; phi = np.arctan2(Y, X); mask = rho <= 1
        grids = []
        for (n, m) in CM._active:
            Z = zernike(n, m, rho, phi); Z[~mask] = 0; grids.append(Z)
        _GRID_CACHE[key] = (xg, np.array(grids))
    return _GRID_CACHE[key]


def order_states(R2, L_mm, th_deg, alpha_as, wl_nm, bl_mm, thx, pmax=None,
                 n_y=None, n_core=None, surface="exact"):
    """States of one order at field angle thx. Returns U (M,nr) clean complex amplitudes, B (M,nr,NACT) accumulated
    phase per unit Zernike coefficient, groups (nr,) core-group label per partial beam, MEAN_VAR (closed_model3).
    surface="exact" (default): analytic Zernike basis, so the theory does not depend on the discretization of the
    pipeline; surface="grid": heights as in the pipeline (grid + bilinear interpolation), for diagnosis only.
    The two differ in practice only at the first round trip of partial beams that enter at the circular border of
    the Zernike disk (edge effect); at sigma_w >= 35 nm this amounts to 5-20 %."""
    n_y = G.N_Y if n_y is None else n_y
    n_core = G.N_CORE if n_core is None else n_core
    th = np.deg2rad(th_deg); k0 = 2 * np.pi / (wl_nm * 1e-9); R = L_mm * 1e-3 / 2
    Dx = 2 * h * np.tan(th); Ls = 2 * h / np.cos(th)
    a = alpha_as / 3600 * np.pi / 180; r2 = np.sqrt(R2)
    tdiv = np.arctan(bl_mm * 1e-3 / (2 * f_cyl))
    chord = np.sqrt(max(0.0, 1.0 - (thx / tdiv) ** 2))
    if pmax is None:
        pmax = G.pmax_for_order(L_mm, th_deg, R2, thx)
    R_s = G.surface_radius(L_mm)          # radius of the Zernike disk
    CM.R_mirror = R_s
    # start points: core image across x at the plate edge, chord points across y; equal weights w
    xi_c = np.linspace(-1, 1, n_core) if n_core > 1 else np.array([0.0])
    xi_y = np.linspace(-1, 1, n_y)
    XC, XY = np.meshgrid(xi_c, xi_y)
    x = (-R + G.D_CORE_IMAGE / 2 + XC * G.D_CORE_IMAGE / 2).ravel()
    y = (XY * chord * bl_mm * 1e-3 / 2).ravel()
    groups = np.searchsorted(xi_c, XC.ravel())
    nr = len(x); w = np.full(nr, 1.0 / nr)
    tx = np.full(nr, float(thx)); alive = np.ones(nr, bool)
    acc = np.zeros(nr); accB = np.zeros((nr, CM.NACT))
    x0 = x.copy(); Theta0 = th + tx                     # entrance positions and order direction for the exact wedge phase
    M = pmax + 1
    U = np.zeros((M, nr), complex); B = np.zeros((M, nr, CM.NACT))
    last = M
    # follow the partial beams round trip by round trip until all have left the plate
    for m in range(M):
        alive &= (np.abs(x) <= R)
        if not alive.any():
            last = m; break
        if surface == "grid":
            xg, grids = _gridded_basis(R_s)
            ix = np.clip(np.searchsorted(xg, x) - 1, 0, len(xg) - 2)
            iy = np.clip(np.searchsorted(xg, y) - 1, 0, len(xg) - 2)
            fx = (x - xg[ix]) / (xg[ix + 1] - xg[ix]); fy = (y - xg[iy]) / (xg[iy + 1] - xg[iy])
            Zb = (grids[:, iy, ix] * (1 - fx) * (1 - fy) + grids[:, iy, ix + 1] * fx * (1 - fy)
                  + grids[:, iy + 1, ix] * (1 - fx) * fy + grids[:, iy + 1, ix + 1] * fx * fy)
        else:
            Zb, _, _ = CM.basis_and_grad(x, y)        # (NACT, nr), 0 outside the disk
        # reflection phase 2 k0 h per unit coefficient; wedge phase exact (mirror images) or first order at the hit points
        accB += 2 * k0 * Zb.T
        if WEDGE_PHASE == "exact":
            acc = wedge_phase(m, a, Theta0, x0, k0)
        else:
            acc += 2 * k0 * a * x * np.cos(th)
        U[m] = w * r2 ** m * alive * np.exp(1j * acc)
        B[m] = accB
        # wedge drift of the ray angle, then step by the round-trip offset
        tx += 2 * a * np.cos(th); x = x + Dx + tx * Ls
    return U[:last], B[:last], groups, CM.MEAN_VAR


def s2_of(sw_nm, MV):
    """Return the variance s^2 per Zernike coefficient [m^2] for surface RMS sw_nm."""
    return (sw_nm * 1e-9) ** 2 / MV


# ------------------------------------------------------------------ ensemble kernel and spectrum
_SPH_TAB = {}


def _sphere_cf(x, n):
    """Like _sphere_cf_exact, interpolated from a fine table (spacing 0.005; error ~1e-6)."""
    x = np.asarray(x, float)
    if n not in _SPH_TAB:
        xt = np.arange(0, 400.0, 0.005); _SPH_TAB[n] = (xt, _sphere_cf_exact(xt, n))
    xt, yt = _SPH_TAB[n]
    out = np.interp(x, xt, yt)
    big = x >= xt[-1]
    if np.any(big):
        out[big] = _sphere_cf_exact(x[big], n)
    return out


def _sphere_cf_exact(x, n):
    """Return the characteristic function of the projection of a vector uniformly distributed on the sphere in
    n dimensions (radius such that |x| = rho |v|): Gamma(n/2) (2/x)^(n/2-1) J_(n/2-1)(x)."""
    from scipy.special import jv, gammaln
    nu = n / 2 - 1
    x = np.asarray(x, float); out = np.ones_like(x)
    m = x > 1e-8
    out[m] = np.exp(gammaln(n / 2) + nu * np.log(2 / x[m])) * jv(nu, x[m])
    return out


ENSEMBLE = "sphere"   # "sphere": |c| fixed (as in the pipeline, each surface scaled exactly to sigma_w);
                      # "gauss": c ~ N(0, s^2 1) (Gaussian random field with fluctuating RMS)


def T_matrix(U, B, s2, groups, ensemble=None):
    """Return the (M, M) matrix T(m,n) = sum_g sum_{rho,rho' in g} u_{rho m} u*_{rho' n} exp(-s2/2 |B_{rho m}-B_{rho' n}|^2)
    (Gaussian ensemble; for the sphere ensemble the exponential is replaced by _sphere_cf).
    Core groups are incoherent (only pairs within one group)."""
    M, nr = U.shape
    T = np.zeros((M, M), complex)
    for gi in np.unique(groups):
        idx = np.nonzero(groups == gi)[0]
        Ug = U[:, idx]; Bg = B[:, idx, :]
        Bf = Bg.reshape(M * len(idx), -1); Uf = Ug.reshape(-1); nB = np.sum(Bf ** 2, 1)
        for m in range(M):
            um = Ug[m]
            if not np.any(um):
                continue
            if s2 == 0:
                K = np.ones((len(idx), len(Uf)))
            else:
                Bm = Bg[m]
                D2 = np.clip(np.sum(Bm ** 2, 1)[:, None] + nB[None, :] - 2 * Bm @ Bf.T, 0, None)
                if (ensemble or ENSEMBLE) == "sphere":
                    n = B.shape[2]; K = _sphere_cf(np.sqrt(n * s2 * D2), n)
                else:
                    K = np.exp(-0.5 * s2 * D2)
            T[m] += ((um @ K) * np.conj(Uf)).reshape(M, len(idx)).sum(1)
    return T


def spectrum_from_T(T, ph=None, nfft=2 ** 15):
    """Return (delta, I): I(delta) = sum_{m,n} T(m,n) e^{i(m-n)delta} (optionally with extra phases ph per round trip),
    evaluated by summing T along its diagonals (lags) and one FFT; delta sorted in [-pi, pi)."""
    M = T.shape[0]
    A = T if ph is None else T * np.exp(1j * (ph[:, None] - ph[None, :]))
    idx = (np.arange(M)[:, None] - np.arange(M)[None, :]).ravel() + (M - 1)
    S = (np.bincount(idx, weights=A.real.ravel(), minlength=2 * M - 1)
         + 1j * np.bincount(idx, weights=A.imag.ravel(), minlength=2 * M - 1))
    lags = np.arange(-(M - 1), M); arr = np.zeros(nfft, complex); arr[lags % nfft] = S
    I = np.real(np.fft.ifft(arr)) * nfft
    d = np.fft.fftfreq(nfft) * 2 * np.pi; o = np.argsort(d)
    return d[o], np.clip(I[o], 0, None)


def measure(d, I):
    """Return (F_flankfit, F_eqwidth) as in the grid pipeline: window = one period around the circular centroid,
    flank fit of the FWHM and equivalent width; F = 2 pi / width, NaN if a width is missing."""
    dc = np.angle(np.sum(I * np.exp(1j * d)))
    dd = (d - dc + np.pi) % (2 * np.pi) - np.pi; o = np.argsort(dd); dd, Is = dd[o], I[o]
    ff = G.flank_fit_fwhm(dd, Is); fe = G.equivalent_width_fwhm(dd, Is)
    F1 = 2 * np.pi / ff if ff is not None and np.isfinite(ff) and ff > 0 else np.nan
    F2 = 2 * np.pi / fe if fe is not None and np.isfinite(fe) and fe > 0 else np.nan
    return F1, F2


# ------------------------------------------------------------------ quantities
def F_clean(U, B, groups):
    """Return (F_flankfit, F_eqwidth) of the clean order (no surface error)."""
    return measure(*spectrum_from_T(T_matrix(U, B, 0.0, groups)))


def F_ensemble(U, B, groups, MV, sw_nm, ensemble=None):
    """Return (F_flankfit, F_eqwidth) of the ensemble-mean profile for surface RMS sw_nm."""
    return measure(*spectrum_from_T(T_matrix(U, B, s2_of(sw_nm, MV), groups, ensemble)))


def project_K(U, B, K):
    """Weighted projection of the figure-error phase onto polynomials in m up to order K (weights |u_a|^2).
    Returns (residual B, coefficients, normalized round-trip coordinate tm)."""
    M, nr, _ = B.shape
    W = (np.abs(U) ** 2).reshape(-1); W = W / W.sum()
    mm = np.repeat(np.arange(M), nr).astype(float)
    mb = np.sum(W * mm); ts = np.sqrt(np.sum(W * (mm - mb) ** 2))
    t = (mm - mb) / ts
    X = np.stack([t ** k for k in range(K + 1)], 1)
    Bf = B.reshape(M * nr, -1); XtW = X.T * W
    coef = np.linalg.solve(XtW @ X, XtW @ Bf)
    tm = (np.arange(M) - mb) / ts
    return (Bf - X @ coef).reshape(B.shape), coef, tm


def sigma2_common_res(U, B, MV, sw_nm, K):
    """Return the variance [rad^2] of the common (averaged over partial beams) phase after removing polynomials up to K."""
    s2 = s2_of(sw_nm, MV)
    W = np.abs(U) ** 2; wm = W.sum(1); keep = wm > 0
    Bbar = np.einsum('mk,mkj->mj', W[keep], B[keep]) / wm[keep][:, None]
    w = wm[keep] / wm[keep].sum(); m = np.arange(U.shape[0])[keep].astype(float)
    mb = np.sum(w * m); ts = np.sqrt(np.sum(w * (m - mb) ** 2)); t = (m - mb) / ts
    X = np.stack([t ** k for k in range(K + 1)], 1); XtW = X.T * w
    res = Bbar - X @ np.linalg.solve(XtW @ X, XtW @ Bbar)
    return float(s2 * np.sum(w * np.sum(res ** 2, 1)))


def K_rule(U, B, MV, sw_nm, eps=0.1, Kmax=10):
    """Return the smallest K for which the common residual variance is <= eps rad^2 (at most Kmax)."""
    for K in range(1, Kmax + 1):
        if sigma2_common_res(U, B, MV, sw_nm, K) <= eps:
            return K
    return Kmax


def F_single_K(U, B, groups, MV, sw_nm, K, n_samp=64, seed=0, return_all=False):
    """Single realization: polynomial orders 2..K drawn as samples (jointly Gaussian), rest in the ensemble.
    Returns the medians (F_flankfit, F_eqwidth) over the samples (and the array of all samples if return_all)."""
    s2 = s2_of(sw_nm, MV)
    Bres, coef, tm = project_K(U, B, K)
    T = T_matrix(U, Bres, s2, groups)
    if K < 2:
        return measure(*spectrum_from_T(T))
    Q = coef[2:]
    C = np.random.default_rng(seed).normal(size=(n_samp, Q.shape[1])) * np.sqrt(s2)
    res = []
    for c in C:
        q = Q @ c
        ph = sum(q[j] * tm ** (j + 2) for j in range(len(q)))
        res.append(measure(*spectrum_from_T(T, ph)))
    res = np.array(res)
    med = (float(np.nanmedian(res[:, 0])), float(np.nanmedian(res[:, 1])))
    return (med, res) if return_all else med


# ------------------------------------------------------------------ hierarchy with cross term (across the chord)
def ray_eta(n_y=None, n_core=None):
    """Return the normalized transverse coordinate eta in [-1,1] (y chord) per partial beam, in the order of order_states."""
    n_y = G.N_Y if n_y is None else n_y; n_core = G.N_CORE if n_core is None else n_core
    return np.repeat(np.linspace(-1, 1, n_y), n_core)


def project_KJ(U, B, K, J, eta):
    """Weighted projection of the figure-error phase onto t^k eta^j (k<=K, j<=J), weights |u_a|^2.
    Returns: residual B, coefficients (nterm, NACT), basis values X (M*nr, nterm), list of terms (k, j)."""
    M, nr, _ = B.shape
    W = (np.abs(U) ** 2).reshape(-1); W = W / W.sum()
    mm = np.repeat(np.arange(M), nr).astype(float)
    mb = np.sum(W * mm); ts = np.sqrt(np.sum(W * (mm - mb) ** 2)); t = (mm - mb) / ts
    e = np.tile(eta, M)
    terms = [(k, j) for k in range(K + 1) for j in range(J + 1)]
    from numpy.polynomial import legendre as Lg
    # Legendre products instead of monomials (conditioning); least squares on sqrt(W)-weighted rows
    tn = t / max(np.abs(t).max(), 1e-12)
    Pt = Lg.legvander(tn, K); Pe = Lg.legvander(e, J)
    X = np.stack([Pt[:, k] * Pe[:, j] for (k, j) in terms], 1)
    Bf = B.reshape(M * nr, -1); sw = np.sqrt(W)[:, None]
    coef = np.linalg.lstsq(sw * X, sw * Bf, rcond=None)[0]
    return (Bf - X @ coef).reshape(B.shape), coef, X, terms


def _group_kernels(U, Bres, s2, groups):
    """Per core group: return (flat state index, amplitudes u of the live states, kernel K_ab = exp(-s2/2|Bres_a-Bres_b|^2),
    lag matrix m_a - m_b + M - 1)."""
    M, nr = U.shape; out = []
    for gi in np.unique(groups):
        idx = np.nonzero(groups == gi)[0]
        a_m = np.repeat(np.arange(M), len(idx)); a_r = np.tile(idx, M)
        u = U[:, idx].reshape(-1); live = np.abs(u) > 0
        a_m, a_r, u = a_m[live], a_r[live], u[live]
        Bb = Bres[a_m, a_r, :]
        nB = np.sum(Bb ** 2, 1)
        Kab = np.exp(-0.5 * s2 * np.clip(nB[:, None] + nB[None, :] - 2 * Bb @ Bb.T, 0, None)).astype(np.float32)
        lag = (a_m[:, None] - a_m[None, :]) + (M - 1)
        out.append((a_m * nr + a_r, u, Kab, lag))
    return out


def _spectrum_from_kernels(kers, phase_flat, M, nfft=2 ** 15):
    """Return (delta, I) for the drawn phases phase_flat (per state) combined with the residual ensemble kernels."""
    S = np.zeros(2 * M - 1, complex)
    for flat, u, Kab, lag in kers:
        v = u * np.exp(1j * phase_flat[flat])
        A = (v[:, None] * np.conj(v)[None, :]) * Kab
        S += np.bincount(lag.ravel(), weights=A.real.ravel(), minlength=2 * M - 1) \
             + 1j * np.bincount(lag.ravel(), weights=A.imag.ravel(), minlength=2 * M - 1)
    lags = np.arange(-(M - 1), M); arr = np.zeros(nfft, complex); arr[lags % nfft] = S
    I = np.real(np.fft.ifft(arr)) * nfft
    d = np.fft.fftfreq(nfft) * 2 * np.pi; o = np.argsort(d)
    return d[o], np.clip(I[o], 0, None)


def F_single_KJ(U, B, groups, MV, sw_nm, K, J, eta=None, n_samp=64, seed=0, return_all=False,
                fixed_rms=True, C=None):
    """Single realization with cross term: all terms t^k eta^j are drawn except the pure shifts (k<=1, j=0);
    the rest goes into the ensemble. J=0 corresponds to F_single_K. C = optional given coefficient samples.
    Returns the medians (F_flankfit, F_eqwidth) (and all samples if return_all)."""
    eta = ray_eta() if eta is None else eta
    s2 = s2_of(sw_nm, MV); M = U.shape[0]
    Bres, coef, X, terms = project_KJ(U, B, K, J, eta)
    keep = [i for i, (k, j) in enumerate(terms) if not (j == 0 and k <= 1)]
    Q = coef[keep]; Xk = X[:, keep]
    kers = _group_kernels(U, Bres, s2, groups)
    if C is None:
        C = np.random.default_rng(seed).normal(size=(n_samp, B.shape[2])) * np.sqrt(s2)
        if fixed_rms:
            # as in the pipeline: each realization scaled exactly to sigma_w (|c|^2 = NACT s^2)
            C = C / np.linalg.norm(C, axis=1, keepdims=True) * np.sqrt(B.shape[2] * s2)
    res = np.array([measure(*_spectrum_from_kernels(kers, Xk @ (Q @ c), M)) for c in C])
    med = (float(np.nanmedian(res[:, 0])), float(np.nanmedian(res[:, 1])))
    return (med, res) if return_all else med


def sigma2_diff_res(U, B, MV, sw_nm, K, J, eta=None):
    """Return the weighted variance [rad^2] of the rest after removing t^k eta^j (k<=K, j<=J): criterion for K and J."""
    eta = ray_eta() if eta is None else eta
    s2 = s2_of(sw_nm, MV)
    Bres, _, _, _ = project_KJ(U, B, K, J, eta)
    W = (np.abs(U) ** 2).reshape(-1); W = W / W.sum()
    return float(s2 * np.sum(W * np.sum(Bres.reshape(len(W), -1) ** 2, 1)))


def KJ_rule(U, B, MV, sw_nm, eps=0.1, Kmax=16, Jmax=8, eta=None):
    """Return (K, J) with the smallest K+J for which the WHOLE rest (common and transverse) is <= eps rad^2.
    In this geometry the transverse part does not average out within one realization (partial beams only along
    the y chord), so it has to be drawn as well."""
    cand = sorted(((K, J) for K in range(2, Kmax + 1) for J in range(0, Jmax + 1)), key=lambda kj: (kj[0] + kj[1], kj[1]))
    for K, J in cand:
        if sigma2_diff_res(U, B, MV, sw_nm, K, J, eta) <= eps:
            return K, J
    return Kmax, Jmax


# ------------------------------------------------------------------ semi-closed approximation (instantaneous resonance)
def _fwhm_of(d, I):
    """Return the FWHM (rad) and the equivalent width (rad) of one period around the centroid."""
    dc = np.angle(np.sum(I * np.exp(1j * d)))
    dd = (d - dc + np.pi) % (2 * np.pi) - np.pi; o = np.argsort(dd); dd, Is = dd[o], I[o]
    return G.flank_fit_fwhm(dd, Is), G.equivalent_width_fwhm(dd, Is)


def resonance_form(U, B, eta=None):
    """Instantaneous resonance nu_a = psi_a - psi_{a-1} = c . b_a (b_a: increment of B per round trip).
    Weighted (|u_a|^2) decomposition: mean (shift), linear part in m (random chirp, vector R: d nu/dm = c.R) and
    rest; returns R and the matrix Q of the rest (Var_W(nu_res) = c^T Q c)."""
    M, nr, n = B.shape
    b = np.diff(B, axis=0, prepend=np.zeros((1, nr, n)))          # increment per round trip
    W = (np.abs(U) ** 2).reshape(-1); W = W / W.sum()
    bf = b.reshape(M * nr, n)
    mm = np.repeat(np.arange(M), nr).astype(float)
    mb = np.sum(W * mm); t = mm - mb
    bbar = W @ bf
    R = (W * t) @ bf / np.sum(W * t ** 2)                      # slope per round trip
    res = bf - bbar - np.outer(t, R)
    Q = (res * W[:, None]).T @ res
    return R, Q


def F_semi_closed(U, B, groups, MV, sw_nm, n_samp=2000, seed=0):
    """Semi-closed form FWHM^2 = FWHM_clean(beta + q)^2 + 12 Var_W(nu_res); median over c on the sphere |c| = sigma_w.
    No interference calculation with surface error at all: only the clean pupil with additional chirp and the
    quadratic form Q. Returns (F_flank, F_eq)."""
    s2 = s2_of(sw_nm, MV); n = B.shape[2]; M = U.shape[0]
    R, Q = resonance_form(U, B)
    C = np.random.default_rng(seed).normal(size=(n_samp, n)); C = C / np.linalg.norm(C, axis=1, keepdims=True) * np.sqrt(n * s2)
    q = C @ R                                                # d nu/dm  -> additional chirp beta' = q/2
    V = np.einsum('ij,jk,ik->i', C, Q, C)                     # Var(nu_res) per realization
    m = np.arange(M)
    # clean line with additional chirp: a few nodes in q, then interpolate
    qs = np.quantile(q, np.linspace(0.02, 0.98, 25))
    fw = []
    for qq in qs:
        d, I = spectrum_from_T(T_matrix(U * np.exp(1j * (qq / 2) * (m * (m + 1)))[:, None], B, 0.0, groups))
        fw.append(_fwhm_of(d, I))
    fw = np.array(fw)
    f1 = np.interp(q, qs, fw[:, 0]); f2 = np.interp(q, qs, fw[:, 1])
    tot1 = np.sqrt(f1 ** 2 + 12 * V); tot2 = np.sqrt(f2 ** 2 + 12 * V)
    return float(2 * np.pi / np.median(tot1)), float(2 * np.pi / np.median(tot2))


def F_chirp_only(U, B, groups, MV, sw_nm, n_samp=2000, seed=0, n_q=25):
    """Clean intensity distribution with the drawn additional chirp beta' = q/2 (q = c.R), rest ignored;
    median over c on the sphere. Returns (F_flank, F_eq, std of q, RMS of the residual resonance)."""
    s2 = s2_of(sw_nm, MV); n = B.shape[2]; M = U.shape[0]
    R, Q = resonance_form(U, B)
    C = np.random.default_rng(seed).normal(size=(n_samp, n)); C = C / np.linalg.norm(C, axis=1, keepdims=True) * np.sqrt(n * s2)
    q = C @ R; m = np.arange(M)
    qs = np.quantile(q, np.linspace(0.02, 0.98, n_q)); fw = []
    for qq in qs:
        d, I = spectrum_from_T(T_matrix(U * np.exp(1j * (qq / 2) * (m * (m + 1)))[:, None], B, 0.0, groups))
        fw.append(_fwhm_of(d, I))
    fw = np.array(fw)
    f1 = np.interp(q, qs, fw[:, 0]); f2 = np.interp(q, qs, fw[:, 1])
    return float(2 * np.pi / np.median(f1)), float(2 * np.pi / np.median(f2)), float(np.std(q)), float(np.sqrt(np.trace(Q) * s2))


def F_two_part(U, B, groups, MV, sw_nm, n_samp=2000, seed=0, n_q=25):
    """Two-part semi-closed approximation:
    FWHM^2 = FWHM_clean,apod(beta + q)^2 + 12 Var_W(nu_res,common);
    apodization A_m = sum_rho w exp(-s^2/2 |B_rho m - Bbar_m|^2) (transverse rest in the ensemble, deterministic),
    q = c.R random chirp, common rest = ray-averaged rest of the instantaneous resonance.
    Returns (F_flank, F_eq, and both without the common rest)."""
    s2 = s2_of(sw_nm, MV); n = B.shape[2]; M, nr = U.shape
    # apodization from the transverse part of the phase
    W = np.abs(U) ** 2; wm = W.sum(1); Bbar = np.einsum('mk,mkj->mj', W, B) / np.where(wm > 0, wm, 1)[:, None]
    A = np.zeros(M)
    for mm_ in range(M):
        if wm[mm_] > 0:
            A[mm_] = np.sum(W[mm_] * np.exp(-0.5 * s2 * np.sum((B[mm_] - Bbar[mm_]) ** 2, 1))) / wm[mm_]
    Ua = U * A[:, None]
    # instantaneous resonance: chirp and common rest
    b = np.diff(B, axis=0, prepend=np.zeros((1, nr, n)))
    Wf = W.reshape(-1); Wf = Wf / Wf.sum(); bf = b.reshape(M * nr, n)
    mm = np.repeat(np.arange(M), nr).astype(float); mb = np.sum(Wf * mm); t = mm - mb
    bbar = Wf @ bf; R = (Wf * t) @ bf / np.sum(Wf * t ** 2)
    res = (bf - bbar - np.outer(t, R)).reshape(M, nr, n)
    rbar = np.einsum('mk,mkj->mj', W, res) / np.where(wm > 0, wm, 1)[:, None]      # common rest
    w_m = wm / wm.sum(); Qc = (rbar * w_m[:, None]).T @ rbar
    C = np.random.default_rng(seed).normal(size=(n_samp, n)); C = C / np.linalg.norm(C, axis=1, keepdims=True) * np.sqrt(n * s2)
    q = C @ R; Vc = np.einsum('ij,jk,ik->i', C, Qc, C)
    m = np.arange(M); qs = np.quantile(q, np.linspace(0.02, 0.98, n_q)); fw = []
    for qq in qs:
        d, I = spectrum_from_T(T_matrix(Ua * np.exp(1j * (qq / 2) * (m * (m + 1)))[:, None], B, 0.0, groups))
        fw.append(_fwhm_of(d, I))
    fw = np.array(fw); f1 = np.interp(q, qs, fw[:, 0]); f2 = np.interp(q, qs, fw[:, 1])
    return (float(2 * np.pi / np.median(np.sqrt(f1 ** 2 + 12 * Vc))), float(2 * np.pi / np.median(np.sqrt(f2 ** 2 + 12 * Vc))),
            float(2 * np.pi / np.median(f1)), float(2 * np.pi / np.median(f2)))   # additionally: apodized + chirp without common rest


def F_two_part_v2(U, B, groups, MV, sw_nm, n_samp=2000, seed=0, n_q=25, slow_orders=(2, 3)):
    """Like F_two_part, but: apodization as RMS magnitude of the transverse sum, common rest only with the slow
    orders (nu ~ t^2, t^3). Returns (F_flank, F_eq, F_flank_without_rest, F_eq_without_rest)."""
    from numpy.polynomial import legendre as Lg
    s2 = s2_of(sw_nm, MV); n = B.shape[2]; M, nr = U.shape
    W = np.abs(U) ** 2; wm = W.sum(1)
    A = np.zeros(M)
    for mm_ in range(M):
        if wm[mm_] > 0:
            w = W[mm_] / wm[mm_]; Bm = B[mm_]; nB = np.sum(Bm ** 2, 1)
            K = np.exp(-0.5 * s2 * np.clip(nB[:, None] + nB[None, :] - 2 * Bm @ Bm.T, 0, None))
            A[mm_] = np.sqrt(w @ K @ w)
    Ua = U * A[:, None]
    b = np.diff(B, axis=0, prepend=np.zeros((1, nr, n)))
    Wf = W.reshape(-1); Wf = Wf / Wf.sum(); bf = b.reshape(M * nr, n)
    mm = np.repeat(np.arange(M), nr).astype(float); mb = np.sum(Wf * mm); t = mm - mb
    bbar = Wf @ bf; R = (Wf * t) @ bf / np.sum(Wf * t ** 2)
    res = (bf - bbar - np.outer(t, R)).reshape(M, nr, n)
    rbar = np.einsum('mk,mkj->mj', W, res) / np.where(wm > 0, wm, 1)[:, None]
    w_m = wm / wm.sum(); keep = w_m > 0
    tm = (np.arange(M) - mb); tn = tm / max(np.abs(tm[keep]).max(), 1e-12)
    P = Lg.legvander(tn, max(slow_orders)); X = P[:, list(slow_orders)]
    sw = np.sqrt(w_m)[:, None]
    coef = np.linalg.lstsq(sw * X, sw * rbar, rcond=None)[0]
    slow = X @ coef
    Qc = (slow * w_m[:, None]).T @ slow
    C = np.random.default_rng(seed).normal(size=(n_samp, n)); C = C / np.linalg.norm(C, axis=1, keepdims=True) * np.sqrt(n * s2)
    q = C @ R; Vc = np.einsum('ij,jk,ik->i', C, Qc, C)
    m = np.arange(M); qs = np.quantile(q, np.linspace(0.02, 0.98, n_q)); fw = []
    for qq in qs:
        d, I = spectrum_from_T(T_matrix(Ua * np.exp(1j * (qq / 2) * (m * (m + 1)))[:, None], B, 0.0, groups))
        fw.append(_fwhm_of(d, I))
    fw = np.array(fw); f1 = np.interp(q, qs, fw[:, 0]); f2 = np.interp(q, qs, fw[:, 1])
    return (float(2 * np.pi / np.median(np.sqrt(f1 ** 2 + 12 * Vc))), float(2 * np.pi / np.median(np.sqrt(f2 ** 2 + 12 * Vc))),
            float(2 * np.pi / np.median(f1)), float(2 * np.pi / np.median(f2)))


def F_semi_v3(U, B, groups, MV, sw_nm, n_samp=600, seed=0, n_q=21, slow_orders=(2, 3), per_real_apod=False, nfft=2 ** 14):
    """Semi-closed approximation, variant 3:
    (i) apodization = median over realizations of |sum_rho w e^{i Delta psi}| per m (fixed profile),
        optionally per_real_apod=True: the profile of each realization (upper reference, one FFT per realization);
    (ii) slow common rest (nu ~ t^2, t^3): weighted histogram per realization, convolved with the clean
        (chirped + apodized) intensity distribution;
    (iii) chirp and apodization enter the clean pupil together.
    Returns (F_flank, F_eq) as medians over realizations."""
    from numpy.polynomial import legendre as Lg
    s2 = s2_of(sw_nm, MV); n = B.shape[2]; M, nr = U.shape
    W = np.abs(U) ** 2; wm = W.sum(1); keep = wm > 0
    C = np.random.default_rng(seed).normal(size=(n_samp, n)); C = C / np.linalg.norm(C, axis=1, keepdims=True) * np.sqrt(n * s2)
    # transverse phase per (realization, m, rho)
    Bbar = np.einsum('mk,mkj->mj', W, B) / np.where(keep, wm, 1)[:, None]
    dB = B - Bbar[:, None, :]                                    # (M, nr, n)
    wn = W / np.where(keep, wm, 1)[:, None]
    Aall = np.zeros((n_samp, M))
    for s_ in range(n_samp):
        ph = np.einsum('mkj,j->mk', dB, C[s_])
        Aall[s_] = np.abs(np.sum(wn * np.exp(1j * ph), 1))
    A_med = np.median(Aall, 0)
    # instantaneous resonance: chirp R and slow common rest
    b = np.diff(B, axis=0, prepend=np.zeros((1, nr, n)))
    Wf = W.reshape(-1); Wf = Wf / Wf.sum(); bf = b.reshape(M * nr, n)
    mm = np.repeat(np.arange(M), nr).astype(float); mb = np.sum(Wf * mm); t = mm - mb
    bbar = Wf @ bf; R = (Wf * t) @ bf / np.sum(Wf * t ** 2)
    res = (bf - bbar - np.outer(t, R)).reshape(M, nr, n)
    rbar = np.einsum('mk,mkj->mj', W, res) / np.where(keep, wm, 1)[:, None]
    w_m = wm / wm.sum(); tm = np.arange(M) - mb; tn = tm / max(np.abs(tm[keep]).max(), 1e-12)
    X = Lg.legvander(tn, max(slow_orders))[:, list(slow_orders)]
    sw = np.sqrt(w_m)[:, None]; coef = np.linalg.lstsq(sw * X, sw * rbar, rcond=None)[0]
    slow = X @ coef                                              # (M, n): nu_slow(m) = slow[m].c
    q = C @ R; nu_slow = C @ slow.T                              # (n_samp, M)
    m = np.arange(M); d = np.sort(np.fft.fftfreq(nfft) * 2 * np.pi); dd = d[1] - d[0]
    def clean(qq, A):
        _, I = spectrum_from_T(T_matrix((U * A[:, None]) * np.exp(1j * (qq / 2) * (m * (m + 1)))[:, None], B, 0.0, groups), nfft=nfft)
        return I
    if not per_real_apod:
        qs = np.quantile(q, np.linspace(0.02, 0.98, n_q)); Iq = [clean(qq, A_med) for qq in qs]
    out = []
    for s_ in range(n_samp):
        I = clean(q[s_], Aall[s_]) if per_real_apod else Iq[int(np.argmin(np.abs(qs - q[s_])))]
        # histogram of the slow rest (periodic in delta), convolution via FFT
        h = np.zeros(nfft); idx = np.round((nu_slow[s_] % (2 * np.pi)) / dd).astype(int) % nfft
        np.add.at(h, idx, w_m)
        Ic = np.real(np.fft.ifft(np.fft.fft(np.fft.ifftshift(I)) * np.fft.fft(h)))
        Ic = np.fft.fftshift(Ic)
        out.append(measure(d, np.clip(Ic, 0, None)))
    out = np.array(out)
    return float(np.nanmedian(out[:, 0])), float(np.nanmedian(out[:, 1]))


def F_semi_v4(U, B, groups, MV, sw_nm, n_samp=600, seed=0, n_q=21, nfft=2 ** 14):
    """Like F_semi_v3 (fixed apodization profile), but the shifting part of the common rest is not selected by
    polynomial orders but by a low-pass in m with the clean intensity distribution as transfer function:
    H(f) = I_clean(2 pi f)/I_clean(0) (side bands inside the line broaden it, those outside do not).
    Free of parameters. Returns (F_flank, F_eq) as medians over realizations."""
    s2 = s2_of(sw_nm, MV); n = B.shape[2]; M, nr = U.shape
    W = np.abs(U) ** 2; wm = W.sum(1); keep = wm > 0
    C = np.random.default_rng(seed).normal(size=(n_samp, n)); C = C / np.linalg.norm(C, axis=1, keepdims=True) * np.sqrt(n * s2)
    Bbar = np.einsum('mk,mkj->mj', W, B) / np.where(keep, wm, 1)[:, None]
    dB = B - Bbar[:, None, :]; wn = W / np.where(keep, wm, 1)[:, None]
    Aall = np.zeros((n_samp, M))
    for s_ in range(n_samp):
        Aall[s_] = np.abs(np.sum(wn * np.exp(1j * np.einsum('mkj,j->mk', dB, C[s_])), 1))
    A_med = np.median(Aall, 0)
    b = np.diff(B, axis=0, prepend=np.zeros((1, nr, n)))
    Wf = W.reshape(-1); Wf = Wf / Wf.sum(); bf = b.reshape(M * nr, n)
    mm = np.repeat(np.arange(M), nr).astype(float); mb = np.sum(Wf * mm); t = mm - mb
    bbar = Wf @ bf; R = (Wf * t) @ bf / np.sum(Wf * t ** 2)
    res = (bf - bbar - np.outer(t, R)).reshape(M, nr, n)
    rbar = np.einsum('mk,mkj->mj', W, res) / np.where(keep, wm, 1)[:, None]      # (M, n)
    w_m = wm / wm.sum(); q = C @ R
    m = np.arange(M); d = np.sort(np.fft.fftfreq(nfft) * 2 * np.pi); dd = d[1] - d[0]
    def clean(qq, A):
        _, I = spectrum_from_T(T_matrix((U * A[:, None]) * np.exp(1j * (qq / 2) * (m * (m + 1)))[:, None], B, 0.0, groups), nfft=nfft)
        return I
    qs = np.quantile(q, np.linspace(0.02, 0.98, n_q)); Iq = [clean(qq, A_med) for qq in qs]
    # low-pass: H(f) from the clean line at the median chirp; frequencies in m mapped onto the nfft grid
    I0 = Iq[n_q // 2]; I0 = np.fft.ifftshift(I0)                # I0[k] at delta = 2 pi k/nfft
    fm = np.fft.fftfreq(M)                                       # frequency per round trip for a DFT of length M
    H = np.interp(np.abs(fm) * nfft, np.arange(nfft // 2 + 1), I0[:nfft // 2 + 1]) / I0[0]
    Rw = np.sqrt(w_m)[:, None] * rbar                            # weighting of the nodes (not used further)
    slow = np.real(np.fft.ifft(np.fft.fft(rbar * (w_m[:, None] > 0), axis=0) * H[:, None], axis=0))
    nu_slow = C @ slow.T
    out = []
    for s_ in range(n_samp):
        I = Iq[int(np.argmin(np.abs(qs - q[s_])))]
        h = np.zeros(nfft); idx = np.round((nu_slow[s_] % (2 * np.pi)) / dd).astype(int) % nfft
        np.add.at(h, idx, w_m)
        Ic = np.fft.fftshift(np.real(np.fft.ifft(np.fft.fft(np.fft.ifftshift(I)) * np.fft.fft(h))))
        out.append(measure(d, np.clip(Ic, 0, None)))
    out = np.array(out)
    return float(np.nanmedian(out[:, 0])), float(np.nanmedian(out[:, 1]))


def F_semi_v5(U, B, groups, MV, sw_nm, beta, n_samp=600, seed=0, n_q=21, nfft=2 ** 14, zone_fac=0.5):
    """Variant 5: low-pass = line-shape criterion (as in F_semi_v4) x Fresnel-zone smoothing per realization:
    H_F(f) = exp(-2 (pi f sigma_m)^2), sigma_m = zone_fac * sqrt(pi/|beta + q/2|)  (Gaussian smoothing of nu
    over the Fresnel zone of the total chirp). beta: wedge chirp of the order (rad/round trip^2).
    Returns (F_flank, F_eq) as medians over realizations."""
    s2 = s2_of(sw_nm, MV); n = B.shape[2]; M, nr = U.shape
    W = np.abs(U) ** 2; wm = W.sum(1); keep = wm > 0
    C = np.random.default_rng(seed).normal(size=(n_samp, n)); C = C / np.linalg.norm(C, axis=1, keepdims=True) * np.sqrt(n * s2)
    Bbar = np.einsum('mk,mkj->mj', W, B) / np.where(keep, wm, 1)[:, None]
    dB = B - Bbar[:, None, :]; wn = W / np.where(keep, wm, 1)[:, None]
    Aall = np.zeros((n_samp, M))
    for s_ in range(n_samp):
        Aall[s_] = np.abs(np.sum(wn * np.exp(1j * np.einsum('mkj,j->mk', dB, C[s_])), 1))
    A_med = np.median(Aall, 0)
    b = np.diff(B, axis=0, prepend=np.zeros((1, nr, n)))
    Wf = W.reshape(-1); Wf = Wf / Wf.sum(); bf = b.reshape(M * nr, n)
    mm = np.repeat(np.arange(M), nr).astype(float); mb = np.sum(Wf * mm); t = mm - mb
    bbar = Wf @ bf; R = (Wf * t) @ bf / np.sum(Wf * t ** 2)
    res = (bf - bbar - np.outer(t, R)).reshape(M, nr, n)
    rbar = np.einsum('mk,mkj->mj', W, res) / np.where(keep, wm, 1)[:, None]
    w_m = wm / wm.sum(); q = C @ R
    m = np.arange(M); d = np.sort(np.fft.fftfreq(nfft) * 2 * np.pi); dd = d[1] - d[0]
    def clean(qq, A):
        _, I = spectrum_from_T(T_matrix((U * A[:, None]) * np.exp(1j * (qq / 2) * (m * (m + 1)))[:, None], B, 0.0, groups), nfft=nfft)
        return I
    qs = np.quantile(q, np.linspace(0.02, 0.98, n_q)); Iq = [clean(qq, A_med) for qq in qs]
    fm = np.fft.fftfreq(M)
    nu_res = (C @ rbar.T)                                        # (n_samp, M): common rest per realization
    Fres = np.fft.fft(nu_res * (w_m > 0)[None, :], axis=1)
    out = []
    for s_ in range(n_samp):
        iq = int(np.argmin(np.abs(qs - q[s_]))); I = Iq[iq]; I0 = np.fft.ifftshift(I)
        H_line = np.interp(np.abs(fm) * nfft, np.arange(nfft // 2 + 1), I0[:nfft // 2 + 1]) / I0[0]
        beff = abs(beta + q[s_] / 2)
        sig_m = zone_fac * np.sqrt(np.pi / beff) if beff > 1e-12 else np.inf
        H_F = np.exp(-2 * (np.pi * fm * sig_m) ** 2) if np.isfinite(sig_m) else np.ones(M)
        nu_slow = np.real(np.fft.ifft(Fres[s_] * H_line * H_F))
        h = np.zeros(nfft); idx = np.round((nu_slow % (2 * np.pi)) / dd).astype(int) % nfft
        np.add.at(h, idx, w_m)
        Ic = np.fft.fftshift(np.real(np.fft.ifft(np.fft.fft(I0) * np.fft.fft(h))))
        out.append(measure(d, np.clip(Ic, 0, None)))
    out = np.array(out)
    return float(np.nanmedian(out[:, 0])), float(np.nanmedian(out[:, 1]))


def F_eq_marechal(U, B, groups, MV, sw_nm, n_samp=600, seed=0, n_q=21):
    """Closed structure of the equivalent width: F_eq = F_eq,clean(q, A) * exp(-Var(psi_bar_res)) (Marechal factor),
    Var = weighted variance of the common phase after removing piston, tilt and chirp (quadratic form in c),
    apodization A = median profile. Returns (median F_eq, median F_eq without the Marechal factor)."""
    from numpy.polynomial import legendre as Lg
    s2 = s2_of(sw_nm, MV); n = B.shape[2]; M, nr = U.shape
    W = np.abs(U) ** 2; wm = W.sum(1); keep = wm > 0
    C = np.random.default_rng(seed).normal(size=(n_samp, n)); C = C / np.linalg.norm(C, axis=1, keepdims=True) * np.sqrt(n * s2)
    Bbar = np.einsum('mk,mkj->mj', W, B) / np.where(keep, wm, 1)[:, None]
    dB = B - Bbar[:, None, :]; wn = W / np.where(keep, wm, 1)[:, None]
    Aall = np.zeros((n_samp, M))
    for s_ in range(n_samp):
        Aall[s_] = np.abs(np.sum(wn * np.exp(1j * np.einsum('mkj,j->mk', dB, C[s_])), 1))
    A_med = np.median(Aall, 0)
    # common phase: remove piston, tilt, chirp (Legendre 0..2 in t); residual variance as a quadratic form
    w_m = wm / wm.sum(); mm = np.arange(M).astype(float); mb = np.sum(w_m * mm); tm = mm - mb
    tn = tm / max(np.abs(tm[keep]).max(), 1e-12); X = Lg.legvander(tn, 2); sw = np.sqrt(w_m)[:, None]
    coef = np.linalg.lstsq(sw * X, sw * Bbar, rcond=None)[0]; Bres = Bbar - X @ coef
    Qv = (Bres * w_m[:, None]).T @ Bres
    Xt = np.stack([np.ones(M), tm, tm ** 2], 1); cq = np.linalg.lstsq(sw * Xt, sw * Bbar, rcond=None)[0]
    q = 2 * (C @ cq[2])                                          # chirp: psi ~ (q/2) m^2
    Vpsi = np.einsum('ij,jk,ik->i', C, Qv, C)
    m = np.arange(M); qs = np.quantile(q, np.linspace(0.02, 0.98, n_q)); feq = []
    for qq in qs:
        d, I = spectrum_from_T(T_matrix((U * A_med[:, None]) * np.exp(1j * (qq / 2) * (m * (m + 1)))[:, None], B, 0.0, groups))
        feq.append(measure(d, I)[1])
    f = np.interp(q, qs, np.array(feq))
    return float(np.median(f * np.exp(-Vpsi))), float(np.median(f))


# ------------------------------------------------------------------ scatter loss
def scatter_loss(U, B, groups, MV, sw_nm, n_samp=1000, seed=0):
    """Energy fraction eta = E/E_clean of an order (Parseval: diagonal of the T matrix).
    Returns: eta_ensemble (sphere), eta_gauss, and median and quartiles of the single realization."""
    s2 = s2_of(sw_nm, MV); n = B.shape[2]; M, nr = U.shape
    E0 = 0.0; Es = 0.0; Eg = 0.0
    for gi in np.unique(groups):
        idx = np.nonzero(groups == gi)[0]
        for m in range(M):
            u = U[m, idx]
            if not np.any(u): continue
            Bm = B[m, idx]; nB = np.sum(Bm ** 2, 1)
            D2 = np.clip(nB[:, None] + nB[None, :] - 2 * Bm @ Bm.T, 0, None)
            uu = np.outer(u, np.conj(u)).real
            E0 += uu.sum(); Es += (uu * _sphere_cf(np.sqrt(n * s2 * D2), n)).sum(); Eg += (uu * np.exp(-0.5 * s2 * D2)).sum()
    C = np.random.default_rng(seed).normal(size=(n_samp, n)); C = C / np.linalg.norm(C, axis=1, keepdims=True) * np.sqrt(n * s2)
    eta = np.zeros(n_samp)
    for s_, c in enumerate(C):
        V = U * np.exp(1j * (B @ c)); e = 0.0
        for gi in np.unique(groups):
            e += np.sum(np.abs(V[:, groups == gi].sum(1)) ** 2)
        eta[s_] = e / E0
    return float(Es / E0), float(Eg / E0), float(np.median(eta)), float(np.percentile(eta, 25)), float(np.percentile(eta, 75))


# ------------------------------------------------------------------ periodized Fresnel form
def fresnel_line(delta, gam, beta, M, periodic=True, nwrap=None):
    """Closed Fresnel form of the clean intensity distribution, numerically stable via the Faddeeva function, and
    its periodization (Poisson sum over n_ shifts by 2 pi) for M > M_coh^2.
    E(delta) = sqrt(pi)/(2s) [T(a) - T(b)],  T(x) = exp(A x + i beta x^2) w(i s (x + c)),
    A = -gam + i delta, s = sqrt(-i beta), c = A/(2 i beta), a = -1/2, b = M - 1/2 (midpoint rule for the
    round-trip sum). beta = 0: truncated geometric series (exp(A b) - exp(A a))/A.
    gam = amplitude loss per round trip, beta = chirp (rad/round trip^2, may be complex), M = round trips.
    Returns I(delta) = |E|^2."""
    from scipy.special import wofz
    delta = np.asarray(delta, float)
    if nwrap is None:
        nwrap = int(np.ceil(abs(np.real(beta)) * M / np.pi)) + 2 if periodic else 0
    E = np.zeros(delta.shape, complex)
    for n_ in range(-nwrap, nwrap + 1):
        A = -gam + 1j * (delta + 2 * np.pi * n_)
        a, b = -0.5, M - 0.5
        if abs(beta) < 1e-14:
            E += (np.exp(A * b) - np.exp(A * a)) / A
            continue
        s = np.sqrt(-1j * beta + 0j); c = A / (2j * beta)
        # T(x) uses w(z) for Im z >= 0 and the reflection formula w(z) = 2 exp(-z^2) - w(-z) otherwise (no overflow)
        def T(x):
            z = 1j * s * (x + c)
            pre = np.exp(A * x + 1j * beta * x ** 2)
            out = pre * wofz(np.where(z.imag >= 0, z, 1j))
            neg = z.imag < 0
            if np.any(neg):
                out[neg] = 2 * np.exp((s ** 2 * c ** 2)[neg]) - pre[neg] * wofz(-z[neg])
            return out
        E += np.sqrt(np.pi) / (2 * s) * (T(a) - T(b))
    return np.abs(E) ** 2


def chirp_apod_stats(U, B, groups, MV, sw_nm, n_samp=600, seed=0, n_q=21):
    """Returns per order a dict: F_cA and fwhm_cA (random chirp + median-profile apodization, no rest; median over c)
    for flank fit and equivalent width, and the characteristic quantities of the common rest:
    sig_nu (weighted RMS of nu_res,common, median over c), sig_psi (RMS of the common residual phase after
    removing piston, tilt, chirp), N_eff (effective number of round trips 1/sum w_m^2)."""
    from numpy.polynomial import legendre as Lg
    s2 = s2_of(sw_nm, MV); n = B.shape[2]; M, nr = U.shape
    W = np.abs(U) ** 2; wm = W.sum(1); keep = wm > 0
    C = np.random.default_rng(seed).normal(size=(n_samp, n)); C = C / np.linalg.norm(C, axis=1, keepdims=True) * np.sqrt(n * s2)
    Bbar = np.einsum('mk,mkj->mj', W, B) / np.where(keep, wm, 1)[:, None]
    dB = B - Bbar[:, None, :]; wn = W / np.where(keep, wm, 1)[:, None]
    Aall = np.zeros((n_samp, M))
    for s_ in range(n_samp):
        Aall[s_] = np.abs(np.sum(wn * np.exp(1j * np.einsum('mkj,j->mk', dB, C[s_])), 1))
    A_med = np.median(Aall, 0)
    b = np.diff(B, axis=0, prepend=np.zeros((1, nr, n)))
    Wf = W.reshape(-1); Wf = Wf / Wf.sum(); bf = b.reshape(M * nr, n)
    mm = np.repeat(np.arange(M), nr).astype(float); mb = np.sum(Wf * mm); t = mm - mb
    bbar = Wf @ bf; R = (Wf * t) @ bf / np.sum(Wf * t ** 2)
    res = (bf - bbar - np.outer(t, R)).reshape(M, nr, n)
    rbar = np.einsum('mk,mkj->mj', W, res) / np.where(keep, wm, 1)[:, None]
    w_m = wm / wm.sum(); Qn = (rbar * w_m[:, None]).T @ rbar
    tm = np.arange(M) - mb; tn = tm / max(np.abs(tm[keep]).max(), 1e-12); X = Lg.legvander(tn, 2); sw = np.sqrt(w_m)[:, None]
    Bres = Bbar - X @ np.linalg.lstsq(sw * X, sw * Bbar, rcond=None)[0]; Qp = (Bres * w_m[:, None]).T @ Bres
    q = C @ R; sig_nu = np.sqrt(np.einsum('ij,jk,ik->i', C, Qn, C)); sig_psi = np.sqrt(np.einsum('ij,jk,ik->i', C, Qp, C))
    m = np.arange(M); qs = np.quantile(q, np.linspace(0.02, 0.98, n_q)); fw = []
    for qq in qs:
        d, I = spectrum_from_T(T_matrix((U * A_med[:, None]) * np.exp(1j * (qq / 2) * (m * (m + 1)))[:, None], B, 0.0, groups))
        fw.append(_fwhm_of(d, I))
    fw = np.array(fw); f1 = np.interp(q, qs, fw[:, 0]); f2 = np.interp(q, qs, fw[:, 1])
    return dict(F_cA=(float(2 * np.pi / np.median(f1)), float(2 * np.pi / np.median(f2))),
                fwhm_cA=(float(np.median(f1)), float(np.median(f2))),
                sig_nu=float(np.median(sig_nu)), sig_psi=float(np.median(sig_psi)),
                N_eff=float(1.0 / np.sum(w_m ** 2)))


def F_semi_final(U, B, groups, MV, sw_nm, **kw):
    """Semi-closed form of the single realization (free of parameters):
    Delta^2 = Delta_cA^2 + sigma_nu,res^2, with Delta_cA the width of the clean distribution with random chirp and
    median-profile apodization and sigma_nu,res the weighted RMS of the common residual resonance.
    Returns (F_flank, F_eq). Accuracy against the exact single realization: typ. 6 %, P90 27 %."""
    st = chirp_apod_stats(U, B, groups, MV, sw_nm, **kw)
    return tuple(float(2 * np.pi / np.sqrt(st["fwhm_cA"][i] ** 2 + st["sig_nu"] ** 2)) for i in (0, 1))


def A_s_geometric(U, B, groups, MV, sw_nm, wl_nm, **kw):
    """Return the prefactor A_s of the geometric limit F = A_s lambda/sigma_w of the single realization:
    A_s = k0 sigma_w / sigma_nu,res = sigma_w/(2 sigma_h,res). For comparison, Chabbal (ensemble, no trend removal,
    Gaussian shift): A = 1/(2*2.355) = 0.212."""
    st = chirp_apod_stats(U, B, groups, MV, sw_nm, **kw)
    return float((2 * np.pi / (wl_nm * 1e-9)) * sw_nm * 1e-9 / st["sig_nu"])
