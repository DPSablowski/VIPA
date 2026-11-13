"""VIPA finesse calculator: computational core (no GUI), version 2026-10-18.

Two routes for the same inputs:
  equation(p)            closed-form equation (overlap measure Omega50): net finesse of the median plate
  theory(p, n, seed)     exact line of the theory (round-trip pupil, incoherent across the chord) for n surfaces drawn from the
                         three-band surface model; n = 1: one realisation, n > 1: ensemble with median and spread
  length_series(p, Ls)   equation (and optionally theory) versus plate length
  static_model(p)        comparison model in the style of the LightMachinery calculator (Hu sum truncated at L/dx reflections,
                         convolved with a static Gaussian thickness distribution, FWHM)

Solid VIPA (refractive index n' in the gap): everything between the mirrors happens inside the medium, so the problem is exactly
an air-spaced VIPA with wavelength lambda/n' and the internal angle theta' = asin(sin theta / n') (outside medium n = 1).
Thickness h, geometric wedge angle, beam height and surface heights are unchanged; the surface phase 2 k0 n' dh follows from
k = n' k0. Quantities defined outside the plate are converted:
  - source widths: dlam_eq = dlam * n_g / n'^2 (same fraction of the FSR; n_g group index, default n_g = n'),
  - pixel (far-field angle outside): pixel_eq = pixel / n',
  - divergence in the dispersion direction inside the medium: theta_div / n',
  - temperature: d delta/dT = 2 k0 h [CTE Q + n' dn'/dT / Q], Q = sqrt(n'^2 - sin^2 theta), as a Gaussian of FWHM
    2.3548 |d delta/dT| sigma_T.
Not included: dispersion other than through n_g, refraction at a rough exit surface, the AR-HR transition.
Input mirror R1 < 1: the amplitude per round trip is r1 r2, i.e. the calculation uses R1*R2.
"""
import os, sys, time, math
from dataclasses import dataclass, asdict, fields
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "kernel"))
os.environ.setdefault("OMP_NUM_THREADS", "1")
import warnings; warnings.filterwarnings("ignore")
import numpy as np

C0 = 299792458.0


@dataclass
class Params:
    # light and mirrors
    wl_nm: float = 630.0          # vacuum wavelength
    R1_pct: float = 100.0         # input mirror (HR)
    R2_pct: float = 90.0          # output mirror (PR)
    # plate
    h_mm: float = 2.0             # mirror spacing / plate thickness
    n_gap: float = 1.0            # refractive index in the gap (1 = air, 1.457 = fused silica at 630 nm)
    n_group: float = 0.0          # group index (0: same as n_gap); only FSR in GHz/pm and source widths
    L_mm: float = 25.0            # plate length (illuminable length in the dispersion direction)
    theta_deg: float = 4.0        # angle of incidence outside (air)
    wedge_as: float = 0.0         # thickness wedge along the dispersion direction [arcsec]
    beam_mm: float = 3.5          # collimated beam width across (chord of the central order)
    f_cyl_mm: float = 75.0        # cylinder lens focal length (theory only: divergence, order selection)
    # surface (thickness variation of the gap, RMS; three-band model)
    s_fig_nm: float = 0.0         # figure (smooth over the plate)
    s_mid_nm: float = 0.0         # mid band, power law between 1/D_m and 1/0.1 mm
    alpha_s: float = 3.0          # spectral exponent of the mid band
    bands: str = "stationary"     # convention of s_fig/s_mid: "stationary" (field as such; grids, equation) or "measured" (Zernike fit of
                                  # degrees 2-4 = figure, residual rms less fine band = mid band, as an interferometer delivers them)
    two_surfaces: bool = False    # True: HR and PR are drawn as two independent surfaces (inputs are per surface); False: one field = thickness variation of the gap
    s_fine_nm: float = 0.0        # fine band (structures < 0.1 mm, uncorrelated from reflection to reflection)
    D_m_mm: float = 0.0           # diameter of the surface model (0: from L)
    map_hr: str = ""              # measured height map of the HR surface (plate map .vmap.npz from surface_map.py); adds to the model surface
    map_pr: str = ""              # same for the PR surface (hit points shifted by half the walk-off)
    # source, detector, temperature
    src_gauss_pm: float = 0.0     # Gaussian source, FWHM
    src_lorentz_pm: float = 0.0   # Lorentzian source (laser line), FWHM
    src_rect_pm: float = 0.0      # rectangular spectrum (filter), full width
    pixel_um: float = 0.0         # pixel size
    f_cam_mm: float = 0.0         # camera lens focal length (pixel as an angle)
    sigma_T_mK: float = 0.0       # temperature fluctuation between exposures
    cte_ppm: float = 0.55         # thermal expansion of the thickness (glass or spacer) [1e-6/K]
    dndT_ppm: float = 0.0         # dn'/dT [1e-6/K]

    def copy(self, **kw):
        d = asdict(self); d.update(kw); return Params(**d)


FIELDS = [f.name for f in fields(Params)]


def _gap(p: Params):
    """Parameters of the single gap-thickness field equivalent to two independent surfaces: all rms values times sqrt(2)."""
    if not p.two_surfaces: return p
    import dataclasses
    r = math.sqrt(2.0)
    return dataclasses.replace(p, two_surfaces=False, s_fig_nm=p.s_fig_nm * r, s_mid_nm=p.s_mid_nm * r, s_fine_nm=p.s_fine_nm * r)


def band_params(p: Params):
    """Return (sigma_fig, sigma_mid) [m] in the stationary convention. For p.bands == "measured" the Zernike-fit figure and
    the residual mid band are converted with Noll's shares (flaeche_dreiband.measured_to_stationary):
    sigma_mid^2 = s_mid^2/(1 - p - p_t), sigma_fig^2 = s_fig^2 - p sigma_mid^2."""
    sf, sm = p.s_fig_nm * 1e-9, p.s_mid_nm * 1e-9
    if p.bands == "measured" and sm > 0:
        import run_full_grid_v7 as G
        from flaeche_dreiband import measured_to_stationary
        D_m = p.D_m_mm * 1e-3 if p.D_m_mm > 0 else 2 * G.surface_radius(p.L_mm)
        sf, sm = measured_to_stationary(sf, sm, p.alpha_s, D_m)
    return float(sf), float(sm)


# ---------------------------------------------------------------- conversion and characteristic quantities
def equivalent(p: Params):
    """Air-spaced equivalent and characteristic quantities (SI)."""
    n = float(p.n_gap); ng = float(p.n_group) if p.n_group > 0 else n
    wl = p.wl_nm * 1e-9; h = p.h_mm * 1e-3; L = p.L_mm * 1e-3
    th_out = math.radians(p.theta_deg); s = math.sin(th_out) / n
    if s >= 1: raise ValueError("Internal angle undefined (sin theta / n' >= 1).")
    th_in = math.asin(s)
    R2eff = (p.R1_pct / 100.0) * (p.R2_pct / 100.0)
    if not (0 < R2eff < 1): raise ValueError("R1*R2 must lie between 0 and 1 (R2 < 100 %).")
    dx = 2 * h * math.tan(th_in)
    if dx <= 0: raise ValueError("Angle of incidence must be > 0.")
    r = math.sqrt(R2eff); gam = -math.log(r)
    M = (L - 75e-6 / 2) / dx                                   # round trips on the plate (as in the equation, core image 75 um)
    FSR_m = wl ** 2 / (2 * ng * h * math.cos(th_in)); FSR_Hz = C0 / (2 * ng * h * math.cos(th_in))
    FR = math.pi * math.sqrt(r) / (1 - r)
    N01 = math.log(1e-3) / math.log(R2eff)                     # reflections down to 0.1 % residual intensity
    Q = math.sqrt(n * n - math.sin(th_out) ** 2)
    ddel_dT = 2 * (2 * math.pi / wl) * h * (p.cte_ppm * 1e-6 * Q + n * p.dndT_ppm * 1e-6 / Q)
    fGT = 2.3548 * abs(ddel_dT) * p.sigma_T_mK * 1e-3          # FWHM in rad of the resonance variable
    pix_rad = (p.pixel_um * 1e-6 / (p.f_cam_mm * 1e-3)) if (p.pixel_um > 0 and p.f_cam_mm > 0) else 0.0
    return dict(n=n, ng=ng, wl=wl, wl_eq=wl / n, h=h, L=L, th_in_deg=math.degrees(th_in), R2eff=R2eff, dx=dx, M=M, v=gam * M,
                FSR_pm=FSR_m * 1e12, FSR_GHz=FSR_Hz * 1e-9, F_Airy=FR, N_01=N01, L_min_mm=N01 * dx * 1e3,
                src_scale=ng / n ** 2, ddel_dT=ddel_dT, fG_T_rad=fGT, pixel_rad=pix_rad,
                dlam_T_pm=abs(ddel_dT) / (2 * math.pi) * FSR_m * 1e12)


# ---------------------------------------------------------------- orders
def orders(p: Params, max_orders=41, margin=0.95):
    """All orders inside the divergence cone of the cylinder lens. Each order has its own field angle thx (inside the medium),
    its own chord (round beam: beam * sqrt(1 - (thx/theta_div)^2)) and its own round-trip offset. Sorted by |thx|; the
    first entry is the central order. x_det: position on the detector (camera focal length f_cam), 0 if f_cam = 0."""
    a = equivalent(p); th = math.radians(a["th_in_deg"]); h = a["h"]; wl = a["wl_eq"]; n = a["n"]
    tdiv = math.atan(p.beam_mm * 1e-3 / (2 * p.f_cyl_mm * 1e-3)) / n
    lo, hi = max(-margin * tdiv, -0.8 * th), margin * tdiv
    Nlo = math.ceil(2 * h * math.cos(th + hi) / wl); Nhi = math.floor(2 * h * math.cos(th + lo) / wl)
    out = []
    for N in range(Nlo, Nhi + 1):
        c = N * wl / (2 * h)
        if c >= 1: continue
        thx = math.acos(c) - th
        if not (lo <= thx <= hi): continue
        Th_out = math.asin(min(n * math.sin(th + thx), 1.0)); dth_out = Th_out - math.radians(p.theta_deg)
        chord = p.beam_mm * math.sqrt(max(0.0, 1 - (thx / tdiv) ** 2))
        out.append(dict(N=N, thx=thx, theta_out_mrad=dth_out * 1e3, x_det_mm=(p.f_cam_mm * math.tan(dth_out) if p.f_cam_mm > 0 else 0.0),
                        chord_mm=chord, Theta_out=Th_out, dx=2 * h * math.tan(th + thx)))
    out.sort(key=lambda o: abs(o["thx"]))
    out = out[:max_orders]
    if not out:                                      # cone narrower than the order spacing: order on the axis
        out = [dict(N=0, thx=0.0, theta_out_mrad=0.0, x_det_mm=0.0, chord_mm=p.beam_mm, Theta_out=math.radians(p.theta_deg), dx=a["dx"])]
    return out


def defocus_mm(p: Params, c, order=None):
    """Detector shift dz = z_F - f [mm] behind the focal plane that imprints the pupil phase c m^2:
    beta_def = k0 v^2 (1 - z_F/f) / (2 f) with the transverse spacing v = dx' cos(Theta_out) of the virtual sources
    (air: v = 2 h sin(Theta)). None if f_cam = 0."""
    if p.f_cam_mm <= 0: return None
    o = order or orders(p)[0]; f = p.f_cam_mm * 1e-3; k0 = 2 * math.pi / (p.wl_nm * 1e-9)
    v = o["dx"] * math.cos(o["Theta_out"])
    return -2 * f * f * c / (k0 * v * v) * 1e3


# ---------------------------------------------------------------- closed-form equation
def equation(p: Params, with_details=True, order=None):
    """Net finesse (Omega50) of the closed-form equation in the focal plane and in best focus (wedge and mean random chirp
    compensated), plus the clean line, the validity check and the detector shift for best focus. order: entry of orders(p)
    (default: central order)."""
    import finesse_vipa as FV
    p = _gap(p)
    a = equivalent(p); sq = a["src_scale"]
    o = order or orders(p)[0]
    arg = (a["R2eff"], a["L"], a["th_in_deg"] + math.degrees(o["thx"]), p.wedge_as, a["wl_eq"])
    sf_st, sm_st = band_params(p)
    kw = dict(h=a["h"], blende=p.beam_mm * 1e-3, sehne=o["chord_mm"] * 1e-3, sigma_fig=sf_st, sigma_mid=sm_st,
              alpha_s=p.alpha_s, sigma_fein=p.s_fine_nm * 1e-9, dlam_Q=p.src_gauss_pm * 1e-12 * sq, dlam_L=p.src_lorentz_pm * 1e-12 * sq,
              dlam_R=p.src_rect_pm * 1e-12 * sq, f_G_T=a["fG_T_rad"], pixel_rad=a["pixel_rad"] / a["n"])
    if p.D_m_mm > 0: kw["D_m"] = p.D_m_mm * 1e-3
    t0 = time.time()
    F, det = FV.finesse(*arg, details=True, **kw)
    F_bf = FV.finesse(*arg, best_focus=True, **kw)
    F0 = FV.finesse(*arg, h=a["h"], blende=p.beam_mm * 1e-3, sehne=o["chord_mm"] * 1e-3)       # clean line without source (edge and wedge)
    try:
        va = FV.gueltigkeit(*arg, h=a["h"], sigma_mid=sm_st, sigma_fig=sf_st, alpha_s=p.alpha_s,
                            D_m=kw.get("D_m"))
        va["phi_mf"] = 2 * (2 * math.pi / a["wl_eq"]) * math.hypot(p.s_mid_nm, p.s_fine_nm) * 1e-9
        va["phi_ok"] = va["phi_mf"] <= 1.5
    except Exception as ex:
        va = dict(error=str(ex))
    beta = float(det.get("beta", 0.0)); s_q = float(det.get("s_q", 0.0))
    dz = defocus_mm(p, -beta, o); dz_sd = (abs(defocus_mm(p, s_q, o)) if dz is not None else None)
    out = dict(F_net=float(F), F_best=float(F_bf), F_clean=float(F0), loss=float(F / F0), static=static_model(p),
               dlam_pm=a["FSR_pm"] / float(F), resolving_power=p.wl_nm * 1e3 / (a["FSR_pm"] / float(F)),
               dz_best_mm=dz, dz_spread_mm=dz_sd, order=o, time_s=time.time() - t0, validity=va, equivalent=a)
    if with_details: out["details"] = {k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in det.items()}
    return out


def equation_orders(p: Params, max_orders=41):
    """Equation for every order (focal plane and best focus)."""
    rows = []
    for o in orders(p, max_orders):
        g = equation(p, with_details=False, order=o)
        rows.append(dict(o, F_net=g["F_net"], F_best=g["F_best"], F_clean=g["F_clean"], dz_best_mm=g["dz_best_mm"], dz_spread_mm=g["dz_spread_mm"]))
    return rows


def warnings_text(g):
    """Plain-text notes on the validity range of the equation."""
    va = g.get("validity", {}); H = []
    if "error" in va: return ["Validity check not possible: " + va["error"]]
    if not va.get("phi_ok", True): H.append(f"Phase per reflection (mid+fine) {va['phi_mf']:.2f} rad > 1.5: outside the validity range.")
    if not va.get("fig_ok", True): H.append(f"Figure phase {va['phi_fig']:.2f} rad > 1.5: outside the validity range.")
    if va.get("plateau"): H.append("Fresnel plateau (wedge, v < 2, N_F > 4): equation off by about 9 % on average.")
    if va.get("grosschirp"): H.append(f"Large chirp t = {va['t']:.0f} > 30: the line fills the period, equation too low.")
    if va.get("track_ok") is False: H.append(f"Hit-point shift from surface slope: phase error {va['eps_track']:.2f} rad > 0.8 (note only).")
    F = g["F_net"]
    if F < 5: H.append("Predicted finesse < 5: accuracy of the equation about 15 % (P90 60 %).")
    elif F < 8: H.append("Predicted finesse 5-8: accuracy about 4 % (P90 14 %).")
    v = g["equivalent"]["v"]; d = g.get("details", {})
    if v < 1 and (d.get("fG", 0) > 0 or d.get("sig_res", 0) > 0):
        H.append("Edge-limited line (v < 1) with broadening: Voigt rule in the overlap measure up to about 4 % pessimistic.")
    H.append("Best focus (equation): wedge and mean random chirp compensated; a freely optimised focus is typically 3-7 % higher (10 % at R2 >= 0.95).")
    if g["equivalent"]["n"] > 1.0001: H.append("Solid VIPA: computed as the air-spaced equivalent (lambda/n', internal angle).")
    return H


# ---------------------------------------------------------------- theory
_GEO = {}


def _geometry(p: Params, thx=None):
    """Theory geometry (clean pupil, hit points) of the air-spaced equivalent for the order with field angle thx
    (default: central order); cached per process."""
    if thx is None: thx = orders(p)[0]["thx"]
    key = tuple(getattr(p, k) for k in ("wl_nm", "R1_pct", "R2_pct", "h_mm", "n_gap", "L_mm", "theta_deg", "wedge_as", "beam_mm", "f_cyl_mm", "D_m_mm")) + (round(thx, 12),)
    if key in _GEO: return _GEO[key]
    import pupil_theory_v7 as PT, run_full_grid_v7 as G
    from theorie_grid_dreiband import order_tracks, set_h
    a = equivalent(p); set_h(a["h"]); G.D_CORE_IMAGE = 75e-6
    PT.f_cyl = p.f_cyl_mm * 1e-3
    PT.theta_div_override = math.atan(p.beam_mm * 1e-3 / (2 * p.f_cyl_mm * 1e-3)) / a["n"]
    wl_eq_nm = a["wl_eq"] * 1e9; R2 = a["R2eff"]
    U, X, y, g, dx, chord = order_tracks(R2, p.L_mm, a["th_in_deg"], p.wedge_as, wl_eq_nm, p.beam_mm, thx, 61)
    D_m = p.D_m_mm * 1e-3 if p.D_m_mm > 0 else 2 * G.surface_radius(p.L_mm)
    # weights for the mean random chirp of a plate (sweep of the strips)
    W = np.abs(U) ** 2; wm = W.sum(1); keep = wm > 0; w = wm[keep] / wm[keep].sum(); wy = W[keep].sum(0) / W[keep].sum()
    m = np.arange(keep.sum(), dtype=float); t = m - np.sum(w * m); r = w * t / np.sum(w * t ** 2)
    beta = 2 * (2 * math.pi / a["wl_eq"]) * math.radians(p.wedge_as / 3600) * a["h"] * math.sin(math.radians(a["th_in_deg"]))
    geo = dict(a=a, U=U, X=X, y=y, g=g, dx=dx, chord=chord, D_m=D_m, thx=thx, wl_eq_nm=wl_eq_nm, keep=keep, wy=wy, r=r, beta=beta,
               mm=np.arange(U.shape[0])[:, None])
    if len(_GEO) > 64: _GEO.clear()
    _GEO[key] = geo
    return geo


def _conv_widths(p, a):
    """Convolution widths in rad of the resonance variable (real quantities against the real FSR)."""
    k = 2 * math.pi / a["FSR_pm"]
    g = math.hypot(k * p.src_gauss_pm, a["fG_T_rad"])
    rect_s = k * p.src_rect_pm
    rect_p = 2 * math.pi * a["pixel_rad"] * a["dx"] / a["wl"]          # pixel: 2 pi theta_p dx'/lambda
    lor = k * p.src_lorentz_pm
    return g, rect_s, rect_p, lor


def _convolve(I, g, rs, rp, lor):
    if g <= 0 and rs <= 0 and rp <= 0 and lor <= 0: return I
    NF = len(I); kk = np.fft.fftfreq(NF) * NF; A = np.fft.fft(I)
    if g > 0: A = A * np.exp(-0.5 * (g / 2.3548 * kk) ** 2)
    if rs > 0: A = A * np.sinc(kk * rs / (2 * np.pi))
    if rp > 0: A = A * np.sinc(kk * rp / (2 * np.pi))
    if lor > 0: A = A * np.exp(-0.5 * lor * np.abs(kk))
    return np.maximum(np.real(np.fft.ifft(A)), 0)


def measures(I):
    """Finesse of a line on the FFT grid: overlap Omega50, half maximum (FWHM), equivalent width."""
    import messung as MS
    Is = np.asarray(I)[MS.ORD]
    F_om = MS.overlap_F(Is)[0]
    F_hm = MS.hm_zentriert(np.asarray(I))
    F_eq, _ = MS.energy_measures(MS.DS, Is)
    return float(F_om), float(F_hm), float(F_eq)


def _line(p, geo, P, c=0.0):
    from detection_modes import intensity
    import messung as MS
    Pc = P if c == 0 else P * np.exp(1j * c * geo["mm"] ** 2)
    return _convolve(intensity(Pc, geo["g"], MS.NF, "integrated"), *_conv_widths(p, geo["a"]))


def _best_focus(p, geo, P, c0, scale, scan=False):
    """Defocus c (pupil phase c m^2, common to all strips) with the highest Omega50 finesse: grid around c0, then Brent.
    Returns (c_best, I_best, [(c, F) scan])."""
    import messung as MS
    from scipy.optimize import minimize_scalar
    cache = {}
    def F(c):
        if c not in cache: cache[c] = MS.overlap_F(_line(p, geo, P, c)[MS.ORD])[0]
        return cache[c]
    cs = c0 + np.linspace(-3, 3, 13) * scale; Fs = [F(c) for c in cs]; k = int(np.argmax(Fs))
    rr = minimize_scalar(lambda c: -F(c), bounds=(cs[max(k - 1, 0)], cs[min(k + 1, 12)]), method="bounded", options=dict(xatol=scale * 1e-3))
    cb = rr.x if -rr.fun >= Fs[k] else cs[k]
    curve = sorted(cache.items()) if scan else None
    return cb, _line(p, geo, P, cb), curve


def _scale(geo, F_clean_om):
    D0 = 2 * math.pi / max(F_clean_om, 1.0)
    return max(D0 * D0 / 8, 2e-5)


def _clean(p, thx, best_focus):
    from detection_modes import intensity
    import messung as MS
    geo = _geometry(p, thx); a = geo["a"]
    I0_raw = intensity(geo["U"], geo["g"], MS.NF, "integrated"); I0 = _convolve(I0_raw, *_conv_widths(p, a))
    out = dict(I0=I0, F=measures(I0), F_raw=measures(I0_raw))
    if best_focus:
        cb, Ib, _ = _best_focus(p, geo, geo["U"], -geo["beta"], max(abs(geo["beta"]) * 0.5, _scale(geo, out["F"][0])))
        out.update(c_best=cb, F_best=measures(Ib))
    return out


def line_for_surface(p: Params, j: int, seed: int = 1, thx=None, best_focus=False, F_clean_om=None, scan=False):
    """Exact line for surface no. j (random seed `seed`) in the order thx. Returns dict with I and measures in the focal plane,
    and with best_focus also c_best, I_best and the measures in best focus (scan: the evaluated (c, F) pairs)."""
    from flaeche_dreiband import Flaeche, FlaecheStreifen
    geo = _geometry(p, thx); a = geo["a"]; X = geo["X"]; y = geo["y"]
    k_eq = 2 * math.pi / a["wl_eq"]
    Y = np.repeat(y[None, :], X.shape[0], 0); D_m = geo["D_m"]
    Fl = Flaeche if p.L_mm <= 40 else FlaecheStreifen
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 0, int(j)]))
    # measured bands: the projected generator (figure and mid band exactly as a Zernike fit returns them); stationary: the field as such
    has_model = (p.s_fig_nm > 0 or p.s_mid_nm > 0 or p.s_fine_nm > 0)
    F = Fl(rng, D_m, p.s_fig_nm * 1e-9, p.s_mid_nm * 1e-9, p.alpha_s, p.s_fine_nm * 1e-9, 2 * D_m + 2e-3, 2 * D_m, project_figure=(p.bands == "measured"))
    hh = F.h(X, Y) if has_model else np.zeros(X.shape)
    if p.two_surfaces and has_model:      # PR: second independent field with the same band parameters, hit points shifted by dx/2
        rng2 = np.random.default_rng(np.random.SeedSequence([int(seed), 1, int(j)]))
        F2 = Fl(rng2, D_m, p.s_fig_nm * 1e-9, p.s_mid_nm * 1e-9, p.alpha_s, p.s_fine_nm * 1e-9, 2 * D_m + 2e-3, 2 * D_m, project_figure=(p.bands == "measured"))
        hh = hh + F2.h(X + geo["dx"] / 2, Y)
    if p.map_hr or p.map_pr:
        import surface_map as SM
        if p.map_hr: hh = hh + SM.load_plate_map(p.map_hr).h(X, Y)
        if p.map_pr: hh = hh + SM.load_plate_map(p.map_pr).h(X + geo["dx"] / 2, Y)
    nu = 2 * k_eq * hh; P = geo["U"] * np.exp(1j * np.cumsum(nu, axis=0))
    I = _line(p, geo, P)
    res = dict(I=I, F=measures(I))
    if best_focus:
        qr = geo["r"] @ nu[geo["keep"]]; qm = float(np.sum(geo["wy"] * qr))
        c0 = -(geo["beta"] + qm / 2)
        sc = max(math.sqrt(float(np.sum(geo["wy"] * (qr - qm) ** 2))) / 2, _scale(geo, F_clean_om or res["F"][0]))
        cb, Ib, curve = _best_focus(p, geo, P, c0, sc, scan)
        res.update(c0=c0, c_best=cb, I_best=Ib, F_best=measures(Ib), scan=curve)
    return res


def _worker(args):
    pd, j, seed, k_ord, thx, bf, Fc, keep_line = args
    p = Params(**pd); r = line_for_surface(p, j, seed, thx, bf, Fc, scan=keep_line and j == 0)
    out = dict(k=k_ord, j=j, F=r["F"], I=r["I"].astype(np.float32))
    if bf: out.update(c_best=r["c_best"], c0=r["c0"], F_best=r["F_best"], I_best=(r["I_best"].astype(np.float32) if keep_line else None), scan=r["scan"])
    return out


def _stat(x):
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)], float)
    if len(x) == 0: return dict(median=np.nan, p10=np.nan, p90=np.nan, mean=np.nan)
    return dict(median=float(np.median(x)), p10=float(np.percentile(x, 10)), p90=float(np.percentile(x, 90)), mean=float(np.mean(x)))


def theory(p: Params, n: int = 1, seed: int = 1, processes: int = 1, n_lines: int = 5, best_focus=False, all_orders=False,
           max_orders=41, progress=None, cancel=None):
    """Theory for n surfaces (the same plates in every order). Returns per order: clean line, statistics of Omega50/FWHM/F_eq in the
    focal plane and (best_focus) in the best focus with the detector shift dz; for the central order also example lines, the
    ensemble-average line and, for a single realisation, the focus scan. progress(k, n): callback; cancel(): True stops."""
    t0 = time.time()
    a = equivalent(p)
    ords = orders(p, max_orders) if all_orders else orders(p)[:1]
    has_map = bool(p.map_hr or p.map_pr)
    has_surface = (p.s_fig_nm > 0 or p.s_mid_nm > 0 or p.s_fine_nm > 0 or has_map)
    if not has_surface: n = 0
    elif has_map and not (p.s_fig_nm > 0 or p.s_mid_nm > 0 or p.s_fine_nm > 0): n = 1      # a measured plate is one plate
    res = []
    for k, o in enumerate(ords):
        cl = _clean(p, o["thx"], best_focus)
        res.append(dict(order=o, clean=cl, F=np.full((n, 3), np.nan), F_best=np.full((n, 3), np.nan), c_best=np.full(n, np.nan),
                        lines=[], lines_best=[], I_sum=None, scan=None, line0=None, line0_best=None))
    jobs = [(asdict(p), j, seed, k, o["thx"], best_focus, res[k]["clean"]["F"][0], (k == 0 and j < n_lines) or j == 0)
            for k, o in enumerate(ords) for j in range(n)]
    total = len(jobs); done = 0

    def take(r):
        R = res[r["k"]]; j = r["j"]; R["F"][j] = r["F"]
        if j == 0: R["line0"] = r["I"]; R["line0_best"] = r.get("I_best")
        if r["k"] == 0:
            R["I_sum"] = r["I"].astype(np.float64) if R["I_sum"] is None else R["I_sum"] + r["I"]
            if j < n_lines: R["lines"].append((j, r["I"]))
        if best_focus:
            R["F_best"][j] = r["F_best"]; R["c_best"][j] = r["c_best"]
            if r["k"] == 0 and r.get("I_best") is not None: R["lines_best"].append((j, r["I_best"]))
            if r.get("scan"): R["scan"] = r["scan"]

    if processes > 1 and total > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=processes) as ex:
            futs = [ex.submit(_worker, jb) for jb in jobs]
            for fu in as_completed(futs):
                take(fu.result()); done += 1
                if progress: progress(done, total)
                if cancel and cancel():
                    for f2 in futs: f2.cancel()
                    break
    else:
        for jb in jobs:
            take(_worker(jb)); done += 1
            if progress: progress(done, total)
            if cancel and cancel(): break
    import messung as MS
    out_orders = []
    for R in res:
        o = R["order"]; ok = ~np.isnan(R["F"][:, 0]) if n else np.zeros(0, bool)
        Fo = R["F"][ok]; Fb = R["F_best"][ok]
        dz = [defocus_mm(p, c, o) for c in R["c_best"][ok]] if (best_focus and p.f_cam_mm > 0) else []
        cl = R["clean"]
        entry = dict(order=o, n=int(ok.sum()), I0=cl["I0"], line0=R["line0"], line0_best=R["line0_best"],
                     F_clean=dict(zip(("om50", "fwhm", "eq"), cl["F"])),
                     F_clean_no_source=dict(zip(("om50", "fwhm", "eq"), cl["F_raw"])),
                     om50=_stat(Fo[:, 0]), fwhm=_stat(Fo[:, 1]), eq=_stat(Fo[:, 2]), per_surface=Fo)
        if best_focus:
            entry.update(F_clean_best=dict(zip(("om50", "fwhm", "eq"), cl["F_best"])),
                         dz_clean_mm=defocus_mm(p, cl["c_best"], o),
                         om50_best=_stat(Fb[:, 0]), fwhm_best=_stat(Fb[:, 1]), eq_best=_stat(Fb[:, 2]), per_surface_best=Fb,
                         dz_mm=_stat(dz) if dz else None, dz_per_surface=np.array(dz, float))
        out_orders.append(entry)
    R0 = res[0]; e0 = out_orders[0]; nn = e0["n"]
    I_mean = R0["I_sum"] / max(nn, 1) if R0["I_sum"] is not None else None
    scan = None
    if R0["scan"] and p.f_cam_mm > 0:
        scan = [(defocus_mm(p, c, ords[0]), F) for c, F in R0["scan"]]
    elif R0["scan"]:
        scan = [(c, F) for c, F in R0["scan"]]
    dpm = MS.D0 / (2 * np.pi) * a["FSR_pm"]                     # detuning in pm on the (unsorted) FFT grid
    return dict(equivalent=a, n=nn, seed=seed, best_focus=best_focus, all_orders=all_orders, orders=out_orders,
                # central order (compatible with the single-order display)
                F_clean=e0["F_clean"], F_clean_no_source=e0["F_clean_no_source"], om50=e0["om50"], fwhm=e0["fwhm"], eq=e0["eq"],
                per_surface=e0["per_surface"], I0=R0["clean"]["I0"], lines=[I for _, I in sorted(R0["lines"], key=lambda t: t[0])],
                lines_best=[I for _, I in sorted(R0["lines_best"], key=lambda t: t[0])], I_mean=I_mean,
                F_mean_line=(measures(I_mean) if I_mean is not None else None), focus_scan=scan, focus_scan_in_mm=p.f_cam_mm > 0,
                dlam_pm=dpm, order=MS.ORD, time_s=time.time() - t0, has_surface=has_surface)


# ---------------------------------------------------------------- detector spectrum (all orders with envelope)
def envelope_width(p: Params):
    """1/e^2 half angle theta_0 of Hu's Gaussian envelope (outside, rad): the angular spectrum of the Gaussian beam focused by the
    cylinder lens, theta_0 = lambda / (pi w_0) = w / f_cyl with the waist w_0 = lambda f_cyl / (pi w) and the 1/e^2 radius
    w = beam/2 of the collimated beam. On the detector W_c = f_cam theta_0 (Hu et al. 2015, Eq. (10))."""
    return (p.beam_mm / 2) / p.f_cyl_mm


def spectrum(p: Params, theory_result=None, dlam2_pm=0.0, npts=200000, span=2.5, use_best=False):
    """Intensity on the detector over the field of the cylinder-lens cone, after Hu et al. (2015), Eqs. (9)-(10):
        I(theta_x) = exp(-2 theta_x^2 / theta_0^2) * (1 - 2 x cos(M delta) + x^2) / (1 - 2 r cos(delta) + r^2) * (1-r)^2/(1-x)^2
    with delta(theta_x) = 2 k0 h sqrt(n'^2 - sin^2(theta + theta_x)), r = sqrt(R1 R2), x = r^M, M = L / (2 h tan theta'(theta_x)).
    Normalised so that a clean line on the axis peaks at 1. With a theory result, each order is replaced by the exact line of
    the theory (surface 1 of the ensemble, or of the single realisation; the clean line if no surface error was given), taken
    from the computed order nearest to theta_x (all orders) or from the central order, and multiplied by the same envelope;
    the theory line is scaled so that its clean line peaks at 1. dlam2_pm > 0 adds a second wavelength lambda + dlam2.
    Returns the coordinate (x on the detector in mm if f_cam > 0, else field angle in mrad) and the curves."""
    a = equivalent(p); n = a["n"]; h = a["h"]; th0 = math.radians(p.theta_deg); w0 = envelope_width(p)
    lo = max(-span * w0, -0.9 * th0); hi = span * w0
    tx = np.linspace(lo, hi, int(npts))
    env = np.exp(-2 * tx ** 2 / w0 ** 2)
    Th = th0 + tx; sin_in = np.sin(Th) / n; th_in = np.arcsin(np.clip(sin_in, -1, 1))
    Q = np.sqrt(np.maximum(n * n - np.sin(Th) ** 2, 0))
    M = a["L"] / (2 * h * np.tan(np.maximum(th_in, 1e-9)))
    r = math.sqrt(a["R2eff"]); x = r ** M
    coord = p.f_cam_mm * np.tan(tx) if p.f_cam_mm > 0 else tx * 1e3

    def hu(wl):
        d = 2 * (2 * math.pi / wl) * h * Q
        I = (1 - 2 * x * np.cos(M * d) + x * x) / (1 - 2 * r * np.cos(d) + r * r)
        return env * I * (1 - r) ** 2 / (1 - x) ** 2, d

    out = dict(coord=coord, coord_unit=("mm" if p.f_cam_mm > 0 else "mrad"), envelope=env, theta0_mrad=w0 * 1e3,
               W_c_mm=(p.f_cam_mm * w0 if p.f_cam_mm > 0 else None))
    wls = [a["wl"]] + ([a["wl"] + dlam2_pm * 1e-12] if dlam2_pm > 0 else [])
    out["hu"] = []; deltas = []
    for wl in wls:
        I, d = hu(wl); out["hu"].append(I); deltas.append(d)
    if theory_result is not None:
        import messung as MS
        E = theory_result["orders"]
        tx_ord = np.array([e["order"]["theta_out_mrad"] for e in E]) * 1e-3
        grid = MS.D0                                       # delta grid of the theory lines (unsorted FFT order)
        ordg = MS.ORD; gs = grid[ordg]
        prof = []
        for e in E:
            if use_best and theory_result.get("best_focus"):
                L_ = e["line0_best"] if e["line0_best"] is not None else e["I0"]
            else:
                L_ = e["line0"] if e["line0"] is not None else e["I0"]
            prof.append((np.asarray(L_, float)[ordg], float(np.max(e["I0"]))))
        # nearest computed order for every field angle
        idx = np.abs(tx[:, None] - tx_ord[None, :]).argmin(1) if len(E) > 1 else np.zeros(len(tx), int)
        out["theory"] = []
        for d in deltas:
            dw = np.angle(np.exp(1j * d))              # wrap to [-pi, pi)
            I = np.zeros_like(tx)
            for k, (y, nrm) in enumerate(prof):
                sel = idx == k
                if sel.any(): I[sel] = np.interp(dw[sel], gs, y / nrm, period=2 * np.pi)
            out["theory"].append(env * I)
        out["theory_label"] = ("best focus per order, " if (use_best and theory_result.get("best_focus")) else "") + \
            ("surface 1" if theory_result["has_surface"] else "clean line") + (", all orders" if len(E) > 1 else ", central-order line for all orders")
    out["wavelengths_nm"] = [w * 1e9 for w in wls]
    return out


# ---------------------------------------------------------------- static comparison model
def static_model(p: Params, NG=2 ** 16):
    """Comparison model like the LightMachinery calculator (reproduced from its outputs on 2026-10-18): Hu sum with
    N = length / round-trip offset reflections, numerically convolved with a static Gaussian thickness distribution (RMS of all
    surface bands together, phase 2 k0 n' sigma), measure FWHM. Knows neither walk-off nor spatial frequencies."""
    p = _gap(p)
    a = equivalent(p); r = math.sqrt(a["R2eff"]); N = max(int(a["L"] / a["dx"]) + 1, 1)
    d = np.linspace(-np.pi, np.pi, NG, endpoint=False); x = r ** N
    I = (1 - 2 * x * np.cos(N * d) + x * x) / (1 - 2 * r * np.cos(d) + r * r)
    s = 2 * (2 * math.pi / a["wl_eq"]) * math.sqrt(p.s_fig_nm ** 2 + p.s_mid_nm ** 2 + p.s_fine_nm ** 2) * 1e-9

    def fwhm_F(y):
        y = y / y.max(); i0 = int(np.argmax(y)); j = i0
        while y[j % NG] > 0.5 and j - i0 < NG // 2: j += 1
        a0, b0 = y[(j - 1) % NG], y[j % NG]; w = 2 * ((j - 1 - i0) + (a0 - 0.5) / (a0 - b0 + 1e-300)) * (2 * np.pi / NG)
        return 2 * np.pi / w
    F_raw = fwhm_F(I)
    if s > 0:
        k = np.fft.fftfreq(NG) * NG
        I = np.fft.fftshift(np.real(np.fft.ifft(np.fft.fft(np.fft.ifftshift(I)) * np.exp(-0.5 * (s * k) ** 2))))
    return dict(N=N, F_clean=float(F_raw), F_net=float(fwhm_F(I)), sigma_phase=s)


# ---------------------------------------------------------------- length series
def length_series(p: Params, Ls_mm, theory_n=0, seed=1, processes=1, best_focus=False, progress=None, cancel=None):
    rows = []; N = len(Ls_mm)
    for i, L in enumerate(Ls_mm):
        pi = p.copy(L_mm=float(L))
        g = equation(pi, with_details=False); sm = g["static"]
        row = dict(L_mm=float(L), M=g["equivalent"]["M"], v=g["equivalent"]["v"], F_eq_net=g["F_net"], F_eq_best=g["F_best"],
                   F_eq_clean=g["F_clean"], F_static_fwhm=sm["F_net"], F_static_clean=sm["F_clean"])
        if theory_n > 0:
            t = theory(pi, n=theory_n, seed=seed, processes=processes, n_lines=0, best_focus=best_focus, cancel=cancel)
            e = t["orders"][0]
            row.update(F_th_om50=e["om50"]["median"] if t["has_surface"] else e["F_clean"]["om50"],
                       F_th_fwhm=e["fwhm"]["median"] if t["has_surface"] else e["F_clean"]["fwhm"],
                       F_th_clean=e["F_clean"]["om50"])
            if best_focus:
                row.update(F_th_best=e["om50_best"]["median"] if t["has_surface"] else e["F_clean_best"]["om50"])
        rows.append(row)
        if progress: progress(i + 1, N)
        if cancel and cancel(): break
    return rows


if __name__ == "__main__":
    p = Params(s_mid_nm=10.0)
    g = equation(p); print("equation", round(g["F_net"], 2), "clean", round(g["F_clean"], 2), f"({g['time_s']:.1f} s)")
    t = theory(p, n=10, seed=1); print("theory median", round(t["om50"]["median"], 2), "clean", round(t["F_clean"]["om50"], 2), f"({t['time_s']:.1f} s)")
