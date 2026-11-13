"""
Theory grid with the THREE-BAND surface model: both coherence models (coherent and incoherent across the chord) and
source widths in pm, with the three-band surface model (flaeche_dreiband.py):
  figure (Zernike degree 2-4, sigma_fig) + mid band (power law alpha_s, 1/D_m .. 1/0.1 mm, sigma_mid)
  + fine band (white, sigma_fein).

For each grid point (R2, L_eff, theta_i, alpha, lambda, aperture | sigma_mid, alpha_s, sigma_fein, sigma_fig, h)
NSURF surfaces are drawn. For each surface the profile of the central order is computed in both modes
(axis = coherent across the chord; inc = incoherent across the chord), convolved with each source width and measured
(measures listed in NAMES: hull, equivalent width, half maximum, overlap measures, side maxima, one-sided overlaps,
one-sided half widths). In addition, for each surface the characteristic quantities are computed from the actual phase
steps nu_{m rho} = 2 k0 h(x_{m rho}, y_rho) of the same realisation (single-realisation form of the matrix forms in
closed_integrated.py):
  var_q_s  : weighted slope of the steps over m per strip, squared, averaged over the strips   (random chirp)
  sig2_s   : weighted variance of the rest after removing mean and slope per strip, averaged    (residual resonance)
  sperp2   : weighted variance of the strip means about the chord mean                          (strip offset)
  sig_h_strip : RMS of the height at all hit points (check against sigma_mid, sigma_fein, sigma_fig)
Median and mean over the surfaces are stored (columns *_med, *_mean). No formula, no fitted constant.

Inputs: command-line options (see below); kernel modules pupil_theory_v7, run_full_grid_v7, messung,
detection_modes, raytrace_2d and flaeche_dreiband.
Output: theorie_grid_dreiband_results.csv next to this script (or --out); resumable: repeat the same call and only
missing points are computed. One row per grid point.

    python theorie_grid_dreiband.py --probe                                   # quick test (3 points, 3 surfaces)
    python theorie_grid_dreiband.py --n 800 --nsurf 50 --procs 60 --src_pm 0,2,4,6

Used in the main document: sec. 4.2 (Das Werkzeug: wie die Theorie gerechnet wird). Pages: 38.
"""
import os, sys, time, argparse
os.environ.setdefault("OMP_NUM_THREADS", "1")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
import multiprocessing as mp

# Grid axes and their levels (lengths in mm, wedge alpha in arcsec, wavelength in nm, RMS values in nm,
# h = mirror spacing); a grid point is addressed by its flat index pidx into SHAPE
AX = ["R2", "L_eff_mm", "theta_i_deg", "alpha", "wl_nm", "blende_mm", "sigma_mid_nm", "alpha_s", "sigma_fein_nm", "sigma_fig_nm", "h_mm"]
LEVELS = dict(R2=[0.7, 0.8, 0.85, 0.9, 0.95, 0.98], L_eff_mm=[10.0, 15.0, 25.0, 35.0, 50.0, 75.0, 100.0, 150.0], theta_i_deg=[2.0, 3.0, 4.0, 6.0], alpha=[0.0, 0.5, 1.5, 3.0],
              wl_nm=[400, 630, 800, 1000], blende_mm=[1.5, 2.5, 3.5, 5.5],
              sigma_mid_nm=[5.0, 10.0, 15.0, 25.0, 35.0, 50.0], alpha_s=[2.0, 2.5, 3.0, 3.5], sigma_fein_nm=[0.0, 1.0, 3.0], sigma_fig_nm=[0.0, 10.0, 25.0, 50.0, 75.0, 100.0, 150.0], h_mm=[0.5, 1.0, 2.0, 4.0, 8.0, 12.0])
SHAPE = tuple(len(LEVELS[a]) for a in AX)
# Measures per profile: the first entries come from messung.measures, then the number of side peaks (n10),
# one-sided overlaps for a neighbour of strength 0.3 / 0.1 (left/right) and one-sided half widths
NAMES = ("hull", "eq", "hm", "e50", "om50", "om25", "om10", "n10", "oL03", "oR03", "oL01", "oR01", "hwL", "hwR")
MODES = ("axis", "inc")
KENN = ("var_q_s", "sig2_s", "sperp2", "sig_h_strip")
H_PLATE = 2e-3


def set_h(h_m):
    """Set the mirror spacing h [m] in the kernel modules (PT.h, rt2d.h); all geometry functions read it at run time.
    Returns the new PT.h."""
    import pupil_theory_v7 as PT
    PT.h = float(h_m)
    try:
        # The geometric phase (geometric_phase_nominal) and the order positions use t0
        import raytrace_2d as rt2d; rt2d.h = float(h_m); rt2d.t0 = float(h_m)
    except Exception: pass
    return PT.h


def om_side(p, r, sign, thr=0.5):
    """Return the one-sided overlap finesse N/s for a neighbour of relative strength r on one side (sign -1 = left,
    +1 = right): s is the shift in bins (interpolated) at which the overlap min(p, r*shifted p)/r falls to thr.
    p: binned profile normalised to unit sum. Coarse search, then fine search with linear interpolation."""
    N = len(p); half = N // 2; cs = max(1, N // 512); cache = {}
    def om(k):
        """Return the normalised overlap for a shift of k bins (cached)."""
        k = int(k)
        if k not in cache: cache[k] = float(np.minimum(p, r * np.roll(p, sign * k)).sum()) / r
        return cache[k]
    coarse = np.arange(1, half + 1, cs); omc = np.array([om(k) for k in coarse]); bad = np.where(omc > thr)[0]
    if len(bad) == 0: return float(N)
    kc = int(coarse[bad[-1]])
    if kc + cs >= half and om(half) > thr: return N / half
    ks = np.arange(kc, min(kc + cs, half) + 1); oms = np.array([om(k) for k in ks]); b2 = np.where(oms > thr)[0]; kl = int(ks[b2[-1]])
    if kl >= half: return N / half
    o1, o2 = om(kl), om(kl + 1); s_ = kl + (o1 - thr) / (o1 - o2) if o1 > o2 else kl + 1.0
    return N / s_


def hw_sides(I):
    """Return the one-sided half-width finesses (left, right): 2 pi / (2 x half width at half maximum on that side),
    with the profile centred on its peak; at least one grid step."""
    ip = int(np.argmax(I)); n = len(I); J = np.roll(I, n // 2 - ip); J = J / J.max(); idx = np.where(J >= 0.5)[0]; dx = 2 * np.pi / n
    return 2 * np.pi / (2 * max((n // 2 - idx[0]) * dx, dx)), 2 * np.pi / (2 * max((idx[-1] - n // 2) * dx, dx))


def point_params(pidx):
    """Return the list of parameter values (in the order of AX) of grid point pidx."""
    ix = np.unravel_index(pidx, SHAPE); return [LEVELS[a][i] for a, i in zip(AX, ix)]


def phi_h_of(p):
    """Return the rms phase per reflection 4 pi sigma_h / lambda of a grid point (index tuple p), with sigma_h the
    total RMS of figure, mid band and fine band."""
    v = {a: LEVELS[a][i] for a, i in zip(AX, p)}
    return 4 * np.pi * np.sqrt(v["sigma_fig_nm"] ** 2 + v["sigma_mid_nm"] ** 2 + v["sigma_fein_nm"] ** 2) / v["wl_nm"]


# Upper limit of the number of round trips (kernel: 2500); points above it (thin, long plates at small angle) are skipped
M_MAX = 2400
# Total rms phase per reflection; the equation is valid for phi_mf <= 1.5 and phi_fig <= 1.5, the margin up to 2.5
# checks the correction tables
PHI_MAX = 2.5


def select_points(n, seed, phi_max=None):
    """Return a sorted list of n grid indices: a balanced random sample over all axes (each level appears about equally
    often), keeping only points with phi_h <= phi_max (default PHI_MAX) and at most M_MAX round trips; an oversized
    pool (8 n) is drawn and filtered."""
    phi_max = PHI_MAX if phi_max is None else phi_max
    rng = np.random.default_rng(seed); cols = []; n_pool = 8 * n
    for a, s in zip(AX, SHAPE):
        v = np.resize(np.arange(s), n_pool); rng.shuffle(v); cols.append(v)
    pts = []; seen = set()
    for t in zip(*cols):
        if phi_h_of(t) > phi_max: continue
        vv = {ax: LEVELS[ax][i] for ax, i in zip(AX, t)}
        # Above the round-trip limit of the kernel (M = L / (2 h tan theta_i))
        if "h_mm" in vv and vv["L_eff_mm"] / (2 * vv["h_mm"] * np.tan(np.deg2rad(vv["theta_i_deg"]))) > M_MAX: continue
        p = int(np.ravel_multi_index(t, SHAPE))
        if p not in seen: seen.add(p); pts.append(p)
        if len(pts) >= n: break
    return sorted(pts)


def srctag(v):
    """Return the column tag of source width v [pm], e.g. 2.5 -> "p2p5pm"."""
    return f"p{v:g}pm".replace(".", "p")


def columns(sources):
    """Return the ordered list of output CSV columns for the given source widths."""
    cols = ["pidx"] + AX + ["D_m_mm", "theta_x", "M", "ny", "nsurf", "n_ok", "D_ideal", "F_ideal", "FSR_pm", "D_ideal_pm", "runtime_s"]
    for md in MODES:
        for v in sources:
            tag = f"{md}_{srctag(v)}"
            for nm in NAMES: cols += [f"{tag}_{nm}", f"{tag}_{nm}_q25", f"{tag}_{nm}_q75", f"{tag}_{nm}_clean"]
            cols += [f"{tag}_frag"]
    cols += ["M_k", "gamma", "v", "beta", "dx_mm", "chord_mm"]
    for k in KENN: cols += [f"{k}_med", f"{k}_mean"]
    return cols


def order_tracks(R2, L_mm, th_deg, alpha_as, wl_nm, bl_mm, thx, n_y):
    """Return (U, X, y, groups, Dx, chord) for the central order: clean pupil field U (M, nr) with wedge and plate
    edge, hit positions X (M, nr) along the plate [m], chord coordinates y [m], core groups g, round-trip offset
    Dx [m] and chord length [m].

    Exactly as PT.order_states (n_core = G.N_CORE core points, grouped incoherently), but also returning positions.
    thx: exit angle of the order; n_y: points across the chord."""
    import pupil_theory_v7 as PT, run_full_grid_v7 as G
    th = np.deg2rad(th_deg); k0 = 2 * np.pi / (wl_nm * 1e-9); R = L_mm * 1e-3 / 2
    Dx = 2 * PT.h * np.tan(th); Ls = 2 * PT.h / np.cos(th); a = alpha_as / 3600 * np.pi / 180; r2 = np.sqrt(R2)
    # Chord: round aperture (default) or rectangle; divergence optionally independent (PT.theta_div_override)
    chord = PT.chord_factor(thx, bl_mm)
    pmax = G.pmax_for_order(L_mm, th_deg, R2, thx); n_core = G.N_CORE
    xi_c = np.linspace(-1, 1, n_core) if n_core > 1 else np.array([0.0]); xi_y = np.linspace(-1, 1, n_y)
    XC, XY = np.meshgrid(xi_c, xi_y)
    x = (-R + G.D_CORE_IMAGE / 2 + XC * G.D_CORE_IMAGE / 2).ravel(); y = (XY * chord * bl_mm * 1e-3 / 2).ravel()
    groups = np.searchsorted(xi_c, XC.ravel()); nr = len(x); w = 1.0 / nr
    tx = np.full(nr, float(thx)); alive = np.ones(nr, bool); acc = np.zeros(nr)
    x0 = x.copy(); Theta0 = th + tx                     # for the exact wedge phase (PT.wedge_phase)
    M = pmax + 1; U = np.zeros((M, nr), complex); X = np.zeros((M, nr)); last = M
    # Propagate the rays round trip by round trip: a ray dies when it leaves the plate (|x| > R); the wedge phase is
    # the exact mirror-image phase PT.wedge_phase (or, with PT.WEDGE_PHASE = "hitpoint", the old first-order sum
    # 2 k0 a x cos(th) over the hit points); the wedge tilts the ray by 2 a cos(th) per round trip; amplitude r2^m
    for m in range(M):
        alive &= (np.abs(x) <= R)
        if not alive.any(): last = m; break
        X[m] = x
        if PT.WEDGE_PHASE == "exact":
            acc = PT.wedge_phase(m, a, Theta0, x0, k0, PT.h)
        else:
            acc += 2 * k0 * a * x * np.cos(th)
        U[m] = w * r2 ** m * alive * np.exp(1j * acc)
        tx += 2 * a * np.cos(th); x = x + Dx + tx * Ls
    return U[:last], X[:last], y, groups, Dx, chord * bl_mm * 1e-3


def kennzahlen_realisierung(nu, U, g):
    """Return (var_q_s, sig2_s, sperp2): single-realisation form of the characteristic quantities from the phase steps
    nu (M, ny), with the window weights taken from U (w over round trips, wy over strips)."""
    W = np.abs(U) ** 2; wm = W.sum(1); keep = wm > 0; nu = nu[keep]; W = W[keep]; wm = wm[keep]
    w = wm / wm.sum(); wy = W.sum(0); wy = wy / wy.sum(); Mk = len(w)
    # Weighted linear regression over m: t = centred round-trip index, r = regression weights for the slope
    m = np.arange(Mk, dtype=float); t = m - np.sum(w * m); r = w * t / np.sum(w * t ** 2)
    # Slope per strip (ny,)
    q = r @ nu
    # Strip mean (ny,)
    mean_s = w @ nu
    res = nu - mean_s[None, :] - np.outer(t, q)
    var_q_s = float(np.sum(wy * q ** 2)); sig2_s = float(np.sum(wy * (w @ res ** 2)))
    sperp2 = float(np.sum(wy * (mean_s - np.sum(wy * mean_s)) ** 2))
    return var_q_s, sig2_s, sperp2


def worker(arg):
    """Compute one grid point and return a list with one result dict (one CSV row).

    arg = (pidx, number of surfaces NS, seed, source widths [pm], ny, D_m_mm). The clean profiles (no surface error)
    are stored as *_clean; for the surfaces the median and quartiles of each measure, the fraction of fragmented
    lines (*_frag: >= 2 peaks), the window parameters and the characteristic quantities are stored."""
    pidx, NS, seed, sources, ny, D_m_mm = arg
    import pupil_theory_v7 as PT, messung as MS, run_full_grid_v7 as G
    from detection_modes import intensity
    from scipy.signal import find_peaks
    from flaeche_dreiband import Flaeche
    NF = MS.NF; d = np.fft.fftfreq(NF) * 2 * np.pi
    R2, L, th, al, wl, bl, s_mid, al_s, s_fein, s_fig, h_mm = point_params(pidx); set_h(h_mm * 1e-3); t0 = time.time()
    k0 = 2 * np.pi / (wl * 1e-9)
    thx = PT.order_angles(R2, L, th, wl, bl); thx = float(thx[np.argmin(np.abs(thx))])
    U, X, y, g, Dx, chord = order_tracks(R2, L, th, al, wl, bl, thx, ny)
    # Ideal line (no wedge, no surface error): half-maximum finesse hm0 and line width D_ideal = 2 pi / hm0
    U0 = order_tracks(R2, L, th, 0.0, wl, bl, thx, ny)[0]
    I0 = intensity(U0, g, NF, "axis"); hm0 = MS.measures_hm(I0)[NAMES.index("hm")]; D_ideal = 2 * np.pi / hm0
    # FSR in pm: lambda^2 / (2 h cos Theta)
    Theta = np.deg2rad(th) + thx; FSR_pm = (wl * 1e-9) ** 2 / (2 * PT.h * np.cos(Theta)) * 1e12
    # Diameter of the RMS definition and lowest spatial frequency 1/D_m
    D_m = (D_m_mm * 1e-3) if D_m_mm > 0 else 2 * G.surface_radius(L)
    def conv(I, v):
        """Return profile I convolved with a Gaussian source of FWHM v [pm] (periodised over 5 orders)."""
        w = 2 * np.pi * v / FSR_pm
        if w <= 0: return I
        sg = w / 2.3548; K = sum(np.exp(-0.5 * ((d + 2 * np.pi * k) / sg) ** 2) for k in (-2, -1, 0, 1, 2)); K /= K.sum()
        return np.real(np.fft.ifft(np.fft.fft(I) * np.fft.fft(K)))
    def meas(I):
        """Return the list of all measures in NAMES for profile I (NaN on failure)."""
        try:
            m = list(MS.measures(I)); pk, _ = find_peaks(np.fft.fftshift(I / I.max()), prominence=0.10)
            p = np.asarray(I, float).reshape(MS.OV_N, -1).sum(1); p = p / p.sum(); hl, hr = hw_sides(I)
            return m + [len(pk), om_side(p, 0.3, -1), om_side(p, 0.3, +1), om_side(p, 0.1, -1), om_side(p, 0.1, +1), hl, hr]
        except Exception: return [np.nan] * len(NAMES)
    def profiles(V):
        """Return the profiles of field V for both detection modes (axis = coherent, inc = integrated)."""
        return {"axis": intensity(V, g, NF, "axis"), "inc": intensity(V, g, NF, "integrated")}
    rec = dict(pidx=int(pidx), **dict(zip(AX, (R2, L, th, al, wl, bl, s_mid, al_s, s_fein, s_fig))), D_m_mm=D_m * 1e3, theta_x=thx, M=int(U.shape[0]), ny=ny, nsurf=NS,
               D_ideal=D_ideal, F_ideal=hm0, FSR_pm=FSR_pm, D_ideal_pm=D_ideal / (2 * np.pi) * FSR_pm)
    clean = {(md, v): meas(conv(I, v)) for md, I in profiles(U).items() for v in sources}
    arr = {k: [] for k in clean}; kenn = []
    # Size of the generated surface patch
    Lx = 2 * D_m + 2e-3; Ly = 2 * D_m
    Y = np.repeat(y[None, :], X.shape[0], 0)
    for j in range(NS):
        rng = np.random.default_rng(np.random.SeedSequence([seed, int(pidx), j]))
        F = Flaeche(rng, D_m, s_fig * 1e-9, s_mid * 1e-9, al_s, s_fein * 1e-9, Lx, Ly)
        # Reflection phase step nu = 2 k0 h per hit, accumulated over round trips
        hh = F.h(X, Y); nu = 2 * k0 * hh; psi = np.cumsum(nu, axis=0)
        P = profiles(U * np.exp(1j * psi))
        for (md, v) in clean: arr[(md, v)].append(meas(conv(P[md], v)))
        vq, sg, sp = kennzahlen_realisierung(nu, U, g); alive = np.abs(U) > 0
        kenn.append((vq, sg, sp, float(np.sqrt(np.mean(hh[alive] ** 2)))))
    n_ok = None
    for (md, v), cl in clean.items():
        tag = f"{md}_{srctag(v)}"; A = np.array(arr[(md, v)])
        for j, nm in enumerate(NAMES):
            rec[f"{tag}_{nm}_clean"] = cl[j]; vv = A[:, j]; vv = vv[np.isfinite(vv)]
            if n_ok is None: n_ok = len(vv)
            rec[f"{tag}_{nm}"] = float(np.median(vv)) if len(vv) else np.nan; rec[f"{tag}_{nm}_q25"] = float(np.percentile(vv, 25)) if len(vv) else np.nan; rec[f"{tag}_{nm}_q75"] = float(np.percentile(vv, 75)) if len(vv) else np.nan
        rec[f"{tag}_frag"] = float(np.mean(A[:, NAMES.index("n10")] >= 2))
    rec["n_ok"] = n_ok if n_ok is not None else 0
    # Window parameters: gamma = amplitude loss per round trip, M_k = live round trips, v = gamma M_k,
    # beta = wedge phase slope 2 k0 alpha h sin(theta_i)
    gamma = -np.log(np.sqrt(R2)); Mk = int((np.abs(U) ** 2).sum(1).nonzero()[0].size)
    rec.update(M_k=Mk, gamma=gamma, v=gamma * Mk, beta=2 * k0 * np.deg2rad(al / 3600) * PT.h * np.sin(np.deg2rad(th)), dx_mm=Dx * 1e3, chord_mm=chord * 1e3)
    K = np.array(kenn)
    for i, k in enumerate(KENN): rec[f"{k}_med"] = float(np.median(K[:, i])); rec[f"{k}_mean"] = float(np.mean(K[:, i]))
    rec["runtime_s"] = time.time() - t0
    return [rec]


def main():
    """Parse the options, select the grid points, skip those already in the output file and compute the rest in a
    process pool; rows are appended to the CSV as they finish."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=800); ap.add_argument("--nsurf", type=int, default=50); ap.add_argument("--procs", type=int, default=max(1, mp.cpu_count() - 1))
    ap.add_argument("--seed", type=int, default=20261104, help="Samen der Punktauswahl UND der Flaechen")
    ap.add_argument("--ny", type=int, default=61, help="Strahlen ueber die Sehne")
    ap.add_argument("--src_pm", default="0,2,4,6", help="Quellbreiten (Halbwertsbreite, Gauss) in Pikometern, kommagetrennt")
    ap.add_argument("--D_m", type=float, default=0.0, help="Durchmesser [mm] der RMS-Definition und untere Spektralgrenze; 0 = bisherige Zernike-Scheibe 2 R_s(L_eff)")
    ap.add_argument("--probe", action="store_true"); ap.add_argument("--out", default=None)
    a = ap.parse_args(); sources = [float(x) for x in a.src_pm.split(",") if x.strip()]
    out = a.out or os.path.join(HERE, "theorie_grid_dreiband_probe.csv" if a.probe else "theorie_grid_dreiband_results.csv")
    if a.probe: pts = select_points(3, 7); a.nsurf = 3
    else: pts = select_points(a.n, a.seed)
    done = set(pd.read_csv(out, usecols=["pidx"])["pidx"]) if os.path.exists(out) else set()
    todo = [p for p in pts if p not in done]
    # Most expensive points first (cost ~ L / tan(theta_i), i.e. number of round trips)
    cost = lambda p: (lambda x: x[1] / np.tan(np.deg2rad(x[2])))(point_params(p)); todo.sort(key=lambda p: -cost(p)); COLS = columns(sources)
    t0 = time.time(); print(f"{len(pts)} Punkte, {len(todo)} offen, {a.nsurf} Flaechen je Punkt, ny={a.ny}, Quellen={sources} pm, {a.procs} Prozesse, Ausgabe {out}", flush=True)
    tasks = [(p, a.nsurf, a.seed, sources, a.ny, a.D_m) for p in todo]; n = 0
    with mp.Pool(min(a.procs, 61) if sys.platform == "win32" else a.procs) as pool:
        for rows in pool.imap_unordered(worker, tasks, chunksize=1):
            pd.DataFrame(rows).reindex(columns=COLS).to_csv(out, mode="a", header=not os.path.exists(out), index=False); n += 1
            if n % 50 == 0 or n == len(tasks) or a.probe:
                el = time.time() - t0; print(f"  {n}/{len(tasks)} Punkte, {el/60:.1f} min, Rest ~{el/n*(len(tasks)-n)/60:.0f} min", flush=True)
    print(f"fertig nach {(time.time()-t0)/60:.1f} min -> {out}")


if __name__ == "__main__":
    mp.freeze_support(); main()
