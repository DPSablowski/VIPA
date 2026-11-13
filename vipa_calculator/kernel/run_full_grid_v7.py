"""
Full 7-parameter ray-tracing grid for the global finesse model.

Purpose
-------
Runs the 2D multimode ray tracing of a real VIPA on the grid
    R2 x L_eff x theta_i x alpha x sigma_w x lambda x F# (aperture)
and stores, for every grid point and every random surface realisation, the finesse of
up to N_ORDERS_MAX diffraction orders measured with six width methods. These results are
the reference ("ray tracing") against which the theory and the closed equation are validated.

Physics of the run
------------------
  - The plate has a physical mirror length L_eff; a ray that leaves the mirror
    [-L_eff/2, +L_eff/2] does not contribute from that round trip on (hard edge cutoff).
  - Surface error: random Zernike surface (piston, tilt, defocus and astigmatism removed),
    scaled to RMS sigma_w over the Zernike disk of radius
        R_s = SURFACE_MARGIN * sqrt(R^2 + (d_Bl,max/2)^2),   R = L_eff/2, d_Bl,max = max(BLENDE_VALUES_MM).
    The disk encloses the whole illuminated strip, so no partial ray starts at the disk
    boundary where the surface is discontinuous. R_s deliberately does NOT depend on the
    aperture of the point (otherwise the surface statistics would depend on the aperture).
    The physical mirror edge stays at |x| <= R (L_eff).
  - Wedge: wedge phase 2 k0 alpha x cos(theta) per round trip plus the angle correction.
  - Aperture cone: for one focus the virtual sources are exact mirror images, the same for
    all rays; in detector direction theta_x only the ray that leaves at theta_x contributes.
    Therefore each order is computed with ONE x angle = field angle of the order
    (core and y extent are kept; y chord of the round aperture at theta_x).
  - Order positions are analytic from the resonance condition (no reference simulation);
    the measurement window is one full period around the circular centroid in delta.
    Only orders with |theta_x| <= min(0.9 theta_div, 0.5 sin theta_i) are measured.
  - The number of round trips per order follows from the walk-off to the mirror edge or the
    reflectivity decay, whichever is smaller (pmax_for_order).
This model was validated against the simulation-free mirror-image calculation (validate_v6.py).

Width methods (all six are stored, so the method can be chosen afterwards without new runs)
-------------------------------------------------------------------------------------------
  (1) adaptive (rms_fwhm): half-maximum crossing for a single peak, power-weighted RMS
      width when the line is fragmented (standard value F_values);
  (2) pure half-maximum crossing;
  (3) pure power-weighted RMS width;
  (4) Lorentz flank fit: Lorentz branches fitted separately to the outermost left and right
      flank of the highest sub-peak; width = sum of both half widths;
  (5) equivalent width: total area / peak height, scaled to the FWHM scale (factor pi/2
      for a Lorentz shape); integrates all fragmented power;
  (6) participation ratio: N_eff = (sum I)^2 / sum I^2, calibrated analytically against a
      Lorentz line (factor pi), no shape assumption.
The window resolution is chosen from the expected finesse (estimate_window_and_resolution),
so that narrow lines at high R2 are resolved with POINTS_PER_FWHM_TARGET points.
Optionally the raw spectra (xF, I) of all orders are archived (SAVE_ALL_SPECTRA), so that
any later evaluation method can be applied without repeating the expensive ray tracing.

Inputs
------
  Command line:  python run_full_grid_v7.py [N_REAL] [N_PROC]
      N_REAL: number of random realisations of the Zernike surface per grid point with
              sigma_w > 0. Without the argument it is taken from the checkpoint log
              (highest real index + 1), otherwise the default N_REAL_DEFAULT = 5.
              For sigma_w = 0 (perfect mirrors) only 1 realisation is computed (deterministic).
      N_PROC: number of parallel worker processes (default: all cores; on Windows at most 61).
  Examples:
      python run_full_grid_v7.py 40            # 40 realisations, all cores
      python run_full_grid_v7.py 40 32         # ... with 32 worker processes
  Required modules/files next to this script: raytrace_2d.py, raytrace_2d_multimode_v5_leff.py
  (read as source text at import), pupil_theory_v7.py (mirror spacing h), zernike_hr_surface.py.

Outputs (all relative to the folder of this script)
-------
  results_v4/<key>.npz        one file per grid point and realisation (version = 7), with the
                              grid parameters, F values of the six methods, fragmentation
                              diagnosis, field angles and pmax per order, optionally raw spectra;
                              files with version < 7 are recomputed and overwritten;
  completed_points_v7.log     checkpoint log, one key per finished point (primary resume mechanism);
  refcache_v2/                cache of the clean reference spectra (only used by the
                              legacy functions get_clean_reference / find_orders);
  progress lines on stdout.
Afterwards the results are aggregated with aggregate_full_grid_v7.py.

Run time: about 9200 parameter combinations before the multiplication with the
realisations; with a few seconds per call several hours on many cores. The script can be
interrupted and resumed at any time (checkpointing).
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import sys
import time
from multiprocessing import Pool, cpu_count
from scipy.optimize import curve_fit

# ============================================================
# Grid values -- seven physical parameters
# ============================================================
# Further runs with other values can be added as additional commented-out blocks and
# computed one after the other (RESULTS_DIR stays the same, checkpointing prevents
# duplicate work).

R2_VALUES = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.98]
L_EFF_VALUES_MM = [10.0, 15.0, 20.0, 25.0, 35.0]     # physical mirror length
# Tilt angle; 1 degree deliberately left out: numerically unstable at very small angle / few orders
THETA_I_VALUES_DEG = [2.0, 3.0, 4.0, 5.0, 6.0]
ALPHA_VALUES = [0.0, 0.5, 1.0, 1.5, 3.0]            # wedge angle, arcseconds
# Surface error RMS; 0.0 = perfect mirror (deterministic)
SIGMA_VALUES_NM = [0.0, 15.0, 25.0, 35.0, 50.0]
WL_VALUES_NM = [400, 500, 630, 700, 800, 900, 1000]
BLENDE_VALUES_MM = [1.5, 2.5, 3.5, 4.5, 5.5]             # aperture; F# = 75/aperture -> 50...13.6
SURFACE_MARGIN = 1.02                                  # Zernike disk slightly larger than needed


def surface_radius(L_eff_mm):
    """Returns the radius (m) of the Zernike disk. It encloses the whole illuminated strip
    (|x| <= L_eff/2, |y| <= d_Bl,max/2) for ALL apertures of the grid; independent of the
    aperture of the single point."""
    R = L_eff_mm * 1e-3 / 2
    y_max = max(BLENDE_VALUES_MM) * 1e-3 / 2
    return SURFACE_MARGIN * np.sqrt(R ** 2 + y_max ** 2)

N_TERMS = 36
N_REAL_DEFAULT = 5
# Paths relative to the folder of THIS script, not to the current working directory,
# so that a start from another folder (IDE, shortcut) still finds the checkpoint log.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(_SCRIPT_DIR, "results_v4")   # result folder (name kept; files carry version=7)
MIN_FREE_GB = 3.0   # below this free disk space the workers wait before saving
# Checkpoint log outside RESULTS_DIR, so that it stays untouched when finished .npz files
# are moved from there to an external disk and deleted locally (see move_to_external.py).
# One line per finished point (the "key" from process_one_point). The log, not a file
# system check of RESULTS_DIR, is the primary resume mechanism; otherwise moved/deleted
# result files would count as "not yet computed" and be recomputed.
CHECKPOINT_LOG = os.path.join(_SCRIPT_DIR, "completed_points_v7.log")


def _all_checkpoint_logs():
    """Returns the list of checkpoint logs to read (completed_points_v7*.log next to the script).
    Logs of earlier model versions are deliberately NOT read, so that their points are
    recomputed with the corrected surface."""
    import glob as _glob
    return sorted(_glob.glob(os.path.join(_SCRIPT_DIR, "completed_points_v7*.log")))


def _read_done_keys():
    """Returns the set of keys of all finished points found in the checkpoint logs."""
    keys = set()
    for fn in _all_checkpoint_logs():
        with open(fn, "r") as f:
            keys.update(l.strip() for l in f if l.strip())
    return keys

GRID_N = 151                # simulation grid of the mirror surfaces (GRID_N x GRID_N)
# Ray discretisation: with n_sub < 25 aliasing ghost orders appear, mainly for large
# aperture combined with wedge error (errors of a factor 2-10 possible)
N_SUB_COARSE = 25
N_SUB_FINE = 29
PMAX_HARD_CAP = 2500        # upper safety limit of round trips, prevents outlier run times
PMAX_SAFETY_MARGIN = 1.3    # multiple of the limiting round-trip estimate
PMAX_MIN_EXTRA = 30         # additive safety margin (divergence of the ray bundle)
DECAY_EPS = 1e-2            # reflectivity decay criterion (1 % residual amplitude)
# True: per point additionally store a small (500 points, float32) diagnostic spectrum
# of the CENTRAL order (~8 kB/point, ~2.6 GB in total); see SAVE_ALL_SPECTRA for ALL orders.
SAVE_DIAGNOSTIC_SPECTRA = False

# Store the full raw spectra of ALL orders, not only one diagnostic order, so that any
# evaluation method found later (such as participation_ratio_fwhm()) can be applied
# afterwards WITHOUT new simulation. The expensive ray tracing has already been done for
# the six standard methods; saving the (xF, I) arrays that are already in memory costs no
# extra computing time, only disk space.
#
# SIZE ESTIMATE (measured at a reference point). IMPORTANT: the sub-arrays MUST be written
# element by element into a preallocated object array (see process_one_point); otherwise
# numpy silently drops the float32 subtype for orders of equal length and the storage
# doubles. A typical grid point (R2 = 0.9, 24 orders) takes ~388 kB at FULL resolution,
# ~68 kB with RAW_SPECTRA_MAX_POINTS = 300. Extrapolated to the full grid (~9200 parameter
# combinations, on average ~4.2 realisations depending on the share of sigma_w = 0 / > 0):
#   RAW_SPECTRA_MAX_POINTS=None (full resolution): about 15 GB in total
#   RAW_SPECTRA_MAX_POINTS=300 (compromise):       about 3 GB in total
# Both are practicable.
# True: store the full (or reduced, see RAW_SPECTRA_MAX_POINTS) raw spectra of ALL orders
# of every grid point.
SAVE_ALL_SPECTRA = True
# None = full resolution (~15 GB in total, see estimate above); int = reduce every order to
# at most this many points (sampled evenly from the full window, ~3 GB at 300).
# Only relevant if SAVE_ALL_SPECTRA=True.
RAW_SPECTRA_MAX_POINTS = None

# Adaptive window resolution, see estimate_window_and_resolution()
MIN_WINDOW_POINTS = 2000    # lower limit of the number of window points
# Upper safety limit (prevents outliers at unexpectedly high finesse due to fragmentation)
MAX_WINDOW_POINTS = 20000
POINTS_PER_FWHM_TARGET = 40 # target number of points inside the expected FWHM

# Construction without astigmatism: piston, tilt, defocus and astigmatism are excluded
EXCLUDED = {(0, 0), (1, 1), (1, -1), (2, 0), (2, 2), (2, -2)}


def build_surface_no_astig(R_mirror, sigma_target, rng, n_terms, grid_n):
    """Builds a random Zernike surface from the first n_terms Noll terms without the
    EXCLUDED terms, with N(0, 1) coefficients, scaled to RMS sigma_target over the disk of
    radius R_mirror. Returns (xg, surf, mask, active_terms): grid axis (m), height map (m,
    zero outside the disk), disk mask and the list of (n, m) terms used."""
    from zernike_hr_surface import zernike, noll_to_nm
    xg = np.linspace(-R_mirror, R_mirror, grid_n)
    X, Y = np.meshgrid(xg, xg)
    rho = np.sqrt(X**2 + Y**2) / R_mirror
    phi = np.arctan2(Y, X)
    mask = rho <= 1.0

    active_terms = []
    for j in range(1, n_terms + 1):
        n, m = noll_to_nm(j)
        if (n, m) in EXCLUDED:
            continue
        active_terms.append((n, m))

    coeffs = rng.normal(0, 1, size=len(active_terms))
    surf = np.zeros_like(rho)
    for c, (n, m) in zip(coeffs, active_terms):
        Z = zernike(n, m, rho, phi)
        Z[~mask] = 0
        surf += c * Z

    current_rms = np.std(surf[mask])
    surf *= sigma_target / current_rms
    surf[~mask] = 0
    return xg, surf, mask, active_terms


def compute_pmax(L_eff_mm, theta_i_deg, R2, h=2e-3, blende_mm=None, f_cyl=75e-3):
    """Automatic p_max (number of round trips): the SMALLER of the two physical limits
    (round trips to the mirror edge, reflectivity decay length) determines convergence.
    Returns (pmax, M_eff) with M_eff = L_eff / Delta_x of the nominal ray.

    The edge limit is computed for the SLOWEST sub-ray, not only for the nominal ray. The
    input cone (half angle theta_div = arctan(d_aperture/(2 f_cyl))) lets sub-rays with
    negative thx walk more slowly: rate = Delta_x - theta_div*L_step. These rays need
    L_eff/rate > M_eff round trips to the edge; for rate <= 0 (theta_div >= sin(theta_i),
    only theta_i = 2 deg with large aperture) they never leave the mirror and are limited
    by the reflectivity alone. Without this, such rays would be cut off by pmax at
    R2 = 0.98 with up to ~40 % residual amplitude (truncation artefact).
    blende_mm=None -> nominal ray only.
    Not used by the current per-order measurement (see pmax_for_order)."""
    theta_rad = np.deg2rad(theta_i_deg)
    Delta_x = 2 * h * np.tan(theta_rad)
    M_eff = (L_eff_mm * 1e-3) / Delta_x
    r2 = np.sqrt(R2)
    decay_terms = np.log(DECAY_EPS) / np.log(r2)
    M_walk = M_eff
    if blende_mm is not None:
        theta_div = np.arctan(blende_mm * 1e-3 / (2 * f_cyl))
        L_step = 2 * h / np.cos(theta_rad)
        rate_min = Delta_x - theta_div * L_step
        M_walk = (L_eff_mm * 1e-3) / rate_min if rate_min > 0 else np.inf
    limiting = min(max(M_eff, M_walk), decay_terms)
    pmax = int(np.ceil(limiting * PMAX_SAFETY_MARGIN)) + PMAX_MIN_EXTRA
    return min(pmax, PMAX_HARD_CAP), M_eff


def F_reflectivity(R2):
    """Returns the standard Fabry-Perot (Airy) finesse pi*sqrt(R)/(1-R), R = sqrt(R2).
    It holds EXACTLY for the VIPA in the reflectivity-dominated, defect-free limit; here it
    is used ONLY as a conservative UPPER bound for the adaptive window resolution (the actual
    finesse is, because of M_eff and fragmentation, normally NOT higher than this value)."""
    R = np.sqrt(R2)
    return np.pi * np.sqrt(R) / (1 - R)


def estimate_window_and_resolution(R2, spacing_m):
    """Adaptive window width and resolution. Returns (window, n_points): half width of the
    window (m) and number of points.
    The window half width is 0.5*spacing (one full period, without gaps or overlap with the
    neighbouring window). Reason: with strong wedge fragmentation side peaks reach up to
    0.49*spacing from the centre; a narrower window would lose them and overestimate F_eq.
    With 0.5 the raw spectrum archive (SAVE_ALL_SPECTRA) is also complete: any later choice
    of window or threshold (e.g. for flank_fit_fwhm or the frag_info threshold) can be
    recomputed from the archive without new simulation. Only the RESOLUTION (number of
    points) is raised according to the expected finesse, so that the expected FWHM is
    resolved with at least POINTS_PER_FWHM_TARGET points."""
    F_est = F_reflectivity(R2)
    expected_fwhm = spacing_m / F_est
    window = spacing_m * 0.5
    n_points = int(np.ceil(2 * window / expected_fwhm * POINTS_PER_FWHM_TARGET))
    n_points = int(np.clip(n_points, MIN_WINDOW_POINTS, MAX_WINDOW_POINTS))
    return window, n_points


# ============================================================
# Measurement pipeline: no artificial envelope; widths measured per order.
#
# The envelope (field-of-view window) of the original ray tracer is pure post-processing,
# not part of the actual ray sum. Applied as a hard window it cuts orders near its border
# and makes their width measurement useless; without it a complete, valid peak is present
# there. In addition, with stronger wedge error the interference pattern fragments
# into several side maxima of similar height; a simple peak search then
# picks one of them almost at random. The measurement therefore uses the ray sum without
# envelope and measures the width with the methods below.
#
# The ray tracer without envelope is created here by patching the source text of
# raytrace_2d_multimode_v5_leff.py: the envelope lines are removed, the function is renamed
# to _raytrace_v5_noenv and the code is executed into a private namespace.
# ============================================================
_src_v5 = open(os.path.join(_SCRIPT_DIR, "raytrace_2d_multimode_v5_leff.py")).read()
_src_v5_noenv = _src_v5.replace(
    "    theta_x_field = np.arctan(xF / rt2d.F)\n"
    "    envelope = (np.abs(theta_x_field) <= theta_div).astype(float)\n"
    "    return I_raw * envelope",
    "    return I_raw"
).replace("def raytrace_2d_multimode_v5_leff", "def _raytrace_v5_noenv")
_ns_v5 = {}
exec(compile(_src_v5_noenv, "_raytrace_v5_noenv_module", "exec"), _ns_v5)
_raytrace_v5_noenv = _ns_v5["_raytrace_v5_noenv"]


def raytrace_noenv(xg, surf, mask, blende_mm, xF, n_sub, pmax, alpha):
    """Returns the detector intensity I(xF) of the multimode ray tracer without envelope, with
    the surface error on the HR mirror only (the other two surfaces are flat and unmasked) and
    with the hard cutoff at the mirror edge. (Redefined identically further below.)"""
    zero = np.zeros_like(surf); fullmask = np.ones_like(mask)
    return _raytrace_v5_noenv(xg, xg, surf, mask, xg, xg, zero, fullmask, xg, xg, zero, fullmask,
                                blende_mm, xF, n_sub=n_sub, pmax=pmax, alpha_arcsec=alpha,
                                enforce_Leff_cutoff=True)


def rms_fwhm(xF_win, I_win):
    """ADAPTIVE width measurement (standard method). Returns the FWHM (in units of xF) or nan.
    The pure power-weighted RMS width strongly overestimates the width of a clean
    (unfragmented) Airy peak, because its flanks enter the RMS quadratically. Example:
    at R2=0.8, L_eff=10mm, theta_i=2 deg, alpha=0, sigma_w=0 the pure RMS method gave
    F=8.54 instead of the correct F=28.4 (~F_reflectivity=28.1), i.e. an UNDERESTIMATE by
    a factor ~3.3 that limits the finesse in the whole alpha~0/sigma_w~0 region to an
    artificial saturation value.
    The pure half-maximum crossing, on the other hand, fails for fragmented wedge spectra
    with erratic order-to-order scatter.
    SOLUTION: detect the peak multiplicity in the window (find_peaks, threshold 40 % of the
    maximum). Exactly one peak (clean case): normal half-maximum crossing. Several peaks of
    similar height (fragmentation): robust RMS width (FWHM = 2.3548 sigma, Gaussian scale).
    Validated at both extremes (clean: 28.39 against expected 28.1; fragmented: relative
    order scatter 10.0 %)."""
    from scipy.signal import find_peaks as _find_peaks_local
    pk, _ = _find_peaks_local(I_win, height=I_win.max() * 0.4)
    if len(pk) <= 1:
        half = I_win.max() / 2
        above = xF_win[I_win >= half]
        return above.max() - above.min() if len(above) >= 2 else np.nan
    base = np.percentile(I_win, 10)
    w = np.clip(I_win - base, 0, None)
    if w.sum() <= 0:
        return np.nan
    mean = np.sum(xF_win * w) / np.sum(w)
    var = np.sum(w * (xF_win - mean) ** 2) / np.sum(w)
    return np.sqrt(var) * 2.3548


# ============================================================
# SPEED-UP 1: cache the common reference spectrum (legacy order search).
#
# The clean reference spectrum (alpha=0, sigma=0) depends only on
# (R2, L_eff, theta_i, wl, aperture) -- NOT on alpha, sigma_w or the realisation.
# Up to 5 (alpha) x 21 (sigma + realisations) = 105 tasks share the same reference;
# without caching it would be recomputed 105 times.
# Stored as a small .npz file (only peak positions + spacing).
# ============================================================
REFCACHE_DIR = os.path.join(_SCRIPT_DIR, "refcache_v2")


def get_clean_reference(pmax, R2, L_eff_mm, theta_i_deg, wl_nm, blende_mm):
    """Returns (positions, spacing) of the orders of the clean plate (alpha=0, sigma=0) on
    the detector (m), from a coarse ray tracing and a peak search; cached in REFCACHE_DIR.
    Used only by the legacy order search find_orders()."""
    from scipy.signal import find_peaks
    key = f"R2{R2}_Leff{L_eff_mm}_th{theta_i_deg}_wl{wl_nm}_bl{blende_mm}_pmax{pmax}"
    cache_file = os.path.join(REFCACHE_DIR, f"{key}.npz")
    if os.path.exists(cache_file):
        d = np.load(cache_file)
        return d["positions"], d["spacing"]

    GRID_N = 151
    R_mirror = L_eff_mm * 1e-3 / 2
    xg = np.linspace(-R_mirror, R_mirror, GRID_N)
    zero = np.zeros((GRID_N, GRID_N)); fullmask = np.ones((GRID_N, GRID_N), dtype=bool)
    xF_coarse = np.linspace(-2e-3, 2e-3, 6000)
    I_clean = raytrace_noenv(xg, zero, fullmask, blende_mm, xF_coarse, N_SUB_COARSE, pmax, 0.0)
    peaks_clean, _ = find_peaks(I_clean, height=I_clean.max() * 0.05, distance=100)
    positions = xF_coarse[peaks_clean]
    spacing = np.gradient(positions) if len(positions) >= 2 else np.array([])

    os.makedirs(REFCACHE_DIR, exist_ok=True)
    # write atomically (temp file + rename), since several processes could compute
    # and write the same cache entry at the same time
    tmp = cache_file + f".tmp{os.getpid()}"
    np.savez(tmp, positions=positions, spacing=spacing)
    try:
        os.replace(tmp, cache_file)
    except OSError:
        pass
    return positions, spacing


# ============================================================
# SPEED-UP 2: measure all windows in ONE call.
#
# Inside the round-trip loop the quantities summed per round trip (A_m, B_m -- amplitude
# and phase over all rays) are completely INDEPENDENT of xF; xF enters only at the end
# through the (cheap) geometric phase. One separate full call per order would run the
# (expensive) round-trip loop each time. Benchmark: 35 single calls of 2000 points cost
# ~35x a base call; ONE combined call with 35x2000 points costs only ~4.8x (7.3x faster).
# Therefore all order windows are joined into one xF array, ONE call is made, and the
# result is split again (measure_all_orders_combined).
# ============================================================
def raytrace_noenv(xg, surf, mask, blende_mm, xF, n_sub, pmax, alpha):
    """Same as the definition above: detector intensity I(xF) without envelope."""
    zero = np.zeros_like(surf); fullmask = np.ones_like(mask)
    return _raytrace_v5_noenv(xg, xg, surf, mask, xg, xg, zero, fullmask, xg, xg, zero, fullmask,
                                blende_mm, xF, n_sub=n_sub, pmax=pmax, alpha_arcsec=alpha,
                                enforce_Leff_cutoff=True)


def halfmax_fwhm(xF_win, I_win):
    """Pure half-maximum crossing without fragmentation detection. Returns the FWHM or nan."""
    half = I_win.max() / 2
    above = xF_win[I_win >= half]
    return above.max() - above.min() if len(above) >= 2 else np.nan


def pure_rms_fwhm(xF_win, I_win):
    """Pure power-weighted RMS width without peak detection, background = 10th percentile.
    Returns 2.3548 * RMS (FWHM on a Gaussian scale) or nan."""
    base = np.percentile(I_win, 10)
    w = np.clip(I_win - base, 0, None)
    if w.sum() <= 0:
        return np.nan
    mean = np.sum(xF_win * w) / np.sum(w)
    var = np.sum(w * (xF_win - mean) ** 2) / np.sum(w)
    return np.sqrt(var) * 2.3548


def flank_fit_fwhm(xF_win, I_win, frac_lo=0.05, frac_hi=0.6):
    """LORENTZ flank fit. Returns the FWHM (sum of both fitted half widths) or nan.
    Fits a Lorentz branch I(x)=Imax/(1+((x-x_peak)/gamma)^2) (fixed at position and height
    of the highest sub-peak) separately to the outermost, connected left and right flank
    (fit range frac_lo..frac_hi = 5-60 % of the peak height; the walk inward stops at the
    FIRST point reaching frac_hi, which ignores inner rises between fragmented sub-peaks).
    For a Lorentzian gamma is directly the HWHM. Why Lorentz instead of Gauss: VIPA
    transmission peaks have Lorentz-shaped (heavier) flanks; a Gaussian fit to the same
    flank points needs a too large sigma and overestimates the width by about 15-20 %."""
    Imax = I_win.max()
    if Imax < 1e-8:
        return np.nan
    imax_idx = np.argmax(I_win)
    x_peak = xF_win[imax_idx]

    # Left flank: from the left window edge inward until the first point >= frac_hi*Imax
    i = 0
    while i < imax_idx and I_win[i] < frac_hi * Imax:
        i += 1
    left_seg = np.arange(0, i + 1)
    left_seg = left_seg[I_win[left_seg] >= frac_lo * Imax]
    x_left, I_left = xF_win[left_seg], I_win[left_seg]

    # Right flank: same from the right window edge
    j = len(I_win) - 1
    while j > imax_idx and I_win[j] < frac_hi * Imax:
        j -= 1
    right_seg = np.arange(j, len(I_win))
    right_seg = right_seg[I_win[right_seg] >= frac_lo * Imax]
    x_right, I_right = xF_win[right_seg], I_win[right_seg]

    if len(x_left) < 2 or len(x_right) < 2:
        return np.nan

    # Lorentz branch with fixed position and height; only gamma (HWHM) is fitted
    def lorentz_x0fixed(x, gamma):
        return Imax / (1 + ((x - x_peak) / gamma) ** 2)

    xspan = xF_win.max() - xF_win.min()
    try:
        popt_l, _ = curve_fit(lorentz_x0fixed, x_left, I_left, p0=[xspan / 10],
                                bounds=(xspan * 1e-4, xspan * 2))
        popt_r, _ = curve_fit(lorentz_x0fixed, x_right, I_right, p0=[xspan / 10],
                                bounds=(xspan * 1e-4, xspan * 2))
    except Exception:
        return np.nan

    return popt_l[0] + popt_r[0]


def equivalent_width_fwhm(xF_win, I_win, calib=np.pi / 2, bg_percentile=5):
    """EQUIVALENT WIDTH (total area / peak height), scaled to the FWHM scale (factor pi/2 for
    a pure Lorentz shape: eq_width = pi*HWHM = (pi/2)*FWHM). Returns the FWHM-equivalent or nan.
    Integrates the WHOLE (fragmented) power automatically, without detecting peaks or flanks
    individually -- a standard measure in spectroscopy. Background = bg_percentile-th
    percentile of the window.

    Validated on a full test grid (383 238 orders): at EVERY degree of fragmentation a clearly
    smaller bias than half-maximum and Lorentz flank fit -- constantly about 46-49 % smaller
    log deviation than half-maximum (flank fit only 21-26 %, decreasing at high
    fragmentation), over the whole tested range from 2 to >10 sub-peaks.

    IMPORTANT CAVEAT: the equivalent width integrates over the WHOLE measurement window; if
    the window is too narrow relative to the line width, Lorentz flanks are cut and the
    area (and therefore F) is biased. The window is one full order period (see
    estimate_window_and_resolution()); keep this in mind for extreme fragmentation."""
    base = np.percentile(I_win, bg_percentile)
    I_bg = np.clip(I_win - base, 0, None)
    if I_bg.max() <= 0:
        return np.nan
    area = np.trapezoid(I_bg, xF_win)
    return area / I_bg.max() / calib


def participation_ratio_fwhm(xF_win, I_win):
    """PARTICIPATION RATIO (a measure taken from solid-state physics for the "effective number
    of contributing states/bins"), the sixth width method. Returns the FWHM-equivalent.
    Definition:
        N_eff = (sum I_k)^2 / sum(I_k^2)
    -- purely energy/statistics based, WITHOUT any shape assumption, WITHOUT window-edge or
    minimum search (unlike the flank fit) and WITHOUT the pure peak-height normalisation of
    the equivalent width (which weights wing energy differently under strong fragmentation).

    CALIBRATION (analytic): for a pure Lorentz shape I(x)=1/(1+(x/gamma)^2) in the continuum
    limit N_eff = (int I dx)^2 / int I^2 dx = 2*pi*gamma = pi*FWHM (exact) -- hence
    FWHM = N_eff*dx/pi, with dx = point spacing in the window (determined PER ORDER from the
    actual xF_win, since n_points varies per order).

    NO background subtraction: the long, low flanks of a profile carry real, necessary width
    information for THIS method (unlike for the equivalent width). Cutting them makes N_eff
    artificially small and F artificially large -- exactly the fragmentation bias this method
    is meant to avoid. A test against the exact first-principles ensemble theory gave a
    factor 3.3-8.4 deviation with background subtraction and 1.1-1.6 without it.

    VALIDATED (small-aperture test series, several sigma_w and aperture values, against the
    EXACT first-principles ensemble theory): clearly better agreement than the flank fit
    (factor 2.4-4.2 deviation) and the equivalent width (factor 1.4-2.4); the participation
    ratio stays in the range 1.1-1.6, without an extra parameter to tune.

    CAVEAT (as for equivalent_width_fwhm, but weaker): the analytic pi calibration is exact
    only for an INFINITE window; in a finite window a numerical test shows a moderate but
    SYSTEMATIC bias of about -5 % (FWHM slightly underestimated, i.e. F slightly
    overestimated). Deliberately not recalibrated (the pure constant pi keeps the method
    independent of the window size). Without background subtraction a constant zero offset
    of a detector would be a residual risk -- not relevant here, since the simulation
    delivers exact intensities without a noise floor; to be checked for real measured data."""
    dx = np.median(np.diff(xF_win))
    N_eff = np.sum(I_win)**2 / np.sum(I_win**2)
    return N_eff * dx / np.pi


def measure_all_orders_combined(pmax, alpha, surf, mask, xg, blende_mm, positions, spacing, shift,
                                  R2=None, save_diagnostic=False, save_all_spectra=False):
    """Measures all orders of one grid point with ONE combined ray-tracing call over all
    order windows (instead of one call per order). Legacy path: the current main path uses
    measure_orders_v6().

    Returns (F_adaptive, F_halfmax, F_rms, F_flankfit, F_eqwidth, F_participation,
    diagnostic, frag_info, raw_spectra): lists of finesse values (order spacing / FWHM) for
    the SIX width methods, an optional diagnostic spectrum, the fragmentation diagnosis per
    order (n_peaks, second-highest/highest peak) and optionally the raw spectra.

    Storing all six methods is nearly free, because the expensive ray tracing has already
    been done; the method can thus be changed afterwards without new simulation.

    save_all_spectra=True additionally stores the FULL raw spectrum (xF, I) of EVERY order
    (see RAW_SPECTRA_MAX_POINTS and the module docstring for the size estimate).

    R2 is used for the ADAPTIVE window resolution (see estimate_window_and_resolution());
    window half width 0.5*spacing. With R2=None the fixed value of 2000 points is used.

    Optional (save_diagnostic=True): a REDUCED (500 points, float32) diagnostic spectrum of
    the central order (closest to the origin) -- for visual spot checks without new simulation.

    Fragmentation diagnosis per order (frag_info): multi-peak structures occur NOT only with
    wedge error (alpha>0) but also from surface errors alone (sigma_w>0) -- a reproducible
    physical phenomenon (correlated phase errors over many round trips of the same ray
    trajectory), not a measurement uncertainty. For these cases a single "finesse" number is
    inherently fuzzy. Instead of "solving" this in the simulation, the RAW diagnostic data are
    stored so that the weighting can be defined freely per parameter combination in the FIT
    step (see fit_global_model_v2.py, compute_weight())."""
    valid_idx = [i for i in range(len(positions)) if i < len(spacing) and spacing[i] > 0]
    if not valid_idx:
        return [], [], [], [], [], [], None, [], None

    windows = []
    n_points_list = []
    for i in valid_idx:
        center = positions[i] + shift
        if R2 is not None:
            window, n_points = estimate_window_and_resolution(R2, spacing[i])
        else:
            # Fallback (R2 unknown): 0.5*spacing, consistent with
            # estimate_window_and_resolution() -- see its docstring.
            window, n_points = spacing[i] * 0.5, MIN_WINDOW_POINTS
        xFf = np.linspace(center - window, center + window, n_points)
        windows.append(xFf)
        n_points_list.append(n_points)
    xF_combined = np.concatenate(windows)

    I_combined = raytrace_noenv(xg, surf, mask, blende_mm, xF_combined, N_SUB_FINE, pmax, alpha)

    F_values_adaptive, F_values_halfmax, F_values_rms, F_values_flankfit, F_values_eqwidth = [], [], [], [], []
    F_values_participation = []
    frag_info = []  # per order: (n_peaks, second highest peak / highest peak)
    diagnostic = None
    center_dist = np.inf
    offset = 0
    raw_spectra = [] if save_all_spectra else None
    for j, i in enumerate(valid_idx):
        n_points = n_points_list[j]
        seg = I_combined[offset:offset + n_points]
        xFf = windows[j]
        offset += n_points
        if seg.max() < 1e-8:
            continue
        fwhm_a = rms_fwhm(xFf, seg)          # adaptive (standard, see above)
        fwhm_h = halfmax_fwhm(xFf, seg)
        fwhm_r = pure_rms_fwhm(xFf, seg)
        fwhm_f = flank_fit_fwhm(xFf, seg)     # Lorentz flank fit
        fwhm_eq = equivalent_width_fwhm(xFf, seg)  # equivalent width
        fwhm_pr = participation_ratio_fwhm(xFf, seg)  # participation ratio
        if not np.isnan(fwhm_a) and fwhm_a > 0:
            F_values_adaptive.append(spacing[i] / fwhm_a)
        if not np.isnan(fwhm_h) and fwhm_h > 0:
            F_values_halfmax.append(spacing[i] / fwhm_h)
        if not np.isnan(fwhm_r) and fwhm_r > 0:
            F_values_rms.append(spacing[i] / fwhm_r)
        if not np.isnan(fwhm_f) and fwhm_f > 0:
            F_values_flankfit.append(spacing[i] / fwhm_f)
        if not np.isnan(fwhm_eq) and fwhm_eq > 0:
            F_values_eqwidth.append(spacing[i] / fwhm_eq)
        if not np.isnan(fwhm_pr) and fwhm_pr > 0:
            F_values_participation.append(spacing[i] / fwhm_pr)

        # Fragmentation: number of peaks above 40 % of the maximum and height ratio of the two highest
        from scipy.signal import find_peaks as _fp
        pk, props = _fp(seg, height=seg.max() * 0.4)
        if len(pk) >= 2:
            heights_sorted = np.sort(props["peak_heights"])[::-1]
            second_ratio = float(heights_sorted[1] / heights_sorted[0])
        else:
            second_ratio = 0.0
        frag_info.append((len(pk), second_ratio))

        if save_diagnostic:
            d = abs(positions[i] + shift)
            if d < center_dist:
                center_dist = d
                idx = np.linspace(0, len(xFf) - 1, 500).astype(int)
                diagnostic = (xFf[idx].astype(np.float32), seg[idx].astype(np.float32))

        if save_all_spectra:
            # Store the full raw spectrum of THIS order (see the module docstring for the size
            # estimate and the RAW_SPECTRA_MAX_POINTS compromise). Done AFTER the method
            # evaluation; only the arrays already in memory are copied.
            if RAW_SPECTRA_MAX_POINTS is not None and len(xFf) > RAW_SPECTRA_MAX_POINTS:
                idx_r = np.linspace(0, len(xFf) - 1, RAW_SPECTRA_MAX_POINTS).astype(int)
                raw_spectra.append((xFf[idx_r].astype(np.float32), seg[idx_r].astype(np.float32)))
            else:
                raw_spectra.append((xFf.astype(np.float32), seg.astype(np.float32)))

    return (F_values_adaptive, F_values_halfmax, F_values_rms, F_values_flankfit,
            F_values_eqwidth, F_values_participation, diagnostic, frag_info, raw_spectra)


def _delta_of_xF(xF):
    """Returns the resonance variable delta (phase per round trip) as a function of the
    detector position xF, directly from the imaging phase of the ray tracer:
    delta(xF) = phi_2 - phi_1. At z_F = F the imaging phase is exactly linear in m, so this is exact."""
    import raytrace_2d as rt2d   # the same module object configured by process_one_point
    return rt2d.geometric_phase_nominal(2, xF) - rt2d.geometric_phase_nominal(1, xF)


def _local_fsr(xF):
    """Returns the local order spacing on the detector: 2*pi / |d delta / d xF| (central
    difference; analytically lambda*F / (2 t0 (sin th + cos th xF/F)) at z_F = F)."""
    h = 1e-7
    d = (_delta_of_xF(np.asarray(xF) + h) - _delta_of_xF(np.asarray(xF) - h)) / (2 * h)
    return 2 * np.pi / np.abs(d)


def find_orders(pmax, alpha, surf, mask, xg, blende_mm, R2, L_eff_mm, theta_i_deg, wl_nm):
    """Returns centres and local order spacing of the orders to be measured:
    (positions, spacing, shift) with shift = 0.0. Legacy order search (the current main path
    uses order_centers()).

    Wedge (and roughness) shift the lines by a fixed amount in the RESONANCE VARIABLE delta;
    on the detector this is proportional to the local order spacing, which varies strongly
    across the image field at small theta_i (theta_i = 2 deg: factor 3). A single shift in
    xF applied to all orders would put windows next to the lines (measurement errors up to
    -21 % at theta_i = 2, alpha = 3''). Therefore: cross-correlation in delta (where the shift
    is the same for all orders), shift limited to |d_delta| <= pi (a full period would only
    be the neighbouring order), centre of each order via the inverse mapping delta -> xF,
    order spacing analytic at the new centre."""
    from scipy.signal import correlate
    positions, spacing = get_clean_reference(pmax, R2, L_eff_mm, theta_i_deg, wl_nm, blende_mm)
    if len(positions) < 2:
        return positions, spacing, 0.0

    zero = np.zeros_like(surf); fullmask = np.ones_like(mask)
    xF_coarse = np.linspace(-2e-3, 2e-3, 6000)
    I_real = raytrace_noenv(xg, surf, mask, blende_mm, xF_coarse, N_SUB_COARSE, pmax, alpha)
    I_clean_full = raytrace_noenv(xg, zero, fullmask, blende_mm, xF_coarse, N_SUB_COARSE, pmax, 0.0)

    # Resample onto a uniform delta grid (delta(xF) is monotonic in the image field)
    d_x = _delta_of_xF(xF_coarse)
    o = np.argsort(d_x)
    d_sorted = d_x[o]
    d_grid = np.linspace(d_sorted[0], d_sorted[-1], len(xF_coarse))
    Ir = np.interp(d_grid, d_sorted, I_real[o])
    Ic = np.interp(d_grid, d_sorted, I_clean_full[o])
    xcorr = correlate(Ir - Ir.mean(), Ic - Ic.mean(), mode="same")
    dd = d_grid[1] - d_grid[0]
    lags = (np.arange(len(xcorr)) - len(xcorr) // 2) * dd
    ok = np.abs(lags) <= np.pi
    shift_delta = lags[ok][np.argmax(xcorr[ok])]

    # New centres: delta of the reference line + shift, mapped back to xF
    x_sorted = xF_coarse[o]
    new_pos = np.interp(_delta_of_xF(np.asarray(positions)) + shift_delta, d_sorted, x_sorted)
    new_spacing = _local_fsr(new_pos)
    return new_pos, new_spacing, 0.0


def measure_fwhm_single(pmax, alpha, surf, mask, xg, blende_mm, center, window):
    """Single-order measurement (2000 points, adaptive width) for single tests. Returns the FWHM or nan."""
    xF_fine = np.linspace(center - window, center + window, 2000)
    I_fine = raytrace_noenv(xg, surf, mask, blende_mm, xF_fine, N_SUB_FINE, pmax, alpha)
    if I_fine.max() < 1e-8:
        return np.nan
    return rms_fwhm(xF_fine, I_fine)


# ============================================================
# Ray bundle per order (one x angle), see module docstring
# ============================================================
D_CORE_IMAGE = 75e-6      # image of the fibre core in x (m)
N_CORE = 3                # sample points across the core image (small effect)
N_Y = N_SUB_FINE          # sample points across the y chord of the aperture
N_ORDERS_MAX = 11         # at most this many orders (those closest to the centre)
# True: add the core points incoherently (multimode fibre: modes mutually incoherent).
# Open model assumption; effect < 1 % except at R2 ~ 0.98 with strong wedge (up to ~10 %).
CORE_INCOHERENT = True


def _interp2d_vec(x, y, xg, yg, Z):
    """Returns Z bilinearly interpolated at the points (x, y) on the grid (xg, yg) (indices clipped to the grid)."""
    ix = np.clip(np.searchsorted(xg, x) - 1, 0, len(xg) - 2)
    iy = np.clip(np.searchsorted(yg, y) - 1, 0, len(yg) - 2)
    fx = (x - xg[ix]) / (xg[ix + 1] - xg[ix])
    fy = (y - yg[iy]) / (yg[iy + 1] - yg[iy])
    return (Z[iy, ix] * (1 - fx) * (1 - fy) + Z[iy, ix + 1] * fx * (1 - fy)
            + Z[iy + 1, ix] * (1 - fx) * fy + Z[iy + 1, ix + 1] * fx * fy)


def raytrace_order(xg, surf, mask, blende_mm, xF, pmax, alpha_arcsec, theta_x0):
    """Coherent round-trip sum for ONE ray bundle with x angle theta_x0. Returns the detector
    intensity I(xF) (incoherent sum over the core groups if CORE_INCOHERENT).
    Physics: surface error on the HR surface only, height phase 2 k0 h, slope deflection,
    wedge phase 2 k0 alpha x cos(theta), wedge angle correction, hard cutoff at the mirror
    edge; all rays start with thx = theta_x0 (field angle of the measured order)."""
    import raytrace_2d as rt2d
    alpha_rad = alpha_arcsec / 3600 * np.pi / 180
    dx = xg[1] - xg[0]
    surf_m = np.where(mask, surf, 0.0)
    g0, g1 = np.gradient(surf_m, dx); sl_x, sl_y = g1, g0
    theta_div = np.arctan(blende_mm * 1e-3 / (2 * rt2d.f_cyl))
    chord = np.sqrt(max(0.0, 1.0 - (theta_x0 / theta_div) ** 2))   # round aperture: relative y chord at theta_x0
    xi_c = np.linspace(-1, 1, N_CORE) if N_CORE > 1 else np.array([0.0])
    xi_y = np.linspace(-1, 1, N_Y)
    XC, XY = np.meshgrid(xi_c, xi_y)
    # Start points: core image at the entrance edge of the mirror (x) and across the y chord
    x = (-rt2d.R_mirror + D_CORE_IMAGE / 2 + XC * D_CORE_IMAGE / 2).ravel()
    y = (XY * chord * blende_mm * 1e-3 / 2).ravel()
    n = len(x)
    w = np.full(n, 1.0 / n)
    thx = np.full(n, float(theta_x0)); thy = np.zeros(n)
    alive = np.ones(n, dtype=bool)
    acc = np.zeros(n)
    # Groups that add coherently inside, incoherently among each other (one group per core point)
    groups = [np.arange(n)] if not CORE_INCOHERENT else [np.nonzero(XC.ravel() == c)[0] for c in xi_c]
    Er = np.zeros((len(groups), len(xF))); Ei = np.zeros((len(groups), len(xF)))
    for m in range(pmax + 1):
        # Hard edge: a ray that has left the mirror does not contribute any more
        alive &= (np.abs(x) <= rt2d.R_mirror)
        if not alive.any():
            break
        hh = _interp2d_vec(x, y, xg, xg, surf_m)
        sx = _interp2d_vec(x, y, xg, xg, sl_x)
        sy = _interp2d_vec(x, y, xg, xg, sl_y)
        # Accumulated phase: double pass of the height error + wedge phase per round trip
        acc = acc + rt2d.k0 * 2 * hh + rt2d.k0 * 2 * alpha_rad * x * np.cos(rt2d.theta)
        amp = w * rt2d.r2 ** m * alive
        # Nominal geometric (imaging) phase of round trip m at the detector positions
        pg = rt2d.geometric_phase_nominal(m, xF)
        cg = np.cos(pg); sg = np.sin(pg)
        for gi, g in enumerate(groups):
            A = np.sum(amp[g] * np.cos(acc[g])); B = np.sum(amp[g] * np.sin(acc[g]))
            Er[gi] += A * cg - B * sg; Ei[gi] += A * sg + B * cg
        # Ray direction: reflection doubles the surface slope; the wedge tilts by 2 alpha cos(theta) per round trip
        thx = thx + 2 * sx + 2 * alpha_rad * np.cos(rt2d.theta)
        thy = thy + 2 * sy
        x = x + rt2d.Delta_x + thx * rt2d.L_step
        y = y + thy * rt2d.L_step
    return np.sum(Er ** 2 + Ei ** 2, axis=0)


def order_centers(theta_i_deg, blende_mm):
    """Analytic order positions: delta(xF) = 2 pi N (phase exactly linear in m at z_F=F).
    Only |theta_x| <= min(0.9 theta_div, 0.5 sin theta_i). Returns (centres sorted in xF,
    at most N_ORDERS_MAX closest to the centre; theta_div)."""
    import raytrace_2d as rt2d
    theta_div = np.arctan(blende_mm * 1e-3 / (2 * rt2d.f_cyl))
    thx_max = min(0.9 * theta_div, 0.5 * np.sin(np.deg2rad(theta_i_deg)))
    xmax = rt2d.F * np.tan(thx_max)
    xs = np.linspace(-xmax, xmax, 20001)
    d = _delta_of_xF(xs)
    # Order index k = floor(delta/2pi); a change of k between neighbouring points marks a line centre
    k = np.floor(d / (2 * np.pi))
    idx = np.nonzero(np.diff(k) != 0)[0]
    centers = []
    for i in idx:
        target = 2 * np.pi * max(k[i], k[i + 1])
        centers.append(np.interp(target, [d[i], d[i + 1]] if d[i + 1] > d[i] else [d[i + 1], d[i]],
                                 [xs[i], xs[i + 1]] if d[i + 1] > d[i] else [xs[i + 1], xs[i]]))
    centers = np.array(sorted(centers, key=abs))[:N_ORDERS_MAX]
    return np.sort(centers), theta_div


def pmax_for_order(L_eff_mm, theta_i_deg, R2, theta_x0):
    """Returns the number of round trips for the bundle of this order: walk-off rate
    Delta_x + theta_x L_step (hard edge) or reflectivity decay, whichever is smaller (capped)."""
    import pupil_theory_v7 as _PT; th = np.deg2rad(theta_i_deg); h = _PT.h   # mirror spacing from the kernel module (set_h)
    rate = 2 * h * np.tan(th) + theta_x0 * 2 * h / np.cos(th)
    M_walk = (L_eff_mm * 1e-3) / rate if rate > 0.05 * 2 * h * np.tan(th) else np.inf
    decay = np.log(DECAY_EPS) / np.log(np.sqrt(R2))
    lim = min(M_walk * 1.1 + 10, decay * PMAX_SAFETY_MARGIN + PMAX_MIN_EXTRA)
    return int(min(np.ceil(lim), PMAX_HARD_CAP))


def measure_orders_v6(R2, L_eff_mm, theta_i_deg, alpha, surf, mask, xg, blende_mm,
                      save_all_spectra=False):
    """Measures all orders of one grid point, one ray bundle per order (current main path).
    Returns (res, frag_info, raw, theta_x per order, pmax per order, number of centres);
    res maps the six method names to lists of finesse values (local order spacing / FWHM)."""
    centers, theta_div = order_centers(theta_i_deg, blende_mm)
    import raytrace_2d as rt2d
    res = dict(adaptive=[], halfmax=[], rms=[], flankfit=[], eqwidth=[], participation=[])
    frag_info, raw, thx_list, pmax_list = [], [] if save_all_spectra else None, [], []
    for xc in centers:
        thx0 = np.arctan(xc / rt2d.F)
        sp = float(_local_fsr(xc))
        _, n_pts = estimate_window_and_resolution(R2, sp)
        # Window of +- one order spacing around the centre (twice the points), cut below to one period
        xw = np.linspace(xc - sp, xc + sp, 2 * n_pts)
        pm = pmax_for_order(L_eff_mm, theta_i_deg, R2, thx0)
        I = raytrace_order(xg, surf, mask, blende_mm, xw, pm, alpha, thx0)
        if I.max() < 1e-12:
            continue
        drel = _delta_of_xF(xw) - _delta_of_xF(np.array([xc]))[0]
        dc = np.angle(np.sum(I * np.exp(1j * drel)))          # circular centroid in delta
        sel = np.abs(drel - dc) <= np.pi                       # exactly one period
        xs, Is = xw[sel], I[sel]
        if len(xs) < 50:
            continue
        sp_c = float(_local_fsr(xs[np.argmin(np.abs(drel[sel] - dc))]))
        for key, fn in (("adaptive", rms_fwhm), ("halfmax", halfmax_fwhm), ("rms", pure_rms_fwhm),
                        ("flankfit", flank_fit_fwhm), ("eqwidth", equivalent_width_fwhm),
                        ("participation", participation_ratio_fwhm)):
            fw = fn(xs, Is)
            if fw is not None and not np.isnan(fw) and fw > 0:
                res[key].append(sp_c / fw)
        from scipy.signal import find_peaks as _fp
        pk, props = _fp(Is, height=Is.max() * 0.4)
        second = float(np.sort(props["peak_heights"])[::-1][1] / Is.max()) if len(pk) >= 2 else 0.0
        frag_info.append((len(pk), second))
        thx_list.append(thx0); pmax_list.append(pm)
        if save_all_spectra:
            raw.append((xs.astype(np.float32), Is.astype(np.float32)))
    return res, frag_info, raw, np.array(thx_list), np.array(pmax_list), len(centers)


def process_one_point(args):
    """Worker: computes one grid point / realisation and saves results_v4/<key>.npz.
    args = (R2, L_eff_mm, theta_i_deg, alpha, sigma_nm, wl_nm, blende_mm, real_idx).
    Returns (key, message)."""
    R2, L_eff_mm, theta_i_deg, alpha, sigma_nm, wl_nm, blende_mm, real_idx = args
    key = (f"R2{R2}_Leff{L_eff_mm}_th{theta_i_deg}_a{alpha}_s{sigma_nm}"
           f"_wl{wl_nm}_bl{blende_mm}_real{real_idx}")
    out_file = os.path.join(RESULTS_DIR, f"{key}.npz")
    if os.path.exists(out_file):
        # Only an existing version-7 file counts as finished (it is then added to the
        # log); files of earlier versions are recomputed and overwritten.
        try:
            with np.load(out_file, allow_pickle=True) as _z:
                _is_cur = "version" in _z.files and int(_z["version"]) == 7
        except Exception:
            _is_cur = False
        if _is_cur:
            return key, f"FERTIG (v7-Datei bereits vorhanden, im Log nachgetragen): {key}"

    import raytrace_2d as rt2d

    # Configure the ray-tracing module (module globals) for this point
    wl = wl_nm * 1e-9
    rt2d.wavelength = wl
    rt2d.k0 = 2 * np.pi / wl
    rt2d.R2 = R2
    rt2d.r2 = np.sqrt(R2)

    # --- set theta_i consistently in all four places ---
    rt2d.tilt = theta_i_deg
    rt2d.theta = np.deg2rad(theta_i_deg)
    rt2d.Delta_x = 2 * rt2d.h * np.tan(rt2d.theta)
    rt2d.L_step = 2 * rt2d.h / np.cos(rt2d.theta)

    # --- set L_eff (physical mirror length) consistently ---
    L_eff = L_eff_mm * 1e-3
    R_mirror = L_eff / 2
    rt2d.L_eff = L_eff
    rt2d.R_mirror = R_mirror

    M_eff = L_eff / rt2d.Delta_x

    R_surf = surface_radius(L_eff_mm)       # Zernike disk encloses the illuminated strip
    xg = np.linspace(-R_surf, R_surf, GRID_N)

    # Reproducible seed per realisation index (same surface for all parameter combinations)
    seed = 5000 + real_idx
    rng = np.random.default_rng(seed)
    sigma_w = sigma_nm * 1e-9
    if sigma_nm > 0:
        xg_s, surf, mask, _ = build_surface_no_astig(R_surf, sigma_w, rng, N_TERMS, GRID_N)
    else:
        surf = np.zeros((GRID_N, GRID_N))
        mask = np.ones((GRID_N, GRID_N), dtype=bool)

    t0 = time.time()
    res, frag_info, raw_spectra, thx_orders, pmax_orders, n_centers = measure_orders_v6(
        R2, L_eff_mm, theta_i_deg, alpha, surf, mask, xg, blende_mm, save_all_spectra=SAVE_ALL_SPECTRA)
    F_adaptive, F_halfmax, F_rms = res["adaptive"], res["halfmax"], res["rms"]
    F_flankfit, F_eqwidth, F_participation = res["flankfit"], res["eqwidth"], res["participation"]
    diag = None
    positions = thx_orders
    pmax = int(pmax_orders.max()) if len(pmax_orders) else 0
    F_values = F_adaptive  # standard method, used for the aggregation

    # Fragmentation summary of this point (see docstring of measure_all_orders_combined):
    # n_peaks_per_order and second_ratio_per_order are stored raw (small: at most ~40 pairs
    # of numbers), so that ANY weighting rule can be defined per parameter combination in
    # the fit step without new simulation.
    if frag_info:
        n_peaks_arr = np.array([f[0] for f in frag_info], dtype=np.int16)
        second_ratio_arr = np.array([f[1] for f in frag_info], dtype=np.float32)
    else:
        n_peaks_arr = np.array([], dtype=np.int16)
        second_ratio_arr = np.array([], dtype=np.float32)
    frag_fraction = float(np.mean(n_peaks_arr >= 2)) if len(n_peaks_arr) else 0.0
    frag_severity = float(np.median(second_ratio_arr[n_peaks_arr >= 2])) if np.any(n_peaks_arr >= 2) else 0.0

    os.makedirs(RESULTS_DIR, exist_ok=True)
    save_kwargs = dict(R2=R2, L_eff_mm=L_eff_mm, theta_i_deg=theta_i_deg, alpha=alpha,
              sigma_nm=sigma_nm, wl_nm=wl_nm, blende_mm=blende_mm, real_idx=real_idx,
              seed=seed, pmax=pmax, M_eff=M_eff, F_values=F_values, n_orders=len(positions),
              # Multiple methods (see docstring of measure_all_orders_combined):
              # allow choosing the method afterwards WITHOUT new simulation.
              F_values_halfmax=F_halfmax, F_values_rms=F_rms, F_values_flankfit=F_flankfit,
              F_values_eqwidth=F_eqwidth, F_values_participation=F_participation,
              # Fragmentation diagnosis (raw + summary):
              n_peaks_per_order=n_peaks_arr, second_ratio_per_order=second_ratio_arr,
              frag_fraction=frag_fraction, frag_severity=frag_severity,
              theta_x_orders=thx_orders, pmax_orders=pmax_orders, version=7, R_surf=R_surf)
    if diag is not None:
        save_kwargs["diag_xF"] = diag[0]
        save_kwargs["diag_I"] = diag[1]
    if raw_spectra is not None and len(raw_spectra) > 0:
        # Full (or reduced to RAW_SPECTRA_MAX_POINTS) raw spectra of ALL orders of this
        # grid point, as object arrays of (xF, I) pairs of different length (orders can
        # have slightly different numbers of points because of the adaptive resolution /
        # edge effects). See the module docstring for the size estimate; loaded e.g. by
        # load_raw_spectra() in the aggregation script.
        #
        # IMPORTANT: np.array([...], dtype=object) would, for sub-arrays of EQUAL length
        # (common, since n_points mostly depends only on R2), make numpy build a regular
        # multi-dimensional object array and lose the float32 subtype of the single arrays
        # (~2x more storage). Fix: preallocate an empty object array and fill it element by
        # element, which reliably prevents this automatic detection.
        xF_list = np.empty(len(raw_spectra), dtype=object)
        I_list = np.empty(len(raw_spectra), dtype=object)
        for idx_s, (xF_s, I_s) in enumerate(raw_spectra):
            xF_list[idx_s] = xF_s
            I_list[idx_s] = I_s
        save_kwargs["raw_spectra_xF"] = xF_list
        save_kwargs["raw_spectra_I"] = I_list
    # Save atomically (temp file + rename) instead of writing directly to out_file, so that
    # move_to_external.py (or any other script watching RESULTS_DIR) never gets a half
    # written .npz file -- np.savez is not atomic, os.replace() on the same file system is.
    tmp_file = out_file + f".tmp{os.getpid()}.npz"
    # If disk space is low, wait until move_to_external.py has freed space, instead of
    # crashing with "disk full" and losing the result.
    import shutil as _sh
    _warned = False
    while _sh.disk_usage(RESULTS_DIR).free < MIN_FREE_GB * 1e9:
        if not _warned:
            print(f"[{key}] Weniger als {MIN_FREE_GB} GB frei -- warte auf move_to_external.py ...", flush=True)
            _warned = True
        time.sleep(30)
    for _attempt in range(20):
        try:
            np.savez(tmp_file, **save_kwargs)
            os.replace(tmp_file, out_file)
            break
        except OSError as _e:
            # e.g. disk full after all (race between several workers): remove the temp
            # file, wait, try again.
            try:
                if os.path.exists(tmp_file):
                    os.remove(tmp_file)
            except OSError:
                pass
            print(f"[{key}] Speichern fehlgeschlagen ({_e}), neuer Versuch in 60 s ...", flush=True)
            time.sleep(60)
    else:
        raise OSError(f"Speichern von {key} nach 20 Versuchen fehlgeschlagen")

    dt = time.time() - t0
    n_valid = len(F_values)
    med = np.median(F_values) if n_valid > 0 else np.nan
    msg = (f"FERTIG: {key} -> pmax={pmax} (M_eff={M_eff:.0f}), {n_valid} Ordnungen, "
            f"Median F={med:.2f} ({dt:.1f}s)")
    return key, msg


def main():
    """Builds the task list of the full grid, removes points already in the checkpoint log
    and runs the remaining points in a process pool; every finished key is appended to the log."""
    # Without argument N_real is taken from the checkpoint log (highest real index + 1),
    # so that a restart without "40" does not silently compute a different grid (N_real=5).
    _log_nreal = None
    done_keys = _read_done_keys()
    if done_keys:
        _mx = -1
        for _l in done_keys:
            if "_real" in _l:
                try:
                    _mx = max(_mx, int(_l.rsplit("_real", 1)[1]))
                except ValueError:
                    pass
        _log_nreal = _mx + 1 if _mx >= 0 else None
    write_log = CHECKPOINT_LOG   # one process, one log
    if len(sys.argv) > 1:
        N_real = int(sys.argv[1])
        if _log_nreal is not None and _log_nreal > N_real:
            print(f"WARNUNG: Das Checkpoint-Log enthaelt Realisierungen bis real{_log_nreal-1}, "
                  f"angegeben ist N_real={N_real}. Absicht?", flush=True)
    else:
        N_real = _log_nreal if _log_nreal else N_REAL_DEFAULT
        print(f"N_real = {N_real} ({'aus dem Checkpoint-Log' if _log_nreal else 'Standardwert'})", flush=True)
    print(f"Ergebnisordner: {RESULTS_DIR}\nCheckpoint-Log (schreiben): {write_log}\n"
          f"Checkpoint-Logs (lesen): {[os.path.basename(f) for f in _all_checkpoint_logs()]}", flush=True)
    # Remove temp files left over from an interrupted run (older than 2 hours, as a
    # safety margin).
    if os.path.isdir(RESULTS_DIR):
        _n = 0; _now = time.time()
        for _e in os.scandir(RESULTS_DIR):
            if ".tmp" in _e.name:
                try:
                    if _now - _e.stat().st_mtime > 7200:
                        os.remove(_e.path); _n += 1
                except OSError:
                    pass
        if _n:
            print(f"{_n} verwaiste Temp-Dateien aus einem abgebrochenen Lauf entfernt.", flush=True)
    n_proc = int(sys.argv[2]) if len(sys.argv) > 2 else cpu_count()
    if sys.platform == "win32" and n_proc > 61:
        # multiprocessing on Windows: at most 61 workers (WaitForMultipleObjects limit)
        print(f"Windows: Worker-Zahl von {n_proc} auf 61 begrenzt.", flush=True)
        n_proc = 61
    # Range of realisations computed by this process. To split over several computers,
    # real_start/real_end can be set to a part of the realisations (e.g. computer A 0..3,
    # computer B 3..5); copy the result folders together afterwards and run the aggregation
    # ONCE. sigma_w=0 needs only real_idx=0 (deterministic) -- compute it on only ONE of
    # the computers (include_sigma0) to avoid duplicate work.
    real_start, real_end, include_sigma0 = 0, N_real, True

    tasks = []
    for R2 in R2_VALUES:
        for L_eff_mm in L_EFF_VALUES_MM:
            for theta_i in THETA_I_VALUES_DEG:
                for alpha in ALPHA_VALUES:
                    for sigma_nm in SIGMA_VALUES_NM:
                        for wl_nm in WL_VALUES_NM:
                            for blende_mm in BLENDE_VALUES_MM:
                                if sigma_nm == 0.0:
                                    if not include_sigma0:
                                        continue
                                    real_range = [0]
                                else:
                                    real_range = range(real_start, min(real_end, N_real))
                                for real_idx in real_range:
                                    tasks.append((R2, L_eff_mm, theta_i, alpha,
                                                  sigma_nm, wl_nm, blende_mm, real_idx))

    # Load the checkpoint log and remove finished points from the task list. This is the
    # PRIMARY resume mechanism (not the file system check in process_one_point) -- it also
    # works when finished .npz files have meanwhile been moved to an external disk and
    # deleted locally (see move_to_external.py). The key is formatted exactly as in
    # process_one_point to guarantee consistency.
    def task_key(t):
        R2, L_eff_mm, theta_i, alpha, sigma_nm, wl_nm, blende_mm, real_idx = t
        return (f"R2{R2}_Leff{L_eff_mm}_th{theta_i}_a{alpha}_s{sigma_nm}"
                f"_wl{wl_nm}_bl{blende_mm}_real{real_idx}")

    if done_keys:
        n_before = len(tasks)
        tasks = [t for t in tasks if task_key(t) not in done_keys]
        print(f"Checkpoint-Logs gefunden: {len(done_keys)} "
              f"Punkte bereits erledigt, {n_before - len(tasks)} davon aus "
              f"dieser Task-Liste entfernt. Verbleibend: {len(tasks)}.")

    print(f"Vollstaendiges 7-Parameter-Raster: {len(tasks)} Simulationsaufrufe "
          f"(Realisierungen {real_start}..{min(real_end,N_real)-1} von {N_real}, "
          f"{n_proc} Prozesse parallel).")
    print(f"R2: {R2_VALUES}")
    print(f"L_eff [mm]: {L_EFF_VALUES_MM}")
    print(f"theta_i [deg]: {THETA_I_VALUES_DEG}")
    print(f"alpha [\"]: {ALPHA_VALUES}")
    print(f"sigma_w [nm]: {SIGMA_VALUES_NM}")
    print(f"lambda [nm]: {WL_VALUES_NM}")
    print(f"Blende [mm]: {BLENDE_VALUES_MM}")
    print("p_max: automatisch pro Punkt (siehe compute_pmax), Hard-Cap="
          f"{PMAX_HARD_CAP}")
    print()

    if not tasks:
        print("Nichts zu tun -- alle Punkte laut Checkpoint-Log bereits fertig.")
        return

    t_start = time.time()
    # Keep the checkpoint log open in append mode; after EVERY finished point write one
    # line and flush (+ os.fsync, so that the entry is safely on disk even after a hard
    # abort/power failure, not only in the OS buffer). The order of the lines follows the
    # completion order (imap_unordered), not the task order -- irrelevant for resuming,
    # since the keys are read into a set.
    with open(write_log, "a") as logf, Pool(n_proc) as pool:
        for i, (key, result) in enumerate(pool.imap_unordered(process_one_point, tasks)):
            if not result.startswith("UEBERSPRUNGEN"):
                logf.write(key + "\n")
                logf.flush()
                os.fsync(logf.fileno())
            elapsed = time.time() - t_start
            if (i + 1) % 100 == 0 or (i + 1) == len(tasks):
                remaining = (len(tasks) - (i + 1)) * (elapsed / (i + 1)) / 3600 if i > 0 else 0
                print(f"[{i+1}/{len(tasks)}] ({elapsed/3600:.2f}h, ~{remaining:.1f}h "
                      f"verbleibend) {result}")

    print(f"\nAlles fertig nach {(time.time()-t_start)/3600:.2f} Stunden.")
    print("Fuehre jetzt 'python aggregate_full_grid_v2.py' aus (nach dem "
          "Zusammenkopieren der Ergebnisordner, falls auf mehreren Rechnern gerechnet).")


if __name__ == "__main__":
    main()
