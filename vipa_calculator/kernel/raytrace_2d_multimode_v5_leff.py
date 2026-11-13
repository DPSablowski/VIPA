"""Multimode 2D ray tracer of the VIPA with physical cut-off at the mirror edge.

Purpose: sums the round trips of a bundle of rays (sub-apertures of the input spot and of the beam across the chord)
coherently and returns the detector intensity I(xF). Surface errors of the HR and PR coatings enter as phase and as
deflection (slope) at every reflection; the back surface enters as a phase only. A wedge alpha changes both the phase
and the ray direction per reflection (law of reflection on a tilted mirror, as in Teav, Jean-Ruel & Steinberg,
Appl. Opt. 64, 7834 (2025)).

Edge cut-off: a ray whose position x leaves the mirror [-R_mirror, +R_mirror] no longer contributes from that round
trip on (amplitude 0 for all following round trips). This implements the effective plate length L_eff, i.e. the
number of round trips M_eff = L_eff/Delta_x (cf. Hu et al. 2015, Eq. 10). Without it only the parameter pmax would
limit the sum.

Inputs: surface maps on regular grids (HR, PR, back side), masks, beam width, detector positions; geometry constants
(k0, theta, Delta_x, L_step, F, f_cyl, r2, R_mirror, pmax) come from the module raytrace_2d.

Used by run_full_grid_v7.py, which reads this file as source text, removes the field-of-view envelope at the end of
the function and compiles it as _raytrace_v5_noenv. Keep this file next to run_full_grid_v7.py. The ray tracing is
used in the main document to validate the theory (ray-tracing grids, sec. on validation).
"""
import numpy as np
import raytrace_2d as rt2d

# diameter of the input spot (image of the fibre core) on the plate, in m
d_core_image = 75e-6


def interp2d_vec(x, y, xg, yg, Z):
    """Bilinear interpolation of the map Z (rows = y, columns = x) at the points (x, y); returns an array.
    Indices are clipped to the grid, so points outside get the edge value."""
    ix = np.clip(np.searchsorted(xg, x) - 1, 0, len(xg) - 2)
    iy = np.clip(np.searchsorted(yg, y) - 1, 0, len(yg) - 2)
    fx = (x - xg[ix]) / (xg[ix + 1] - xg[ix])
    fy = (y - yg[iy]) / (yg[iy + 1] - yg[iy])
    return (Z[iy, ix] * (1 - fx) * (1 - fy) + Z[iy, ix + 1] * fx * (1 - fy)
            + Z[iy + 1, ix] * (1 - fx) * fy + Z[iy + 1, ix + 1] * fx * fy)


def raytrace_2d_multimode_v5_leff(xg_hr, yg_hr, surf_hr, mask_hr, xg_pr, yg_pr, surf_pr, mask_pr,
                                    xg_bk, yg_bk, surf_bk, mask_bk, blende_mm, xF, n_sub=13, pmax=None,
                                    alpha_arcsec=0.0, enforce_Leff_cutoff=True):
    """Returns the detector intensity I(xF) (array over xF) of the coherent round-trip sum.

    Arguments: grids, maps and masks of the HR surface, the PR surface and the back surface (heights in m);
    blende_mm = beam width across the chord in mm; xF = detector positions in m; n_sub = sub-apertures per
    direction of the input spot; pmax = number of round trips (default rt2d.pmax); alpha_arcsec = wedge angle;
    enforce_Leff_cutoff = stop rays that leave the mirror (default True).

    The wedge enters twice per round trip: as a phase k0*2*alpha*x*cos(theta) and as a change 2*alpha*cos(theta) of
    the ray direction, so wedge and surface slopes follow the same law of reflection. Checked with sigma_w = 0 and
    alpha = 0.3''-8'': the direction term changes the measured finesse by less than 1 %. At the end the spectrum is
    multiplied by the field-of-view envelope |theta_x| <= theta_div (removed by run_full_grid_v7.py)."""
    alpha_rad = alpha_arcsec / 3600 * np.pi / 180
    if pmax is None:
        pmax = rt2d.pmax
    # surface slopes (gradients) of HR and PR maps: they deflect the rays at each reflection
    dx_hr = xg_hr[1] - xg_hr[0]
    surf_hr_m = np.where(mask_hr, surf_hr, 0.0)
    g0, g1 = np.gradient(surf_hr_m, dx_hr); slope_hr_x, slope_hr_y = g1, g0
    dx_pr = xg_pr[1] - xg_pr[0]
    surf_pr_m = np.where(mask_pr, surf_pr, 0.0)
    g0, g1 = np.gradient(surf_pr_m, dx_pr); slope_pr_x, slope_pr_y = g1, g0
    dx_bk = xg_bk[1] - xg_bk[0]
    surf_bk_m = np.where(mask_bk, surf_bk, 0.0)

    d_blende = blende_mm * 1e-3
    # half divergence angle behind the cylinder lens (beam width / 2 over f_cyl)
    theta_div = np.arctan(d_blende / (2 * rt2d.f_cyl))

    # sub-aperture grid of the input spot: n_sub x n_sub points, only those inside the unit circle, equal weights
    xi0 = np.linspace(-1, 1, n_sub)
    XI_X, XI_Y = np.meshgrid(xi0, xi0)
    circle_mask = (XI_X**2 + XI_Y**2) <= 1.0
    weights_full = circle_mask.astype(float).ravel()
    active = weights_full > 0
    weights = weights_full[active]
    weights /= weights.sum()

    # Place the input spot completely on the mirror (from -R_mirror to -R_mirror + d_core_image) instead of centring it
    # on the edge; otherwise the alive check at m = 0 would remove all rays with XI_X < 0 (negative thx) and the input
    # cone would become one-sided.
    x = (-rt2d.R_mirror + d_core_image / 2 + XI_X * (d_core_image / 2)).ravel()[active]
    y = (XI_Y * (d_blende / 2)).ravel()[active]
    thx = (XI_X * theta_div).ravel()[active]
    thy = np.zeros_like(thx)
    n_rays = len(weights)

    alive = np.ones(n_rays, dtype=bool)  # True while the ray is still on the mirror

    m_arr = np.arange(pmax + 1)
    Ereal = np.zeros(len(xF)); Eimag = np.zeros(len(xF))
    acc_phase = np.zeros(n_rays)

    n_walked_off_at = None

    # loop over round trips m: accumulate phase, add the field of this round trip, then propagate the rays
    for m in m_arr:
        if enforce_Leff_cutoff:
            alive = alive & (np.abs(x) <= rt2d.R_mirror)
            if n_walked_off_at is None and not alive.any():
                n_walked_off_at = m

        h_hr = interp2d_vec(x, y, xg_hr, xg_hr, surf_hr_m)
        sx_hr = interp2d_vec(x, y, xg_hr, xg_hr, slope_hr_x)
        sy_hr = interp2d_vec(x, y, xg_hr, xg_hr, slope_hr_y)
        h_pr = interp2d_vec(x, y, xg_pr, xg_pr, surf_pr_m)
        sx_pr = interp2d_vec(x, y, xg_pr, xg_pr, slope_pr_x)
        sy_pr = interp2d_vec(x, y, xg_pr, xg_pr, slope_pr_y)
        h_bk = interp2d_vec(x, y, xg_bk, xg_bk, surf_bk_m)

        # accumulated phase per ray: double pass over the height errors of HR and PR, plus the wedge phase
        acc_phase = acc_phase + rt2d.k0 * 2 * (h_hr + h_pr)
        acc_phase = acc_phase + rt2d.k0 * 2 * alpha_rad * x * np.cos(rt2d.theta)

        # nominal geometric phase of round trip m at the detector; back surface enters with (n - 1) = 0.5
        phase_geom = rt2d.geometric_phase_nominal(m, xF)
        phase_backside = rt2d.k0 * (1.5 - 1) * h_bk
        phase_per_ray = acc_phase + phase_backside
        amp = weights * rt2d.r2**m
        if enforce_Leff_cutoff:
            amp = amp * alive  # rays that have left the mirror no longer contribute

        cos_g = np.cos(phase_geom); sin_g = np.sin(phase_geom)
        cos_r = np.cos(phase_per_ray); sin_r = np.sin(phase_per_ray)
        # the ray phases are summed first (A + iB), then multiplied with the geometric phase factor
        A = np.sum(amp * cos_r)
        B = np.sum(amp * sin_r)

        Ereal += A * cos_g - B * sin_g
        Eimag += A * sin_g + B * cos_g

        thx = thx + 2 * (sx_hr + sx_pr) + 2 * alpha_rad * np.cos(rt2d.theta)  # wedge: change of ray direction per round trip
        thy = thy + 2 * (sy_hr + sy_pr)
        # propagate to the next reflection: nominal step Delta_x plus the walk from the ray angle
        x = x + rt2d.Delta_x + thx * rt2d.L_step
        y = y + thy * rt2d.L_step

    I_raw = Ereal**2 + Eimag**2
    # field-of-view envelope (removed by run_full_grid_v7.py, which measures on the raw sum)
    theta_x_field = np.arctan(xF / rt2d.F)
    envelope = (np.abs(theta_x_field) <= theta_div).astype(float)
    return I_raw * envelope
