"""Full 2D ray tracing through a VIPA plate with surface errors on HR, PR and back side (validation of the theory).

Purpose
    Independent ray-tracing model used to validate the wave-optical theory. A Gaussian input beam focused by a
    cylindrical lens is split into n_sub x n_sub sub-rays (in x AND y). Each sub-ray runs through pmax+1 round trips
    between the HR and PR mirrors; at every reflection it picks up the phase 2 k0 h of the local surface height and
    is deflected by twice the local slopes (dh/dx, dh/dy). The far field in the focal plane of the output lens is the
    coherent sum over round trips and sub-rays of r^m times the geometric phase of order m plus the accumulated
    surface phase. Finally the Hu envelope (Hu et al., Eq. 10) is applied.

Fixed parameters (module level, SI units)
    wavelength 630 nm, R^2 = 0.90, mirror spacing h = 2 mm, tilt 4 deg, focal lengths F = f_cyl = 75 mm,
    effective plate length L_eff = 25 mm (mirror radius R_mirror = L_eff/2), beam waist w0x = 37.5 um, pmax = 80.

Functions
    get_fwhm                 FWHM of a sampled curve (after light smoothing)
    geometric_phase_nominal  nominal (error-free) phase of order m at the focal-plane positions xF
    raytrace_2d              intensity in the focal plane for given HR, PR and back-side surfaces

How to run
    python3 raytrace_2d.py   (demo: one random Zernike HR surface with sigma = lambda/10; prints run time and FWHM)
    Otherwise imported as a library.

"""
import numpy as np
import time

wavelength = 630e-9
n=1; R2=0.90; t0=2e-3; tilt=4.0; d0=0.05; F=0.075; zF=F
pmax = 80
h = 2.0e-3; theta = np.deg2rad(4.0)
Delta_x = 2*h*np.tan(theta)
L_step = 2*h/np.cos(theta)
w0x = 37.5e-6
f_cyl = 75e-3
L_eff = 25e-3
R_mirror = L_eff/2
r2 = np.sqrt(R2)
k0 = 2*np.pi/wavelength
# Collimated beam width in y behind the cylindrical lens (Gaussian beam: lambda f / (pi w0)).
W_y = wavelength*f_cyl/(np.pi*w0x)

def get_fwhm(x, I, smooth_frac=0.005):
    """Return the FWHM of I(x) after a moving-average smoothing over smooth_frac of the samples (NaN if no crossing)."""
    from scipy.ndimage import uniform_filter1d
    smooth_window = max(3, int(len(x)*smooth_frac))
    I_s = uniform_filter1d(I, size=smooth_window)
    imax = I_s.max(); half = imax/2
    idx_peak = np.argmax(I_s)
    # Walk outward from the peak to the half-maximum crossings, then interpolate linearly.
    left = idx_peak
    while left>0 and I_s[left]>half: left -= 1
    right = idx_peak
    while right<len(I_s)-1 and I_s[right]>half: right += 1
    if left==idx_peak or right==idx_peak: return np.nan
    xl = np.interp(half, [I_s[left],I_s[left+1]], [x[left],x[left+1]])
    xr = np.interp(half, [I_s[right],I_s[right-1]], [x[right],x[right-1]])
    return xr-xl

def geometric_phase_nominal(m, xF_arr):
    """Return the nominal phase k0 * (optical path) of order m at the focal-plane positions xF_arr (no surface errors).

    Order m is a virtual source shifted by voffset (transverse) and zoffset (along the axis) relative to order 0;
    d0 is the distance of the beam waist from the lens, zF the observation plane.
    """
    if m==0:
        return -1/(2*zF)*d0*xF_arr**2/(zF+d0*(1-zF/F))*k0
    voffset = 2*m*t0*np.sin(np.radians(tilt))
    zoffset = 2*m*t0*np.cos(np.radians(tilt))
    denom = zF + (d0+zoffset)*(1-zF/F)
    term = (zoffset + 0.5*(1-zF/F)*voffset**2/denom
            - voffset*xF_arr/denom
            - 1/(2*zF)*(d0+zoffset)*xF_arr**2/denom)
    return k0*term

def raytrace_2d(xg_hr, yg_hr, surf_hr, mask_hr, xg_pr, yg_pr, surf_pr, mask_pr,
                 xg_bk, yg_bk, surf_bk, mask_bk, blende_mm, xF, n_sub=9, pmax=pmax):
    """Full 2D ray tracing: sub-rays in x AND y, with angular deflection in both directions by the local slopes
    (dh/dx, dh/dy).

    Inputs: grids (xg, yg), height maps surf [m] and masks for the HR surface (_hr), the PR surface (_pr) and the
    back side (_bk); aperture blende_mm [mm] (sets the angular spread of the sub-rays); focal-plane positions xF [m];
    n_sub sub-rays per direction; pmax highest round trip.
    Returns the intensity I(xF) in the focal plane including the Hu envelope.
    """
    dx_hr = xg_hr[1]-xg_hr[0]
    surf_hr_m = np.where(mask_hr, surf_hr, 0.0)
    # np.gradient returns (d/drow, d/dcolumn) = (y, x) order!
    slope_hr_x, slope_hr_y = np.gradient(surf_hr_m, dx_hr)
    # correct it: np.gradient(Z) -> (dZ/daxis0, dZ/daxis1) = (dZ/dy, dZ/dx)
    slope_hr_x, slope_hr_y = slope_hr_y, slope_hr_x

    dx_pr = xg_pr[1]-xg_pr[0]
    surf_pr_m = np.where(mask_pr, surf_pr, 0.0)
    g0,g1 = np.gradient(surf_pr_m, dx_pr); slope_pr_x, slope_pr_y = g1, g0

    dx_bk = xg_bk[1]-xg_bk[0]
    surf_bk_m = np.where(mask_bk, surf_bk, 0.0)
    g0,g1 = np.gradient(surf_bk_m, dx_bk); slope_bk_x, slope_bk_y = g1, g0

    def interp2d(x,y,xg,yg,Z):
        """Return the bilinear interpolation of Z (indexed [y, x]) at the point (x, y), clamped to the grid."""
        ix = np.clip(np.searchsorted(xg,x)-1, 0, len(xg)-2)
        iy = np.clip(np.searchsorted(yg,y)-1, 0, len(yg)-2)
        fx = (x-xg[ix])/(xg[ix+1]-xg[ix])
        fy = (y-yg[iy])/(yg[iy+1]-yg[iy])
        return (Z[iy,ix]*(1-fx)*(1-fy) + Z[iy,ix+1]*fx*(1-fy)
                + Z[iy+1,ix]*(1-fx)*fy + Z[iy+1,ix+1]*fx*fy)

    # Sub-rays: Gaussian-weighted sample points xi in [-2.5, 2.5] (units of the beam radius) in x and y;
    # the start angle in x spans the divergence set by the aperture d over f_cyl.
    d = blende_mm*1e-3
    theta_div = np.arctan(d/f_cyl)
    xi0 = np.linspace(-2.5,2.5,n_sub)
    w1d = np.exp(-xi0**2/2)
    XI_X, XI_Y = np.meshgrid(xi0, xi0)
    W2D = np.outer(w1d,w1d); W2D/=W2D.sum()

    x_start = (-R_mirror + XI_X*w0x).ravel()
    y_start = (XI_Y*W_y).ravel()
    thx_start = (XI_X*theta_div/2.5).ravel()
    weights = W2D.ravel()
    n_rays = len(weights)

    m_arr = np.arange(pmax+1)
    Ereal = np.zeros(len(xF)); Eimag = np.zeros(len(xF))

    # Coherent sum of all sub-rays over all round trips.
    for ridx in range(n_rays):
        x = x_start[ridx]; y = y_start[ridx]
        thx = thx_start[ridx]; thy = 0.0
        acc_phase = 0.0
        for m in m_arr:
            h_hr = interp2d(x,y,xg_hr,xg_hr,surf_hr_m)
            sx_hr = interp2d(x,y,xg_hr,xg_hr,slope_hr_x)
            sy_hr = interp2d(x,y,xg_hr,xg_hr,slope_hr_y)
            h_pr = interp2d(x,y,xg_pr,xg_pr,surf_pr_m)
            sx_pr = interp2d(x,y,xg_pr,xg_pr,slope_pr_x)
            sy_pr = interp2d(x,y,xg_pr,xg_pr,slope_pr_y)
            h_bk = interp2d(x,y,xg_bk,xg_bk,surf_bk_m)

            # HR + PR reflection: accumulated, repeats on every round trip
            acc_phase += k0*2*(h_hr+h_pr)

            phase_geom = geometric_phase_nominal(m, xF)
            # Back-side transmission (glass index 1.5): ONLY for THIS order, not accumulated
            phase_backside = k0*(1.5-1)*h_bk
            total_phase = phase_geom + acc_phase + phase_backside
            # Amplitude decays by r = sqrt(R^2) per round trip.
            amp = weights[ridx]*r2**m
            Ereal += amp*np.cos(total_phase)
            Eimag += amp*np.sin(total_phase)

            # Reflection deflects the ray by twice the local slope; then propagate one round trip (step Delta_x in x).
            thx = thx + 2*(sx_hr+sx_pr)
            thy = thy + 2*(sy_hr+sy_pr)
            x = x + Delta_x + thx*L_step
            y = y + thy*L_step

    I_raw = Ereal**2+Eimag**2
    # Apply the envelope (Hu, Eq. 10) afterwards -- W = aperture radius
    # d = blende_mm*1e-3, defined above
    W_envelope = d/2
    theta_x_field = np.arctan(xF/F)
    envelope = np.exp(-2*f_cyl**2/W_envelope**2*np.sin(theta_x_field)**2)
    return I_raw*envelope

if __name__=="__main__":
    # Demo: random Zernike HR surface with RMS lambda/10, flat PR and back side, aperture 3.5 mm.
    from zernike_hr_surface import build_zernike_surface
    xF = np.linspace(-0.1e-3,0.1e-3,2000)
    sigma_hr = wavelength/10
    rng = np.random.default_rng(1)
    xg,surf,mask,_ = build_zernike_surface(R_mirror, sigma_hr, rng, grid_n=201)

    t0=time.time()
    I = raytrace_2d(xg,xg,surf,mask, xg,xg,surf*0,mask, xg,xg,surf*0,mask, 3.5, xF, n_sub=9)
    t1=time.time()
    print(f'Eine 2D-Simulation (n_sub=9 -> {9*9} Unterstrahlen): {t1-t0:.2f}s')
    print(f'FWHM (nur HR, PR=0): {get_fwhm(xF,I)*1e6:.4f}um')
