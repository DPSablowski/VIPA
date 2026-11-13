"""flaeche_dreiband.py -- three-band surface model for the theory grid.

Height h(x, y) = figure + mid band + fine band, all bands independent:
  figure    : Zernike terms of radial degrees 2..4 (defocus, astigmatism, coma, trefoil, spherical, ...; 12 terms,
              equal variance per term, scaled to RMS sigma_fig over the disk of diameter D_m); evaluated analytically
              at the hit point.
  mid band  : stationary, isotropic random field with power spectrum S2(f) ~ f^(-alpha_s) for 1/D_m <= f <= f_max
              (FFT synthesis on a periodic grid of spacing dg; each realization scaled exactly to RMS sigma_mid),
              interpolated bilinearly at the hit point.
  fine band : white noise on the grid (periods < 2 dg, i.e. far below the round-trip offset), RMS sigma_fein;
              at the hit point the nearest grid value is used (no interpolation, which would average the variance).
D_m is the diameter over which the RMS values are defined (default in the grid: the Zernike disk 2 R_s); sigma_mid is
the RMS over the central disk D_m of each realization (the measurable quantity). The box is 2 D_m wide so that the
modes near 1/D_m are resolved.

Two conventions for the mid band (module switch MID_PROJECTED, or the argument project_figure):
  False (grids up to 22-10-2026): the mid band is the stationary power-law field as such. Its low-frequency end then
        carries variance in the Zernike terms of degrees 0-4 (about 46 % for alpha_s = 3 on D_m), which a Zernike fit of
        a measured map books as figure; extracting the parameters from a map therefore needs the projection correction
        of Part I, Sec. 3B (Noll's integral).
  True  (procedure-based definition): the Zernike terms of degrees 0-4 (piston, tilt, figure) are fitted on the disk
        D_m and removed from the mid band, which is then scaled to RMS sigma_mid over the disk. Figure and mid band are
        then exactly what a least-squares Zernike fit of a height map returns (figure = fit of degrees 2-4, mid band =
        residual rms less the fine band), and no projection correction is needed. The removed polynomial is subtracted
        on the whole box (the plate lies inside the disk).
Units: metres. Seeds from (seed, pidx, surface index): equal seeds give equal surfaces.

Contents: class Flaeche (full box), noll_share and measured_to_stationary (conversion between the two conventions), psd_mid (normalized mid-band PSD for the spectral integrals of the characteristic
quantities), class FlaecheStreifen (memory-saving variant for long plates). Library module, no main program.

"""
import numpy as np
from zernike_hr_surface import zernike

# figure: the 12 Zernike terms (n, m) of radial degrees 2..4
FIG_TERMS = [(2, 0), (2, 2), (2, -2), (3, 1), (3, -1), (3, 3), (3, -3), (4, 0), (4, 2), (4, -2), (4, 4), (4, -4)]
DG_DEFAULT = 25e-6          # grid spacing of mid and fine band [m]
FMAX_DEFAULT = 1 / 0.1e-3   # upper limit of the mid band: period 0.1 mm
MID_PROJECTED = False       # default convention of the mid band (see module docstring); True = figure-free (procedure-based) mid band
PROJ_TERMS = [(0, 0), (1, 1), (1, -1)] + FIG_TERMS   # terms removed from the mid band when projected: piston, tilt, figure


def _remove_zernike(h, xg, yg, D_m, terms=PROJ_TERMS):
    """Fit the Zernike terms on the disk of diameter D_m (least squares on the grid points inside) and subtract the
    fitted polynomial on the whole grid. Returns (h_residual, coefficients)."""
    X, Y = np.meshgrid(xg, yg); rho = np.sqrt(X ** 2 + Y ** 2) / (D_m / 2); phi = np.arctan2(Y, X); disk = rho <= 1
    Z = np.array([zernike(n, m, rho[disk], phi[disk]) for n, m in terms]).T
    c = np.linalg.lstsq(Z, h[disk], rcond=None)[0]
    for ci, (n, m) in zip(c, terms):
        h = h - ci * zernike(n, m, rho, phi)
    return h, c


class Flaeche:
    """One random three-band surface on a box Lx x Ly [m] centred at the origin; height via h(x, y).

    rng = numpy Generator; D_m = reference diameter; sigma_fig, sigma_mid, sigma_fein = RMS of the three bands [m]
    (0 switches a band off); alpha_s = spectral exponent of the mid band; dg = grid spacing; f_max = upper frequency
    of the mid band [1/m].
    """
    def __init__(self, rng, D_m, sigma_fig, sigma_mid, alpha_s, sigma_fein, Lx, Ly, dg=DG_DEFAULT, f_max=FMAX_DEFAULT, project_figure=None):
        self.D_m = D_m; self.dg = dg; self.c_fig = None
        self.projected = MID_PROJECTED if project_figure is None else bool(project_figure)
        if sigma_fig > 0:
            c = rng.normal(size=len(FIG_TERMS)); self.c_fig = c / np.sqrt(np.sum(c ** 2)) * sigma_fig   # Noll normalization: mean square of each term = 1 on the disk, so sum c^2 = sigma_fig^2
        nx = int(np.ceil(Lx / dg)); ny = int(np.ceil(Ly / dg))
        self.x0 = -Lx / 2; self.y0 = -Ly / 2; self.nx = nx; self.ny = ny; self.mid = None; self.fein = None
        if sigma_mid > 0:
            # FFT synthesis: complex white noise times amplitude sqrt(S2) = f^(-alpha_s/2) inside the band; float32 to save memory
            fx = np.fft.fftfreq(nx, dg).astype(np.float32); fy = np.fft.fftfreq(ny, dg).astype(np.float32); F = np.sqrt(fx[None, :] ** 2 + fy[:, None] ** 2)
            A = np.zeros_like(F); band = (F >= 1 / D_m) & (F <= f_max); A[band] = F[band] ** (-alpha_s / 2); del F, band
            Z = (rng.normal(size=(ny, nx)) + 1j * rng.normal(size=(ny, nx))).astype(np.complex64) * A; del A
            h = np.real(np.fft.ifft2(Z)).astype(np.float64); del Z
            # RMS over the central disk of diameter D_m (the measurable quantity), not over the box
            xg = self.x0 + dg * np.arange(nx); yg = self.y0 + dg * np.arange(ny); disk = (xg[None, :] ** 2 + yg[:, None] ** 2) <= (D_m / 2) ** 2
            if self.projected:
                h, self.c_removed = _remove_zernike(h, xg, yg, D_m)      # piston, tilt and figure terms taken out on the disk
            h -= h[disk].mean(); self.mid = h * (sigma_mid / np.sqrt(np.mean(h[disk] ** 2)))
        if sigma_fein > 0:
            self.fein = (rng.normal(size=(ny, nx)) * sigma_fein).astype(np.float32)
        self.alpha_s = alpha_s; self.sigma = (sigma_fig, sigma_mid, sigma_fein)

    def h(self, x, y):
        """Return the height [m] at the points (x, y) (arrays of equal shape); the grids are periodic."""
        x = np.asarray(x, float); y = np.asarray(y, float); out = np.zeros_like(x)
        if self.c_fig is not None:
            rho = np.sqrt(x ** 2 + y ** 2) / (self.D_m / 2); phi = np.arctan2(y, x); inside = rho <= 1
            for c, (n, m) in zip(self.c_fig, FIG_TERMS): out += c * zernike(n, m, np.minimum(rho, 1.0), phi) * inside
        if self.mid is not None:
            # bilinear interpolation on the periodic grid
            gx = (x - self.x0) / self.dg; gy = (y - self.y0) / self.dg
            ix = np.floor(gx).astype(int) % self.nx; iy = np.floor(gy).astype(int) % self.ny
            fx = gx - np.floor(gx); fy = gy - np.floor(gy); ix1 = (ix + 1) % self.nx; iy1 = (iy + 1) % self.ny
            out += (self.mid[iy, ix] * (1 - fx) * (1 - fy) + self.mid[iy, ix1] * fx * (1 - fy)
                    + self.mid[iy1, ix] * (1 - fx) * fy + self.mid[iy1, ix1] * fx * fy)
        if self.fein is not None:
            ix = np.rint((x - self.x0) / self.dg).astype(int) % self.nx; iy = np.rint((y - self.y0) / self.dg).astype(int) % self.ny
            out += self.fein[iy, ix]
        return out


def noll_share(alpha_s, D_m, f_max=FMAX_DEFAULT, degrees=(2, 3, 4)):
    """Return the expected share of the variance of the stationary mid band (power law f^-alpha_s between 1/D_m and
    f_max) that a least-squares Zernike fit on the disk D_m books in the given radial degrees: p = sum_n (n+1)^2
    int S(f) [2 J_{n+1}(pi f D_m)/(pi f D_m)]^2 d^2f / int S d^2f (Noll 1976, JOSA 66, 207; Bachmann et al. 2024,
    arXiv:2402.04826). degrees=(2,3,4) gives the figure share p, degrees=(1,) the tilt share p_t, (0,) the piston."""
    from scipy.special import jv
    ff = np.geomspace(1 / D_m, f_max, 200000); xx = np.pi * ff * D_m; S = ff ** (-alpha_s); den = np.trapezoid(S * 2 * np.pi * ff, ff)
    return float(sum((n + 1) ** 2 * np.trapezoid(S * (2 * jv(n + 1, xx) / xx) ** 2 * 2 * np.pi * ff, ff) / den for n in degrees))


def measured_to_stationary(sigma_fig_fit, sigma_mid_res, alpha_s, D_m, f_max=FMAX_DEFAULT):
    """Convert the procedure-based band parameters of a height map (sigma_fig_fit = rms of the Zernike fit of degrees
    2-4, sigma_mid_res = rms of the residual less the fine band, i.e. the parameters of the projected generator,
    MID_PROJECTED = True) into the parameters of the stationary convention used by the closed-form equation and by
    the grids (MID_PROJECTED = False): sigma_mid^2 = sigma_mid_res^2/(1 - p - p_t), sigma_fig^2 = sigma_fig_fit^2 -
    p sigma_mid^2 (Part I, Sec. 3B, Eq. 11). Returns (sigma_fig, sigma_mid)."""
    p = noll_share(alpha_s, D_m, f_max, (2, 3, 4)); pt = noll_share(alpha_s, D_m, f_max, (1,))
    sm2 = sigma_mid_res ** 2 / (1 - p - pt); sf2 = max(sigma_fig_fit ** 2 - p * sm2, 0.0)
    return np.sqrt(sf2), np.sqrt(sm2)


def psd_mid(f, D_m, alpha_s, f_max=FMAX_DEFAULT):
    """Return the isotropic PSD of the mid band normalized to variance 1 (integral of S * 2 pi f df = 1) on f in
    [1/D_m, f_max], zero outside; used for the spectral integrals of the characteristic quantities."""
    f = np.asarray(f, float); S = np.where((f >= 1 / D_m) & (f <= f_max), np.maximum(f, 1e-30) ** (-alpha_s), 0.0)
    fr = np.geomspace(1 / D_m, f_max, 20000); norm = np.trapezoid(fr ** (-alpha_s) * 2 * np.pi * fr, fr)
    return S / norm


class FlaecheStreifen:
    """Memory-saving version of the surface model for long plates (L > 40 mm), same interface as Flaeche
    (h(x, y); attributes c_fig, mid, fein).

    The mid band is generated in two parts, both with the same physical spectral density S2 ~ f^(-alpha_s):
      low  (1/D_m <= f <= F_C = 0.25/mm, periods > 4 mm): on a coarse grid (0.25 mm) over the whole box 2 D_m,
           so the modes near 1/D_m stay resolved;
      high (F_C < f <= f_max, periods 0.1-4 mm): on a strip (length D_m + 4 mm, width 32 mm, grid 25 um), because
           this part is statistically homogeneous and has no periods above 4 mm; only |y| <= 2.75 mm is illuminated.
    Each realization is scaled to RMS sigma_mid: root of the sum of squares of the low part's RMS over the central
    disk D_m and the high part's RMS over the strip.
    Fine band: white noise on the strip grid. Figure: Zernike, analytic. Memory per surface ~ 0.3 GB at L = 150 mm
    (instead of ~ 8 GB with the full box).
    Bilinear interpolation as in Flaeche; the coarse grid has >= 16 points per shortest period of the low part."""
    F_C = 250.0              # 1/m; split frequency between low and high part
    DG_LOW = 250e-6          # m; coarse grid of the low part
    LY_STREIFEN = 32e-3      # m; width of the strip in y

    def __init__(self, rng, D_m, sigma_fig, sigma_mid, alpha_s, sigma_fein, Lx, Ly, dg=DG_DEFAULT, f_max=FMAX_DEFAULT, project_figure=None):
        self.D_m = D_m; self.dg = dg; self.c_fig = None; self.mid = None; self.fein = None
        self.projected = MID_PROJECTED if project_figure is None else bool(project_figure)
        if sigma_fig > 0:
            c = rng.normal(size=len(FIG_TERMS)); self.c_fig = c / np.sqrt(np.sum(c ** 2)) * sigma_fig
        # strip grid (high part and fine band)
        Lxs = D_m + 4e-3; self.nx = int(np.ceil(Lxs / dg)); self.ny = int(np.ceil(self.LY_STREIFEN / dg)); self.x0 = -Lxs / 2; self.y0 = -self.LY_STREIFEN / 2
        if sigma_mid > 0:
            # marker only (band_heights sets mid to None to isolate the other bands); a plain flag rather than a
            # reference to the arrays avoids a reference cycle, so the memory of each surface is freed
            self.mid = True
            def synth(nx, ny, dgx, f_lo, f_hi, Lbox_disk=None):
                """Return a random field on an nx x ny grid (spacing dgx) with S2 ~ f^(-alpha_s) for f_lo <= f <= f_hi;
                the factor sqrt(nx*ny)/dgx makes the amplitude independent of the grid, so low and high part share
                the same physical spectral density."""
                fx = np.fft.fftfreq(nx, dgx).astype(np.float32); fy = np.fft.fftfreq(ny, dgx).astype(np.float32); F = np.sqrt(fx[None, :] ** 2 + fy[:, None] ** 2)
                A = np.zeros_like(F); band = (F >= f_lo) & (F <= f_hi); A[band] = F[band] ** (-alpha_s / 2); del F, band
                Z = (rng.normal(size=(ny, nx)) + 1j * rng.normal(size=(ny, nx))).astype(np.complex64) * A; del A
                return np.real(np.fft.ifft2(Z)).astype(np.float64) * (np.sqrt(nx * ny) / dgx)
            nl = int(np.ceil(Lx / self.DG_LOW)); nly = int(np.ceil(Ly / self.DG_LOW)); self.nxl = nl; self.nyl = nly; self.x0l = -Lx / 2; self.y0l = -Ly / 2
            low = synth(nl, nly, self.DG_LOW, 1 / D_m, self.F_C)
            xg = self.x0l + self.DG_LOW * np.arange(nl); yg = self.y0l + self.DG_LOW * np.arange(nly); disk = (xg[None, :] ** 2 + yg[:, None] ** 2) <= (D_m / 2) ** 2
            if self.projected:      # piston, tilt and figure terms live in the low part (periods > 4 mm): remove them there
                low, self.c_removed = _remove_zernike(low, xg, yg, D_m)
            low -= low[disk].mean(); rms_l2 = float(np.mean(low[disk] ** 2))
            high = synth(self.nx, self.ny, dg, self.F_C, f_max); high -= high.mean(); rms_h2 = float(np.mean(high ** 2))
            s = sigma_mid / np.sqrt(rms_l2 + rms_h2); self.low = low * s; self.high = high * s
        if sigma_fein > 0:
            self.fein = (rng.normal(size=(self.ny, self.nx)) * sigma_fein).astype(np.float32)
        self.alpha_s = alpha_s; self.sigma = (sigma_fig, sigma_mid, sigma_fein)

    @staticmethod
    def _bilin(A, gx, gy):
        """Return the bilinear interpolation of the periodic grid A at the fractional grid indices (gx, gy)."""
        ny, nx = A.shape; ix = np.floor(gx).astype(int); iy = np.floor(gy).astype(int); fx = gx - ix; fy = gy - iy; ix %= nx; iy %= ny; ix1 = (ix + 1) % nx; iy1 = (iy + 1) % ny
        return A[iy, ix] * (1 - fx) * (1 - fy) + A[iy, ix1] * fx * (1 - fy) + A[iy1, ix] * (1 - fx) * fy + A[iy1, ix1] * fx * fy

    def h(self, x, y):
        """Return the height [m] at the points (x, y): figure + low and high mid-band part + fine band."""
        x = np.asarray(x, float); y = np.asarray(y, float); out = np.zeros_like(x)
        if self.c_fig is not None:
            rho = np.sqrt(x ** 2 + y ** 2) / (self.D_m / 2); phi = np.arctan2(y, x); inside = rho <= 1
            for c, (n, m) in zip(self.c_fig, FIG_TERMS): out += c * zernike(n, m, np.minimum(rho, 1.0), phi) * inside
        if self.mid is not None:
            out += self._bilin(self.low, (x - self.x0l) / self.DG_LOW, (y - self.y0l) / self.DG_LOW)
            out += self._bilin(self.high, (x - self.x0) / self.dg, (y - self.y0) / self.dg)
        if self.fein is not None:
            ix = np.rint((x - self.x0) / self.dg).astype(int) % self.nx; iy = np.rint((y - self.y0) / self.dg).astype(int) % self.ny
            out += self.fein[iy, ix]
        return out
