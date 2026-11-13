"""surface_map.py -- measured surface maps as input of the VIPA calculator (no GUI), version 2026-11-13.

Workflow
  raw = read_int(path)                       # CODE V interferogram file (.int, GRD format), e.g. an export of Zygo Mx
  raw.set_scale(diameter_mm=...)             # the file has no lateral scale: diameter of the valid aperture or pixel size
  pm = prepare(raw, L_mm, W_mm, ...)         # cut out the strip used by the VIPA, remove piston/tilt, fill gaps
  pm.save("hr.vmap.npz")                     # plate map: the file the calculator reads (Params.map_hr / Params.map_pr)
  h = load_plate_map("hr.vmap.npz").h(x, y)  # heights [m] at plate coordinates [m] (bilinear)

Conventions
  Raw map: array h_nm[row, column] with NaN for invalid points; column = +x_map, row = +y_map of the file, pixel size
           pixel_mm. Heights in nm: integer / SSZ * WVL (CODE V: SSZ data units per wavelength), times `factor`
           (e.g. 0.5 if the file is a reflected wavefront and not a surface, -1 to change the sign).
  Plate map: heights on a regular grid in PLATE coordinates. x runs along the dispersion direction, from the entrance
           side (-L/2) to the far end (+L/2); y runs across (the chord), y = 0 in the middle of the beam. The strip is
           placed on the raw map by its centre (cx_mm, cy_mm, measured from the centre of the valid aperture) and by the
           angle of the plate x axis against the map x axis (counter-clockwise, degrees).
  Tilt:    the tilt of an interferogram is the alignment of the part in the interferometer, not the wedge of the VIPA.
           It is removed on the strip; the wedge is a separate input of the calculator.
"""
import os, json, math
import numpy as np

FORMAT_VERSION = 1


# ---------------------------------------------------------------------------------------------- raw map
class RawMap:
    def __init__(self, h_nm, header=None, source=""):
        self.h_nm = np.asarray(h_nm, float)               # (ny, nx), NaN = invalid
        self.header = dict(header or {})
        self.source = source
        self.pixel_mm = None
        ok = np.isfinite(self.h_nm)
        if not ok.any(): raise ValueError("the map contains no valid point")
        yy, xx = np.nonzero(ok)
        self.cx_px, self.cy_px = float(xx.mean()), float(yy.mean())                     # centre of the valid area
        self.r_px = float(np.hypot(xx - self.cx_px, yy - self.cy_px).max())            # radius of the valid area

    # -- scale
    def set_scale(self, diameter_mm=None, pixel_mm=None):
        """Lateral scale: either the diameter of the valid aperture or the pixel size."""
        if pixel_mm: self.pixel_mm = float(pixel_mm)
        elif diameter_mm: self.pixel_mm = float(diameter_mm) / (2 * self.r_px)
        else: raise ValueError("give the diameter of the valid aperture or the pixel size")
        return self.pixel_mm

    @property
    def diameter_mm(self):
        return None if self.pixel_mm is None else 2 * self.r_px * self.pixel_mm

    # -- coordinates of the valid points [mm], origin at the centre of the valid area
    def coords_mm(self):
        self._need_scale()
        ny, nx = self.h_nm.shape
        x = (np.arange(nx) - self.cx_px) * self.pixel_mm; y = (np.arange(ny) - self.cy_px) * self.pixel_mm
        return x, y

    def _need_scale(self):
        if self.pixel_mm is None: raise ValueError("lateral scale missing: call set_scale(diameter_mm=...) or set_scale(pixel_mm=...)")

    # -- statistics
    def stats(self, factor=1.0):
        """PV and rms of the valid area: raw, minus piston and tilt, minus Zernike terms of degree 0-4 (disk of the valid area)."""
        ok = np.isfinite(self.h_nm); yy, xx = np.nonzero(ok); hv = self.h_nm[ok] * factor
        u, w = (xx - self.cx_px) / self.r_px, (yy - self.cy_px) / self.r_px
        A = np.c_[np.ones_like(u), u, w]; c = np.linalg.lstsq(A, hv, rcond=None)[0]; r1 = hv - A @ c
        B, idx = zernike_basis(u, w, 4); cz = np.linalg.lstsq(B, hv, rcond=None)[0]; r4 = hv - B @ cz
        pv = lambda v: float(v.max() - v.min())
        return dict(n_valid=int(ok.sum()), fill=float(ok.sum() / (math.pi * self.r_px ** 2)),
                    pv_raw=pv(hv), rms_raw=float(hv.std()),
                    tilt_pv=float(2 * math.hypot(c[1], c[2])), tilt_x=float(c[1]), tilt_y=float(c[2]),
                    pv_notilt=pv(r1), rms_notilt=float(r1.std()),
                    fig_rms=float(np.sqrt(sum(cz[i] ** 2 for i, (n, m) in enumerate(idx) if n >= 2))),
                    zernike={f"Z({n},{m:+d})": float(v) for (n, m), v in zip(idx, cz)},
                    pv_res=pv(r4), rms_res=float(r4.std()))

    def without_tilt(self, factor=1.0):
        """Map minus piston and tilt of the valid area (for display)."""
        ok = np.isfinite(self.h_nm); yy, xx = np.nonzero(ok)
        u, w = (xx - self.cx_px) / self.r_px, (yy - self.cy_px) / self.r_px
        A = np.c_[np.ones_like(u), u, w]; hv = self.h_nm[ok] * factor; c = np.linalg.lstsq(A, hv, rcond=None)[0]
        out = np.full(self.h_nm.shape, np.nan); out[yy, xx] = hv - A @ c
        return out

    def without_zernike(self, nmax=4, factor=1.0):
        ok = np.isfinite(self.h_nm); yy, xx = np.nonzero(ok)
        u, w = (xx - self.cx_px) / self.r_px, (yy - self.cy_px) / self.r_px
        B, _ = zernike_basis(u, w, nmax); hv = self.h_nm[ok] * factor; c = np.linalg.lstsq(B, hv, rcond=None)[0]
        out = np.full(self.h_nm.shape, np.nan); out[yy, xx] = hv - B @ c
        return out


def zernike_basis(u, w, nmax):
    """Zernike polynomials (Noll normalization: unit rms on the unit disk) of radial degree 0..nmax at the points (u, w)."""
    from math import factorial as f
    rho = np.hypot(u, w); th = np.arctan2(w, u); cols = []; idx = []
    for n in range(nmax + 1):
        for m in range(-n, n + 1, 2):
            a = abs(m)
            R = sum((-1) ** k * f(n - k) / (f(k) * f((n + a) // 2 - k) * f((n - a) // 2 - k)) * rho ** (n - 2 * k) for k in range((n - a) // 2 + 1))
            N = math.sqrt(n + 1) if m == 0 else math.sqrt(2 * (n + 1))
            cols.append(N * R * (np.cos(a * th) if m >= 0 else np.sin(a * th))); idx.append((n, m))
    return np.array(cols).T, idx


def read_int(path):
    """Read a CODE V interferogram file with grid data (GRD nx ny [SUR|WFR|FIL] WVL w SSZ s NDA n ...).
    Value = integer / SSZ in waves at WVL [micrometre]; NDA marks invalid points. Returns a RawMap (heights in nm).
    The first data row is taken as the first row of the array (plotted with the origin at the top, as the interferometer shows it)."""
    with open(path, errors="replace") as fh: lines = fh.read().splitlines()
    try: ih = next(i for i, l in enumerate(lines) if l.strip().upper().startswith("GRD"))
    except StopIteration: raise ValueError("no GRD line: not a CODE V grid interferogram file")
    tok = lines[ih].split(); nx, ny = int(tok[1]), int(tok[2])
    up = [t.upper() for t in tok]
    def val(key, default=None):
        return float(tok[up.index(key) + 1]) if key in up else default
    wvl = val("WVL"); ssz = val("SSZ", 1.0); nda = val("NDA", -32768.0)
    if wvl is None: raise ValueError("no WVL in the header: the height unit is unknown")
    kind = next((t for t in up[3:] if t in ("SUR", "WFR", "FIL")), "?")
    v = np.array(" ".join(lines[ih + 1:]).split(), dtype=float)
    if v.size < nx * ny: raise ValueError(f"file has {v.size} values, header announces {nx}x{ny}")
    Z = v[:nx * ny].reshape(ny, nx)
    h = np.where(Z == nda, np.nan, Z / ssz * wvl * 1e3)
    hdr = dict(format="CODE V INT (GRD)", nx=nx, ny=ny, kind=kind, wvl_um=wvl, ssz=ssz, nda=nda, step_nm=wvl * 1e3 / ssz,
               title=" | ".join(l.strip() for l in lines[:ih] if l.strip() and l.strip() != "!"), header_line=lines[ih].strip())
    return RawMap(h, hdr, source=os.path.basename(path))


# ---------------------------------------------------------------------------------------------- plate map
class PlateMap:
    """Heights on a regular grid in plate coordinates. h_nm[iy, ix] at x = x0_mm + ix*pixel_mm, y = y0_mm + iy*pixel_mm."""

    def __init__(self, h_nm, pixel_mm, x0_mm, y0_mm, meta=None):
        self.h_nm = np.asarray(h_nm, float); self.pixel_mm = float(pixel_mm); self.x0_mm = float(x0_mm); self.y0_mm = float(y0_mm)
        self.meta = dict(meta or {})
        if not np.isfinite(self.h_nm).all(): raise ValueError("the plate map contains invalid points")

    @property
    def extent_mm(self):
        ny, nx = self.h_nm.shape
        return (self.x0_mm, self.x0_mm + (nx - 1) * self.pixel_mm, self.y0_mm, self.y0_mm + (ny - 1) * self.pixel_mm)

    def covers(self, x_m, y_m, tol_mm=1e-6):
        x0, x1, y0, y1 = self.extent_mm; x = np.asarray(x_m) * 1e3; y = np.asarray(y_m) * 1e3
        return bool(x.min() >= x0 - tol_mm and x.max() <= x1 + tol_mm and y.min() >= y0 - tol_mm and y.max() <= y1 + tol_mm)

    def h(self, x_m, y_m):
        """Heights [m] at plate coordinates [m], bilinear interpolation. Raises if a point lies outside the map."""
        x = np.asarray(x_m, float) * 1e3; y = np.asarray(y_m, float) * 1e3
        if not self.covers(x_m, y_m):
            x0, x1, y0, y1 = self.extent_mm
            raise ValueError(f"surface map too small: hits need x = {x.min():.2f}..{x.max():.2f} mm, y = {y.min():.2f}..{y.max():.2f} mm; "
                             f"the map covers x = {x0:.2f}..{x1:.2f} mm, y = {y0:.2f}..{y1:.2f} mm")
        ny, nx = self.h_nm.shape
        fx = np.clip((x - self.x0_mm) / self.pixel_mm, 0, nx - 1); fy = np.clip((y - self.y0_mm) / self.pixel_mm, 0, ny - 1)
        ix = np.minimum(fx.astype(int), nx - 2); iy = np.minimum(fy.astype(int), ny - 2); tx = fx - ix; ty = fy - iy
        H = self.h_nm
        return (H[iy, ix] * (1 - tx) * (1 - ty) + H[iy, ix + 1] * tx * (1 - ty) + H[iy + 1, ix] * (1 - tx) * ty + H[iy + 1, ix + 1] * tx * ty) * 1e-9

    def stats(self):
        v = self.h_nm
        return dict(pv=float(v.max() - v.min()), rms=float(v.std()), mean=float(v.mean()))

    def save(self, path):
        np.savez_compressed(path, h_nm=self.h_nm.astype(np.float32), pixel_mm=self.pixel_mm, x0_mm=self.x0_mm, y0_mm=self.y0_mm,
                            meta=json.dumps(dict(self.meta, format_version=FORMAT_VERSION)))
        return path


_CACHE = {}


def load_plate_map(path):
    """Read a plate map (.npz written by PlateMap.save). Cached per process by path and modification time."""
    key = (os.path.abspath(path), os.path.getmtime(path))
    if key not in _CACHE:
        with np.load(path, allow_pickle=False) as z:
            _CACHE[key] = PlateMap(z["h_nm"].astype(float), float(z["pixel_mm"]), float(z["x0_mm"]), float(z["y0_mm"]), json.loads(str(z["meta"])))
    return _CACHE[key]


def _fill_invalid(h):
    """Replace invalid points by the nearest valid one. Returns (filled, number filled)."""
    bad = ~np.isfinite(h)
    if not bad.any(): return h, 0
    from scipy.ndimage import distance_transform_edt
    idx = distance_transform_edt(bad, return_distances=False, return_indices=True)
    return h[tuple(idx)], int(bad.sum())


def strip_corners(L_mm, W_mm, cx_mm=0.0, cy_mm=0.0, angle_deg=0.0, margin_mm=0.0):
    """Corners of the strip on the raw map [mm] (closed polygon, 5 points) and the entrance point (middle of the -L/2 edge)."""
    a = math.radians(angle_deg); ex = np.array([math.cos(a), math.sin(a)]); ey = np.array([-math.sin(a), math.cos(a)])
    c = np.array([cx_mm, cy_mm]); hx, hy = L_mm / 2 + margin_mm, W_mm / 2 + margin_mm
    P = [c - hx * ex - hy * ey, c + hx * ex - hy * ey, c + hx * ex + hy * ey, c - hx * ex + hy * ey]
    return np.array(P + [P[0]]), c - L_mm / 2 * ex, ex


def prepare(raw, L_mm, W_mm, cx_mm=0.0, cy_mm=0.0, angle_deg=0.0, remove="tilt", factor=1.0, flip_y=False,
            margin_mm=0.5, fill=True, max_fill_fraction=0.05):
    """Cut the strip used by the VIPA out of a raw map and return it as a PlateMap in plate coordinates.

    L_mm, W_mm   length (dispersion direction) and width (across) of the strip; W must cover the beam (chord)
    cx_mm, cy_mm centre of the strip on the raw map, measured from the centre of the valid aperture (map axes: x to the right,
                 y downwards as plotted, i.e. along increasing row index)
    angle_deg    direction of the plate x axis (entrance -> far end) on the raw map, from +x_map towards +y_map
    remove       "none", "piston", "tilt" (piston and tilt, default) or "power" (piston, tilt and the paraboloid) on the strip
    factor       height factor (0.5: the file is a reflected wavefront; -1: opposite sign)
    flip_y       mirror the strip across its long axis (a surface that is seen from the other side in the assembled cavity)
    margin_mm    extra border around L x W (the beams of the outer orders and of a wedge walk slightly beyond L)
    fill         replace invalid points inside the strip by the nearest valid one (error if more than max_fill_fraction)"""
    raw._need_scale()
    p = raw.pixel_mm
    xs = np.arange(-(L_mm / 2 + margin_mm), L_mm / 2 + margin_mm + p / 2, p); ys = np.arange(-(W_mm / 2 + margin_mm), W_mm / 2 + margin_mm + p / 2, p)
    Xp, Yp = np.meshgrid(xs, ys)
    a = math.radians(angle_deg); Ys = -Yp if flip_y else Yp
    xm = cx_mm + Xp * math.cos(a) - Ys * math.sin(a); ym = cy_mm + Xp * math.sin(a) + Ys * math.cos(a)        # map coordinates [mm]
    fx = xm / p + raw.cx_px; fy = ym / p + raw.cy_px
    ny, nx = raw.h_nm.shape
    inside = (fx >= 0) & (fx <= nx - 1) & (fy >= 0) & (fy <= ny - 1)
    ix = np.clip(fx.astype(int), 0, nx - 2); iy = np.clip(fy.astype(int), 0, ny - 2); tx = fx - ix; ty = fy - iy
    H = raw.h_nm
    h = H[iy, ix] * (1 - tx) * (1 - ty) + H[iy, ix + 1] * tx * (1 - ty) + H[iy + 1, ix] * (1 - tx) * ty + H[iy + 1, ix + 1] * tx * ty
    h = np.where(inside, h, np.nan) * factor
    bad = ~np.isfinite(h); frac = float(bad.mean())
    if frac > 0:
        if not fill or frac > max_fill_fraction:
            raise ValueError(f"{100 * frac:.1f} % of the strip lies outside the valid area of the map: move, turn or shorten the strip")
    ok = ~bad; A = [np.ones(ok.sum())]
    if remove in ("tilt", "power"): A += [Xp[ok], Yp[ok]]
    if remove == "power": A += [Xp[ok] ** 2 + Yp[ok] ** 2]
    removed = {}
    if remove != "none":
        A = np.array(A).T; c = np.linalg.lstsq(A, h[ok], rcond=None)[0]
        full = [np.ones_like(Xp)] + ([Xp, Yp] if remove in ("tilt", "power") else []) + ([Xp ** 2 + Yp ** 2] if remove == "power" else [])
        h = h - sum(ci * fi for ci, fi in zip(c, full))
        removed = dict(piston_nm=float(c[0]))
        if len(c) >= 3: removed.update(tilt_x_nm_per_mm=float(c[1]), tilt_y_nm_per_mm=float(c[2]),
                                       tilt_x_arcsec=float(math.degrees(c[1] * 1e-6) * 3600), tilt_y_arcsec=float(math.degrees(c[2] * 1e-6) * 3600))
        if len(c) == 4: removed.update(power_nm_per_mm2=float(c[3]))
    h, nfill = _fill_invalid(h)
    meta = dict(source=raw.source, header=raw.header, pixel_mm=p, diameter_mm=raw.diameter_mm, L_mm=L_mm, W_mm=W_mm, cx_mm=cx_mm, cy_mm=cy_mm,
                angle_deg=angle_deg, remove=remove, removed=removed, factor=factor, flip_y=bool(flip_y), margin_mm=margin_mm, filled_points=nfill)
    return PlateMap(h, p, xs[0], ys[0], meta)
