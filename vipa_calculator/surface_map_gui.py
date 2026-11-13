"""Window 'Surface map': load a CODE V .int map, remove tilt, supply missing parameters, choose the strip, save a plate map."""
import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import surface_map as SM


class MapWindow(tk.Toplevel):
    def __init__(self, master, surface, L_mm, W_mm, on_saved):
        super().__init__(master)
        self.title(f"Surface map - {surface}"); self.geometry("1100x720")
        self.surface, self.on_saved, self.raw = surface, on_saved, None
        self.v = {k: tk.StringVar(value=val) for k, val in dict(diam="", pix="", factor="1", L=f"{L_mm:g}", W=f"{W_mm:g}", cx="0", cy="0", ang="0").items()}
        self.remove = tk.StringVar(value="tilt"); self.view = tk.StringVar(value="raw"); self.flip = tk.BooleanVar(value=False)
        self.info = tk.StringVar(value="Load a CODE V .int file.")
        left = ttk.Frame(self); left.pack(side="left", fill="y", padx=6, pady=6)
        ttk.Button(left, text="Load .int ...", command=self.load).pack(fill="x")
        f = ttk.LabelFrame(left, text="Missing parameters"); f.pack(fill="x", pady=4)
        for i, (k, lab, tip) in enumerate([("diam", "aperture diameter [mm]", "diameter of the valid area on the part"), ("pix", "or pixel size [mm]", "used if given"),
                                           ("factor", "height factor", "0.5: reflected wavefront; -1: opposite sign")]):
            ttk.Label(f, text=lab).grid(row=i, column=0, sticky="w"); ttk.Entry(f, textvariable=self.v[k], width=9).grid(row=i, column=1)
        f = ttk.LabelFrame(left, text="Strip (plate coordinates)"); f.pack(fill="x", pady=4)
        for i, (k, lab) in enumerate([("L", "length L [mm]"), ("W", "width [mm] (>= chord)"), ("cx", "centre x [mm]"), ("cy", "centre y [mm]"), ("ang", "angle [deg]")]):
            ttk.Label(f, text=lab).grid(row=i, column=0, sticky="w"); ttk.Entry(f, textvariable=self.v[k], width=9).grid(row=i, column=1)
        ttk.Checkbutton(f, text="mirror across the long axis", variable=self.flip, command=self.redraw).grid(row=5, column=0, columnspan=2, sticky="w")
        f = ttk.LabelFrame(left, text="Remove from the strip"); f.pack(fill="x", pady=4)
        for t, val in [("nothing", "none"), ("piston", "piston"), ("piston + tilt", "tilt"), ("piston + tilt + power", "power")]:
            ttk.Radiobutton(f, text=t, variable=self.remove, value=val, command=self.redraw).pack(anchor="w")
        f = ttk.LabelFrame(left, text="Map shown"); f.pack(fill="x", pady=4)
        for t, val in [("raw", "raw"), ("minus piston, tilt", "tilt"), ("minus Zernike deg. 0-4", "zern"), ("strip (result)", "strip")]:
            ttk.Radiobutton(f, text=t, variable=self.view, value=val, command=self.redraw).pack(anchor="w")
        ttk.Button(left, text="Update", command=self.redraw).pack(fill="x", pady=2)
        ttk.Button(left, text="Save plate map and use ...", command=self.save).pack(fill="x", pady=2)
        ttk.Label(left, textvariable=self.info, wraplength=260, justify="left").pack(fill="x", pady=4)
        self.fig = Figure(figsize=(7, 6), dpi=100); self.ax = self.fig.add_subplot(111)
        cv = FigureCanvasTkAgg(self.fig, self); cv.get_tk_widget().pack(side="right", fill="both", expand=True)
        NavigationToolbar2Tk(cv, self).update(); self.cv = cv

    def _num(self, k):
        try: return float(self.v[k].get().replace(",", "."))
        except ValueError: raise ValueError(f"'{k}' is not a number")

    def load(self):
        fn = filedialog.askopenfilename(filetypes=[("CODE V interferogram", "*.int"), ("all", "*.*")])
        if not fn: return
        try: self.raw = SM.read_int(fn)
        except Exception as e: messagebox.showerror("Load", str(e)); return
        self.info.set(f"{self.raw.source}: {self.raw.h_nm.shape[1]}x{self.raw.h_nm.shape[0]} points, {self.raw.header['kind']}, step {self.raw.header['step_nm']:.3f} nm. "
                      "The file has no lateral scale: enter the aperture diameter or the pixel size.")
        self.redraw()

    def _scale(self):
        pix, dia = self.v["pix"].get().strip(), self.v["diam"].get().strip()
        if pix: self.raw.set_scale(pixel_mm=self._num("pix"))
        elif dia: self.raw.set_scale(diameter_mm=self._num("diam"))
        else: raise ValueError("enter the aperture diameter or the pixel size")

    def _strip(self):
        return SM.prepare(self.raw, self._num("L"), self._num("W"), self._num("cx"), self._num("cy"), self._num("ang"), remove=self.remove.get(),
                          factor=self._num("factor"), flip_y=self.flip.get())

    def redraw(self):
        if self.raw is None: return
        ax = self.ax; ax.clear()
        try:
            self._scale(); fa = self._num("factor"); x, y = self.raw.coords_mm(); ext = (x[0], x[-1], y[-1], y[0]); view = self.view.get()
            if view == "strip":
                pm = self._strip(); x0, x1, y0, y1 = pm.extent_mm; im = ax.imshow(pm.h_nm, extent=(x0, x1, y1, y0), cmap="RdBu_r"); ax.set_xlabel("plate x [mm]"); ax.set_ylabel("plate y [mm]")
                self.info.set(f"strip: rms {pm.stats()['rms']:.1f} nm, PV {pm.stats()['pv']:.0f} nm; removed tilt {pm.meta['removed'].get('tilt_x_arcsec', 0):.0f}'' / {pm.meta['removed'].get('tilt_y_arcsec', 0):.0f}''")
            else:
                h = {"raw": self.raw.h_nm * fa, "tilt": self.raw.without_tilt(fa), "zern": self.raw.without_zernike(4, fa)}[view]
                im = ax.imshow(h, extent=ext, cmap="RdBu_r"); ax.set_xlabel("map x [mm]"); ax.set_ylabel("map y [mm]")
                try:
                    P, c, ex = SM.strip_corners(self._num("L"), self._num("W"), self._num("cx"), self._num("cy"), self._num("ang"))
                    if self.flip.get(): pass
                    ax.plot(P[:, 0], P[:, 1], "k-", lw=1.5); ax.plot(*c, "ko", ms=5); ax.annotate("entrance", c, color="k", fontsize=8)
                except Exception: pass
                s = self.raw.stats(fa)
                self.info.set(f"valid area {s['fill'] * 100:.0f} %, diameter {self.raw.diameter_mm:.1f} mm; raw rms {s['rms_raw']:.0f} nm; tilt PV {s['tilt_pv']:.0f} nm; "
                              f"minus tilt: PV {s['pv_notilt']:.0f}, rms {s['rms_notilt']:.1f} nm; Zernike deg. 2-4: {s['fig_rms']:.1f} nm; rest {s['rms_res']:.1f} nm")
            self.fig.colorbar(im, ax=ax, label="height [nm]") if not hasattr(self, "_cb") or True else None
            ax.set_aspect("equal")
        except Exception as e:
            self.info.set(str(e))
        self.cv.draw()

    def save(self):
        if self.raw is None: return
        try: self._scale(); pm = self._strip()
        except Exception as e: messagebox.showerror("Strip", str(e)); return
        fn = filedialog.asksaveasfilename(defaultextension=".vmap.npz", initialfile=f"{self.surface.lower()}.vmap.npz", filetypes=[("plate map", "*.npz")])
        if not fn: return
        pm.save(fn); self.on_saved(fn); self.info.set("saved and entered: " + fn)
