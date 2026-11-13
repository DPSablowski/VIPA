"""VIPA finesse calculator: GUI (Tkinter + matplotlib). Start with start_calculator.py."""
import json, os, queue, threading, traceback, csv
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from dataclasses import asdict
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

import calculator_core as CC
import materials as MAT

# input groups: (field, label, unit, tooltip)
GROUPS = [
    ("Light and mirrors", [
        ("wl_nm", "Wavelength", "nm", "vacuum wavelength"),
        ("R1_pct", "Input mirror R1", "%", "HR side; the calculation uses R1*R2 per round trip"),
        ("R2_pct", "Output mirror R2", "%", "partially reflecting side"),
    ]),
    ("Plate", [
        ("h_mm", "Mirror spacing / thickness h", "mm", ""),
        ("n_gap", "Refractive index in gap n'", "", "1 = air-spaced"),
        ("n_group", "Group index n_g", "", "0 = same as n'; only FSR and source widths"),
        ("L_mm", "Length L", "mm", "illuminable length in the dispersion direction"),
        ("theta_deg", "Angle of incidence (outside)", "deg", ""),
        ("wedge_as", "Wedge (along dispersion)", "arcsec", "thickness wedge in the dispersion direction"),
        ("beam_mm", "Beam width (across)", "mm", "collimated beam = chord"),
        ("f_cyl_mm", "Cylinder lens focal length", "mm", "theory only (divergence, order selection)"),
    ]),
    ("Surface (thickness variation, RMS)", [
        ("s_fig_nm", "Figure", "nm", "smooth over the plate (Zernike)"),
        ("s_mid_nm", "Mid band", "nm", "power law 1/D_m ... 1/0.1 mm"),
        ("alpha_s", "Spectral exponent", "", "mid band, typically 2-3.5"),
        ("s_fine_nm", "Fine band", "nm", "< 0.1 mm, uncorrelated between reflections"),
        ("D_m_mm", "Surface model diameter D_m", "mm", "0 = from L"),
    ]),
    ("Source, detector, temperature", [
        ("src_gauss_pm", "Source Gaussian (FWHM)", "pm", ""),
        ("src_lorentz_pm", "Source Lorentzian (FWHM)", "pm", "laser line"),
        ("src_rect_pm", "Source rectangular (width)", "pm", "filter"),
        ("pixel_um", "Pixel size", "um", ""),
        ("f_cam_mm", "Camera focal length", "mm", "0 = no pixel"),
        ("sigma_T_mK", "Temperature fluctuation", "mK", "between exposures"),
        ("cte_ppm", "CTE of the thickness", "1e-6/K", "glass or spacer"),
        ("dndT_ppm", "dn'/dT", "1e-6/K", ""),
    ]),
]

HELP = """Calculation routes
  Equation: closed-form equation of the main document (overlap measure Omega50). Net finesse of the median
            plate; fractions of a second.
  Theory:   exact line (round-trip pupil, incoherent across the chord) for surfaces drawn from the three-band model.
            'Single realisation' shows one device; 'Ensemble' gives the median over N devices with P10-P90 spread,
            a histogram (tab 'Distribution') and the ensemble-average line. Roughly 0.3-4 s per surface and process.
  Random seed: start value of the random-number generator that draws the surfaces. The same seed gives the same
            surfaces, so runs are reproducible and can be compared with each other.
Measures
  Omega50 (overlap) is the measure of the document; FWHM is the measure of LightMachinery/Hu; F_eq is the
  equivalent width (energy measure, chapter 6).
Best focus (tick box)
  The detector is moved behind the focal plane until the line is sharpest. Equation: wedge and mean random chirp
  compensated (main document, best-focus equation). Theory: the defocus that maximises Omega50 is searched for every
  device. With the camera focal length f the shift is dz = -2 f^2 c / (k0 v^2), v = dx' cos(Theta) the transverse
  spacing of the virtual sources (air: v = 2 h sin Theta); tab 'Focus' shows dz per device or the focus scan.
Detector spectrum (tab 'Spectrum')
  Intensity on the detector over the whole field after Hu et al. (2015), Eqs. (9)-(10): Gaussian envelope
  exp(-2 theta_x^2/theta_0^2) with theta_0 = (beam/2)/f_cyl (1/e^2 radius of a Gaussian beam), times Hu's line of finite
  round-trip number with delta(theta_x) = 2 k0 h sqrt(n'^2 - sin^2(theta + theta_x)). Optional second wavelength. After a
  theory run the exact lines (surface 1, or the clean line) are overlaid under the same envelope, per order if 'All orders'
  was ticked. Note: the theory models a round top-hat beam (chord), Hu a Gaussian beam; the envelope is Hu's.
All orders (tick box)
  Every order inside the divergence cone of the cylinder lens, each with its own field angle, chord and round-trip
  offset; table in the results and tab 'Orders' (finesse over the detector position). The theory uses the same
  plates in every order. Cost: orders x surfaces lines (best focus about 20 lines per surface).
Static model
  Hu sum truncated at L/dx reflections, convolved with a static Gaussian thickness distribution, FWHM: the way
  the LightMachinery VIPA designer computes (reproduced from its outputs to < 1 %).
Solid VIPA
  Computed as an air-spaced VIPA with lambda/n' and the internal angle (exact conversion); source widths scaled
  with n_g/n'^2, pixel and divergence with 1/n'. Dispersion only through n_g; no rough exit surface.
"""


class Calculator(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("VIPA Finesse Calculator")
        self.geometry("1400x900")
        self.vars = {}; self.q = queue.Queue(); self.job = None; self.stop = threading.Event()
        self.last = dict(equation=None, theory=None, series=None, params=None, eq_orders=None)
        self._build()
        self.after(100, self._poll)
        self.run_equation()

    # ------------------------------------------------------------ layout
    def _build(self):
        pw = ttk.PanedWindow(self, orient="horizontal"); pw.pack(fill="both", expand=True)
        left = ttk.Frame(pw); right = ttk.Frame(pw); pw.add(left, weight=0); pw.add(right, weight=1)

        # input tabs (no scrolling): device, surface, source/detector, run; buttons and status stay below
        low = ttk.Frame(left); low.pack(side="bottom", fill="x")
        nbl = ttk.Notebook(left); nbl.pack(side="top", fill="both", expand=True)
        T = {}
        for key, name in (("dev", "Device"), ("surf", "Surface"), ("src", "Source"), ("run", "Run")):
            T[key] = ttk.Frame(nbl); nbl.add(T[key], text=name)
        self.nbl = nbl
        TAB_OF = {"Light and mirrors": "dev", "Plate": "dev", "Surface (thickness variation, RMS)": "surf", "Source, detector, temperature": "src"}

        std = CC.Params()
        fm = ttk.LabelFrame(T["dev"], text="Gap material"); fm.pack(side="top", fill="x", padx=6, pady=4)
        self.material = tk.StringVar(value="Air")
        ttk.Combobox(fm, textvariable=self.material, values=MAT.MATERIALS, state="readonly", width=26).grid(row=0, column=0, padx=4, pady=3)
        ttk.Button(fm, text="Set n', n_g", command=self._material).grid(row=0, column=1, padx=4)
        fbands = ttk.LabelFrame(T["surf"], text="Surface band convention"); fbands.pack(fill="x", padx=6, pady=4); fb = fbands
        self.bands = tk.StringVar(value="stationary")
        ttk.Radiobutton(fb, text="stationary (model field; grids, equation)", variable=self.bands, value="stationary").grid(row=0, column=0, sticky="w", padx=4)
        ttk.Radiobutton(fb, text="measured (Zernike fit = figure, residual = mid band)", variable=self.bands, value="measured").grid(row=1, column=0, sticky="w", padx=4)
        fmp = ttk.LabelFrame(T["surf"], text="Measured surface maps (plate maps)"); fmp.pack(fill="x", padx=6, pady=4)
        self.maps = {"map_hr": tk.StringVar(value=""), "map_pr": tk.StringVar(value="")}
        for i, (k, lab) in enumerate([("map_hr", "HR"), ("map_pr", "PR")]):
            ttk.Label(fmp, text=lab).grid(row=i, column=0, sticky="w", padx=4)
            ttk.Entry(fmp, textvariable=self.maps[k], width=24).grid(row=i, column=1, padx=2, pady=1)
            ttk.Button(fmp, text="Prepare...", width=9, command=lambda k=k, lab=lab: self._map_window(k, lab)).grid(row=i, column=2, padx=1)
            ttk.Button(fmp, text="Open", width=5, command=lambda k=k: self._map_open(k)).grid(row=i, column=3, padx=1)
            ttk.Button(fmp, text="X", width=2, command=lambda k=k: self.maps[k].set("")).grid(row=i, column=4, padx=1)
        ttk.Label(fmp, text="A map is added to the model surface; with maps only, one plate is computed.", foreground="#555", wraplength=400).grid(row=2, column=0, columnspan=5, sticky="w", padx=4)
        for title, flds in GROUPS:
            lf = ttk.LabelFrame(T[TAB_OF[title]], text=title)
            if title.startswith("Surface"): lf.pack(fill="x", padx=6, pady=4, before=fbands)
            else: lf.pack(fill="x", padx=6, pady=4)
            for i, (k, lab, unit, tip) in enumerate(flds):
                ttk.Label(lf, text=lab).grid(row=i, column=0, sticky="w", padx=4)
                v = tk.StringVar(value=_fmt(getattr(std, k))); self.vars[k] = v
                en = ttk.Entry(lf, textvariable=v, width=10); en.grid(row=i, column=1, padx=2, pady=1)
                en.bind("<Return>", lambda ev: self.run_equation())
                ttk.Label(lf, text=unit, foreground="#555").grid(row=i, column=2, sticky="w")
                if tip: _Tip(en, tip)
        ft = ttk.LabelFrame(T["run"], text="Theory"); ft.pack(fill="x", padx=6, pady=4)
        self.mode = tk.StringVar(value="ensemble")
        ttk.Radiobutton(ft, text="Single realisation", variable=self.mode, value="single").grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(ft, text="Ensemble (median)", variable=self.mode, value="ensemble").grid(row=0, column=1, sticky="w")
        self.n_surf = tk.StringVar(value="50"); self.seed = tk.StringVar(value="1"); self.nproc = tk.StringVar(value=str(max(1, (os.cpu_count() or 2) - 1)))
        self.bf = tk.BooleanVar(value=False); self.allord = tk.BooleanVar(value=False); self.maxord = tk.StringVar(value="41")
        cb1 = ttk.Checkbutton(ft, text="Best focus (detector shift from camera focal length)", variable=self.bf, command=self.run_equation)
        cb1.grid(row=4, column=0, columnspan=2, sticky="w")
        _Tip(cb1, "Detector moved behind the focal plane until the line is sharpest (maximum Omega50 per device).\n"
                  "Equation: wedge and mean random chirp compensated. The shift dz needs the camera focal length.")
        cb2 = ttk.Checkbutton(ft, text="All orders in the divergence cone", variable=self.allord, command=self.run_equation)
        cb2.grid(row=5, column=0, columnspan=2, sticky="w")
        _Tip(cb2, "Every order inside the cone of the cylinder lens: own field angle, chord and round-trip offset")
        ttk.Label(ft, text="Max. orders").grid(row=6, column=0, sticky="w"); ttk.Entry(ft, textvariable=self.maxord, width=8).grid(row=6, column=1, sticky="w")
        tips = ["number of surfaces (devices) in the ensemble", "start value of the random-number generator; same seed = same surfaces",
                "parallel processes for the ensemble"]
        for i, (lab, var) in enumerate([("Surfaces N (ensemble)", self.n_surf), ("Random seed", self.seed), ("Processes", self.nproc)]):
            ttk.Label(ft, text=lab).grid(row=1 + i, column=0, sticky="w")
            en = ttk.Entry(ft, textvariable=var, width=8); en.grid(row=1 + i, column=1, sticky="w"); _Tip(en, tips[i])
        fs = ttk.LabelFrame(T["run"], text="Detector spectrum (Hu, Gaussian envelope)"); fs.pack(fill="x", padx=6, pady=4)
        self.dlam2 = tk.StringVar(value="0"); self.sp_th = tk.BooleanVar(value=True); self.sp_bf = tk.BooleanVar(value=False)
        ttk.Label(fs, text="Second wavelength: offset [pm]").grid(row=0, column=0, sticky="w")
        en = ttk.Entry(fs, textvariable=self.dlam2, width=8); en.grid(row=0, column=1, sticky="w"); _Tip(en, "0 = only one wavelength")
        en.bind("<Return>", lambda ev: self.show_spectrum())
        ttk.Checkbutton(fs, text="Overlay theory lines (after a theory run)", variable=self.sp_th, command=self.show_spectrum).grid(row=1, column=0, columnspan=2, sticky="w")
        ttk.Checkbutton(fs, text="Theory lines in best focus (tilted detector)", variable=self.sp_bf, command=self.show_spectrum).grid(row=2, column=0, columnspan=2, sticky="w")
        ttk.Button(fs, text="Show spectrum", command=lambda: (self.show_spectrum(), self.nb.select(5))).grid(row=3, column=0, columnspan=2, sticky="ew", pady=2)
        fl = ttk.LabelFrame(T["run"], text="Length series"); fl.pack(fill="x", padx=6, pady=4)
        self.L_from = tk.StringVar(value="5"); self.L_to = tk.StringVar(value="80"); self.L_n = tk.StringVar(value="12"); self.L_th = tk.StringVar(value="0")
        for i, (lab, var) in enumerate([("L from [mm]", self.L_from), ("L to [mm]", self.L_to), ("Points", self.L_n), ("Theory surfaces per point (0 = equation only)", self.L_th)]):
            ttk.Label(fl, text=lab).grid(row=i, column=0, sticky="w"); ttk.Entry(fl, textvariable=var, width=8).grid(row=i, column=1, sticky="w")
        fb = ttk.Frame(low); fb.pack(fill="x", padx=6, pady=6)
        for i, (t, c) in enumerate([("Equation", self.run_equation), ("Theory", self.run_theory), ("Length series", self.run_series),
                                    ("Cancel", self.cancel), ("Save results (CSV)", self.save_csv), ("Save inputs", self.save_json),
                                    ("Load inputs", self.load_json), ("Help", lambda: messagebox.showinfo("Help", HELP))]):
            ttk.Button(fb, text=t, command=c).grid(row=i // 2, column=i % 2, sticky="ew", padx=2, pady=2)
        fb.columnconfigure(0, weight=1); fb.columnconfigure(1, weight=1)
        self.status = tk.StringVar(value="ready"); self.pb = ttk.Progressbar(low, mode="determinate")
        self.pb.pack(fill="x", padx=6); ttk.Label(low, textvariable=self.status, wraplength=420).pack(fill="x", padx=6, pady=2)

        vp = ttk.PanedWindow(right, orient="vertical"); vp.pack(fill="both", expand=True)
        ft2 = ttk.Frame(vp); vp.add(ft2, weight=0)
        self.txt = tk.Text(ft2, height=24, font=("Courier New", 9), wrap="none"); self.txt.pack(fill="both", expand=True)
        nb = ttk.Notebook(vp); vp.add(nb, weight=1)
        self.figs = {}
        for name in ("Line", "Distribution", "Orders", "Focus", "Length series", "Spectrum"):
            fr = ttk.Frame(nb); nb.add(fr, text=name)
            fig = Figure(figsize=(8, 4.5), dpi=100); ax = fig.add_subplot(111)
            can = FigureCanvasTkAgg(fig, master=fr); can.get_tk_widget().pack(fill="both", expand=True)
            NavigationToolbar2Tk(can, fr).update()
            self.figs[name] = (fig, ax, can)
        self.nb = nb

    # ------------------------------------------------------------ inputs
    def params(self):
        d = {}
        for k, v in self.vars.items():
            s = v.get().strip().replace(",", ".")
            try: d[k] = float(s)
            except ValueError: raise ValueError(f"Input '{k}' is not a number: {v.get()!r}")
        d["bands"] = self.bands.get()
        for k, v in self.maps.items(): d[k] = v.get().strip()
        return CC.Params(**d)

    def _map_window(self, key, lab):
        import surface_map_gui as SG
        try: p = self.params()
        except Exception: p = CC.Params()
        SG.MapWindow(self, lab, p.L_mm, max(6.0, p.beam_mm + 2.0), lambda fn, k=key: self.maps[k].set(fn))

    def _map_open(self, key):
        fn = filedialog.askopenfilename(filetypes=[("plate map", "*.npz")])
        if fn: self.maps[key].set(fn)

    def _material(self):
        try: wl = float(self.vars["wl_nm"].get().replace(",", "."))
        except ValueError: return
        n, ng = MAT.index(self.material.get(), wl)
        self.vars["n_gap"].set(f"{n:.5f}"); self.vars["n_group"].set(f"{ng:.5f}" if self.material.get() != "Air" else "0")
        self.run_equation()

    # ------------------------------------------------------------ calculations
    def run_equation(self):
        try:
            p = self.params(); g = CC.equation(p)
            self.last["eq_orders"] = CC.equation_orders(p, int(self.maxord.get())) if self.allord.get() else None
        except Exception as ex:
            self.status.set("Error: " + str(ex)); return
        self.last["equation"] = g; self.last["params"] = p
        if self.last.get("theory_params") != p: self.last["theory"] = None          # theory belongs to other inputs
        self._show_text(); self._show_orders(); self.show_spectrum(); self.status.set(f"Equation computed ({g['time_s']:.2f} s)")

    def _start(self, name, fn):
        if self.job and self.job.is_alive():
            messagebox.showwarning("Busy", "A calculation is already running."); return
        self.stop.clear(); self.pb["value"] = 0; self.status.set(name + " running ...")
        def work():
            try: self.q.put(("done", name, fn()))
            except Exception:
                self.q.put(("error", name, traceback.format_exc()))
        self.job = threading.Thread(target=work, daemon=True); self.job.start()

    def _progress(self, k, n): self.q.put(("progress", k, n))

    def run_theory(self):
        try:
            p = self.params(); n = 1 if self.mode.get() == "single" else max(2, int(self.n_surf.get()))
            seed = int(self.seed.get()); pr = max(1, int(self.nproc.get())); bf = self.bf.get(); ao = self.allord.get(); mo = int(self.maxord.get())
        except Exception as ex: self.status.set("Error: " + str(ex)); return
        self.run_equation(); self._pending = p
        self._start("Theory", lambda: CC.theory(p, n=n, seed=seed, processes=pr, n_lines=min(n, 5), best_focus=bf, all_orders=ao,
                                                max_orders=mo, progress=self._progress, cancel=self.stop.is_set))

    def run_series(self):
        try:
            p = self.params(); a, b, n = float(self.L_from.get()), float(self.L_to.get()), int(self.L_n.get())
            nth = int(self.L_th.get()); pr = max(1, int(self.nproc.get())); seed = int(self.seed.get())
        except Exception as ex: self.status.set("Error: " + str(ex)); return
        Ls = np.geomspace(a, b, n) if b / a > 4 else np.linspace(a, b, n)
        bf = self.bf.get()
        self._start("Length series", lambda: CC.length_series(p, Ls, theory_n=nth, seed=seed, processes=pr, best_focus=bf,
                                                              progress=self._progress, cancel=self.stop.is_set))

    def cancel(self): self.stop.set(); self.status.set("Cancelling ...")

    def _poll(self):
        try:
            while True:
                m = self.q.get_nowait()
                if m[0] == "progress":
                    k, n = m[1], m[2]; self.pb["maximum"] = max(n, 1); self.pb["value"] = k; self.status.set(f"line {k}/{n}")
                elif m[0] == "error":
                    self.status.set(m[1] + ": error"); messagebox.showerror("Error", m[2][-2000:])
                elif m[0] == "done":
                    if m[1] == "Theory": self.last["theory"] = m[2]; self.last["theory_params"] = self._pending; self._show_theory()
                    else: self.last["series"] = m[2]; self._show_series()
                    self._show_text(); self.status.set(m[1] + " finished")
        except queue.Empty: pass
        self.after(150, self._poll)

    # ------------------------------------------------------------ display
    def _show_text(self):
        g = self.last["equation"]; t = self.last["theory"]; T = []
        if g:
            a = g["equivalent"]
            T += ["CHARACTERISTICS",
                  f"  FSR                    {a['FSR_GHz']:10.3f} GHz   {a['FSR_pm']:10.3f} pm",
                  f"  Internal angle         {a['th_in_deg']:10.4f} deg   round-trip offset {a['dx']*1e3:8.4f} mm",
                  f"  Round trips on L       {a['M']:10.1f}       v = gamma M  {a['v']:8.3f}",
                  f"  Reflections to 0.1 %   {a['N_01']:10.1f}       minimum length {a['L_min_mm']:8.2f} mm",
                  f"  Airy finesse           {a['F_Airy']:10.2f}       (R1 R2 = {a['R2eff']:.5f})",
                  f"  Temperature drift      {a['dlam_T_pm']:10.4f} pm/K   (resonance at fixed angle)",
                  "", "CLOSED-FORM EQUATION (Omega50, median plate)",
                  f"  Net finesse            {g['F_net']:10.2f}       clean {g['F_clean']:8.2f}   loss factor {g['loss']:.3f}",
                  f"  Bandwidth              {g['dlam_pm']:10.4f} pm    {a['FSR_GHz']/g['F_net']*1e3:10.2f} MHz",
                  f"  Resolving power        {g['resolving_power']:12.0f}"]
            if self.bf.get():
                T.append(f"  Best focus             {g['F_best']:10.2f}       gain x{g['F_best']/g['F_net']:.3f}")
                if g["dz_best_mm"] is not None:
                    T.append(f"  Detector shift dz      {g['dz_best_mm']:+10.3f} mm   behind the focal plane (wedge); spread over plates +-{g['dz_spread_mm']:.3f} mm")
                else: T.append("  Detector shift dz: enter the camera focal length")
            sm = g["static"]
            T.append(f"  Static model (LightMachinery style, FWHM): {sm['F_net']:.2f}   clean {sm['F_clean']:.2f}   N = {sm['N']}")
            va = g.get("validity", {})
            if "t" in va: T.append(f"  chirp t~ {va['t']:.2f}   N_F {va['N_F']:.2f}   phi_mf {va.get('phi_mf', 0):.3f} rad   phi_fig {va.get('phi_fig', 0):.3f} rad")
            for h in CC.warnings_text(g): T.append("  ! " + h)
        if t:
            if t["n"] > 1: head = f"THEORY (ensemble, N = {t['n']} surfaces, random seed {t['seed']}, {t['time_s']:.1f} s)"
            elif t["n"] == 1: head = f"THEORY (single realisation, random seed {t['seed']}, {t['time_s']:.1f} s)"
            else: head = f"THEORY (clean line only: no surface error given, {t['time_s']:.1f} s)"
            T += ["", head, f"  clean line       Omega50 {t['F_clean']['om50']:8.2f}   FWHM {t['F_clean']['fwhm']:8.2f}   F_eq {t['F_clean']['eq']:8.2f}"]
            if t["n"] > 1:
                for key, lab in (("om50", "Omega50"), ("fwhm", "FWHM   "), ("eq", "F_eq   ")):
                    s = t[key]; T.append(f"  {lab}  median {s['median']:8.2f}   P10 {s['p10']:8.2f}   P90 {s['p90']:8.2f}   mean {s['mean']:8.2f}")
                fm = t["F_mean_line"]
                T.append(f"  ensemble-average line   Omega50 {fm[0]:8.2f}   FWHM {fm[1]:8.2f}   (broader than the median device)")
            elif t["n"] == 1:
                T.append(f"  this surface     Omega50 {t['om50']['median']:8.2f}   FWHM {t['fwhm']['median']:8.2f}   F_eq {t['eq']['median']:8.2f}")
            if t["best_focus"]:
                e = t["orders"][0]
                T.append(f"  BEST FOCUS: clean line Omega50 {e['F_clean_best']['om50']:8.2f}" + (f"   dz {e['dz_clean_mm']:+.3f} mm" if e['dz_clean_mm'] is not None else ""))
                if t["n"] > 1:
                    s = e["om50_best"]; T.append(f"  Omega50  median {s['median']:8.2f}   P10 {s['p10']:8.2f}   P90 {s['p90']:8.2f}   gain x{s['median']/t['om50']['median']:.3f}")
                    s = e["fwhm_best"]; T.append(f"  FWHM     median {s['median']:8.2f}   P10 {s['p10']:8.2f}   P90 {s['p90']:8.2f}")
                    if e["dz_mm"]: s = e["dz_mm"]; T.append(f"  dz [mm]  median {s['median']:+8.3f}   P10 {s['p10']:+8.3f}   P90 {s['p90']:+8.3f}")
                elif t["n"] == 1:
                    T.append(f"  this surface     Omega50 {e['om50_best']['median']:8.2f}   FWHM {e['fwhm_best']['median']:8.2f}" + (f"   dz {e['dz_mm']['median']:+.3f} mm" if e['dz_mm'] else ""))
                if g and t["n"] >= 1: T.append(f"  equation / theory (best focus, Omega50)  {g['F_best']/e['om50_best']['median']:.3f}")
            if g and t["n"] >= 1:
                T.append(f"  equation / theory (Omega50)  {g['F_net']/t['om50']['median']:.3f}" + ("   (single surface: scatter of devices!)" if t["n"] == 1 else ""))
        T += self._orders_table()
        self.txt.delete("1.0", "end"); self.txt.insert("1.0", "\n".join(T))

    def _orders_table(self):
        eo = self.last.get("eq_orders"); t = self.last["theory"]
        if not eo and not (t and t["all_orders"]): return []
        T = ["", "ALL ORDERS (sorted by |field angle|; x_det on the camera)"]
        th = {round(e["order"]["thx"], 10): e for e in t["orders"]} if (t and t["all_orders"]) else {}
        bf = self.bf.get()
        hdr = "   N   angle[mrad]  x_det[mm]  chord[mm]   Eq.focal" + ("  Eq.best   dz[mm]" if bf else "") + ("   Th.median" + ("  Th.best" if bf else "") if th else "")
        T.append(hdr)
        rows = eo or [dict(e["order"]) for e in t["orders"]]
        for r in rows:
            e = th.get(round(r["thx"], 10))
            s = f"  {r['N']:4d} {r['theta_out_mrad']:+11.2f} {r['x_det_mm']:+10.3f} {r['chord_mm']:10.2f}"
            s += f" {r['F_net']:10.2f}" if "F_net" in r else " " * 11
            if bf: s += (f" {r['F_best']:8.2f} {r['dz_best_mm']:+8.3f}" if ("F_best" in r and r["dz_best_mm"] is not None) else (f" {r['F_best']:8.2f}      -" if "F_best" in r else " " * 18))
            if e is not None:
                v = e["om50"]["median"] if e["n"] else e["F_clean"]["om50"]; s += f" {v:11.2f}"
                if bf: s += f" {(e['om50_best']['median'] if e['n'] else e['F_clean_best']['om50']):8.2f}"
            T.append(s)
        return T

    def _show_theory(self):
        t = self.last["theory"]; g = self.last["equation"]
        fig, ax, can = self.figs["Line"]; ax.clear()
        o = t["order"]; x = t["dlam_pm"][o]; I0 = t["I0"][o]; nrm = I0.max()
        ax.plot(x, I0 / nrm, "k-", lw=1.5, label="clean line")
        curves = [I0 / nrm]
        for i, I in enumerate(t["lines"]):
            y = np.asarray(I)[o] / nrm; curves.append(y)
            ax.plot(x, y, lw=0.9, alpha=0.45 if t["n"] > 1 else 0.9, label=f"surface {i+1}")
        for i, I in enumerate(t["lines_best"]):
            y = np.asarray(I)[o] / nrm; curves.append(y)
            ax.plot(x, y, lw=1.1, ls="--", alpha=0.6 if t["n"] > 1 else 1.0, label=f"surface {i+1}, best focus")
        if t["n"] > 1 and t["I_mean"] is not None:
            y = t["I_mean"][o] / nrm; curves.append(y)
            ax.plot(x, y, color="#c0392b", lw=2.2, label=f"ensemble average (N = {t['n']})")
        sig = np.any(np.array([c > 0.03 * c.max() for c in curves]), axis=0)
        FSR = t["equivalent"]["FSR_pm"]
        lim = max(np.abs(x[sig]).max() * 1.15 if sig.any() else 1.0, 2.5 * FSR / max(t["F_clean"]["om50"], 1))
        ax.set_xlim(-min(lim, FSR / 2), min(lim, FSR / 2))
        ax.set_xlabel("Detuning [pm]"); ax.set_ylabel("Intensity (clean peak = 1)"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
        ax.set_title("Line of one order" + (f" - ensemble of {t['n']} surfaces" if t["n"] > 1 else (" - single realisation" if t["n"] == 1 else "")))
        fig.tight_layout(); can.draw()
        fig, ax, can = self.figs["Distribution"]; ax.clear()
        if t["n"] > 1:
            F = t["per_surface"][:, 0]; bins = min(30, max(8, t["n"] // 3))
            if t["best_focus"]:
                Fb = t["orders"][0]["per_surface_best"][:, 0]; bins = np.histogram_bin_edges(np.r_[F, Fb], bins=bins)
                ax.hist(Fb, bins=bins, color="#e08e2b", alpha=0.6, label=f"best focus, median {np.median(Fb):.2f}")
                if g: ax.axvline(g["F_best"], color="#e08e2b", lw=1.5, ls="--", label=f"equation best focus {g['F_best']:.2f}")
            ax.hist(F, bins=bins, color="#4a7bb7", alpha=0.7, label="focal plane")
            ax.axvline(np.median(F), color="k", lw=1.5, label=f"theory median {np.median(F):.2f}")
            ax.axvspan(t["om50"]["p10"], t["om50"]["p90"], color="#4a7bb7", alpha=0.12, label="P10-P90")
            if g: ax.axvline(g["F_net"], color="#c0392b", lw=1.5, ls="--", label=f"equation {g['F_net']:.2f}")
            ax.axvline(t["F_clean"]["om50"], color="#777", lw=1, ls=":", label=f"clean {t['F_clean']['om50']:.2f}")
            ax.set_xlabel("Finesse (Omega50) per surface"); ax.set_ylabel("Count"); ax.legend(fontsize=8)
            ax.set_title(f"Ensemble of {t['n']} surfaces (random seed {t['seed']})")
        else:
            ax.text(0.5, 0.5, "Distribution only in ensemble mode", ha="center", va="center", transform=ax.transAxes)
        fig.tight_layout(); can.draw()
        self._show_focus(); self._show_orders(); self.show_spectrum()
        self.nb.select(2 if t["all_orders"] else (1 if t["n"] > 1 else (3 if t["best_focus"] else 0)))

    def show_spectrum(self):
        p = self.last.get("params")
        if p is None: return
        fig, ax, can = self.figs["Spectrum"]; ax.clear()
        try: d2 = float(self.dlam2.get().replace(",", "."))
        except ValueError: d2 = 0.0
        t = self.last["theory"] if self.sp_th.get() else None
        try: s = CC.spectrum(p, theory_result=t, dlam2_pm=d2, use_best=self.sp_bf.get())
        except Exception as ex:
            self.status.set("Spectrum: " + str(ex)); return
        x = s["coord"]
        ax.plot(x, s["envelope"], color="#999", lw=1, ls="--", label=f"Gaussian envelope (theta_0 = {s['theta0_mrad']:.1f} mrad" + (f", W_c = {s['W_c_mm']:.2f} mm)" if s["W_c_mm"] else ")"))
        cols = ["#2c5aa0", "#e08e2b"]
        for i, (I, wl) in enumerate(zip(s["hu"], s["wavelengths_nm"])):
            ax.plot(x, I, color=cols[i], lw=0.8, alpha=0.55 if "theory" in s else 1.0, label=f"Hu, clean, {wl:.4f} nm")
        if "theory" in s:
            for i, (I, wl) in enumerate(zip(s["theory"], s["wavelengths_nm"])):
                ax.plot(x, I, color=["#0b2a5c", "#9a4f00"][i], lw=1.1, label=f"theory ({s['theory_label']}), {wl:.4f} nm")
        ax.set_xlabel("Position on the detector [mm]" if s["coord_unit"] == "mm" else "Field angle [mrad] (enter camera focal length for mm)")
        ax.set_ylabel("Intensity (clean line on the axis = 1)"); ax.legend(fontsize=7, loc="upper right"); ax.grid(alpha=0.25)
        ax.set_title("Detector spectrum: all orders under the envelope (zoom with the toolbar)")
        fig.tight_layout(); can.draw()

    def _show_focus(self):
        t = self.last["theory"]; g = self.last["equation"]; fig, ax, can = self.figs["Focus"]; ax.clear()
        if not t or not t["best_focus"]:
            ax.text(0.5, 0.5, "Tick 'Best focus' and run the theory", ha="center", va="center", transform=ax.transAxes)
        elif t["n"] > 1:
            e = t["orders"][0]
            if e["dz_mm"]:
                dz = e["dz_per_surface"]; ax.hist(dz, bins=min(30, max(8, t["n"] // 3)), color="#4a7bb7", alpha=0.8)
                ax.axvline(np.median(dz), color="k", lw=1.5, label=f"median {np.median(dz):+.3f} mm")
                if g and g["dz_best_mm"] is not None: ax.axvline(g["dz_best_mm"], color="#c0392b", ls="--", label=f"wedge pocket {g['dz_best_mm']:+.3f} mm")
                ax.set_xlabel("Detector shift for best focus dz [mm] (behind the focal plane > 0)"); ax.set_ylabel("Count"); ax.legend(fontsize=8)
                ax.set_title("Best-focus position per device")
            else:
                ax.text(0.5, 0.5, "Enter the camera focal length for the detector shift", ha="center", va="center", transform=ax.transAxes)
        elif t["focus_scan"]:
            xs, Fs = zip(*t["focus_scan"]); ax.plot(xs, Fs, "o-", color="#2c5aa0", ms=3)
            ax.set_xlabel("Detector shift dz [mm]" if t["focus_scan_in_mm"] else "Defocus phase c [rad] (enter camera focal length for mm)")
            ax.set_ylabel("Finesse (Omega50)"); ax.grid(alpha=0.3); ax.set_title("Focus scan of this surface (central order)")
        fig.tight_layout(); can.draw()

    def _show_orders(self):
        fig, ax, can = self.figs["Orders"]; ax.clear(); eo = self.last.get("eq_orders"); t = self.last["theory"]; bf = self.bf.get()
        p = self.last["params"]; use_x = p is not None and p.f_cam_mm > 0
        key = "x_det_mm" if use_x else "theta_out_mrad"
        if eo:
            X = [r[key] for r in eo]; i = np.argsort(X)
            ax.plot(np.array(X)[i], np.array([r["F_net"] for r in eo])[i], "o-", color="#c0392b", ms=4, label="equation, focal plane")
            if bf: ax.plot(np.array(X)[i], np.array([r["F_best"] for r in eo])[i], "s--", color="#e08e2b", ms=4, label="equation, best focus")
        if t and t["all_orders"]:
            E = t["orders"]; X = np.array([e["order"][key] for e in E]); i = np.argsort(X)
            Y = np.array([e["om50"]["median"] if e["n"] else e["F_clean"]["om50"] for e in E])
            ax.plot(X[i], Y[i], "o-", color="#2c5aa0", ms=4, label="theory median, focal plane" if t["n"] else "theory clean")
            if t["n"] > 1:
                ax.fill_between(X[i], np.array([e["om50"]["p10"] for e in E])[i], np.array([e["om50"]["p90"] for e in E])[i], color="#2c5aa0", alpha=0.12, label="theory P10-P90")
            if t["best_focus"]:
                Yb = np.array([e["om50_best"]["median"] if e["n"] else e["F_clean_best"]["om50"] for e in E])
                ax.plot(X[i], Yb[i], "s:", color="#2c5aa0", ms=4, label="theory, best focus")
        if not eo and not (t and t["all_orders"]):
            ax.text(0.5, 0.5, "Tick 'All orders' and press Equation or Theory", ha="center", va="center", transform=ax.transAxes)
        else:
            ax.set_xlabel("Position on the detector x_det [mm]" if use_x else "Field angle of the order [mrad] (outside)")
            ax.set_ylabel("Finesse (Omega50)"); ax.grid(alpha=0.3); ax.legend(fontsize=8); ax.set_title("Finesse of all orders")
        fig.tight_layout(); can.draw()

    def _show_series(self):
        R = self.last["series"]; fig, ax, can = self.figs["Length series"]; ax.clear()
        L = [r["L_mm"] for r in R]
        ax.plot(L, [r["F_eq_net"] for r in R], "o-", color="#c0392b", label="equation, net (Omega50)")
        ax.plot(L, [r["F_eq_clean"] for r in R], "s--", color="#777", ms=4, label="equation, clean")
        if self.bf.get(): ax.plot(L, [r["F_eq_best"] for r in R], "o--", color="#e08e2b", ms=4, label="equation, best focus")
        if "F_th_best" in R[0]: ax.plot(L, [r["F_th_best"] for r in R], "s:", color="#e08e2b", ms=4, label="theory median, best focus")
        if "F_th_om50" in R[0]:
            ax.plot(L, [r["F_th_om50"] for r in R], "o-", color="#2c5aa0", label="theory median (Omega50)")
            ax.plot(L, [r["F_th_fwhm"] for r in R], "^:", color="#2c5aa0", ms=4, label="theory median (FWHM)")
            ax.plot(L, [r["F_th_clean"] for r in R], "s:", color="#9bb3d6", ms=4, label="theory, clean (Omega50)")
        ax.plot(L, [r["F_static_fwhm"] for r in R], "d-.", color="#8e6c00", ms=4, label="static model (LightMachinery style, FWHM)")
        ax.set_xlabel("Length L [mm]"); ax.set_ylabel("Finesse"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
        fig.tight_layout(); can.draw(); self.nb.select(4)

    # ------------------------------------------------------------ files
    def save_csv(self):
        fn = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if not fn: return
        p = self.last["params"]; g = self.last["equation"]; t = self.last["theory"]; R = self.last["series"]
        with open(fn, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, delimiter=";")
            if p:
                w.writerow(["# inputs"])
                for k, v in asdict(p).items(): w.writerow([k, v])
            if g:
                w.writerow(["# equation"])
                for k in ("F_net", "F_clean", "loss", "dlam_pm", "resolving_power"): w.writerow([k, g[k]])
                w.writerow(["static_F_fwhm", g["static"]["F_net"]])
                for k, v in g["equivalent"].items(): w.writerow([k, v])
            if g and g.get("F_best") is not None:
                w.writerow(["F_best_focus_equation", g["F_best"]]); w.writerow(["dz_best_mm_equation", g["dz_best_mm"]])
            eo = self.last.get("eq_orders")
            if eo:
                w.writerow(["# equation, all orders"]); keys = ["N", "theta_out_mrad", "x_det_mm", "chord_mm", "F_net", "F_best", "dz_best_mm"]
                w.writerow(keys); [w.writerow([r[k] for k in keys]) for r in eo]
            if t:
                for e in t["orders"]:
                    o = e["order"]
                    w.writerow([f"# theory, order N={o['N']}, angle {o['theta_out_mrad']:+.3f} mrad, x_det {o['x_det_mm']:+.3f} mm, random seed {t['seed']}"])
                    w.writerow(["clean_om50", e["F_clean"]["om50"], "clean_fwhm", e["F_clean"]["fwhm"]])
                    if e["n"] > 0:
                        cols = ["surface", "F_om50", "F_fwhm", "F_eq"] + (["F_om50_best", "F_fwhm_best", "F_eq_best", "dz_mm"] if t["best_focus"] else [])
                        w.writerow(cols)
                        for i, row in enumerate(e["per_surface"]):
                            extra = list(e["per_surface_best"][i]) + [e["dz_per_surface"][i] if len(e["dz_per_surface"]) else ""] if t["best_focus"] else []
                            w.writerow([i, *row, *extra])
            if R:
                w.writerow(["# length series"]); w.writerow(list(R[0].keys()))
                for r in R: w.writerow(list(r.values()))
        self.status.set("saved: " + fn)

    def save_json(self):
        fn = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON", "*.json")])
        if fn:
            with open(fn, "w", encoding="utf-8") as f: json.dump(asdict(self.params()), f, indent=1)

    def load_json(self):
        fn = filedialog.askopenfilename(filetypes=[("JSON", "*.json")])
        if not fn: return
        with open(fn, encoding="utf-8") as f: d = json.load(f)
        for k, v in d.items():
            if k in self.vars: self.vars[k].set(_fmt(v))
            elif k in self.maps: self.maps[k].set(v)
        self.run_equation()


def _fmt(v):
    return f"{v:g}" if isinstance(v, (int, float)) else str(v)


class _Tip:
    def __init__(self, w, text):
        self.w, self.text, self.tw = w, text, None
        w.bind("<Enter>", self.show); w.bind("<Leave>", self.hide)
    def show(self, ev=None):
        x, y = self.w.winfo_rootx() + 20, self.w.winfo_rooty() + 22
        self.tw = tk.Toplevel(self.w); self.tw.wm_overrideredirect(True); self.tw.geometry(f"+{x}+{y}")
        tk.Label(self.tw, text=self.text, background="#ffffe0", relief="solid", borderwidth=1, font=("Arial", 9)).pack()
    def hide(self, ev=None):
        if self.tw: self.tw.destroy(); self.tw = None


def main():
    Calculator().mainloop()
