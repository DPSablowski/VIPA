# VIPA Finesse Calculator

The calculator computes the finesse of a VIPA from one set of inputs in two ways:

- **Equation**: the closed-form equation (overlap measure Ω₅₀). It returns the net finesse of the median plate in a fraction of a second.
- **Theory**: the exact line (round-trip pupil, incoherent across the chord) for surfaces drawn from the three-band surface model. Two modes:
  - **Single realisation**: one device.
  - **Ensemble**: N devices. The calculator then reports the median, P10–P90 and mean, shows a histogram (tab *Distribution*, opened automatically) and plots the ensemble-average line together with five example surfaces (tab *Line*). The theory gives Ω₅₀, FWHM and F_eq.

Two options apply to both routes:

- **Best focus** (tick box): the detector is moved behind the focal plane until the line is sharpest.
  - Equation: the best-focus equation (wedge and mean random chirp compensated).
  - Theory: for every device the defocus that maximises Ω₅₀ is searched.
  - With the camera focal length f, the detector shift is dz = −2f²c/(k₀v²), where c is the defocus phase per round trip squared and v = Δx'·cosΘ is the transverse spacing of the virtual sources (air: v = 2h·sinΘ). For a 0.5″ wedge the shift is 0.7 mm at f = 200 mm, 0.1 mm at 75 mm.
  - The equation also gives the spread of dz over the plates. Tab *Focus* shows dz per device (ensemble) or the focus scan (single realisation).
- **All orders** (tick box): every order inside the divergence cone of the cylinder lens is computed, each with its own field angle, chord (round beam) and round-trip offset.
  - The theory uses the same plates in every order.
  - Results are listed in a table, and tab *Orders* plots the finesse against the position on the detector, with the camera focal length.
  - The cost is orders × surfaces lines; best focus needs about 20 line evaluations per surface.

**Detector spectrum** (tab *Spectrum*, panel *Detector spectrum*): the intensity on the detector over the whole field after Hu et al. (2015), Eqs. (9)–(10):

    I(θx) = exp(−2θx²/θ₀²) · (1 − 2x·cos(Mδ) + x²)/(1 − 2r·cos δ + r²)

with these quantities:
- δ(θx) = 2k₀h·√(n'² − sin²(θ+θx)), r = √(R1R2), x = r^M, M = L/(2h·tan θ'(θx));
- θ₀ = (beam/2)/f_cyl, the 1/e² half angle of the Gaussian beam behind the cylinder lens;
- on the detector the envelope width is W_c = f_cam·θ₀.

Options and behaviour:
- A second wavelength can be added (offset in pm).
- After a theory run, the exact lines (surface 1, or the clean line) are overlaid under the same envelope. They are taken per order if *All orders* was ticked, otherwise the central-order line is used for all orders; optionally the best-focus lines are shown.
- Without surface errors the theory lines coincide with Hu's to 0.2 % rms (self-test).
- The envelope is Hu's Gaussian beam, whereas the theory computes the line of each order with a round top-hat beam (chord). A theory of the envelope itself is still open.

For comparison it also shows the **static model in the style of the LightMachinery VIPA designer**. This model truncates the Hu sum at L/Δx reflections, convolves it with a static Gaussian thickness distribution and measures the FWHM. It reproduces the LightMachinery outputs to better than 1 %.

## Start

    python start_calculator.py          # GUI
    python start_calculator.py --test   # self-test without GUI (about 40 s, 10 checks)

Requirements: Python 3 with tkinter (included in the standard Windows installer), numpy, scipy, matplotlib and pandas.

## Use

- **Inputs** are in the left column; Enter recomputes the equation. *Gap material* sets n' and n_g from Sellmeier data (fused silica, N-BK7, CaF₂) at the chosen wavelength.
- **Random seed**: the start value of the random-number generator that draws the surfaces. The same seed gives the same surfaces, so runs are reproducible and comparable.
- **Theory, ensemble mode**: the N surfaces are distributed over several processes. One surface takes about 0.3–4 s per process. *Cancel* stops after the current surface.
- **Length series**: the equation and the static model versus L. With *theory surfaces per point* > 0, the theory median is computed as well.
- **Saving**: *Save results (CSV)* writes the inputs, the characteristics, the theory per surface and the length series. The inputs can be saved and loaded as JSON.

## Inputs

**Surfaces.** The thickness variation of the gap (RMS) is split into three bands:
- figure: smooth over the plate;
- mid band: a power law between 1/D_m and 1/0.1 mm with exponent α_s;
- fine band: structures below 0.1 mm, uncorrelated from reflection to reflection.

The figure RMS refers to the whole plate of diameter D_m (D_m = 0: derived from L).

**Band convention** (radio buttons, or `Params.bands`). *Stationary*: figure and mid band are the independent
components of the model, the mid band being a stationary power-law field whose RMS refers to the disk D_m; this is
the convention of the theory grids and of the closed-form equation. *Measured*: figure = RMS of a least-squares
Zernike fit of degrees 2–4 of a height map, mid band = RMS of the residual less the fine band, i.e. what an
interferometer delivers. In the measured convention the theory draws surfaces with the mid band orthogonal to the
figure terms (projected generator), and the equation converts the inputs with Noll's shares
(`flaeche_dreiband.measured_to_stationary`). The two conventions differ: at the same nominal mid-band
RMS the measured convention gives 8–24 % lower finesse, so measured values must not be entered as stationary ones.

**Input mirror R1 < 1.** The calculation uses R1·R2 per round trip.

**Solid VIPA (n' > 1).** Inside the plate the problem is exactly an air-spaced VIPA with wavelength λ/n' and the internal angle. Quantities defined outside the plate are converted:
- source widths: n_g/n'²
- pixel angle: 1/n'
- divergence: 1/n'

The self-test checks this against the air-spaced equivalent.

Not included:
- dispersion other than through n_g
- a rough exit surface
- the AR-HR transition

**Temperature.** dδ/dT = 2k₀h·[CTE·Q + n'·dn'/dT / Q] with Q = √(n'² − sin²θ). It enters as a Gaussian of FWHM 2.355·|dδ/dT|·σ_T.

## Accuracy

**Equation.** 2.7 % mean absolute deviation on the full grid. The calculator lists the notes that apply below the result. Two cases to watch:
- predicted finesse below 5: about 15 %;
- edge-limited lines with broadening: up to about 4 % pessimistic.

**Theory, ensemble mode.** Limited by the statistical error of the median: about ±5 % for 50 surfaces.

## Measured surface maps

The GUI frame *Measured surface maps* takes one map per surface (HR, PR). *Prepare...* opens the map window:
load a CODE V `.int` file (grid phase map, e.g. from a Zygo export), enter the aperture diameter or pixel size (the file has no lateral scale),
choose the strip (length, width, centre, angle), remove piston and tilt (the alignment of the part, not the wedge), and save a plate map
(`.vmap.npz`). The calculator then computes the plate with these maps added to the model surface; with maps only, one plate (a measured plate is one plate).
Height factor: 0.5 for a reflected wavefront, -1 to change the sign. A strip that leaves the valid area is refused. The wedge stays a separate input.
The equation route ignores maps. In scripts: `surface_map.read_int`, `prepare`, `load_plate_map`; `Params(map_hr=..., map_pr=...)`.

## Files

- `surface_map.py`, `surface_map_gui.py`: reading, preparing and the window for surface maps

- `start_calculator.py`: start script
- `calculator_gui.py`: GUI
- `calculator_core.py`: computational core, also usable without the GUI
- `materials.py`: Sellmeier data
- `test_calculator.py`: self-test
- `kernel/`: modules of the theory and the equation (English comments; each module docstring describes what it does)

The core can also be used without the GUI:

    import calculator_core as CC
    p = CC.Params(R2_pct=95, L_mm=20, s_mid_nm=10)
    CC.equation(p)["F_net"]; CC.theory(p, n=50, processes=8)["om50"]["median"]
