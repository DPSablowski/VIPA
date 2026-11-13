"""Self-test of the core (no GUI):  python start_calculator.py --test
Checks: (1) clean equation against Airy for a long plate, (2) solid VIPA = air-spaced equivalent (equation and theory),
(3) source width as a fraction of the FSR gives the same finesse for glass and air, (4) equation against theory for the
standard VIPA, (5) static model against the LightMachinery outputs of 2026-10-18, (6) best-focus shift, (7) orders, (8) detector spectrum against Hu."""
import time
import numpy as np
import calculator_core as CC


def check(name, ok, text):
    print(f"[{'ok' if ok else 'FAILED'}] {name}: {text}"); return ok


def main():
    t0 = time.time(); allok = True
    p = CC.Params(R2_pct=90, L_mm=60, theta_deg=4)
    g = CC.equation(p); a = g["equivalent"]
    allok &= check("Airy", abs(g["F_clean"] / a["F_Airy"] - 1) < 0.02, f"equation clean {g['F_clean']:.2f}, Airy {a['F_Airy']:.2f}")
    pg = CC.Params(n_gap=1.457, wl_nm=630, theta_deg=4, R2_pct=92, L_mm=20, s_mid_nm=8, s_fig_nm=10, wedge_as=0.5)
    ag = CC.equivalent(pg)
    pl = pg.copy(n_gap=1.0, wl_nm=630 / 1.457, theta_deg=ag["th_in_deg"], f_cyl_mm=pg.f_cyl_mm * 1.457)   # divergence inside: / n'
    g1, g2 = CC.equation(pg)["F_net"], CC.equation(pl)["F_net"]
    allok &= check("glass = air equivalent (equation)", abs(g1 / g2 - 1) < 1e-6, f"{g1:.4f} vs {g2:.4f}")
    t1, t2 = CC.theory(pg, n=3, seed=5), CC.theory(pl, n=3, seed=5)
    d = np.max(np.abs(t1["per_surface"] - t2["per_surface"]))
    allok &= check("glass = air equivalent (theory)", d < 1e-5, f"max. difference {d:.2e}")
    q = 0.05
    pgq = pg.copy(s_mid_nm=0, s_fig_nm=0, wedge_as=0); pgq = pgq.copy(src_gauss_pm=q * CC.equivalent(pgq)["FSR_pm"])
    plq = pl.copy(s_mid_nm=0, s_fig_nm=0, wedge_as=0); plq = plq.copy(src_gauss_pm=q * CC.equivalent(plq)["FSR_pm"])
    f1, f2 = CC.equation(pgq)["F_net"], CC.equation(plq)["F_net"]
    allok &= check("source relative to FSR", abs(f1 / f2 - 1) < 1e-6, f"glass {f1:.3f}, air {f2:.3f}")
    ps = CC.Params(s_mid_nm=10.0)
    gs = CC.equation(ps)["F_net"]; ts = CC.theory(ps, n=40, seed=1, processes=1)
    r = gs / ts["om50"]["median"]
    allok &= check("equation vs theory (standard VIPA, 10 nm, N = 40)", abs(r - 1) < 0.15,
                   f"equation {gs:.2f}, theory median {ts['om50']['median']:.2f} (P10 {ts['om50']['p10']:.1f}, P90 {ts['om50']['p90']:.1f}), ratio {r:.3f}")
    lm = {10: 42.9, 20: 50.4, 40: 52.9}
    st = {L: CC.static_model(CC.Params(wl_nm=532, R2_pct=99, h_mm=3.43, theta_deg=1.5, L_mm=L, s_mid_nm=2))["F_net"] for L in lm}
    dev = max(abs(st[L] / lm[L] - 1) for L in lm)
    allok &= check("static model vs LightMachinery (air, 532 nm, R2 99 %, 2 nm)", dev < 0.01,
                   ", ".join(f"L={L}: {st[L]:.1f} ({lm[L]})" for L in lm))
    pw = CC.Params(wedge_as=0.5, f_cam_mm=200)
    dz200, dz75 = CC.equation(pw)["dz_best_mm"], CC.equation(pw.copy(f_cam_mm=75))["dz_best_mm"]
    allok &= check("best-focus shift vs main document (0.5 arcsec: 0.7 mm at f = 200 mm, 0.1 mm at 75 mm)",
                   abs(dz200 - 0.70) < 0.01 and abs(dz75 - 0.10) < 0.005, f"{dz200:.3f} mm, {dz75:.3f} mm")
    tc = CC.theory(pw, n=0, best_focus=True)["orders"][0]
    allok &= check("theory best focus of the clean line: same shift, finesse back to the unwedged value",
                   abs(tc["dz_clean_mm"] / dz200 - 1) < 0.02 and abs(tc["F_clean_best"]["om50"] / CC.theory(CC.Params(), n=0)["F_clean"]["om50"] - 1) < 0.01,
                   f"dz {tc['dz_clean_mm']:.3f} mm, F {tc['F_clean_best']['om50']:.2f}")
    od = CC.orders(CC.Params())
    allok &= check("orders in the divergence cone", len(od) > 3 and abs(od[0]["thx"]) <= min(abs(o["thx"]) for o in od),
                   f"{len(od)} orders, central at {od[0]['theta_out_mrad']:+.2f} mrad")
    ps_ = CC.Params(f_cam_mm=200); Tc = CC.theory(ps_, n=0, all_orders=True, max_orders=5)
    sp = CC.spectrum(ps_, theory_result=Tc, npts=400000); xm = np.abs(sp["coord"]) < 1.3
    H, Th = sp["hu"][0][xm], sp["theory"][0][xm]; rms = np.sqrt(np.mean((H - Th) ** 2)) / H.max()
    allok &= check("detector spectrum: clean theory lines under the envelope = Hu Eq. (9)-(10)", rms < 0.01, f"rms difference {rms*100:.2f} % of the peak")
    import tempfile, os, surface_map as SM
    pz = CC.Params(L_mm=20); fl = os.path.join(tempfile.gettempdir(), "vipa_flat_test.vmap.npz")
    SM.PlateMap(np.zeros((40, 300)), 0.1, -15.0, -2.0).save(fl)
    tz = CC.theory(pz.copy(map_hr=fl, map_pr=fl), n=1); Fc = tz["F_clean"]["om50"]
    d = abs(float(tz["per_surface"][0, 0]) / Fc - 1)
    allok &= check("flat surface map = clean line", d < 1e-3, f"relative difference {d:.1e}")
    rr = np.random.default_rng(1); hmap = rr.normal(0, 10, (40, 300)); SM.PlateMap(hmap, 0.1, -15.0, -2.0).save(fl + "2.npz")
    pm = SM.load_plate_map(fl + "2.npz"); xs = -15e-3 + 0.1e-3 * np.array([3.0, 17.0]); ys = -2e-3 + 0.1e-3 * np.array([5.0, 9.0])
    allok &= check("plate map interpolation at grid points", np.allclose(pm.h(xs, ys), [hmap[5, 3] * 1e-9, hmap[9, 17] * 1e-9]), "bilinear = grid value")
    print(f"Duration {time.time() - t0:.0f} s. " + ("All checks passed." if allok else "At least one check failed."))
    return 0 if allok else 1


if __name__ == "__main__":
    raise SystemExit(main())
